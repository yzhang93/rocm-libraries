#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Tune GEMM sizes with Geko and add the kernels that win to a user kernel library.

Takes the same config YAML as `geko --tune --list` (see
utilities/geko/geko/config_generator/config.yaml) and runs:

  1. tune      geko --tune on the sizes in the config
  2. measure   for every size Geko found a faster kernel for, register Geko's
               payload in a scratch copy of the library and time, through the
               hipBLASLt Python bindings, the tuned kernel against what
               hipBLASLt selects today: the shipped kernel, or a user kernel an
               earlier run already added for that size
  3. add       sizes whose tuned kernel is correct and at least --min-speedup
               faster than today's selection are registered and mapped into
               --library-root; nothing else is written there
  4. confirm   a fresh process opens the library, refreshes it, and reports
               what heuristic() now selects for every size

Any process that opens the same library and calls refresh gets the new kernels
from then on. Rerunning with the same library is safe: a size is only
remapped when the new kernel beats the one already there.

With --replay nothing is tuned and the library is only read: a fresh process
opens it, calls refresh, and for every size reports whether heuristic() returns
a user kernel, how fast it is against the shipped kernel, and whether their
outputs agree. The sizes are the config's if one is given, otherwise every size
the library maps (its data type and transposes are found by probing).

Usage:
    python tune_and_register.py CONFIG.yaml [--library-root DIR] [--devices 0,1,2,3]
    python tune_and_register.py [CONFIG.yaml] --replay --library-root DIR
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

DEFAULT_HIPBLASLT = Path(__file__).resolve().parents[2]

# hipBLASLt dtype names used by Geko -> (hipDataType name, numpy dtype name).
DTYPES = {"bf16_r": ("R_16BF", "bfloat16"), "f16_r": ("R_16F", "float16"),
          "f32_r": ("R_32F", "float32")}

# (A/B type, C/D type) pairs tried when --replay has to recover a journalled
# size's type: E records carry only M, N, batch and K.
PROBE_TYPES = [("bf16_r", "bf16_r"), ("f16_r", "f16_r"), ("f32_r", "f32_r"),
               ("bf16_r", "f32_r"), ("f16_r", "f32_r")]


def size_key(trans, m, n, batch, k, a_type, c_type, compute_type):
    return f"{trans}:{a_type}:{c_type}:{compute_type}:{m}x{n}x{batch}x{k}"


def label(e):
    return f"M={e['m']} N={e['n']} B={e['batch']} K={e['k']} {e['trans']} {e['a_type']}"


# ---------------------------------------------------------------------------
# Orchestration (runs in the plain venv environment)
# ---------------------------------------------------------------------------

class Env:
    def __init__(self, args):
        self.hipblaslt = args.hipblaslt.resolve()
        self.rocm = Path(args.rocm_path)
        self.python = Path(sys.executable)
        self.geko = self.python.parent / "geko"
        self.install_lib = self.hipblaslt / "hipblaslt-install/lib"

        base = dict(os.environ)
        # LD_LIBRARY_PATH at the install tree makes Geko's tensilelite client
        # load a host library that cannot read the YAML Geko writes; an
        # inherited HIP_VISIBLE_DEVICES remaps the GPU ids Geko is given; a
        # leftover HIPBLASLT_TENSILE_LIBPATH redirects the shipped baseline.
        for var in ("LD_LIBRARY_PATH", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES",
                    "CUDA_VISIBLE_DEVICES", "HIPBLASLT_TENSILE_LIBPATH"):
            base.pop(var, None)
        base["ROCM_PATH"] = str(self.rocm)
        base["PATH"] = f"{self.python.parent}:{self.rocm}/bin:{base.get('PATH', '')}"
        self.tune = base

        self.py = dict(base)
        self.py["LD_LIBRARY_PATH"] = f"{self.install_lib}:{self.rocm}/lib"
        self.py["HIP_VISIBLE_DEVICES"] = str(args.eval_gpu)


def step(n, title):
    print(f"\n[{n}/4] {title}", flush=True)


def check_setup(env, library_root, need_geko=True):
    problems = []
    if need_geko and not env.geko.is_file():
        problems.append(f"geko not found at {env.geko} (pip install -e utilities/geko)")
    if not (env.install_lib / "libhipblaslt.so").exists():
        problems.append(f"no hipBLASLt install at {env.install_lib}")
    cache = env.hipblaslt / "build/release/CMakeCache.txt"
    if cache.is_file():
        for line in cache.read_text().splitlines():
            if line.startswith("TENSILELITE_LOGIC_FILTER:") and line.split("=", 1)[1]:
                problems.append(
                    f"{cache} has {line.split(':', 1)[0]}={line.split('=', 1)[1]!r}, so the "
                    "shipped kernel library is incomplete; reconfigure with "
                    "-DTENSILELITE_LOGIC_FILTER= and rebuild + install")
    for d in (library_root, library_root / "objects"):
        if d.exists() and d.stat().st_mode & 0o022:
            problems.append(f"{d} is group- or world-writable; hipBLASLt refuses to load "
                            f"from it (chmod go-w {d})")
    return problems


def requested_sizes(config, arch):
    """Every size the config asks for, via Geko's own loader."""
    from geko.config_generator.load_input_config import load_prepared_config_from_yaml
    prepared = load_prepared_config_from_yaml(config_path=config, arch=arch)
    out = {}
    for gc in prepared["GemmProblems"]:
        t = gc.gemm_type
        trans = f"{t.transA}{t.transB}"
        for m, n, b, k in gc.sizes:
            e = {"trans": trans, "m": m, "n": n, "batch": b, "k": k, "a_type": t.a_type,
                 "b_type": t.b_type, "c_type": t.c_type, "compute_type": t.compute_type}
            out.setdefault(size_key(trans, m, n, b, k, t.a_type, t.c_type, t.compute_type), e)
    return out


def run_geko(env, args, config, arch, geko_dir, log):
    shutil.rmtree(geko_dir, ignore_errors=True)
    cmd = [env.geko, "--tune", "--arch", arch, "--hipblaslt", env.hipblaslt,
           "--list", config, "--devices", args.devices, "--workdir", geko_dir]
    print(f"      {' '.join(str(c) for c in cmd)}")
    print(f"      log: {log}", flush=True)
    t0 = time.time()
    # Geko builds its tensilelite client under cwd/build_tmp, keyed by git
    # hash, and may leave other files in cwd; a fixed directory in the build
    # tree lets every run after the first reuse the client.
    geko_cwd = env.hipblaslt / "build/geko"
    geko_cwd.mkdir(parents=True, exist_ok=True)
    with open(log, "w") as f:
        proc = subprocess.Popen([str(c) for c in cmd], env=env.tune, cwd=geko_cwd,
                                stdout=f, stderr=subprocess.STDOUT)
        while proc.poll() is None:
            time.sleep(5)
            if int(time.time() - t0) % 120 < 5:
                print(f"      ... still tuning ({time.time() - t0:.0f}s)", flush=True)
    print(f"      geko finished in {time.time() - t0:.0f}s (exit {proc.returncode})")
    return proc.returncode


def read_geko(geko_dir):
    """Geko's per-size verdicts, keyed like requested_sizes(), and its payloads."""
    raw = geko_dir / "results/raw_results.csv"
    if not raw.is_file():
        return None, []

    def key(r):
        return size_key(r["transA"] + r["transB"], int(r["m"]), int(r["n"]),
                        int(r["batch_count"]), int(r["k"]), r["a_type"], r["c_type"],
                        r["compute_type"])

    final = geko_dir / "results/final_results.csv"
    kept = {key(r) for r in csv.DictReader(open(final))} if final.is_file() else set()
    rows = {}
    for r in csv.DictReader(open(raw)):
        rows[key(r)] = {
            "shipped_us": float(r["us_reference"]), "tuned_us": float(r["us_tuned"]),
            "uplift_pct": float(r["uplift_pct"]), "kept": key(r) in kept,
            "kernel": r["kernel_tuned"], "shipped_source": r["lib_source_reference"],
        }
    payloads = sorted(str(p)[: -len(".dat.zlib")] for p in
                      (geko_dir / "build/library").glob("*/TensileLibrary_*_Contraction_*.dat.zlib"))
    return rows, payloads


def run_worker(env, stage, plan, workdir):
    """Run one hipBLASLt stage in a child process with the install on LD_LIBRARY_PATH."""
    plan_file = workdir / f"{stage}_in.json"
    out_file = workdir / f"{stage}_out.json"
    plan_file.write_text(json.dumps(plan, indent=2))
    out_file.unlink(missing_ok=True)
    log = workdir / f"{stage}.log"
    with open(log, "w") as f:
        proc = subprocess.Popen([str(env.python), __file__, "--_stage", stage, "--_plan",
                                 str(plan_file), "--_out", str(out_file)],
                                env=env.py, stdout=subprocess.PIPE, stderr=f, text=True)
        for line in proc.stdout:
            print(line, end="", flush=True)
            f.write(line)
        proc.wait()
    if proc.returncode != 0 or not out_file.exists():
        tail = log.read_text().strip().splitlines()[-5:]
        print(f"      ERROR: {stage} failed (exit {proc.returncode}); see {log}")
        for line in tail:
            print(f"        {line}")
        return None
    return json.loads(out_file.read_text())


def scratch_copy(library_root, scratch):
    """A throwaway copy of the library, so measuring never writes to the real one."""
    shutil.rmtree(scratch, ignore_errors=True)
    if (library_root / "registry.log").is_file():
        shutil.copytree(library_root, scratch)
    else:
        (scratch / "objects").mkdir(parents=True)
    for d in (scratch, scratch / "objects"):
        d.chmod(0o755)


def report(entries, workdir, dry_run):
    fmt = lambda v, f="{:.2f}": f.format(v) if isinstance(v, (int, float)) else "-"
    print("\n" + "=" * 118)
    print(f"{'size':40} {'Geko':>8} {'shipped':>9} {'tuned':>9} {'tuned vs':>9}  "
          f"{'before':13} {'now':12} result")
    print(f"{'':40} {'uplift':>8} {'us':>9} {'us':>9} {'shipped':>9}  "
          f"{'this run':13} {'selected':12}")
    for e in entries:
        g, x = e.get("geko") or {}, e.get("eval") or {}
        before = "-" if not x.get("current") else (
            f"user {x['current_us']:.2f}" if x["current"] == "user kernel" else "shipped")
        print(f"{label(e):40} {fmt(g.get('uplift_pct'), '{:+.1f}%'):>8} "
              f"{fmt(x.get('shipped_us')):>9} {fmt(x.get('tuned_us')):>9} "
              f"{fmt(x.get('speedup_vs_shipped'), '{:.2f}x'):>9}  "
              f"{before:13} {e.get('selected', '-'):12} {e['result']}")
    print("times are wall-clock us/call through the Python bindings; each call includes a "
          "fixed host sync,\nso ratios understate the kernel-only speedup. 'before this run' is "
          "what a tuned kernel has to beat:\nthe shipped kernel, or a user kernel an earlier "
          "run added (with its time).")

    counts = {}
    for e in entries:
        counts[e["outcome"]] = counts.get(e["outcome"], 0) + 1
    order = ["added", "would add", "kept", "rejected", "skipped", "error"]
    print("\n" + ", ".join(f"{counts[o]} {o}" for o in order if o in counts)
          + f"  (of {len(entries)} sizes)" + ("  [dry run: library not modified]" if dry_run else ""))

    flat = []
    for e in entries:
        g, x = e.get("geko") or {}, e.get("eval") or {}
        flat.append({
            "trans": e["trans"], "m": e["m"], "n": e["n"], "batch": e["batch"], "k": e["k"],
            "a_type": e["a_type"], "c_type": e["c_type"], "compute_type": e["compute_type"],
            "geko_shipped_us": g.get("shipped_us"), "geko_tuned_us": g.get("tuned_us"),
            "geko_uplift_pct": g.get("uplift_pct"), "geko_kept": g.get("kept"),
            "shipped_us": x.get("shipped_us"), "current": x.get("current"),
            "current_us": x.get("current_us"), "tuned_us": x.get("tuned_us"),
            "speedup_vs_current": x.get("speedup_vs_current"),
            "speedup_vs_shipped": x.get("speedup_vs_shipped"), "rel_diff": x.get("rel_diff"),
            "outcome": e["outcome"], "result": e["result"], "selected": e.get("selected"),
            "tuned_kernel": x.get("tuned_kernel") or g.get("kernel"),
        })
    with open(workdir / "report.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(flat[0]))
        w.writeheader()
        w.writerows(flat)
    (workdir / "report.json").write_text(json.dumps(flat, indent=2))
    print(f"report: {workdir / 'report.csv'}")


def skip_reason(e):
    return (f"skipped: Geko found nothing faster than shipped "
            f"({e['geko']['uplift_pct']:+.1f}%)")


def journal_sizes(library_root):
    """(m, n, batch, k) of every E record, in first-mapped order."""
    out = []
    for line in (library_root / "registry.log").read_text().splitlines():
        f = line.split()
        if f[:1] == ["E"] and len(f) >= 7:
            dims = tuple(int(x) for x in f[3:7])
            if dims not in out:
                out.append(dims)
    return out


TENSILE_TYPES = {"BFloat16": "bf16_r", "Half": "f16_r", "Float": "f32_r"}
SIZE_PROPS = {"FreeSizeA": "m", "FreeSizeB": "n", "BatchSize": "batch", "BoundSize": "k"}


def payload_sizes(library_root):
    """Every size a stored payload was tuned for, with its GEMM type.

    Geko's payloads carry a matching table keyed by [M, N, batch, K] and each
    kernel's problemType, so this covers sizes whose kernel was stored but
    never mapped. Payloads that cannot be read are skipped.
    """
    import zlib
    import msgpack
    hashes = []
    for line in (library_root / "registry.log").read_text().splitlines():
        f = line.split()
        if f[:1] == ["R"] and len(f) >= 3 and f[1] not in hashes:
            hashes.append(f[1])

    out = {}
    for dh in hashes:
        base = library_root / "objects" / f"{dh}.dat"
        try:
            raw = zlib.decompress(Path(f"{base}.zlib").read_bytes()) \
                if Path(f"{base}.zlib").exists() else base.read_bytes()
            lib = msgpack.unpackb(raw, raw=False, strict_map_key=False)
        except Exception:
            continue
        ptype = {s["index"]: s.get("problemType", {}) for s in lib.get("solutions", [])}

        def tables(x):
            if isinstance(x, dict):
                if "table" in x and "properties" in x:
                    yield x
                for v in x.values():
                    yield from tables(v)
            elif isinstance(x, list):
                for v in x:
                    yield from tables(v)

        for t in tables(lib.get("library", {})):
            names = [SIZE_PROPS.get(p.get("type")) for p in t["properties"]]
            if sorted(filter(None, names)) != sorted(SIZE_PROPS.values()):
                continue
            for row in t["table"]:
                p = ptype.get(row.get("index"), {})
                d = dict(zip(names, row["key"]))
                a, cc = TENSILE_TYPES.get(p.get("aType")), TENSILE_TYPES.get(p.get("dType"))
                comp = TENSILE_TYPES.get(p.get("computeType"))
                if not (a and cc and comp):
                    continue
                e = {"trans": ("T" if p.get("transA") else "N") + ("T" if p.get("transB") else "N"),
                     "m": d["m"], "n": d["n"], "batch": d["batch"], "k": d["k"],
                     "a_type": a, "b_type": TENSILE_TYPES.get(p.get("bType"), a),
                     "c_type": cc, "compute_type": comp}
                out.setdefault(size_key(e["trans"], e["m"], e["n"], e["batch"], e["k"],
                                        a, cc, comp), e)
    return out


def replay(args, env, config, arch, library_root, workdir):
    if not (library_root / "registry.log").is_file():
        return f"{library_root} has no registry.log; nothing to replay"
    mapped = journal_sizes(library_root)
    tuned = {} if config else payload_sizes(library_root)
    print(f"library   {library_root} ({len(mapped)} mapped sizes"
          + ("" if config else f", tuned kernels for {len(tuned)} sizes") + ")")
    print(f"workdir   {workdir}")
    t0 = time.time()

    groups, extra = [], []
    if config:
        sizes = requested_sizes(config, arch)
        print(f"config    {config} ({len(sizes)} sizes)")
        for key, e in sizes.items():
            dims = (e["m"], e["n"], e["batch"], e["k"])
            g = {"key": key, "dims": dims, "mapped": dims in mapped, "variants": [dict(e, key=key)]}
            if e["a_type"] not in DTYPES or e["c_type"] not in DTYPES or e["compute_type"] != "f32_r":
                g["result"] = (f"skipped: {e['a_type']}/{e['c_type']}/{e['compute_type']} "
                               "is not supported by this script")
            groups.append(g)
        extra = [d for d in mapped if d not in {g["dims"] for g in groups}]
    else:
        for key, e in tuned.items():
            dims = (e["m"], e["n"], e["batch"], e["k"])
            groups.append({"key": key, "dims": dims, "mapped": dims in mapped, "stored": True,
                           "variants": [dict(e, key=key)]})
        covered = {g["dims"] for g in groups}
        for m, n, b, k in (d for d in mapped if d not in covered):
            variants = [{"trans": ta + tb, "m": m, "n": n, "batch": b, "k": k, "a_type": a,
                         "b_type": a, "c_type": cc, "compute_type": "f32_r"}
                        for ta in "NT" for tb in "NT" for a, cc in PROBE_TYPES]
            groups.append({"key": f"{m}x{n}x{b}x{k}", "dims": (m, n, b, k), "mapped": True,
                           "variants": variants})

    print(f"\n[replay] fresh process: open, refresh, select (GPU {args.eval_gpu})")
    todo = [g for g in groups if "result" not in g]
    res = run_worker(env, "replay", {
        "library": str(library_root), "groups": todo, "warmup": args.warmup,
        "iters": args.iters, "passes": args.passes, "rotating_mb": args.rotating,
        "max_workspace": args.max_workspace}, workdir) if todo else {}
    if res is None:
        return "replay failed"

    for g in todo:
        r = g["replay"] = res.get(g["key"], {})
        if r.get("variant") is not None:
            g["entry"] = g["variants"][r["variant"]]
        if r.get("user"):
            rel = r.get("rel_diff")
            if rel is None:
                g["result"] = "user kernel (hipBLASLt ships no kernel to compare with)"
            elif rel > args.tolerance:
                g["result"] = f"WARNING: output differs from the shipped kernel (rel {rel:.3g})"
            else:
                g["result"] = f"user kernel, {r['speedup']:.2f}x vs shipped"
        elif not g["mapped"]:
            g["result"] = ("shipped: the library stores a tuned kernel for this size, "
                           "but does not map it" if g.get("stored")
                           else "shipped: not mapped in the library")
        elif config or g.get("stored"):
            g["result"] = "WARNING: mapped in the library, but heuristic() returns the shipped kernel"
        else:
            g["result"] = ("WARNING: mapped, but no bf16/fp16/fp32 problem of this size selects it "
                           "(another data type, or an unusable payload)")

    fmt = lambda v, f="{:.2f}": f.format(v) if isinstance(v, (int, float)) else "-"
    print("\n" + "=" * 118)
    print(f"{'size':40} {'mapped':>6}  {'selected':12} {'shipped':>9} {'selected':>9} "
          f"{'vs':>8}  result")
    print(f"{'':40} {'':>6}  {'':12} {'us':>9} {'us':>9} {'shipped':>8}")
    rows = []
    for g in groups:
        r = g.get("replay") or {}
        e = g.get("entry") or g["variants"][0]
        name = label(e) if (config or "entry" in g) else \
            "M={} N={} B={} K={} (type unknown)".format(*g["dims"])
        sel = "-" if not r else ("user kernel" if r.get("user") else "shipped")
        print(f"{name:40} {'yes' if g['mapped'] else 'no':>6}  {sel:12} "
              f"{fmt(r.get('shipped_us')):>9} {fmt(r.get('selected_us')):>9} "
              f"{fmt(r.get('speedup'), '{:.2f}x'):>8}  {g['result']}")
        rows.append({"size": name, "m": g["dims"][0], "n": g["dims"][1], "batch": g["dims"][2],
                     "k": g["dims"][3], "trans": e["trans"] if "entry" in g or config else None,
                     "a_type": e["a_type"] if "entry" in g or config else None,
                     "mapped": g["mapped"], "selected": sel, "shipped_us": r.get("shipped_us"),
                     "selected_us": r.get("selected_us"), "speedup_vs_shipped": r.get("speedup"),
                     "rel_diff": r.get("rel_diff"), "selected_kernel": r.get("selected_kernel"),
                     "result": g["result"]})
    print("times are wall-clock us/call through the Python bindings; each call includes a "
          "fixed host sync,\nso ratios understate the kernel-only speedup.")
    if extra:
        print(f"\nthe library also maps {len(extra)} size(s) not in this config: "
              + ", ".join("M={} N={} B={} K={}".format(*d) for d in extra)
              + "\n(run --replay without a config to replay every mapped size)")

    with open(workdir / "replay.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (workdir / "replay.json").write_text(json.dumps(rows, indent=2))
    print(f"report: {workdir / 'replay.csv'}")
    print(f"done in {time.time() - t0:.0f}s")
    return 0


def main(args):
    config = args.config.resolve() if args.config else None
    library_root = args.library_root.resolve()
    default_workdir = (f"tune_{config.stem}" if config else f"replay_{library_root.name}")
    workdir = (args.workdir or Path.cwd() / default_workdir).resolve()
    geko_dir = workdir / "geko"
    workdir.mkdir(parents=True, exist_ok=True)
    args.eval_gpu = args.eval_gpu or args.devices.split(",")[0]
    env = Env(args)

    import yaml
    arch = args.arch or (config and (yaml.safe_load(config.read_text()) or {}).get("ARCH"))
    if config and not arch:
        return "no ARCH in the config; pass --arch"
    problems = check_setup(env, library_root, need_geko=not args.replay)
    if problems:
        return "setup problems:\n  " + "\n  ".join(problems)
    if args.replay:
        return replay(args, env, config, arch, library_root, workdir)
    sizes = requested_sizes(config, arch)
    if not sizes:
        return f"{config} lists no sizes"

    print(f"config    {config} ({len(sizes)} sizes, {arch})")
    print(f"library   {library_root}" + ("" if (library_root / "registry.log").exists()
                                         else " (new)"))
    print(f"workdir   {workdir}")
    t0 = time.time()

    step(1, f"tuning with Geko on GPUs {args.devices}")
    saved = workdir / "config.yaml"
    if args.reuse_tune and (geko_dir / "results/raw_results.csv").is_file():
        if saved.is_file() and saved.read_text() != config.read_text():
            return f"--reuse-tune: {config} differs from the one tuned in {workdir}"
        print(f"      reusing the tune in {geko_dir}")
    else:
        rc = run_geko(env, args, config, arch, geko_dir, workdir / "geko.log")
        shutil.copyfile(config, saved)
        if rc != 0:
            print(f"      WARNING: geko exited {rc}; using whatever results it produced")
    geko, payloads = read_geko(geko_dir)
    if geko is None:
        return f"Geko produced no results; see {workdir / 'geko.log'}"

    entries = []
    for key, e in sizes.items():
        e = dict(e, key=key, geko=geko.get(key))
        if not e["geko"]:
            e.update(outcome="error", result="skipped: Geko produced no result for this size")
        elif e["a_type"] not in DTYPES or e["c_type"] not in DTYPES or e["compute_type"] != "f32_r":
            e.update(outcome="skipped",
                     result=f"skipped: {e['a_type']}/{e['c_type']}/{e['compute_type']} "
                            "is not supported by this script")
        entries.append(e)
    # Sizes Geko did not keep are timed too, for the report, but never added.
    candidates = [e for e in entries if "outcome" not in e]
    kept_n = sum(1 for e in entries if e["geko"] and e["geko"]["kept"])
    print(f"      Geko found faster kernels for {kept_n} of {len(entries)} sizes")

    step(2, f"measuring against today's selection through hipBLASLt (GPU {args.eval_gpu})")
    if kept_n and not payloads:
        return f"Geko kept kernels but left no payload under {geko_dir / 'build/library'}"
    if not payloads:
        for e in candidates:
            e.update(outcome="skipped", result=skip_reason(e))
        candidates = []
    if candidates:
        scratch = workdir / "scratch_lib"
        scratch_copy(library_root, scratch)
        res = run_worker(env, "measure", {
            "library": str(scratch), "payloads": payloads, "entries": candidates,
            "min_speedup": args.min_speedup, "tolerance": args.tolerance,
            "warmup": args.warmup, "iters": args.iters, "passes": args.passes,
            "rotating_mb": args.rotating, "max_workspace": args.max_workspace}, workdir)
        if res is None:
            return "measuring failed"
        for e in candidates:
            x = e["eval"] = res[e["key"]]
            e["outcome"], e["result"] = x["outcome"], x["result"]
            if args.dry_run and x["outcome"] == "add":
                e["outcome"], e["result"] = "would add", "would add: " + x["result"]
    else:
        print("      nothing to measure")

    step(3, f"updating {library_root}")
    to_add = [e for e in entries if e["outcome"] == "add"]
    if args.dry_run:
        print("      dry run: not modifying the library")
    elif not to_add:
        print("      nothing to add")
    else:
        res = run_worker(env, "add", {"library": str(library_root), "entries": to_add,
                                      "max_workspace": args.max_workspace}, workdir)
        if res is None:
            return "adding to the library failed"
        for e in to_add:
            if res.get(e["key"], {}).get("mapped"):
                e.update(outcome="added", result="added: " + e["result"])
            else:
                e.update(outcome="error",
                         result="error: " + res.get(e["key"], {}).get("error", "not mapped"))

    step(4, "checking what a fresh process selects")
    if (library_root / "registry.log").is_file():
        res = run_worker(env, "confirm", {"library": str(library_root),
                                          "entries": [e for e in entries
                                                      if e["a_type"] in DTYPES
                                                      and e["c_type"] in DTYPES]}, workdir)
        for e in entries:
            sel = (res or {}).get(e["key"])
            if sel:
                e["selected"] = "user kernel" if sel["user"] else "shipped"
                if e["outcome"] == "added" and not sel["user"]:
                    print(f"      WARNING: {label(e)} was added but heuristic() still returns "
                          f"the shipped kernel")
    else:
        print(f"      {library_root} has no registry yet; every size uses the shipped kernel")
        for e in entries:
            e["selected"] = "shipped"

    report(entries, workdir, args.dry_run)
    print(f"done in {time.time() - t0:.0f}s")
    return 0


# ---------------------------------------------------------------------------
# hipBLASLt stages (child processes; import the bindings only here)
# ---------------------------------------------------------------------------

def worker(stage, plan_file, out_file):
    import numpy as np
    import ml_dtypes  # noqa: F401  (registers bfloat16 with numpy)
    import hipblaslt
    c = hipblaslt._core
    plan = json.loads(Path(plan_file).read_text())

    def dt(name):
        return getattr(c.DataType, DTYPES[name][0])

    def npdt(name):
        return np.dtype(DTYPES[name][1])

    def op(ch):
        return c.Operation.OP_N if ch == "N" else c.Operation.OP_T

    class Problem:
        """Layouts for one size; buffers only on alloc(), so lookups stay cheap."""

        def __init__(self, e, max_ws):
            self.e = e
            m, n, b, k, t = e["m"], e["n"], e["batch"], e["k"], e["trans"]
            self.desc = c.MatmulDesc(c.ComputeType.COMPUTE_32F, c.DataType.R_32F)
            self.desc.set_attribute_int(c.MatmulDescAttr.TRANSA, op(t[0]))
            self.desc.set_attribute_int(c.MatmulDescAttr.TRANSB, op(t[1]))
            ar, ac = (m, k) if t[0] == "N" else (k, m)
            br, bc = (k, n) if t[1] == "N" else (n, k)
            self.shapes = [(ar, ac, e["a_type"]), (br, bc, e["b_type"]),
                           (m, n, e["c_type"]), (m, n, e["c_type"])]
            self.layouts = []
            for rows, cols, ty in self.shapes:
                lay = c.MatrixLayout(dt(ty), rows, cols, rows)
                if b > 1:
                    lay.set_attribute(c.MatrixLayoutAttr.BATCH_COUNT, b)
                    lay.set_attribute_i64(c.MatrixLayoutAttr.STRIDED_BATCH_OFFSET, rows * cols)
                self.layouts.append(lay)
            self.pref = c.Preference()
            self.pref.set_max_workspace(max_ws)
            self.sets = []

        def alloc(self, rotating_mb):
            b = self.e["batch"]
            nbytes = sum(r * cc * b * npdt(ty).itemsize for r, cc, ty in self.shapes)
            nsets = max(1, -(-rotating_mb * 1024 * 1024 // nbytes))
            rng = np.random.default_rng(0)
            # Small integers make every fp32 partial sum exact, so two correct
            # kernels agree bit for bit whatever their reduction order.
            host = [rng.integers(-2, 3, r * cc * b, dtype=np.int8).astype(npdt(ty))
                    for r, cc, ty in self.shapes[:2]]
            zeros = np.zeros(self.shapes[2][0] * self.shapes[2][1] * b, npdt(self.e["c_type"]))
            tys = [dt(ty) for _, _, ty in self.shapes]
            self.sets = [(hipblaslt.from_numpy(host[0], tys[0]),
                          hipblaslt.from_numpy(host[1], tys[1]),
                          hipblaslt.from_numpy(zeros, tys[2]),
                          hipblaslt.from_numpy(zeros.copy(), tys[3])) for _ in range(nsets)]

        def free(self):
            self.sets = []

        def run(self, h, algo, ws, i=0):
            A, B, C, D = self.sets[i % len(self.sets)]
            la, lb, lc, ld = self.layouts
            c.matmul(h, self.desc, 1.0, A, la, B, lb, 1.0, C, lc, D, ld, algo, ws)

        def output(self, h, algo, ws):
            self.run(h, algo, ws, 0)
            return self.sets[0][3].to_numpy().astype(np.float32)

        def top(self, h):
            got = c.heuristic(h, self.desc, *self.layouts, self.pref, 1)
            return got[0] if got else None

        def supports(self, h, algo):
            la, lb, lc, ld = self.layouts
            return c.is_algo_supported(h, self.desc, 1.0, la, lb, 1.0, lc, ld, algo)

    def workspace(nbytes):
        return hipblaslt.from_numpy(np.zeros(max(1, nbytes), np.uint8), c.DataType.R_8I)

    def time_algo(h, prob, algo, ws):
        for i in range(plan["warmup"]):
            prob.run(h, algo, ws, i)
        t0 = time.perf_counter()
        for i in range(5):
            prob.run(h, algo, ws, i)
        est = (time.perf_counter() - t0) / 5
        iters = max(20, min(plan["iters"], int(0.25 / max(est, 1e-7))))
        best = None
        for _ in range(plan["passes"]):
            t0 = time.perf_counter()
            for i in range(iters):
                prob.run(h, algo, ws, i)
            us = (time.perf_counter() - t0) / iters * 1e6
            best = us if best is None else min(best, us)
        return best

    def register(h, stems):
        """stem -> [(index, kernel name)] for every kernel in each payload."""
        out = {}
        for stem in stems:
            idx = c.user_kernel_register(h, stem + ".dat", stem + ".co")
            out[stem] = []
            for i in idx:
                got = c.get_algos_from_index(h, [i])
                if got:
                    out[stem].append((i, c.kernel_name(h, got[0].algo), got[0].algo))
        return out

    def find(h, prob, registered, name):
        """(stem, index, algo, workspace) of the kernel called `name` that fits prob."""
        for exact in (True, False):
            for stem, kernels in registered.items():
                for i, kname, algo in kernels:
                    if (kname == name) if exact else (name in kname):
                        need = prob.supports(h, algo)
                        if need is not None and need <= plan["max_workspace"]:
                            return stem, i, algo, need
        return None

    out = {}
    with c.Handle() as h:
        c.user_kernel_library_open(h, plan["library"])

        if stage == "measure":
            probs = {e["key"]: Problem(e, plan["max_workspace"]) for e in plan["entries"]}
            # Refresh changes what heuristic() returns for every size at once,
            # so the shipped pick is captured for all of them first.
            shipped = {k: p.top(h) for k, p in probs.items()}
            if (Path(plan["library"]) / "registry.log").exists():
                c.user_kernel_refresh(h)
            registered = register(h, plan["payloads"])
            print(f"      registered {sum(len(v) for v in registered.values())} tuned kernels "
                  f"from {len(registered)} payload(s) in a scratch library", flush=True)

            for key, prob in probs.items():
                e = prob.e
                r = out[key] = {}
                s = shipped[key]
                if s is None:
                    r.update(outcome="skipped", result="skipped: hipBLASLt has no kernel for this size")
                    continue
                cur = prob.top(h)
                cand = find(h, prob, registered, e["geko"]["kernel"])
                if cand is None:
                    if e["geko"]["kept"]:
                        r.update(outcome="error",
                                 result="error: Geko's tuned kernel is not usable from its payload")
                    else:
                        r.update(outcome="skipped", result=skip_reason(e))
                    continue
                stem, idx, algo, need = cand

                prob.alloc(plan["rotating_mb"])
                s_ws, c_ws, t_ws = (workspace(s.workspace_size), workspace(cur.workspace_size),
                                    workspace(need))
                ref = prob.output(h, s.algo, s_ws)
                got = prob.output(h, algo, t_ws)
                rel = float(np.abs(got - ref).max() / max(np.abs(ref).max(), 1e-30))
                t_shipped = time_algo(h, prob, s.algo, s_ws)
                is_user = c.is_user_kernel(cur.algo)
                t_cur = time_algo(h, prob, cur.algo, c_ws) if is_user else t_shipped
                t_tuned = time_algo(h, prob, algo, t_ws)
                prob.free()

                cur_name = c.kernel_name(h, cur.algo)
                r.update(shipped_us=t_shipped, current="user kernel" if is_user else "shipped",
                         current_us=t_cur, current_kernel=cur_name, tuned_us=t_tuned,
                         tuned_kernel=e["geko"]["kernel"], payload=stem,
                         speedup_vs_current=t_cur / t_tuned, speedup_vs_shipped=t_shipped / t_tuned,
                         rel_diff=rel)
                sp = r["speedup_vs_current"]
                if not e["geko"]["kept"]:
                    r.update(outcome="skipped", result=skip_reason(e))
                elif rel > plan["tolerance"]:
                    r.update(outcome="rejected",
                             result=f"rejected: output differs from shipped (rel {rel:.3g})")
                elif is_user and cur_name == e["geko"]["kernel"]:
                    r.update(outcome="kept", result="kept: this kernel is already in the library")
                elif sp < plan["min_speedup"]:
                    r.update(outcome="kept",
                             result=f"kept {r['current']}: tuned is {sp:.2f}x vs it, below "
                                    f"--min-speedup {plan['min_speedup']}")
                else:
                    r.update(outcome="add", result=f"{sp:.2f}x faster than {r['current']}")
                now = f"  user kernel {t_cur:8.2f}" if is_user else ""
                print(f"      {label(e):40} shipped {t_shipped:8.2f}{now}  tuned {t_tuned:8.2f} us"
                      f"  ({t_shipped / t_tuned:.2f}x vs shipped)  -> {r['result']}", flush=True)

        elif stage == "add":
            stems = sorted({e["eval"]["payload"] for e in plan["entries"]})
            registered = register(h, stems)
            for e in plan["entries"]:
                prob = Problem(e, plan["max_workspace"])
                only = {e["eval"]["payload"]: registered[e["eval"]["payload"]]}
                cand = find(h, prob, only, e["eval"]["tuned_kernel"])
                if cand is None:
                    out[e["key"]] = {"mapped": False, "error": "kernel not found on re-register"}
                    continue
                c.user_kernel_set_exact_match(h, cand[1], e["m"], e["n"], e["batch"], e["k"])
                out[e["key"]] = {"mapped": True, "index": cand[1]}
                print(f"      mapped {label(e)}", flush=True)
            print(f"      {len(stems)} payload(s) stored, {sum(v['mapped'] for v in out.values())} "
                  f"size(s) mapped", flush=True)

        elif stage == "replay":
            groups = plan["groups"]
            probs = [[Problem(v, plan["max_workspace"]) for v in g["variants"]] for g in groups]
            shipped = [[p.top(h) for p in ps] for ps in probs]
            n = c.user_kernel_refresh(h)
            print(f"      refresh restored {n} kernels", flush=True)
            for g, ps, ss in zip(groups, probs, shipped):
                r = out[g["key"]] = {"variant": 0 if len(ps) == 1 else None, "user": False}
                hit = None
                for vi, p in enumerate(ps):
                    got = p.top(h)
                    if got is not None and c.is_user_kernel(got.algo):
                        hit = (vi, p, got)
                        break
                if hit is None:
                    if len(ps) == 1 and ss[0] is not None:
                        ps[0].alloc(plan["rotating_mb"])
                        t = time_algo(h, ps[0], ss[0].algo, workspace(ss[0].workspace_size))
                        r.update(shipped_us=t, selected_us=t,
                                 selected_kernel=c.kernel_name(h, ss[0].algo))
                        ps[0].free()
                    print(f"      {label(ps[0].e) if len(ps) == 1 else g['key']:40} shipped",
                          flush=True)
                    continue
                vi, p, got = hit
                s = ss[vi]
                r.update(variant=vi, user=True, selected_kernel=c.kernel_name(h, got.algo))
                p.alloc(plan["rotating_mb"])
                u_ws = workspace(got.workspace_size)
                t_user = time_algo(h, p, got.algo, u_ws)
                r["selected_us"] = t_user
                if s is not None:
                    s_ws = workspace(s.workspace_size)
                    ref = p.output(h, s.algo, s_ws)
                    out_u = p.output(h, got.algo, u_ws)
                    t_shipped = time_algo(h, p, s.algo, s_ws)
                    r.update(shipped_us=t_shipped, speedup=t_shipped / t_user,
                             shipped_kernel=c.kernel_name(h, s.algo),
                             rel_diff=float(np.abs(out_u - ref).max()
                                            / max(np.abs(ref).max(), 1e-30)))
                p.free()
                print(f"      {label(p.e):40} user kernel", flush=True)

        elif stage == "confirm":
            n = c.user_kernel_refresh(h)
            print(f"      refresh restored {n} kernels from {plan['library']}", flush=True)
            for e in plan["entries"]:
                got = Problem(e, 128 * 1024 * 1024).top(h)
                if got is not None:
                    out[e["key"]] = {"user": bool(c.is_user_kernel(got.algo)),
                                     "kernel": c.kernel_name(h, got.algo)}

    Path(out_file).write_text(json.dumps(out, indent=2))
    return 0


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", type=Path, nargs="?",
                    help="Geko tuning YAML, the same file `geko --tune --list` takes")
    ap.add_argument("--library-root", type=Path, default=Path("user_kernel_lib"),
                    help="user kernel library to add winners to; created if missing "
                         "(default: ./user_kernel_lib)")
    ap.add_argument("--workdir", type=Path, default=None,
                    help="where the tune, logs and report go (default: ./tune_<config name>)")
    ap.add_argument("--devices", default="0",
                    help="GPUs for Geko, comma-separated (default: 0)")
    ap.add_argument("--eval-gpu", default=None,
                    help="GPU for the hipBLASLt measurements (default: first of --devices)")
    ap.add_argument("--arch", default=None, help="override the config's ARCH")
    ap.add_argument("--hipblaslt", type=Path, default=DEFAULT_HIPBLASLT,
                    help=f"hipBLASLt project root (default: {DEFAULT_HIPBLASLT})")
    ap.add_argument("--rocm-path", default=os.environ.get("ROCM_PATH", "/opt/rocm"))
    ap.add_argument("--min-speedup", type=float, default=1.03,
                    help="add a kernel only if it is at least this much faster than what "
                         "hipBLASLt selects today (default: 1.03, Geko's threshold)")
    ap.add_argument("--tolerance", type=float, default=0.01,
                    help="max relative difference from the shipped kernel's output")
    ap.add_argument("--reuse-tune", action="store_true",
                    help="skip Geko if --workdir already holds a tune of this config")
    ap.add_argument("--dry-run", action="store_true",
                    help="tune and measure, but do not modify the library")
    ap.add_argument("--replay", action="store_true",
                    help="tune nothing and only read the library: in a fresh process, refresh "
                         "it and report what heuristic() selects for the config's sizes, or for "
                         "every size the library maps if no config is given")
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--iters", type=int, default=1000)
    ap.add_argument("--passes", type=int, default=3)
    ap.add_argument("--rotating", type=int, default=512, metavar="MB",
                    help="rotate inputs through at least this much memory (default: 512, "
                         "as Geko's benchmark does)")
    ap.add_argument("--max-workspace", type=int, default=128 * 1024 * 1024)
    ap.add_argument("--_stage", help=argparse.SUPPRESS)
    ap.add_argument("--_plan", help=argparse.SUPPRESS)
    ap.add_argument("--_out", help=argparse.SUPPRESS)
    args = ap.parse_args()
    if not args._stage and not args.config and not args.replay:
        ap.error("a config YAML is required (optional only with --replay)")
    return args


if __name__ == "__main__":
    a = parse_args()
    r = worker(a._stage, a._plan, a._out) if a._stage else main(a)
    sys.exit(f"\nERROR: {r}" if isinstance(r, str) else r)
