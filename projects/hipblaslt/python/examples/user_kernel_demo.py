#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Register a kernel, run it, and make it selectable -- in one live session.

The point of doing all three in a single interpreter is that the interesting
claims are about a *running* process:

  * the kernel did not exist when the process started
  * running it needs no restart and no library reload
  * making it selectable changes the answer for a shape this process has
    already been computing, which only works if the remembered selection
    result is thrown away

The last one is why the script does a GEMM *before* registering anything. On a
cold cache the demo would look identical whether or not cache invalidation
works, and so would prove nothing.

Usage:
    python user_kernel_demo.py [--payload STEM] [--library-root DIR]

With --library-root the registration is also written to disk, and a second run
with --refresh-only reinstates it without registering again.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

try:
    import ml_dtypes
except ImportError:
    sys.exit("this demo needs ml_dtypes: pip install ml_dtypes")

import hipblaslt

c = hipblaslt._core

# A tall, skinny GEMM: small M against large N and K. Shapes like this are
# memory-bound and common in inference, and small M means the choice of tile
# size matters, so registering a different kernel has something to prove.
M, N, K = 16, 6144, 2048

_LIBRARY_RELATIVE = "hipblaslt-install/lib/hipblaslt/library"

# A bf16 in/out, TN shard. Deliberately not pinned to an architecture or a
# device id: the glob matches whatever this build installed.
_PAYLOAD_GLOB = "TensileLibrary_BB_BB_*_Alik_Bljk_*.dat*"


def device_pci_id() -> str | None:
    """PCI device id of GPU 0, e.g. "75a0", or None if it cannot be read.

    Shard filenames carry the device ids they were built for, so knowing this
    lets the demo register the shard that belongs to this GPU instead of
    discovering by trial which is a waste and muddles the kernel counts.
    """
    try:
        import ctypes

        lib = ctypes.CDLL("libamdhip64.so")
        buf = ctypes.create_string_buffer(64)
        if lib.hipDeviceGetPCIBusId(buf, 64, 0) != 0:
            return None
        bus = buf.value.decode().lower()
        text = (Path("/sys/bus/pci/devices") / bus / "device").read_text().strip()
        return text.removeprefix("0x") or None
    except Exception:
        return None


def candidate_payloads() -> list[str]:
    """Shipped shards usable as a sample payload, most promising first.

    A shipped shard stands in for a tuned kernel: it is a real compiled
    (.dat, .co) pair with the stem pairing registration requires, so the demo
    runs against a stock build with no tuning tool involved. Discovered rather
    than hardcoded so it follows whatever GPU this build targets, instead of
    one the author happened to have.

    Shards built for other device ids deserialize fine but every kernel in them
    is rejected when validated against this GPU, so they are filtered out by
    matching the PCI device id that the filename carries. A CU-count variant
    can still miss, so the caller walks what remains; bigger shards come first
    because they hold more kernels and are likelier to contain a fit.
    """
    pci_id = device_pci_id()
    found: list[tuple[int, str]] = []
    for parent in Path(__file__).resolve().parents:
        library = parent / _LIBRARY_RELATIVE
        if not library.is_dir():
            continue
        for arch_dir in sorted(library.iterdir()):
            if not arch_dir.is_dir():
                continue
            for shard in arch_dir.glob(_PAYLOAD_GLOB):
                if pci_id and pci_id not in shard.name.lower():
                    continue
                # Strip .zlib then .dat to recover the stem; the loader appends
                # the ".zlib" probe itself, so the bare .dat name is what gets
                # passed even when only the compressed form is on disk.
                stem = str(shard)
                for suffix in (".zlib", ".dat"):
                    if stem.endswith(suffix):
                        stem = stem[: -len(suffix)]
                if Path(stem + ".co").exists():
                    found.append((shard.stat().st_size, stem))
        if found:
            break
    return [stem for _, stem in sorted(found, reverse=True)]


def usable_kernel(handle, desc, layouts, indices, max_workspace):
    """First registered index that can actually serve this problem, if any."""
    for idx in indices:
        got = c.get_algos_from_index(handle, [idx])
        if not got:
            continue
        need = c.is_algo_supported(
            handle, desc, 1.0, layouts["a"], layouts["b"], 0.0,
            layouts["c"], layouts["d"], got[0].algo,
        )
        if need is not None and need <= max_workspace:
            return idx, got[0], need
    return None


def build_problem(handle, m=M, n=N, k=K):
    """A bf16 TN GEMM, laid out column-major the way hipBLASLt expects."""
    desc = c.MatmulDesc(c.ComputeType.COMPUTE_32F, c.DataType.R_32F)
    desc.set_attribute_int(c.MatmulDescAttr.TRANSA, c.Operation.OP_T)
    desc.set_attribute_int(c.MatmulDescAttr.TRANSB, c.Operation.OP_N)

    bf = c.DataType.R_16BF
    # Exactly 1.0 everywhere, so every output element is exactly k -- which bf16
    # represents without rounding. Any discrepancy is then a dispatch or launch
    # fault rather than precision.
    a = np.ones((k, m), dtype=np.float32).astype(ml_dtypes.bfloat16)
    b = np.ones((k, n), dtype=np.float32).astype(ml_dtypes.bfloat16)

    buffers = {
        "A": hipblaslt.from_numpy(np.ascontiguousarray(a.T), bf),
        "B": hipblaslt.from_numpy(np.ascontiguousarray(b.T), bf),
        "C": hipblaslt.from_numpy(np.zeros((n, m), ml_dtypes.bfloat16), bf),
        "D": hipblaslt.from_numpy(np.zeros((n, m), ml_dtypes.bfloat16), bf),
    }
    layouts = {
        "a": c.MatrixLayout(bf, k, m, k),
        "b": c.MatrixLayout(bf, k, n, k),
        "c": c.MatrixLayout(bf, m, n, m),
        "d": c.MatrixLayout(bf, m, n, m),
    }
    pref = c.Preference()
    pref.set_max_workspace(64 * 1024 * 1024)
    return desc, buffers, layouts, pref


def top_algo(handle, desc, layouts, pref):
    results = c.heuristic(
        handle, desc, layouts["a"], layouts["b"], layouts["c"], layouts["d"], pref, 1
    )
    if not results:
        raise RuntimeError("no algo for this shape")
    return results[0]


def describe(handle, algo):
    kind = "user" if c.is_user_kernel(algo) else "shipped"
    return f"{algo.solution_index:,} ({kind})"


def run(handle, desc, buffers, layouts, algo, workspace_bytes):
    ws = hipblaslt.from_numpy(
        np.zeros(max(1, workspace_bytes), np.uint8), c.DataType.R_8I
    )
    c.matmul(
        handle, desc, 1.0,
        buffers["A"], layouts["a"], buffers["B"], layouts["b"], 0.0,
        buffers["C"], layouts["c"], buffers["D"], layouts["d"], algo, ws,
    )
    out = buffers["D"].to_numpy().reshape(N, M).T.astype(np.float32)
    return out


def step(number: int, title: str) -> None:
    print(f"\n{number}. {title}")


def detail(label: str, text: str) -> None:
    print(f"     {label:<10} {text}")


def human_bytes(n: int) -> str:
    if n >= 1 << 20:
        return f"{n / (1 << 20):.1f} MiB"
    if n >= 1 << 10:
        return f"{n / (1 << 10):.1f} KiB"
    return f"{n} B"


def tier_state(handle) -> str:
    registered, selectable = c.user_kernel_counts(handle)
    return f"{registered:,} registered, {selectable:,} selectable"


def short_payload(stem: str) -> str:
    """Shard names are long; keep the tail, which is what distinguishes them."""
    name = Path(stem).name
    return name if len(name) <= 56 else "..." + name[-53:]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--payload",
                    help="payload stem; .dat/.dat.zlib and .co are appended. "
                         "Defaults to a shipped shard discovered in this build")
    ap.add_argument("--library-root",
                    help="persist the registration under this directory")
    ap.add_argument("--refresh-only", action="store_true",
                    help="replay an existing library instead of registering")
    args = ap.parse_args()

    if args.refresh_only:
        candidates = []
    elif args.payload:
        candidates = [args.payload]
        if not (Path(args.payload + ".dat").exists()
                or Path(args.payload + ".dat.zlib").exists()):
            return f"payload shard not found: {args.payload}.dat(.zlib)"
        if not Path(args.payload + ".co").exists():
            return f"payload code object not found: {args.payload}.co"
    else:
        candidates = candidate_payloads()
        if not candidates:
            return ("no sample payload found in this build's installed library; "
                    "pass --payload STEM naming a (.dat, .co) pair")

    with c.Handle() as handle:
        desc, buffers, layouts, pref = build_problem(handle)

        print("user kernel registration demo")
        print(f"     {'problem':<10} M={M}  N={N}  K={K}   bf16 TN")
        if args.library_root:
            c.user_kernel_library_open(handle, args.library_root)
            detail("library", args.library_root)

        # -- 1. a GEMM before anything is registered -----------------------
        # This is what makes the demo meaningful: it puts a selection result
        # for this shape in the cache, so step 5 has something stale to beat.
        step(1, "Before registering anything")
        before = top_algo(handle, desc, layouts, pref)
        detail("selection", describe(handle, before.algo))
        out = run(handle, desc, buffers, layouts, before.algo, before.workspace_size)
        detail("result", f"D[0,0] = {out[0, 0]:g}   (expected {K})")
        detail("tier", tier_state(handle))

        # -- 2. register ---------------------------------------------------
        if args.refresh_only:
            count = c.user_kernel_refresh(handle)
            step(2, "Replay the library from disk")
            detail("replayed", f"{count:,} kernels")
            detail("tier", tier_state(handle))
        else:
            # Candidates are tried in turn because a shard built for another
            # device id or CU count registers fine but has no kernel that
            # passes validation here. Which one wins is a detail of finding a
            # sample payload, so only the one that worked is reported.
            chosen = None
            for stem in candidates:
                indices = c.user_kernel_register(handle, stem + ".dat", stem + ".co")
                if not indices:
                    continue
                chosen = usable_kernel(handle, desc, layouts, indices, 64 * 1024 * 1024)
                if chosen:
                    break
            if chosen is None:
                return ("no shipped shard in this build has a kernel for this "
                        "shape; pass --payload to name one")

            step(2, "Register a payload")
            detail("payload", short_payload(stem))
            detail("added", f"{len(indices):,} kernels")
            base = 1 << 30
            first = f"{indices[0]:,}" + ("   (= 2^30, the first user index)"
                                         if indices[0] == base else "")
            detail("first", first)
            detail("tier", tier_state(handle))

        # Replay restores the recorded mappings as well as the kernels, so
        # selection has already moved and the register-only invariant below
        # does not apply on this path.
        if args.refresh_only:
            final = top_algo(handle, desc, layouts, pref)
            step(3, "Selection now returns the kernel reinstated from disk")
            detail("selection", describe(handle, final.algo))
            if not c.is_user_kernel(final.algo):
                return "FAILED: replay did not restore the mapping"
            out = run(handle, desc, buffers, layouts, final.algo, final.workspace_size)
            detail("result", f"D[0,0] = {out[0, 0]:g}   (expected {K})")
            detail("", "the mapping came from an earlier process; this one")
            detail("", "registered nothing itself")
            print("\nOK")
            return 0

        # -- 3. registration alone changes nothing about selection ---------
        after_register = top_algo(handle, desc, layouts, pref)
        step(3, "Registering alone does not change selection")
        detail("selection", f"{describe(handle, after_register.algo)} - unchanged")
        if after_register.algo.solution_index != before.algo.solution_index:
            return "FAILED: registration changed selection on its own"

        # -- 4. run a registered kernel by index, before it is selectable ---
        # Already located in step 2, since finding one is what decided which
        # payload to use.
        idx, result, need = chosen
        step(4, "Run a registered kernel by index, while it is still unselectable")
        detail("kernel", describe(handle, result.algo))
        detail("workspace", human_bytes(need))
        out = run(handle, desc, buffers, layouts, result.algo, need)
        detail("result", f"D[0,0] = {out[0, 0]:g}   (expected {K})")

        # -- 5. now make it selectable -------------------------------------
        c.user_kernel_set_exact_match(handle, idx, M, N, 1, K)
        step(5, "Map it to this exact shape, and selection moves")
        detail("tier", tier_state(handle))

        final = top_algo(handle, desc, layouts, pref)
        detail("selection", describe(handle, final.algo))
        if not c.is_user_kernel(final.algo):
            return "FAILED: selection did not move to the registered kernel"
        detail("", "the answer cached in step 1 was discarded")

        # -- 6. a neighbouring shape is untouched ---------------------------
        desc2, buffers2, layouts2, pref2 = build_problem(handle, m=32)
        other = top_algo(handle, desc2, layouts2, pref2)
        step(6, "A neighbouring shape is untouched")
        detail("M=32", f"{describe(handle, other.algo)} - same N and K")
        if c.is_user_kernel(other.algo):
            return "FAILED: the mapping leaked to an unmapped shape"
        detail("", "matching is exact, so only the mapped shape moved")

        if args.library_root:
            print(f"\nSaved to {args.library_root}")
            print("Replay it in a new process with:")
            print(f"     --refresh-only --library-root {args.library_root}")

    print("\nOK")
    return 0


if __name__ == "__main__":
    result = main()
    if isinstance(result, str):
        sys.exit(f"\nERROR: {result}")
    sys.exit(result)
