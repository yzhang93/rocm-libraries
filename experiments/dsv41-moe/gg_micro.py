"""Time one hipBLASLt grouped MXFP8 GEMM (PR 13581) alone for chosen per-group token counts,
to see how its time depends on the count distribution and on the grid bound."""

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from moe_gg_bench import BLK, DEV, Shim, round_up, swizzle_gfx950  # noqa: E402


def run_case(shim, groups, M, K, counts, total_hint, max_hint, iters=20, ab=0):
    """ab: 0 = MXFP8 x MXFP8, 3 = MXFP4 weights (A) x MXFP8 tokens (B)."""
    kb = round_up(K // BLK, 8)
    seg = [round_up(c, BLK) for c in counts]
    offs = [0]
    for s in seg[:-1]:
        offs.append(offs[-1] + s)
    P = max(sum(seg), BLK)
    a_row = K // 2 if ab == 3 else K
    w = torch.randint(0, 255, (groups, M, a_row), dtype=torch.uint8, device=DEV) & (0xFF if ab == 3 else 0x77)
    ws_ = swizzle_gfx950(torch.full((groups, M, kb), 120, dtype=torch.uint8, device=DEV))
    b = torch.randint(0, 255, (P, K), dtype=torch.uint8, device=DEV) & 0x77
    bs = swizzle_gfx950(torch.full((P, kb), 120, dtype=torch.uint8, device=DEV))
    d = torch.zeros((P, M), dtype=torch.bfloat16, device=DEV)
    ar = torch.arange(groups, device=DEV, dtype=torch.int64)
    o = torch.tensor(offs, device=DEV, dtype=torch.int64)
    n_arr = torch.tensor(counts, device=DEV, dtype=torch.int32)
    a_p = w.data_ptr() + ar * (M * a_row)
    a_sp = ws_.data_ptr() + ar * (M * kb)
    b_p = b.data_ptr() + o * K
    b_sp = bs.data_ptr() + o * kb
    d_p = d.data_ptr() + o * (M * 2)
    wsb = torch.zeros(64 << 20, dtype=torch.uint8, device=DEV)
    plan, kname = shim.create(M, K, max_hint, total_hint, n_arr, a_sp, b_sp, wsb.numel(), groups=groups,
                              ab=ab)
    for _ in range(3):
        shim.run(plan, a_p, b_p, d_p, wsb)
    torch.cuda.synchronize()
    t0, t1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    t0.record()
    for _ in range(iters):
        shim.run(plan, a_p, b_p, d_p, wsb)
    t1.record()
    torch.cuda.synchronize()
    us = t0.elapsed_time(t1) * 1e3 / iters
    active = sum(1 for c in counts if c > 0)
    wbytes = active * M * a_row
    flops = 2.0 * M * K * sum(counts)
    return dict(ab=AB_NAMES[ab], kernel=kname[:48], M=M, K=K, groups=groups, active=active, tokens=sum(counts), total_hint=total_hint,
                max_hint=max_hint, us=us, weight_TBps=wbytes / us / 1e6, TFLOPs=flops / us / 1e6)


AB_NAMES = {0: "mxfp8", 1: "mxfp4", 2: "fp16", 3: "mxfp4xmxfp8"}


def check_case(shim, M, K, counts, total_hint=None, max_hint=None, ab=0):
    """Distinct data per group (ab: 0 = MXFP8, 1 = MXFP4, 3 = MXFP4 A x MXFP8 B, constant scales;
    2 = fp16): every valid D row must match a per-group reference, and rows outside the groups
    (32-row padding) must keep their sentinel."""
    import moe_gg_bench as mb

    groups = len(counts)
    kb = round_up(K // BLK, 8)
    seg = [round_up(c, BLK) for c in counts]
    offs = [0]
    for s in seg[:-1]:
        offs.append(offs[-1] + s)
    P = max(sum(seg), BLK)
    g = torch.Generator(device=DEV).manual_seed(groups * 7919 + sum(counts))
    if ab == 2:
        w = (torch.randn((groups, M, K), device=DEV, generator=g) * 0.1).half()
        b = (torch.randn((P, K), device=DEV, generator=g) * 0.1).half()
        wf, bf, scale, a_row, b_row = w.float(), b.float(), 1.0, K * 2, K * 2
    else:

        def operand(rows, fp4):
            t = torch.randint(0, 256, (*rows, K // 2 if fp4 else K), dtype=torch.uint8, device=DEV,
                              generator=g) & (0xFF if fp4 else 0x77)
            if not fp4:
                return t, t.view(torch.float8_e4m3fn).float(), K
            one = torch.full((*rows, K // BLK), 127, dtype=torch.uint8, device=DEV)
            return t, mb.fp4_dequant(t, one), K // 2

        w, wf, a_row = operand((groups, M), ab in (1, 3))
        b, bf, b_row = operand((P,), ab == 1)
        scale = 2.0 ** (2 * (120 - 127))
    sentinel = 12288.0  # exact in bf16 and fp16
    d = torch.full((P, M), sentinel, dtype=torch.float16 if ab == 2 else torch.bfloat16, device=DEV)
    ar = torch.arange(groups, device=DEV, dtype=torch.int64)
    o = torch.tensor(offs, device=DEV, dtype=torch.int64)
    n_arr = torch.tensor(counts, device=DEV, dtype=torch.int32)
    wsb = torch.zeros(64 << 20, dtype=torch.uint8, device=DEV)
    a_sp = b_sp = None
    if ab != 2:
        ws_ = torch.full((groups, M, kb), 120, dtype=torch.uint8, device=DEV)
        bs = torch.full((P, kb), 120, dtype=torch.uint8, device=DEV)
        # The plan keeps the device addresses of the scale-pointer arrays, so they must outlive it.
        a_sp, b_sp = ws_.data_ptr() + ar * (M * kb), bs.data_ptr() + o * kb
    plan, kname = shim.create(M, K, max_hint or max(max(counts), 1), total_hint or max(sum(counts), 1),
                              n_arr, a_sp, b_sp, wsb.numel(), groups=groups, ab=ab)
    shim.run(plan, w.data_ptr() + ar * (M * a_row), b.data_ptr() + o * b_row,
             d.data_ptr() + o * (M * 2), wsb)
    torch.cuda.synchronize()
    valid = torch.zeros(P, dtype=torch.bool, device=DEV)
    worst = 0.0
    for gi, (c, off) in enumerate(zip(counts, offs)):
        if c == 0:
            continue
        valid[off : off + c] = True
        ref = (bf[off : off + c] @ wf[gi].T) * scale
        err = float((d[off : off + c].float() - ref).norm() / ref.norm().clamp_min(1e-30))
        worst = max(worst, err)
    untouched = bool((d[~valid].float() == sentinel).all())
    return dict(ab=AB_NAMES[ab], kernel=kname[:48], M=M, K=K, groups=groups,
                active=sum(1 for c in counts if c), tokens=sum(counts),
                worst_rel_err=worst, padding_untouched=untouched, ok=worst < 1e-2 and untouched)


def run_checks(shim):
    g = torch.Generator().manual_seed(1)

    def rnd(n, hi, zero_frac):
        c = torch.randint(0, hi + 1, (n,), generator=g)
        c[torch.rand(n, generator=g) < zero_frac] = 0
        return c.tolist()

    cases = [
        ("G=1", [100]),
        ("G=5 with empties", [0, 37, 0, 64, 1]),
        ("G=64", rnd(64, 100, 0.2)),
        ("G=65", rnd(65, 100, 0.2)),
        ("G=100, half empty", rnd(100, 80, 0.5)),
        ("G=130, only first", [77] + [0] * 129),
        ("G=384, only last", [0] * 383 + [70]),
        ("G=384, sparse", rnd(384, 20, 0.8)),
        ("G=384, dense", rnd(384, 70, 0.0)),
        ("G=500, sparse", rnd(500, 20, 0.8)),
        ("G=1000, only last", [0] * 999 + [33]),
        ("G=1000, sparse", rnd(1000, 40, 0.9)),
    ]
    rows = []
    for ab, M, K in ((0, 192, 512), (0, 1152, 5120), (0, 5120, 576),
                     (1, 192, 512), (1, 1152, 5120), (2, 192, 512), (2, 1152, 5120),
                     (3, 192, 512), (3, 1152, 5120), (3, 5120, 576)):
        for name, c in cases:
            r = check_case(shim, M, K, c, ab=ab)
            r["case"] = name
            print("check", json.dumps(r), flush=True)
            rows.append(r)
    print("checks:", "ALL OK" if all(r["ok"] for r in rows) else "FAILURES", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--check", action="store_true", help="correctness cases only")
    ap.add_argument("--ab", type=int, default=0, choices=(0, 3),
                    help="timed types: 0 = MXFP8 x MXFP8, 3 = MXFP4 weights x MXFP8 tokens")
    args = ap.parse_args()
    import moe_gg_bench as mb

    torch.backends.cuda.preferred_blas_library("cublas")
    shim = Shim()
    print("hipblaslt:", shim.hipblaslt_lib, flush=True)
    if args.check:
        rows = run_checks(shim)
        if args.out:
            with open(args.out, "w") as f:
                json.dump(rows, f, indent=1)
        return
    E = mb.E
    g = torch.Generator().manual_seed(0)
    rows = []

    def uniform_counts(T):
        c = [0] * E
        for _ in range(T):
            for e in torch.randperm(E, generator=g)[: mb.TOPK].tolist():
                c[e] += 1
        return c

    for M, K in ((1152, 5120), (5120, 576)):
        cases = []
        for T in (1, 16, 256, 4096):
            c = uniform_counts(T)
            cases.append((f"uniform T={T}, G=384", c, T * mb.TOPK, T))
            G = min(E, T * mb.TOPK)
            if G < E:
                cases.append((f"uniform T={T}, compacted G={G}", sorted(c, reverse=True)[:G], T * mb.TOPK, T))
        cases.append(("1 group n=64, rest 0, G=384", [64] + [0] * (E - 1), 64, 64))
        cases.append(("all 384 groups n=64, G=384", [64] * E, 64 * E, 64))
        cases.append(("single group n=24576, G=1", [64 * E], 64 * E, 64 * E))
        for name, c, tot, mx in cases:
            r = run_case(shim, len(c), M, K, c, tot, mx, ab=args.ab)
            r["case"] = name
            print(json.dumps(r), flush=True)
            rows.append(r)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(rows, f, indent=1)


if __name__ == "__main__":
    main()
