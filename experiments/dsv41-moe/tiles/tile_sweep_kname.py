"""Correctness and timing of each grouped MXFP8 macro tile (hipblaslt-gg-tiles build) on the
DSv4.1 MoE GEMMs, picked by kernel name through GG_KERNEL_MATCH. With --ab 3 the weights are
MXFP4 (the mixed MXFP4 x MXFP8 kernels).

Timing: compacted plans (G = min(384, 6T)) with uniform routing. Every graph replays R
launches, each on a different random set of experts from a full 384-expert weight pool, so
expert weights are mostly not resident in the MALL (as in moe_gg_bench's round-robin graphs).

Run with GG_SHIM=libgg_shim_tiles.so."""

import argparse
import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import moe_gg_bench as mb  # noqa: E402
from gg_micro import check_case  # noqa: E402
from moe_gg_bench import BLK, DEV, Shim, round_up, swizzle_gfx950  # noqa: E402

TILES = ["MT64x64x256", "MT32x128x256", "MT128x32x256", "MT256x32x256", "MT128x64x256", "MT256x64x256",
         "MT128x128x256", "MT256x128x256"]
SHAPES = {"gemm1": (1152, 5120), "gemm2": (5120, 576)}


def uniform_counts(T, g):
    c = [0] * mb.E
    for _ in range(T):
        for e in torch.randperm(mb.E, generator=g)[: mb.TOPK].tolist():
            c[e] += 1
    return c


class Pool:
    """Weights and scales of all 384 experts for one GEMM shape."""

    def __init__(self, M, K, ab):
        self.M, self.K, self.kb, self.ab = M, K, round_up(K // BLK, 8), ab
        E = mb.E
        # Bytes per weight row: MXFP4 packs two values per byte; every FP4 code is finite.
        self.kw = K // 2 if ab == 3 else K
        self.w = torch.randint(0, 255, (E, M, self.kw), dtype=torch.uint8, device=DEV) & (0xFF if ab == 3 else 0x77)
        self.s = swizzle_gfx950(torch.full((E, M, self.kb), 120, dtype=torch.uint8, device=DEV))


def time_case(shim, pool, T, R, reps, seed):
    """µs per launch for T tokens, compacted, averaged over R routing samples."""
    M, K, kb, kw, E = pool.M, pool.K, pool.kb, pool.kw, mb.E
    g = torch.Generator().manual_seed(seed)
    G = min(E, T * mb.TOPK)
    samples = []
    for _ in range(R):
        c = uniform_counts(T, g)
        order = sorted(range(E), key=lambda e: -c[e])[:G]
        samples.append(([c[e] for e in order], order))
    P = round_up(T * mb.TOPK + G * (BLK - 1), BLK)
    b = torch.randint(0, 255, (P, K), dtype=torch.uint8, device=DEV) & 0x77
    bs = swizzle_gfx950(torch.full((P, kb), 120, dtype=torch.uint8, device=DEV))
    d = torch.zeros((P, M), dtype=torch.bfloat16, device=DEV)
    # One plan per routing sample; the plans keep the addresses of their arrays, so the
    # arrays live in `launches` until the timing is done.
    launches = []
    for counts, experts in samples:
        offs, o = [], 0
        for cnt in counts:
            offs.append(o)
            o += round_up(cnt, BLK)
        ex = torch.tensor(experts, device=DEV, dtype=torch.int64)
        of = torch.tensor(offs, device=DEV, dtype=torch.int64)
        n_arr = torch.tensor(counts, device=DEV, dtype=torch.int32)
        a_sp, b_sp = pool.s.data_ptr() + ex * (M * kb), bs.data_ptr() + of * kb
        ws = torch.zeros(64 << 20, dtype=torch.uint8, device=DEV)
        plan, kname = shim.create(M, K, T, T * mb.TOPK, n_arr, a_sp, b_sp, ws.numel(), groups=G, ab=pool.ab)
        launches.append((plan, pool.w.data_ptr() + ex * (M * kw), b.data_ptr() + of * K,
                         d.data_ptr() + of * (M * 2), ws, n_arr, a_sp, b_sp))

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for plan, a_p, b_p, d_p, ws, *_ in launches:
            shim.run(plan, a_p, b_p, d_p, ws)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=s):
        for plan, a_p, b_p, d_p, ws, *_ in launches:
            shim.run(plan, a_p, b_p, d_p, ws)
    for _ in range(2):
        graph.replay()
    torch.cuda.synchronize()
    t0, t1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    t0.record()
    for _ in range(reps):
        graph.replay()
    t1.record()
    torch.cuda.synchronize()
    us = t0.elapsed_time(t1) * 1e3 / (reps * R)

    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(2):
            for plan, a_p, b_p, d_p, ws, *_ in launches:
                shim.run(plan, a_p, b_p, d_p, ws)
        torch.cuda.synchronize()
    gemm_us = sum(getattr(e, "device_time_total", 0) for e in prof.key_averages() if "Cijk" in e.key) / (2 * R)
    tokens = T * mb.TOPK
    active = sum(1 for cnt in samples[0][0] if cnt)
    return dict(us=us, gemm_us=gemm_us, kernel=kname, groups=G, active_first_sample=active,
                weight_TBps=active * M * kw / gemm_us / 1e6, TFLOPs=2.0 * M * K * tokens / gemm_us / 1e6)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiles", default=",".join(TILES),
                    help="comma-separated kernel-name substrings, each optionally as label=substring")
    ap.add_argument("--shapes", default=",".join(SHAPES))
    ap.add_argument("--tokens", default="1,4,16,64,256,1024,4096")
    ap.add_argument("--samples", type=int, default=16)
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--no-check", action="store_true")
    ap.add_argument("--ab", type=int, choices=[0, 3], default=0,
                    help="0 = MXFP8 x MXFP8, 3 = MXFP4 weights x MXFP8 tokens")
    ap.add_argument("--out", default=os.path.join(HERE, "tile_sweep.json"))
    args = ap.parse_args()
    torch.backends.cuda.preferred_blas_library("cublas")
    shim = Shim()
    print("hipblaslt:", shim.hipblaslt_lib, flush=True)
    tiles = [t.split("=", 1) if "=" in t else (t, t) for t in args.tiles.split(",")]
    shapes = {s: SHAPES[s] for s in args.shapes.split(",")}
    tokens = [int(t) for t in args.tokens.split(",")]
    out = dict(hipblaslt_lib=shim.hipblaslt_lib, checks=[], timing=[])

    if not args.no_check:
        gch = torch.Generator().manual_seed(1)

        def rnd(n, hi, zero_frac):
            c = torch.randint(0, hi + 1, (n,), generator=gch)
            c[torch.rand(n, generator=gch) < zero_frac] = 0
            return c.tolist()

        cases = [("G=1", [100]), ("G=5 with empties", [0, 37, 0, 64, 1]), ("G=65", rnd(65, 100, 0.2)),
                 ("G=384, only last", [0] * 383 + [70]), ("G=384, sparse", rnd(384, 20, 0.8)),
                 ("G=384, dense", rnd(384, 150, 0.0)), ("G=1000, sparse", rnd(1000, 40, 0.9))]
        for tile, match in tiles:
            os.environ["GG_KERNEL_MATCH"] = match
            for shape, (M, K) in shapes.items():
                for name, c in cases:
                    r = check_case(shim, M, K, c, ab=args.ab)
                    r.update(case=name, tile=tile, shape=shape)
                    print("check", json.dumps(r), flush=True)
                    out["checks"].append(r)
        print("checks:", "ALL OK" if all(r["ok"] for r in out["checks"]) else "FAILURES", flush=True)

    for shape, (M, K) in shapes.items():
        pool = Pool(M, K, args.ab)
        for T in tokens:
            for tile, match in tiles:
                os.environ["GG_KERNEL_MATCH"] = match
                r = time_case(shim, pool, T, args.samples, args.reps, seed=T)
                r.update(tile=tile, shape=shape, T=T)
                assert match in r["kernel"], r["kernel"]
                r["kernel_name"] = r["kernel"]
                r["kernel"] = tile
                print(json.dumps(r), flush=True)
                out["timing"].append(r)
        del pool
        torch.cuda.empty_cache()
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
