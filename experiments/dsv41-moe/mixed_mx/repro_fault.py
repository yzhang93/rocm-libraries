"""Reproduce the MXFP4 x MXFP8 grouped GEMM2 (5120 x 576) memory fault from gg_micro.py.

Usage: repro_fault.py <variant>
  micro   gg_micro.run_case, uniform T=16 counts, G=384 (the crashing case)
  check   gg_micro.check_case with the same counts (correctness, distinct data)
  ab0     the same run_case with the MXFP8 x MXFP8 kernel
"""

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import gg_micro as gm  # noqa: E402
import moe_gg_bench as mb  # noqa: E402


def uniform_counts(T, seed=0):
    g = torch.Generator().manual_seed(seed)
    c = [0] * mb.E
    for _ in range(T):
        for e in torch.randperm(mb.E, generator=g)[: mb.TOPK].tolist():
            c[e] += 1
    return c


def main():
    variant = sys.argv[1]
    M, K = int(os.environ.get("M", 5120)), int(os.environ.get("K", 576))
    T = int(os.environ.get("T", 16))
    torch.backends.cuda.preferred_blas_library("cublas")
    shim = mb.Shim()
    c = uniform_counts(T) if os.environ.get("LAST_ONLY") is None else [0] * (mb.E - 1) + [T]
    print("groups", len(c), "active", sum(1 for x in c if x), "max", max(c), "last active",
          max(i for i, x in enumerate(c) if x), flush=True)
    if variant == "where":
        # run_case inlined, printing each buffer's range to match against the fault address.
        groups, kb = len(c), mb.round_up(K // mb.BLK, 8)
        seg = [mb.round_up(x, mb.BLK) for x in c]
        offs = [0]
        for s in seg[:-1]:
            offs.append(offs[-1] + s)
        P = max(sum(seg), mb.BLK)
        dev = mb.DEV
        bufs = dict(
            w=torch.randint(0, 255, (groups, M, K // 2), dtype=torch.uint8, device=dev),
            ws=mb.swizzle_gfx950(torch.full((groups, M, kb), 120, dtype=torch.uint8, device=dev)),
            b=torch.randint(0, 255, (P, K), dtype=torch.uint8, device=dev) & 0x77,
            bs=mb.swizzle_gfx950(torch.full((P, kb), 120, dtype=torch.uint8, device=dev)),
            d=torch.zeros((P, M), dtype=torch.bfloat16, device=dev),
        )
        for k_, t in bufs.items():
            n = t.numel() * t.element_size()
            print(f"{k_}: {t.data_ptr():#x} .. {t.data_ptr() + n:#x} ({n} B)", flush=True)
        ar = torch.arange(groups, device=dev, dtype=torch.int64)
        o = torch.tensor(offs, device=dev, dtype=torch.int64)
        n_arr = torch.tensor(c, device=dev, dtype=torch.int32)
        a_sp = bufs["ws"].data_ptr() + ar * (M * kb)
        b_sp = bufs["bs"].data_ptr() + o * kb
        wsb = torch.zeros(64 << 20, dtype=torch.uint8, device=dev)
        plan, _ = shim.create(M, K, T, T * mb.TOPK, n_arr, a_sp, b_sp, wsb.numel(), groups=groups, ab=3)
        shim.run(plan, bufs["w"].data_ptr() + ar * (M * K // 2), bufs["b"].data_ptr() + o * K,
                 bufs["d"].data_ptr() + o * (M * 2), wsb)
        torch.cuda.synchronize()
        print("no fault", flush=True)
        return
    if variant == "check":
        print(gm.check_case(shim, M, K, c, T * mb.TOPK, T, ab=3), flush=True)
    else:
        print(gm.run_case(shim, len(c), M, K, c, T * mb.TOPK, T, ab=0 if variant == "ab0" else 3), flush=True)


if __name__ == "__main__":
    main()
