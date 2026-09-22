#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Register a tuned kernel and check its speedup survives the dispatch path.

A tuning tool's own benchmark number says nothing about whether an application
can reach the kernel. This times the shipped kernel, registers the tuned
payload, times it by index, checks it against an fp32 reference, then maps it
and confirms selection moves.

The payload must come from a tuning run. Unlike user_kernel_demo.py it is not
discovered, since a shipped shard would race the shipped library against
itself. Fixed at bf16 TN batch 1, which the layout and reference assume;
dimensions are configurable.

Usage:
    python verify_registered_kernel.py STEM [--shape M N K]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

try:
    import ml_dtypes
except ImportError:
    sys.exit("this example needs ml_dtypes: pip install ml_dtypes")

import hipblaslt

c = hipblaslt._core

# A tall, skinny GEMM: small M against large N and K. Memory-bound, and small M
# means tile choice matters, so a tuned kernel has room to win.
DEFAULT_SHAPE = (16, 6144, 2048)

# The user tier matches exactly and this example builds an unbatched problem,
# so the mapping is registered at batch 1.
BATCH = 1

DEFAULT_WARMUP = 200
DEFAULT_ITERS = 2000
DEFAULT_PASSES = 3
DEFAULT_MAX_WORKSPACE = 64 * 1024 * 1024

# bf16 rounding alone lands near 4e-3 on shapes this size, so this catches a
# kernel that is wrong rather than one that is merely imprecise.
DEFAULT_TOLERANCE = 0.01


def detail(label: str, text: str) -> None:
    print(f"     {label:<10} {text}")


def build_problem(handle, m: int, n: int, k: int, max_workspace: int):
    """A bf16 TN GEMM, laid out column-major the way hipBLASLt expects.

    Random inputs rather than the all-ones of the demo: this script judges
    numerics against an fp32 reference, and constant inputs would let a kernel
    that mishandles accumulation still look correct.
    """
    desc = c.MatmulDesc(c.ComputeType.COMPUTE_32F, c.DataType.R_32F)
    desc.set_attribute_int(c.MatmulDescAttr.TRANSA, c.Operation.OP_T)
    desc.set_attribute_int(c.MatmulDescAttr.TRANSB, c.Operation.OP_N)

    bf = c.DataType.R_16BF
    a = np.random.rand(k, m).astype(np.float32).astype(ml_dtypes.bfloat16)
    b = np.random.rand(k, n).astype(np.float32).astype(ml_dtypes.bfloat16)

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
    pref.set_max_workspace(max_workspace)
    return desc, buffers, layouts, pref


def run(handle, desc, buffers, layouts, algo, workspace_bytes: int) -> None:
    ws = hipblaslt.from_numpy(
        np.zeros(max(1, workspace_bytes), np.uint8), c.DataType.R_8I
    )
    c.matmul(
        handle, desc, 1.0,
        buffers["A"], layouts["a"], buffers["B"], layouts["b"], 0.0,
        buffers["C"], layouts["c"], buffers["D"], layouts["d"], algo, ws,
    )


def time_algo(handle, desc, buffers, layouts, algo, workspace_bytes: int,
              warmup: int, iters: int, passes: int) -> float:
    """Mean microseconds per call, best of several passes.

    The binding synchronises inside every matmul, so this includes a fixed
    per-call overhead and reads higher than a pure kernel timing. Both kernels
    pay it equally, so the comparison between them is still sound.
    """
    for _ in range(warmup):
        run(handle, desc, buffers, layouts, algo, workspace_bytes)

    best = None
    for _ in range(passes):
        start = time.perf_counter()
        for _ in range(iters):
            run(handle, desc, buffers, layouts, algo, workspace_bytes)
        elapsed = (time.perf_counter() - start) / iters * 1e6
        best = elapsed if best is None else min(best, elapsed)
    return best


def top_algo(handle, desc, layouts, pref):
    results = c.heuristic(
        handle, desc, layouts["a"], layouts["b"], layouts["c"], layouts["d"], pref, 1
    )
    if not results:
        raise RuntimeError("no algo for this shape")
    return results[0]


def usable_kernel(handle, desc, layouts, indices, max_workspace: int):
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


def reference_error(buffers, m: int, n: int, k: int) -> float:
    """Largest relative deviation of D from an fp32 recomputation."""
    out = buffers["D"].to_numpy().reshape(n, m).T.astype(np.float32)
    a32 = buffers["A"].to_numpy().reshape(m, k).T.astype(np.float32)
    b32 = buffers["B"].to_numpy().reshape(n, k).T.astype(np.float32)
    ref = a32.T @ b32
    return float(np.abs(out - ref).max() / np.abs(ref).max())


def main() -> int | str:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("payload", metavar="STEM",
                    help="payload stem; .dat/.dat.zlib and .co are appended")
    ap.add_argument("--shape", nargs=3, type=int, metavar=("M", "N", "K"),
                    default=DEFAULT_SHAPE,
                    help=f"problem dimensions (default: {' '.join(map(str, DEFAULT_SHAPE))})")
    ap.add_argument("--warmup", type=int, default=DEFAULT_WARMUP,
                    help=f"untimed calls before measuring (default: {DEFAULT_WARMUP})")
    ap.add_argument("--iters", type=int, default=DEFAULT_ITERS,
                    help=f"timed calls per pass (default: {DEFAULT_ITERS})")
    ap.add_argument("--passes", type=int, default=DEFAULT_PASSES,
                    help=f"timed passes, best wins (default: {DEFAULT_PASSES})")
    ap.add_argument("--max-workspace", type=int, default=DEFAULT_MAX_WORKSPACE,
                    metavar="BYTES",
                    help=f"workspace cap (default: {DEFAULT_MAX_WORKSPACE})")
    ap.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE,
                    help=f"max relative error allowed (default: {DEFAULT_TOLERANCE})")
    args = ap.parse_args()

    m, n, k = args.shape
    stem = args.payload
    dat, co = stem + ".dat", stem + ".co"
    if not (Path(dat).exists() or Path(dat + ".zlib").exists()):
        return f"payload shard not found: {dat}(.zlib)"
    if not Path(co).exists():
        return f"payload code object not found: {co}"

    with c.Handle() as handle:
        desc, buffers, layouts, pref = build_problem(
            handle, m, n, k, args.max_workspace
        )

        print("registered kernel verification")
        detail("problem", f"M={m}  N={n}  K={k}   bf16 TN")
        detail("payload", Path(stem).name)

        # Time the incumbent first, which also puts a selection result for this
        # shape in the cache so the move at the end has something stale to beat.
        shipped = top_algo(handle, desc, layouts, pref)
        detail("shipped", f"index {shipped.algo.solution_index:,}")
        t_shipped = time_algo(handle, desc, buffers, layouts, shipped.algo,
                              shipped.workspace_size, args.warmup, args.iters,
                              args.passes)
        detail("", f"{t_shipped:8.3f} us/call")

        indices = c.user_kernel_register(handle, dat, co)
        if not indices:
            return f"nothing registered from {stem}"
        chosen = usable_kernel(handle, desc, layouts, indices, args.max_workspace)
        if chosen is None:
            return (f"none of the {len(indices):,} registered kernels can serve "
                    f"M={m} N={n} K={k}; check the payload was tuned for this "
                    "shape and built for this device")
        idx, result, need = chosen

        detail("registered", f"{len(indices):,} kernels")
        detail("tuned", f"index {idx:,} (user={c.is_user_kernel(result.algo)})")
        t_tuned = time_algo(handle, desc, buffers, layouts, result.algo, need,
                            args.warmup, args.iters, args.passes)
        detail("", f"{t_tuned:8.3f} us/call")

        # Correctness before believing the speed.
        run(handle, desc, buffers, layouts, result.algo, need)
        relerr = reference_error(buffers, m, n, k)
        detail("", f"max rel err vs fp32 reference: {relerr:.4g}")

        c.user_kernel_set_exact_match(handle, idx, m, n, BATCH, k)
        picked = top_algo(handle, desc, layouts, pref)
        moved = c.is_user_kernel(picked.algo)
        detail("selection", f"index {picked.algo.solution_index:,} (user={moved})")

        speedup = t_shipped / t_tuned
        print(f"\nspeedup through the real dispatch path: {speedup:.3f}x "
              f"({(speedup - 1) * 100:+.1f}%)")

        if not moved:
            return "selection did not move to the registered kernel"
        if relerr > args.tolerance:
            return f"registered kernel is numerically wrong (rel err {relerr:.3g})"
    return 0


if __name__ == "__main__":
    result = main()
    if isinstance(result, str):
        sys.exit(f"\nERROR: {result}")
    sys.exit(result)
