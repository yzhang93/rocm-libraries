#!/usr/bin/env python3
"""Per-call routed-MoE kernel time in a vLLM torch-profiler trace (one rank).

One MoE call starts at a router launch (topkGatingSoftplusSqrt); the MoE kernels after it on the same stream, up to
the next router launch, belong to that call. ROCm traces carry no grid sizes, so token counts come from the step
names in profiler_out_*.txt instead; the stage-1 tile name gives a coarse size class per call.

usage: serving_mix.py <rank trace .pt.trace.json.gz> [--json out.json]
"""
import argparse
import collections
import gzip
import json
import re
from pathlib import Path

ROUTER = "topkGatingSoftplusSqrt"
STAGES = [
    ("router", re.compile(ROUTER)),
    ("sort", re.compile(r"moe_sorting")),
    ("quant_sort", re.compile(r"fused_mx_quant_moe_sort")),
    ("stage1", re.compile(r"mfma_moe1|flydsl_moe1")),
    ("stage2", re.compile(r"mfma_moe2|flydsl_moe2")),
]


def stage_of(name):
    for s, pat in STAGES:
        if pat.search(name):
            return s
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace", type=Path)
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()

    with gzip.open(args.trace, "rt") as f:
        data = json.load(f)
    kernels = [e for e in data["traceEvents"] if e.get("ph") == "X" and e.get("cat") == "kernel"]
    kernels.sort(key=lambda e: e["ts"])
    total_us = sum(e.get("dur", 0.0) for e in kernels)

    calls = []
    cur = None
    for e in kernels:
        s = stage_of(e["name"])
        if s == "router":
            cur = {"stream": e["args"].get("stream"), "us": collections.Counter(), "names": {}}
            calls.append(cur)
        if s and cur is not None and e["args"].get("stream") == cur["stream"]:
            cur["us"][s] += e.get("dur", 0.0)
            if s.startswith("stage"):
                cur["names"][s] = e["name"]
    print(f"\n{len(calls)} MoE calls; total kernel time {total_us / 1e3:.1f} ms")

    out = {"trace": str(args.trace), "total_kernel_us": total_us,
           "calls": [{"us": dict(c["us"]), "names": c["names"]} for c in calls]}
    if args.json:
        args.json.write_text(json.dumps(out))
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
