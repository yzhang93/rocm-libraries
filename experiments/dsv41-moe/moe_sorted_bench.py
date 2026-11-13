"""One DSv4.1-Flash routed-MoE layer with hipBLASLt grouped GEMM on aiter's sorted token layout
(C7: HIPBLASLT_MATMUL_DESC_MOE_*_POINTER_EXT, users/vivizhan/moe-sorted-layout).

Arms, all with MXFP4 weights and MXFP8 activations:
  aiter    aiter.fused_moe as served.
  hb_moe   aiter sorting and quantization feeding two MoE GEMMs; torch activation and combine:
             sort      aiter moe_sorting, block = the kernels' MacroTile1
             quant     aiter fused MX quant + sort (tokens in token order, scales sorted)
             gemm1     MoESortedLayout 2: expert per block, B rows gathered via sorted_ids,
                       h1 written in sorted order
             act       clamped SiLU * up, MXFP8 quant with swizzled scales (sorted order)
             gemm2     MoESortedLayout 1: expert per block, B read in sorted order
             combine   top-k weighted sum back to token order
Compared with moe_glue_bench.py's hb_aiter arm, the gather and group-setup stages are gone.
Run with GG_SHIM=libgg_shim_moe.so (links hipblaslt-moe)."""

import argparse
import ctypes
import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import moe_gg_bench as mb  # noqa: E402
from aiter import dtypes  # noqa: E402
from aiter.fused_moe import moe_sorting  # noqa: E402
from aiter.ops.quant import fused_dynamic_mx_quant_moe_sort, per_1x32_mx_quant_hip  # noqa: E402
from moe_gg_bench import DEV, E, H, I, TOPK, act, round_up  # noqa: E402
from moe_glue_bench import MATCH, Stage, stage_times  # noqa: E402

STAGES = ["sort", "quant", "act", "combine"]


def kernels(T):
    """(GEMM1 match, GEMM2 match, block): both GEMMs share the sorted layout, so their
    MacroTile1 must equal the sort block."""
    if T <= 1024:
        g2 = "geko_g2_b6n32_128x32" if T <= 16 else "geko_g2_b244n32+b384n32_512x32"
        return MATCH["geko_g1_b244n32_128x32"], MATCH[g2], 32
    return MATCH["geko_g1_b384n96_256x128"], MATCH["geko_g2_b384n96_256x128"], 128


class MoEShim(mb.Shim):
    def __init__(self):
        super().__init__()
        L = self.lib
        vp, i64, u64 = ctypes.c_void_p, ctypes.c_int64, ctypes.c_uint64
        L.moe_create.argtypes = [ctypes.POINTER(vp), i64, i64, i64, i64, i64, vp, vp, vp, u64]
        L.moe_create.restype = ctypes.c_int
        L.moe_run.argtypes = [vp, vp, vp, vp, vp, vp, vp, vp, vp, vp, u64, vp]
        L.moe_run.restype = ctypes.c_int

    def moe_create(self, match, M, K, P, rowsB, aStride, seids, nvalid, sids, ws_bytes):
        os.environ["GG_KERNEL_MATCH"] = match
        try:
            p = ctypes.c_void_p()
            r = self.lib.moe_create(ctypes.byref(p), M, K, P, rowsB, aStride, seids.data_ptr(),
                                    nvalid.data_ptr(), sids.data_ptr() if sids is not None else None,
                                    ws_bytes)
        finally:
            del os.environ["GG_KERNEL_MATCH"]
        if r != 0:
            raise RuntimeError(f"moe_create failed ({r}) M={M} K={K} P={P}")
        return p, self.lib.gg_kernel_name(p).decode()

    def moe_run(self, plan, a, a_s, b, b_s, d, seids, nvalid, sids, ws):
        r = self.lib.moe_run(plan, a.data_ptr(), a_s.data_ptr(), b.data_ptr(), b_s.data_ptr(),
                             d.data_ptr(), seids.data_ptr(), nvalid.data_ptr(),
                             sids.data_ptr() if sids is not None else None, ws.data_ptr(),
                             ws.numel(), torch.cuda.current_stream().cuda_stream)
        if r != 0:
            raise RuntimeError(f"moe_run failed ({r})")


class HipblasltMoESorted:
    """Persistent buffers for T tokens; forward() is graph-capturable."""

    WS_BYTES = 64 << 20

    def __init__(self, shim, w, T):
        assert w.fp4
        self.shim, self.w, self.T = shim, w, T
        self.TK = TK = T * TOPK
        m1, m2, self.blk = kernels(T)
        blk = self.blk
        # At most min(TK, E) experts are active, each adding at most blk - 1 padding rows.
        self.P = P = round_up(TK + min(TK, E) * (blk - 1), blk)
        z = lambda *s, dt: torch.zeros(s, dtype=dt, device=DEV)
        self.ar_rows = torch.arange(P, device=DEV, dtype=torch.int32)
        self.inv = z(TK + 1, dt=torch.int32)
        self.h1 = z(P, 2 * I, dt=torch.bfloat16)
        self.o2 = z(P, H, dt=torch.bfloat16)
        self.ws = z(self.WS_BYTES, dt=torch.uint8)
        ids = z(P + 1, dt=torch.int32)  # stands in for the sort outputs during selection
        self.plan1, self.k1 = shim.moe_create(m1, 2 * I, H, P, T, 2 * I * H, ids, ids, ids, self.WS_BYTES)
        self.plan2, self.k2 = shim.moe_create(m2, H, I, P, 0, H * I, ids, ids, None, self.WS_BYTES)

    def forward(self, x, topk_ids, topk_w):
        w, T, P, blk = self.w, self.T, self.P, self.blk
        with Stage("sort"):
            sids, sw, seids, nvalid, _ = moe_sorting(topk_ids, topk_w, E, H, torch.bfloat16, blk,
                                                     dispatch_policy=2, accumulate=False)
            sids, seids = sids[:P], seids[: P // blk]
        with Stage("quant"):
            xq, xs = fused_dynamic_mx_quant_moe_sort(x, sids, nvalid, T, TOPK, blk, quant_dtype=dtypes.fp8)
        self.shim.moe_run(self.plan1, w.h_w13, w.h_s13, xq, xs, self.h1, seids, nvalid, sids, self.ws)
        with Stage("act"):
            a2q, a2s = per_1x32_mx_quant_hip(act(self.h1), quant_dtype=dtypes.fp8,
                                             scale_type=dtypes.fp8_e8m0, shuffle=True)
        self.shim.moe_run(self.plan2, w.h_w2, w.h_s2, a2q, a2s, self.o2, seids, nvalid, None, self.ws)
        with Stage("combine"):
            # Each (token, slot) pair owns one sorted row; padding rows and rows past
            # num_valid_ids land on the dump entry TK.
            valid = self.ar_rows < nvalid[0]
            tok = torch.where(valid, sids & 0xFFFFFF, T).long()
            pair = torch.where(tok < T, tok * TOPK + (sids >> 24).long(), self.TK)
            self.inv.scatter_(0, pair, self.ar_rows)
            o = self.o2.index_select(0, self.inv[: self.TK]).view(T, TOPK, H).float()
            return (o * topk_w[..., None]).sum(1).to(torch.bfloat16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", default="1,4,16,32,64,256,1024,4096")
    ap.add_argument("--samples", type=int, default=16)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--check-tokens", default="1,7,64,1000,4096")
    ap.add_argument("--out", default=os.path.join(HERE, "results", "moe_sorted_bench.json"))
    args = ap.parse_args()

    torch.manual_seed(0)
    torch.backends.cuda.preferred_blas_library("cublas")
    shim = MoEShim()
    print("hipblaslt:", shim.hipblaslt_lib, flush=True)
    w = mb.Weights(fp4=True)
    res = dict(hipblaslt_lib=shim.hipblaslt_lib, check=[], timing=[])

    for T in [int(t) for t in args.check_tokens.split(",") if t]:
        (x, ti, tw), = mb.make_inputs(T, 1, seed=1000 + T)
        hb = HipblasltMoESorted(shim, w, T)
        y = hb.forward(x, ti, tw)
        ref = w.reference(x, ti, tw, bf16_h=True)
        ref32 = w.reference(x, ti, tw)
        y_a = mb.aiter_forward(w, x, ti, tw)
        torch.cuda.synchronize()
        c = dict(T=T, block=hb.blk, rows=hb.P, kernel1=hb.k1, kernel2=hb.k2,
                 vs_ref_bf16h=mb.rel_err(y, ref), vs_ref=mb.rel_err(y, ref32),
                 aiter_vs_ref=mb.rel_err(y_a, ref32))
        print("check", json.dumps({k: v for k, v in c.items() if not k.startswith("kernel")}), flush=True)
        res["check"].append(c)
        del hb

    bad = [c for c in res["check"] if not c["vs_ref_bf16h"] < 0.01]
    if bad:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(res, f, indent=1)
        print("checks failed, no timing:", json.dumps(bad), flush=True)
        return 1

    for T in [int(t) for t in args.tokens.split(",") if t]:
        inputs = mb.make_inputs(T, args.samples if T <= 1024 else 4, seed=T)
        hb = HipblasltMoESorted(shim, w, T)
        r = dict(T=T, block=hb.blk, kernel1=hb.k1, kernel2=hb.k2,
                 aiter_us=mb.time_graphs(lambda x, ti, tw: mb.aiter_forward(w, x, ti, tw), inputs, args.iters),
                 hb_moe_us=mb.time_graphs(hb.forward, inputs, args.iters))
        st, kern = stage_times(hb.forward, inputs[0])
        r["hb_moe_stage_us"] = {s: st.get(s, 0.0) for s in STAGES}
        # The shim launches the GEMMs outside torch; split them by problem type instead.
        r["hb_moe_stage_us"]["gemm1"] = sum(u for k, (u, _) in kern.items() if "_MoE2_" in k)
        r["hb_moe_stage_us"]["gemm2"] = sum(u for k, (u, _) in kern.items() if "_MoE1_" in k)
        r["hb_moe_kernels"] = [dict(name=k, us=u, calls=c) for k, (u, c) in sorted(kern.items(), key=lambda kv: -kv[1][0])]
        print("time", json.dumps({k: v for k, v in r.items() if not k.startswith(("kernel", "hb_moe_kernels"))}), flush=True)
        res["timing"].append(r)
        del hb, inputs
        torch.cuda.empty_cache()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=1)
    print("wrote", args.out)


if __name__ == "__main__":
    sys.exit(main())
