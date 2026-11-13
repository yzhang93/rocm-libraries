"""One DSv4.1-Flash routed-MoE layer with aiter's routing glue feeding hipBLASLt's grouped GEMMs.

Arms, all with MXFP4 weights and MXFP8 activations:
  aiter      aiter.fused_moe as served (sort, quant+sort, stage 1, quant+sort, stage 2).
  hb_torch   moe_gg_bench.HipblasltMoE: hipBLASLt grouped GEMMs, glue in torch ops.
  hb_aiter   hipBLASLt grouped GEMMs, glue from aiter where today's hipBLASLt API allows:
               sort      aiter moe_sorting (the serving path's two Opus kernels)
               quant     aiter fused MX quant + sort: x quantized in token order, scales
                         written in sorted order and the gfx950 swizzle
               gather    B rows copied into sorted order (hipBLASLt cannot gather)
               groups    per-group token counts and the ten per-group pointer arrays
               gemm1     grouped GEMM
               act       clamped SiLU * up, MXFP8 quant with swizzled scales
               gemm2     grouped GEMM
               combine   top-k weighted sum back to token order
The stages after "quant" are what a sorted-layout hipBLASLt API (gather, groups) and fused
epilogues (act, combine) would remove; aiter has no counterpart for gather and groups, and
fuses act and combine into its stage kernels except the MXFP8 quant (its second quant+sort).

Both hipBLASLt arms use the geko-tuned mixed kernels (hipblaslt-gg-f4f8-geko), selected per
token count (KERNELS). Run with GG_SHIM=libgg_shim_f4f8_geko.so."""

import argparse
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
from moe_gg_bench import BLK, DEV, E, H, I, TOPK, act, round_up  # noqa: E402

STAGES = ["sort", "quant", "gather", "groups", "gemm1", "act", "gemm2", "combine"]

# Fastest geko kernel per GEMM and token count (2026-10-30 report, section 4).
MATCH = json.load(open(os.path.join(HERE, "tiles", "geko_f4f8_match.json")))


def kernels(T):
    g1 = ("geko_g1_b6n32_64x64" if T <= 1 else "geko_g1_b244n32_128x32" if T <= 1024
          else "geko_g1_b384n96_256x128")
    g2 = ("geko_g2_b6n32_128x32" if T <= 16 else "geko_g2_b244n32+b384n32_512x32" if T <= 1024
          else "geko_g2_b384n96_256x128")
    return MATCH[g1], MATCH[g2]


def create(shim, match, *args, **kw):
    os.environ["GG_KERNEL_MATCH"] = match
    try:
        return shim.create(*args, **kw)
    finally:
        del os.environ["GG_KERNEL_MATCH"]


class Stage:
    """record_function labels, so the profiler attributes each kernel to its stage."""

    on = False

    def __init__(self, name):
        self.rf = torch.profiler.record_function("stage:" + name) if Stage.on else None

    def __enter__(self):
        if self.rf:
            self.rf.__enter__()

    def __exit__(self, *a):
        if self.rf:
            self.rf.__exit__(*a)


class HipblasltAiterGlue:
    """Persistent buffers for T tokens; forward() is graph-capturable."""

    WS_BYTES = 64 << 20

    def __init__(self, shim, w, T):
        assert w.fp4
        self.shim, self.w, self.T = shim, w, T
        self.TK = TK = T * TOPK
        self.G = G = min(E, TK)
        # aiter's sorted layout: every active expert padded to 32 rows, experts in id order.
        # At most min(TK, E) experts are active, each adding at most BLK - 1 padding rows, so
        # num_valid_ids <= P and the sort's outputs are sliced to P rows.
        self.P = P = round_up(TK + min(TK, E) * (BLK - 1), BLK)
        self.nblk = P // BLK
        z = lambda *s, dt: torch.zeros(s, dtype=dt, device=DEV)
        self.ar_rows = torch.arange(P, device=DEV, dtype=torch.int32)
        self.inv = z(TK + 1, dt=torch.int32)
        self.ar_blk = torch.arange(self.nblk, device=DEV, dtype=torch.int32)
        self.ones_blk = torch.ones(self.nblk, dtype=torch.int32, device=DEV)
        self.blocks = z(E + 1, dt=torch.int32)
        self.slot_counts = z(G, dt=torch.int32)
        self.xq = z(P, H, dt=torch.uint8)
        self.h1 = z(P, 2 * I, dt=torch.bfloat16)
        self.o2 = z(P, H, dt=torch.bfloat16)
        self.a1, self.as1, self.b1, self.bs1, self.d1 = (z(G, dt=torch.int64) for _ in range(5))
        self.a2, self.as2, self.b2, self.bs2, self.d2 = (z(G, dt=torch.int64) for _ in range(5))
        self.ws1 = z(self.WS_BYTES, dt=torch.uint8)
        self.ws2 = z(self.WS_BYTES, dt=torch.uint8)
        # Padded rows count as tokens of their group, so a group can exceed T.
        maxN = round_up(T, BLK)
        m1, m2 = kernels(T)
        self.plan1, self.k1 = create(shim, m1, 2 * I, H, maxN, P, self.slot_counts, self.as1, self.bs1,
                                     self.WS_BYTES, groups=G, ab=w.ab)
        self.plan2, self.k2 = create(shim, m2, H, I, maxN, P, self.slot_counts, self.as2, self.bs2,
                                     self.WS_BYTES, groups=G, ab=w.ab)

    def forward(self, x, topk_ids, topk_w):
        w, T, P = self.w, self.T, self.P
        with Stage("sort"):
            sids, sw, seids, nvalid, _ = moe_sorting(topk_ids, topk_w, E, H, torch.bfloat16, BLK,
                                                     dispatch_policy=2, accumulate=False)
            sids, seids = sids[:P], seids[: self.nblk]
        with Stage("quant"):
            xq, xs = fused_dynamic_mx_quant_moe_sort(x, sids, nvalid, T, TOPK, BLK, quant_dtype=dtypes.fp8)
        with Stage("gather"):
            # Rows past num_valid_ids and padding rows (token id T) become a dummy token.
            valid = self.ar_rows[: sids.numel()] < nvalid[0]
            tok = torch.where(valid, sids & 0xFFFFFF, T).long()
            self.xq[: tok.numel()].copy_(xq.view(torch.uint8).index_select(0, tok.clamp(max=T - 1)))
        with Stage("groups"):
            # Blocks per expert from the sorted expert id of each 32-row block; groups take
            # the padded rows, so they line up with the sorted, swizzled scales.
            nb = seids.numel()
            eid = torch.where(self.ar_blk[:nb] < nvalid[0] // BLK, seids, E).long()
            self.blocks.zero_()
            self.blocks.scatter_add_(0, eid, self.ones_blk[:nb])
            blk = self.blocks[:E].long()
            first = torch.cumsum(blk, 0) - blk
            cnt, slot_e = torch.sort(blk, descending=True)
            slot_e = slot_e[: self.G]
            self.slot_counts.copy_(cnt[: self.G] * BLK)
            so = first[slot_e] * BLK
            self.a1.copy_(w.p_w13[slot_e])
            self.as1.copy_(w.p_s13[slot_e])
            self.b1.copy_(self.xq.data_ptr() + so * H)
            self.bs1.copy_(xs.data_ptr() + so * w.kb1)
            self.d1.copy_(self.h1.data_ptr() + so * (2 * I * 2))
        with Stage("gemm1"):
            self.shim.run(self.plan1, self.a1, self.b1, self.d1, self.ws1)
        with Stage("act"):
            a2q, a2s = per_1x32_mx_quant_hip(act(self.h1), quant_dtype=dtypes.fp8,
                                             scale_type=dtypes.fp8_e8m0, shuffle=True)
        with Stage("groups"):
            self.a2.copy_(w.p_w2[slot_e])
            self.as2.copy_(w.p_s2[slot_e])
            self.b2.copy_(a2q.data_ptr() + so * I)
            self.bs2.copy_(a2s.data_ptr() + so * w.kb2)
            self.d2.copy_(self.o2.data_ptr() + so * (H * 2))
        with Stage("gemm2"):
            self.shim.run(self.plan2, self.a2, self.b2, self.d2, self.ws2)
        with Stage("combine"):
            # Each (token, slot) pair owns one sorted row; padding rows and rows past
            # num_valid_ids land on the dump entry TK.
            pair = torch.where(tok < T, tok * TOPK + (sids >> 24).long(), self.TK)
            self.inv.scatter_(0, pair, self.ar_rows)
            o = self.o2.index_select(0, self.inv[: self.TK]).view(T, TOPK, H).float()
            return (o * topk_w[..., None]).sum(1).to(torch.bfloat16)


def check_layouts(w, x, topk_ids, topk_w):
    """aiter's sorted, swizzled scales must match what hipBLASLt reads (mb.swizzle_gfx950)."""
    T = x.shape[0]
    sids, _, _, nvalid, _ = moe_sorting(topk_ids, topk_w, E, H, torch.bfloat16, BLK,
                                        dispatch_policy=2, accumulate=False)
    xq, xs = fused_dynamic_mx_quant_moe_sort(x, sids, nvalid, T, TOPK, BLK, quant_dtype=dtypes.fp8)
    nv = int(nvalid[0])
    tok = (sids[:nv] & 0xFFFFFF).long()
    real = tok < T
    q_ref, s_ref = mb.mxfp8_quant(x)
    s_sorted = torch.full((round_up(nv, BLK), w.kb1), 127, dtype=torch.uint8, device=DEV)
    s_sorted[:nv][real, : H // BLK] = s_ref[tok[real]]
    swz = mb.swizzle_gfx950(s_sorted)
    # Compare only the bytes of real rows (padding rows hold arbitrary scales): the swizzle is
    # a permutation, so swizzling a mask of the real rows marks where their bytes land.
    mask = torch.zeros_like(s_sorted)
    mask[:nv][real, : H // BLK] = 1
    m = mb.swizzle_gfx950(mask).bool()
    ok_scale = bool((xs.view(torch.uint8)[: swz.shape[0]][m] == swz[m]).all())
    ok_q = bool((xq.view(torch.uint8) == q_ref.view(torch.uint8)).all())
    h = torch.randn((256, 2 * I), dtype=torch.bfloat16, device=DEV)
    a = act(h)
    q2, s2 = per_1x32_mx_quant_hip(a, quant_dtype=dtypes.fp8, scale_type=dtypes.fp8_e8m0, shuffle=True)
    q2r, s2r = mb.mxfp8_quant(a)
    s2_ref = mb.swizzle_gfx950(mb.pad_cols(s2r, w.kb2))
    m2 = mb.swizzle_gfx950(mb.pad_cols(torch.ones_like(s2r), w.kb2, value=0)).bool()
    ok_act = bool((s2.view(torch.uint8)[:256, : w.kb2][m2] == s2_ref[m2]).all()) and \
        bool((q2.view(torch.uint8) == q2r.view(torch.uint8)).all())
    return dict(T=T, valid_rows=nv, x_quant_equal=ok_q, x_scale_layout_equal=ok_scale,
                act_quant_and_scale_layout_equal=ok_act, act_scale_shape=list(s2.shape))


def stage_times(fn, inp, reps=5):
    from torch.profiler import ProfilerActivity, profile

    Stage.on = True
    try:
        for _ in range(3):
            fn(*inp)
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU]) as prof:
            for _ in range(reps):
                fn(*inp)
            torch.cuda.synchronize()
    finally:
        Stage.on = False
    from torch.autograd import DeviceType

    # Count each kernel once: a stage's CPU range carries its kernels' time, and the GPU-side
    # copy of the range and the aten ops would count it again.
    out, kern = {}, {}
    for ev in prof.events():
        if ev.name.startswith("stage:") and ev.device_type == DeviceType.CPU:
            dt = getattr(ev, "device_time_total", None) or getattr(ev, "cuda_time_total", 0)
            out[ev.name[6:]] = out.get(ev.name[6:], 0.0) + dt / reps
    for ev in prof.key_averages():
        self_dt = getattr(ev, "self_device_time_total", None) or getattr(ev, "self_cuda_time_total", 0)
        if self_dt > 0 and ev.device_type == DeviceType.CUDA and not ev.key.startswith("stage:"):
            kern[ev.key] = (self_dt / reps, ev.count // reps)
    return out, kern


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", default="1,4,16,32,64,256,1024,4096")
    ap.add_argument("--samples", type=int, default=16)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--check-tokens", default="1,7,64,1000")
    ap.add_argument("--out", default=os.path.join(HERE, "results", "moe_glue_bench.json"))
    args = ap.parse_args()

    torch.manual_seed(0)
    torch.backends.cuda.preferred_blas_library("cublas")
    shim = mb.Shim()
    print("hipblaslt:", shim.hipblaslt_lib, flush=True)
    w = mb.Weights(fp4=True)
    res = dict(hipblaslt_lib=shim.hipblaslt_lib, layout=[], check=[], timing=[])

    for T in (1, 64, 1000):
        (x, ti, tw), = mb.make_inputs(T, 1, seed=500 + T)
        r = check_layouts(w, x, ti, tw)
        print("layout", json.dumps(r), flush=True)
        res["layout"].append(r)

    for T in [int(t) for t in args.check_tokens.split(",") if t]:
        (x, ti, tw), = mb.make_inputs(T, 1, seed=1000 + T)
        hb = HipblasltAiterGlue(shim, w, T)
        y = hb.forward(x, ti, tw)
        ref = w.reference(x, ti, tw, bf16_h=True)
        ref32 = w.reference(x, ti, tw)
        y_a = mb.aiter_forward(w, x, ti, tw)
        torch.cuda.synchronize()
        c = dict(T=T, groups=hb.G, kernel1=hb.k1, kernel2=hb.k2, vs_ref_bf16h=mb.rel_err(y, ref),
                 vs_ref=mb.rel_err(y, ref32), aiter_vs_ref=mb.rel_err(y_a, ref32))
        print("check", json.dumps(c), flush=True)
        res["check"].append(c)
        del hb

    bad = [r for r in res["layout"] if not all(v for k, v in r.items() if k.endswith("_equal"))]
    bad += [c for c in res["check"] if not c["vs_ref_bf16h"] < 0.01]
    if bad:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(res, f, indent=1)
        print("checks failed, no timing:", json.dumps(bad), flush=True)
        return 1

    for T in [int(t) for t in args.tokens.split(",") if t]:
        inputs = mb.make_inputs(T, args.samples if T <= 1024 else 4, seed=T)
        hb = HipblasltAiterGlue(shim, w, T)
        ht = mb.HipblasltMoE(shim, w, T)
        m1, m2 = kernels(T)
        ht.plan1, ht.k1 = create(shim, m1, 2 * I, H, T, T * TOPK, ht.slot_counts, ht.as1, ht.bs1,
                                 ht.WS_BYTES, groups=ht.G, ab=w.ab)
        ht.plan2, ht.k2 = create(shim, m2, H, I, T, T * TOPK, ht.slot_counts, ht.as2, ht.bs2,
                                 ht.WS_BYTES, groups=ht.G, ab=w.ab)
        r = dict(T=T, kernel1=hb.k1, kernel2=hb.k2,
                 aiter_us=mb.time_graphs(lambda x, ti, tw: mb.aiter_forward(w, x, ti, tw), inputs, args.iters),
                 hb_torch_us=mb.time_graphs(ht.forward, inputs, args.iters),
                 hb_aiter_us=mb.time_graphs(hb.forward, inputs, args.iters))
        st, kern = stage_times(hb.forward, inputs[0])
        # The shim launches the GEMMs outside torch, so the profiler cannot place them in a
        # stage; their time comes from the kernel list instead.
        r["hb_aiter_stage_us"] = {s: st.get(s, 0.0) for s in STAGES if not s.startswith("gemm")}
        r["hb_aiter_gemm_us"] = sum(u for k, (u, _) in kern.items() if "Cijk" in k)
        r["hb_aiter_kernels"] = [dict(name=k, us=u, calls=c) for k, (u, c) in sorted(kern.items(), key=lambda kv: -kv[1][0])]
        rows = mb.kernel_breakdown(lambda x, ti, tw: mb.aiter_forward(w, x, ti, tw), inputs[0])
        r["aiter_kernels"] = [dict(name=k, us=u, calls=c) for k, u, c in rows]
        rows = mb.kernel_breakdown(ht.forward, inputs[0])
        r["hb_torch_gemm_us"] = sum(u for k, u, _ in rows if "Cijk" in k)
        r["hb_torch_kernel_us"] = sum(u for _, u, _ in rows)
        print("time", json.dumps({k: v for k, v in r.items() if not k.endswith("_kernels")}), flush=True)
        res["timing"].append(r)
        del hb, ht, inputs
        torch.cuda.empty_cache()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=1)
    print("wrote", args.out)


if __name__ == "__main__":
    sys.exit(main())
