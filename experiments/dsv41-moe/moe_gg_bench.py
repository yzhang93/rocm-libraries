"""One DSv4.1-Flash routed-MoE layer per GPU (TP=4): aiter FlyDSL (as served) vs hipBLASLt
grouped MXFP8 GEMM (PR 13581) with the MXFP4 weights upcast to MXFP8.

Both arms use the same MXFP4 weights and MXFP8 (1x32 E8M0) activations, so their math is the
same: every E2M1 value is exact in E4M3 and the block scales do not change.

aiter arm: aiter.fused_moe called with the arguments vLLM passes for DSv4.1
(oracle/mxfp4.py AITER_MXFP4_BF16 + rocm_aiter_fused_experts): weights padded to
intermediate 640 and shuffled gate/up interleaved, AITER_BF16_FP8_MOE_BOUND=0 (FP8
activations), dispatch policy 2, swiglu_limit 10.

hipBLASLt arm (unfused glue in torch/aiter ops):
  route counts -> per-expert 32-aligned token slots -> MXFP8 quant of x (T rows) -> gather
  -> swizzle scales -> grouped GEMM1 (gate_up, 1152x5120) -> clamped SiLU*up -> MXFP8 quant
  -> swizzle -> grouped GEMM2 (down, 5120x576) -> gather + top-k weighted sum.

Timing: CUDA graphs, one per routing sample, replayed round-robin so the 256 MB MALL does
not keep expert weights resident between calls. Kernel breakdown: torch profiler on eager
runs.
"""

import argparse
import ctypes
import json
import math
import os
import sys

os.environ.setdefault("AITER_BF16_FP8_MOE_BOUND", "0")
os.environ.setdefault("TORCH_BLAS_PREFER_HIPBLASLT", "0")

import torch

from aiter import ActivationType, QuantType, dtypes
from aiter.fused_moe import fused_moe
from aiter.ops.flydsl.moe_common import GateMode
from aiter.ops.quant import per_1x32_mx_quant_hip
from aiter.ops.shuffle import shuffle_scale, shuffle_weight

E, TOPK, H, I, I_PAD = 384, 6, 5120, 576, 640
SWIGLU_LIMIT = 10.0
BLK = 32
DEV = "cuda"
HERE = os.path.dirname(os.path.abspath(__file__))

FP4_VALUES = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]


def round_up(v, m):
    return (v + m - 1) // m * m


def swizzle_gfx950(s):
    """Scale tensor [..., R, C] (R % 32 == 0, C % 8 == 0) to the gfx950 pre-swizzled layout
    (DGen::preSwizzleScalesGFX950, the same permutation as aiter e8m0_shuffle)."""
    *lead, R, C = s.shape
    v = s.reshape(-1, R // 32, 2, 16, C // 8, 2, 4)
    return v.permute(0, 1, 4, 6, 3, 5, 2).contiguous().view(*lead, R, C)


def pad_cols(s, cols, value=127):
    if s.shape[-1] == cols:
        return s.contiguous()
    out = torch.full((*s.shape[:-1], cols), value, dtype=s.dtype, device=s.device)
    out[..., : s.shape[-1]] = s
    return out


def fp4_to_fp8(packed):
    """Packed E2M1 pairs (low nibble first) [..., K/2] uint8 -> E4M3 [..., K]."""
    lut = torch.tensor(FP4_VALUES, dtype=torch.float32, device=packed.device).to(
        torch.float8_e4m3fn
    ).view(torch.uint8)
    lo = lut[(packed & 0xF).long()]
    hi = lut[(packed >> 4).long()]
    return torch.stack((lo, hi), dim=-1).reshape(*packed.shape[:-1], -1).view(torch.float8_e4m3fn)


def e8m0_to_f32(s):
    return torch.exp2(s.float() - 127.0)


def fp4_dequant(packed, scale):
    lut = torch.tensor(FP4_VALUES, dtype=torch.float32, device=packed.device)
    v = torch.stack((lut[(packed & 0xF).long()], lut[(packed >> 4).long()]), dim=-1)
    v = v.reshape(*packed.shape[:-1], -1)
    return (v.view(*v.shape[:-1], -1, BLK) * e8m0_to_f32(scale)[..., None]).reshape(v.shape)


def mxfp8_quant(x):
    q, s = per_1x32_mx_quant_hip(x, quant_dtype=dtypes.fp8, scale_type=dtypes.fp8_e8m0, shuffle=False)
    return q, s.view(torch.uint8)


def mxfp8_dequant(q, s):
    v = q.float()
    return (v.view(*v.shape[:-1], -1, BLK) * e8m0_to_f32(s)[..., None]).reshape(v.shape)


def act(h):
    g, u = h[..., :I], h[..., I:]
    return torch.nn.functional.silu(g.clamp(max=SWIGLU_LIMIT)) * u.clamp(-SWIGLU_LIMIT, SWIGLU_LIMIT)


# ----------------------------------------------------------------------------- weights
class Weights:
    def __init__(self, seed=0, fp4=False):
        """fp4: hand hipBLASLt the MXFP4 weights as they are (MXFP4 x MXFP8 grouped kernel)
        instead of upcasting them to MXFP8."""
        self.fp4 = fp4
        g = torch.Generator(device=DEV).manual_seed(seed)
        # Unpadded MXFP4 experts: w13 = [gate; up] (E, 2I, H), w2 (E, H, I).
        self.w13 = torch.randint(0, 256, (E, 2 * I, H // 2), dtype=torch.uint8, device=DEV, generator=g)
        self.w2 = torch.randint(0, 256, (E, H, I // 2), dtype=torch.uint8, device=DEV, generator=g)
        # Exponents chosen so activations stay O(1) through both GEMMs.
        self.s13 = torch.randint(118, 121, (E, 2 * I, H // BLK), dtype=torch.uint8, device=DEV, generator=g)
        self.s2 = torch.randint(119, 122, (E, H, I // BLK), dtype=torch.uint8, device=DEV, generator=g)
        self._aiter()
        self._hipblaslt()

    def _aiter(self):
        # vLLM layout: intermediate padded to 640 per gate/up half, zeros in the padding.
        w13 = torch.zeros((E, 2 * I_PAD, H // 2), dtype=torch.uint8, device=DEV)
        w13[:, :I] = self.w13[:, :I]
        w13[:, I_PAD : I_PAD + I] = self.w13[:, I:]
        s13 = torch.full((E, 2 * I_PAD, H // BLK), 127, dtype=torch.uint8, device=DEV)
        s13[:, :I] = self.s13[:, :I]
        s13[:, I_PAD : I_PAD + I] = self.s13[:, I:]
        w2 = torch.zeros((E, H, I_PAD // 2), dtype=torch.uint8, device=DEV)
        w2[:, :, : I // 2] = self.w2
        s2 = torch.full((E, H, I_PAD // BLK), 127, dtype=torch.uint8, device=DEV)
        s2[:, :, : I // BLK] = self.s2
        fp4 = torch.float4_e2m1fn_x2
        self.a_w13 = shuffle_weight(w13.view(fp4), is_guinterleave=True, gate_up=True)
        self.a_w2 = shuffle_weight(w2.view(fp4), is_guinterleave=True, gate_up=False)
        e8 = torch.float8_e8m0fnu
        self.a_s13 = shuffle_scale(s13.view(e8).reshape(-1, s13.shape[-1]), E, True, True)
        self.a_s2 = shuffle_scale(s2.view(e8).reshape(-1, s2.shape[-1]), E, True, False)
        del w13, w2, s13, s2

    def _hipblaslt(self):
        if self.fp4:
            self.h_w13, self.h_w2, self.ab, kbytes = self.w13, self.w2, 3, 2
        else:
            self.h_w13 = fp4_to_fp8(self.w13)  # (E, 1152, 5120)
            self.h_w2 = fp4_to_fp8(self.w2)  # (E, 5120, 576)
            self.ab, kbytes = 0, 1
        self.kb1 = round_up(H // BLK, 8)  # 160
        self.kb2 = round_up(I // BLK, 8)  # 24
        self.h_s13 = swizzle_gfx950(pad_cols(self.s13, self.kb1))
        self.h_s2 = swizzle_gfx950(pad_cols(self.s2, self.kb2))
        ar = torch.arange(E, device=DEV, dtype=torch.int64)
        self.p_w13 = self.h_w13.data_ptr() + ar * (2 * I * H // kbytes)
        self.p_w2 = self.h_w2.data_ptr() + ar * (H * I // kbytes)
        self.p_s13 = self.h_s13.data_ptr() + ar * (2 * I * self.kb1)
        self.p_s2 = self.h_s2.data_ptr() + ar * (H * self.kb2)

    def reference(self, x, topk_ids, topk_w, bf16_h=False):
        """FP32 math on the dequantized MXFP4 weights and MXFP8-quantized activations.
        bf16_h rounds the gate_up output to bf16 and applies the activation in bf16, as the
        unfused hipBLASLt path does."""
        xq, xs = mxfp8_quant(x)
        xd = mxfp8_dequant(xq, xs)
        y = torch.zeros((x.shape[0], H), dtype=torch.float32, device=DEV)
        for e in topk_ids.unique().tolist():
            rows, slot = (topk_ids == e).nonzero(as_tuple=True)
            w13 = fp4_dequant(self.w13[e], self.s13[e])
            w2 = fp4_dequant(self.w2[e], self.s2[e])
            h = xd[rows] @ w13.T
            h = act(h.to(torch.bfloat16)) if bf16_h else act(h).to(torch.bfloat16)
            hq, hs = mxfp8_quant(h)
            o = mxfp8_dequant(hq, hs) @ w2.T
            y.index_add_(0, rows, o * topk_w[rows, slot, None])
        return y


# ----------------------------------------------------------------------------- hipBLASLt
class Shim:
    def __init__(self):
        # GG_SHIM selects a shim built against another hipBLASLt (RUNPATH), e.g. libgg_shim_tiles.so.
        self.lib = ctypes.CDLL(os.path.join(HERE, os.environ.get("GG_SHIM", "libgg_shim.so")))
        L = self.lib
        vp, i64, u64 = ctypes.c_void_p, ctypes.c_int64, ctypes.c_uint64
        L.gg_create.argtypes = [ctypes.POINTER(vp), ctypes.c_int, i64, i64, i64, i64, ctypes.c_int, vp, vp, vp, u64]
        L.gg_create.restype = ctypes.c_int
        L.gg_run.argtypes = [vp, vp, vp, vp, vp, u64, vp]
        L.gg_run.restype = ctypes.c_int
        L.gg_kernel_name.argtypes = [vp]
        L.gg_kernel_name.restype = ctypes.c_char_p
        with open("/proc/self/maps") as f:
            libs = {l.split()[-1] for l in f if "libhipblaslt.so" in l}
        self.hipblaslt_lib = sorted(libs)[0] if libs else None

    def create(self, M, K, maxN, totalN, n_arr, a_sp, b_sp, ws_bytes, groups=E, ab=0):
        """ab: 0 = MXFP8, 1 = MXFP4, 3 = MXFP4 A x MXFP8 B (bf16 D); 2 = fp16 with fp16 D
        (a_sp, b_sp unused)."""
        p = ctypes.c_void_p()
        r = self.lib.gg_create(ctypes.byref(p), groups, M, K, maxN, totalN, ab, n_arr.data_ptr(),
                               a_sp.data_ptr() if a_sp is not None else None,
                               b_sp.data_ptr() if b_sp is not None else None, ws_bytes)
        if r != 0:
            raise RuntimeError(f"gg_create failed ({r}) M={M} K={K}")
        return p, self.lib.gg_kernel_name(p).decode()

    def run(self, plan, a_p, b_p, d_p, ws):
        r = self.lib.gg_run(plan, a_p.data_ptr(), b_p.data_ptr(), d_p.data_ptr(), ws.data_ptr(),
                            ws.numel(), torch.cuda.current_stream().cuda_stream)
        if r != 0:
            raise RuntimeError(f"gg_run failed ({r})")


class HipblasltMoE:
    """Persistent buffers for T tokens; forward() is graph-capturable.

    Tensile's grouped kernel finds each workgroup's group by a linear search over the
    groups, and the grid is bounded by M-tiles x (total tokens / MT + groups - 1), so every
    empty group costs every surplus workgroup a scan step. At most T * topk experts are
    active, so with compact=True the plan has G = min(E, T * topk) groups and the experts
    are placed in the slots by descending token count on the device.
    """

    WS_BYTES = 64 << 20

    def __init__(self, shim, w, T, compact=True):
        self.shim, self.w, self.T = shim, w, T
        TK = T * TOPK
        self.TK = TK
        self.G = min(E, TK) if compact else E
        G = self.G
        # Each active expert's slot range is padded to 32 rows; at most min(E, TK) are active.
        self.P = round_up(TK + min(E, TK) * (BLK - 1), BLK)
        P = self.P
        z = lambda *s, dt: torch.zeros(s, dtype=dt, device=DEV)
        self.counts = z(E, dt=torch.int32)
        self.slot_counts = z(G, dt=torch.int32)
        self.ones = torch.ones(TK, dtype=torch.int32, device=DEV)
        self.ar_tk = torch.arange(TK, dtype=torch.int64, device=DEV)
        self.tok_of_pair = self.ar_tk // TOPK
        self.xq = z(P, H, dt=torch.uint8)
        self.xs = torch.full((P, w.kb1), 127, dtype=torch.uint8, device=DEV)
        self.xs_swz = z(P, w.kb1, dt=torch.uint8)
        self.h1 = z(P, 2 * I, dt=torch.bfloat16)
        self.a2s = torch.full((P, w.kb2), 127, dtype=torch.uint8, device=DEV)
        self.a2s_swz = z(P, w.kb2, dt=torch.uint8)
        self.o2 = z(P, H, dt=torch.bfloat16)
        self.a1, self.as1, self.b1, self.bs1, self.d1 = (z(G, dt=torch.int64) for _ in range(5))
        self.a2, self.as2, self.b2, self.bs2, self.d2 = (z(G, dt=torch.int64) for _ in range(5))
        self.a2q = z(P, I, dt=torch.uint8)
        self.ws1 = z(self.WS_BYTES, dt=torch.uint8)
        self.ws2 = z(self.WS_BYTES, dt=torch.uint8)
        self.plan1, self.k1 = shim.create(2 * I, H, T, TK, self.slot_counts, self.as1, self.bs1,
                                          self.WS_BYTES, groups=G, ab=w.ab)
        self.plan2, self.k2 = shim.create(H, I, T, TK, self.slot_counts, self.as2, self.bs2,
                                          self.WS_BYTES, groups=G, ab=w.ab)

    def forward(self, x, topk_ids, topk_w):
        w, T, TK = self.w, self.T, self.TK
        flat = topk_ids.reshape(-1).long()
        self.counts.zero_()
        self.counts.scatter_add_(0, flat, self.ones)
        cnt = self.counts.long()
        padded = (cnt + BLK - 1) // BLK * BLK
        offs = torch.cumsum(padded, 0) - padded
        order = torch.argsort(flat, stable=True)
        se = flat[order]
        rank = self.ar_tk - (torch.cumsum(cnt, 0) - cnt)[se]
        dest = torch.empty_like(flat)
        dest[order] = offs[se] + rank
        slot_cnt, slot_e = torch.sort(cnt, descending=True)
        slot_e = slot_e[: self.G]
        self.slot_counts.copy_(slot_cnt[: self.G])
        so = offs[slot_e]

        xq, xs = mxfp8_quant(x)
        self.xq.index_copy_(0, dest, xq.view(torch.uint8).index_select(0, self.tok_of_pair))
        self.xs[:, : H // BLK].index_copy_(0, dest, xs.index_select(0, self.tok_of_pair))
        self.xs_swz.copy_(swizzle_gfx950(self.xs))
        self.a1.copy_(w.p_w13[slot_e])
        self.as1.copy_(w.p_s13[slot_e])
        self.b1.copy_(self.xq.data_ptr() + so * H)
        self.bs1.copy_(self.xs_swz.data_ptr() + so * w.kb1)
        self.d1.copy_(self.h1.data_ptr() + so * (2 * I * 2))
        self.shim.run(self.plan1, self.a1, self.b1, self.d1, self.ws1)

        a2q, a2s = mxfp8_quant(act(self.h1))
        self.a2q.copy_(a2q.view(torch.uint8))
        self.a2s[:, : I // BLK].copy_(a2s)
        self.a2s_swz.copy_(swizzle_gfx950(self.a2s))
        self.a2.copy_(w.p_w2[slot_e])
        self.as2.copy_(w.p_s2[slot_e])
        self.b2.copy_(self.a2q.data_ptr() + so * I)
        self.bs2.copy_(self.a2s_swz.data_ptr() + so * w.kb2)
        self.d2.copy_(self.o2.data_ptr() + so * (H * 2))
        self.shim.run(self.plan2, self.a2, self.b2, self.d2, self.ws2)

        o = self.o2.index_select(0, dest).view(T, TOPK, H).float()
        return (o * topk_w[..., None]).sum(1).to(torch.bfloat16)


def aiter_forward(w, x, topk_ids, topk_w):
    return fused_moe(
        x, w.a_w13, w.a_w2, topk_w, topk_ids,
        activation=ActivationType.Silu, quant_type=QuantType.per_1x32,
        w1_scale=w.a_s13, w2_scale=w.a_s2,
        moe_sorting_dispatch_policy=2, dtype=torch.bfloat16,
        hidden_pad=0, intermediate_pad=I_PAD - I,
        gate_mode=GateMode.INTERLEAVE.value, swiglu_limit=SWIGLU_LIMIT,
    )


# ----------------------------------------------------------------------------- harness
def make_inputs(T, n, seed):
    g = torch.Generator(device=DEV).manual_seed(seed)
    out = []
    for _ in range(n):
        x = torch.randn((T, H), dtype=torch.bfloat16, device=DEV, generator=g)
        logits = torch.rand((T, E), device=DEV, generator=g)
        tw, ti = logits.topk(TOPK, dim=-1)
        tw = (tw / tw.sum(-1, keepdim=True)).float()
        out.append((x, ti.to(torch.int32).contiguous(), tw.contiguous()))
    return out


def rel_err(a, b):
    a, b = a.float(), b.float()
    return float((a - b).norm() / b.norm().clamp_min(1e-30))


def time_graphs(fn, inputs, iters):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for inp in inputs:
            for _ in range(2):
                fn(*inp)
    torch.cuda.synchronize()
    graphs = []
    pool = torch.cuda.graph_pool_handle()
    for inp in inputs:
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, pool=pool, stream=s):
            fn(*inp)
        graphs.append(g)
    torch.cuda.synchronize()
    for g in graphs:
        g.replay()
    torch.cuda.synchronize()
    t0, t1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    n = iters * len(graphs)
    t0.record()
    for i in range(n):
        graphs[i % len(graphs)].replay()
    t1.record()
    torch.cuda.synchronize()
    return t0.elapsed_time(t1) * 1e3 / n  # us per call


def kernel_breakdown(fn, inp, reps=5):
    from torch.profiler import ProfilerActivity, profile

    for _ in range(3):
        fn(*inp)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(reps):
            fn(*inp)
        torch.cuda.synchronize()
    rows = []
    for ev in prof.key_averages():
        dt = getattr(ev, "device_time_total", None) or getattr(ev, "cuda_time_total", 0)
        if dt > 0 and ev.count > 0:
            rows.append((ev.key, dt / reps, ev.count // reps))
    rows.sort(key=lambda r: -r[1])
    return rows


def classify(name):
    n = name.lower()
    if "cijk" in n:
        return "hipblaslt_gemm"
    if "moe1" in n or "stage1" in n:
        return "aiter_gemm1"
    if "moe2" in n or "stage2" in n:
        return "aiter_gemm2"
    return "other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", default="1,2,4,8,16,32,64,96,128,256,512,1024,2048,4096,8192,16384")
    ap.add_argument("--samples", type=int, default=16, help="routing samples (graphs) per token count")
    ap.add_argument("--iters", type=int, default=20, help="round-robin passes over the graphs")
    ap.add_argument("--check-tokens", default="1,7,64,1000")
    ap.add_argument("--out", default=os.path.join(HERE, "results", "moe_gg_bench.json"))
    ap.add_argument("--no-breakdown", action="store_true")
    ap.add_argument("--no-compact", action="store_true", help="plan with all 384 groups")
    ap.add_argument("--fp4-weights", action="store_true",
                    help="MXFP4 weights x MXFP8 activations grouped kernel, no weight upcast")
    args = ap.parse_args()

    torch.manual_seed(0)
    # This hipBLASLt build has only grouped kernels; keep torch's own GEMMs on rocBLAS.
    torch.backends.cuda.preferred_blas_library("cublas")
    shim = Shim()
    print("hipblaslt:", shim.hipblaslt_lib, flush=True)
    w = Weights(fp4=args.fp4_weights)
    print(f"weights: MXFP4 {(w.w13.numel() + w.w2.numel()) / 2**30:.2f} GiB, hipBLASLt reads "
          f"{'MXFP4' if w.fp4 else 'upcast MXFP8'} {(w.h_w13.numel() + w.h_w2.numel()) / 2**30:.2f} GiB",
          flush=True)
    results = {"config": dict(E=E, topk=TOPK, hidden=H, inter=I, inter_pad_aiter=I_PAD,
                              hipblaslt_weights="mxfp4" if w.fp4 else "mxfp8_upcast",
                              swiglu_limit=SWIGLU_LIMIT, samples=args.samples, iters=args.iters,
                              device=torch.cuda.get_device_name(), hipblaslt_lib=shim.hipblaslt_lib),
               "check": [], "timing": []}

    for T in [int(t) for t in args.check_tokens.split(",") if t]:
        (x, ti, tw), = make_inputs(T, 1, seed=1000 + T)
        hb = HipblasltMoE(shim, w, T, compact=not args.no_compact)
        y_h = hb.forward(x, ti, tw)
        y_a = aiter_forward(w, x, ti, tw)
        ref = w.reference(x, ti, tw)
        ref_bf16h = w.reference(x, ti, tw, bf16_h=True)
        torch.cuda.synchronize()
        c = dict(T=T, groups=hb.G, hipblaslt_vs_ref=rel_err(y_h, ref), aiter_vs_ref=rel_err(y_a, ref),
                 hipblaslt_vs_ref_bf16h=rel_err(y_h, ref_bf16h), aiter_vs_ref_bf16h=rel_err(y_a, ref_bf16h),
                 hipblaslt_vs_aiter=rel_err(y_h, y_a), kernel1=hb.k1, kernel2=hb.k2)
        print("check", json.dumps(c), flush=True)
        results["check"].append(c)
        del hb

    for T in [int(t) for t in args.tokens.split(",") if t]:
        # Above ~1K tokens every expert is active, so weights stream from HBM regardless.
        inputs = make_inputs(T, args.samples if T <= 1024 else 4, seed=T)
        hb = HipblasltMoE(shim, w, T, compact=not args.no_compact)
        t_h = time_graphs(hb.forward, inputs, args.iters)
        t_a = time_graphs(lambda x, ti, tw: aiter_forward(w, x, ti, tw), inputs, args.iters)
        r = dict(T=T, groups=hb.G, hipblaslt_us=t_h, aiter_us=t_a, aiter_over_hipblaslt=t_a / t_h)
        if not args.no_breakdown:
            for name, fn in (("hipblaslt", hb.forward),
                             ("aiter", lambda x, ti, tw: aiter_forward(w, x, ti, tw))):
                rows = kernel_breakdown(fn, inputs[0])
                r[f"{name}_kernels"] = [dict(name=k, us=u, calls=c) for k, u, c in rows]
                agg = {}
                for k, u, _ in rows:
                    agg[classify(k)] = agg.get(classify(k), 0.0) + u
                r[f"{name}_by_class_us"] = agg
        print("time", json.dumps({k: v for k, v in r.items() if not k.endswith("_kernels")}), flush=True)
        results["timing"].append(r)
        del hb, inputs
        torch.cuda.empty_cache()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=1)
    print("wrote", args.out)


if __name__ == "__main__":
    sys.exit(main())
