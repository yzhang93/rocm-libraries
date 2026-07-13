# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Pytest suite for the fused GEMM+RstdScale (K3) Subtile epilogue (gfx950, bf16).

Exercises a comprehensive set of M shapes with fixed N_hidden=64, verifying:
  - y output (bf16, tol=2e-2): (h2 @ W1.T) * rstd[:, None]

The fixture is parametrized over wg_n (MIWaveGroup[1]):
  wg_n=1: single-wave, N_out=64
  wg_n=2: two-wave, N_out=128

N_out is pinned to MacroTile1 = 64 * wg_n (row-containment invariant).
N_hidden (GEMM2 contraction dim) is fixed to 64 for all shapes.
"""

import math
import os
import sys

import numpy as np
import pytest

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
TENSILE_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
TOOLS_DIR    = os.path.join(TENSILE_ROOT, "tools")
for _d in (TENSILE_ROOT, TOOLS_DIR):
    if _d not in sys.path:
        sys.path.insert(0, _d)

try:
    import amdgpu_exec
    import ml_dtypes
    _HAVE_DEPS = True
except ImportError:
    _HAVE_DEPS = False

requires_gfx950 = pytest.mark.skipif(
    not _HAVE_DEPS or not (lambda: amdgpu_exec.get_chip().startswith("gfx950"))(),
    reason="requires amdgpu_exec + ml_dtypes and a gfx950 GPU",
)

# MIWaveGroup[1] values to exercise: single-wave and two-wave.
_WG_N = [1, 2]

# Shape list: (M, label). N_hidden is fixed to 64 for all shapes.
_SHAPES = [
    # full-tile M (multiples of MT0=64).
    (   64,  "M64"),
    (  128,  "M128"),
    (  256,  "M256"),
    (  512,  "M512"),
    ( 1024,  "M1024"),
    ( 2048,  "M2048"),
    # edge-tile M (non-multiples of MT0=64).
    (    1,  "M1"),
    (   16,  "M16"),
    (   32,  "M32"),
    (   48,  "M48"),
    (   63,  "M63"),
    (   65,  "M65"),
    (   80,  "M80"),
    (  100,  "M100"),
    (  130,  "M130"),
    (  200,  "M200"),
    (  513,  "M513"),
    ( 1000,  "M1000"),
]

# N_hidden for GEMM2 contraction dim (fixed per solution).
_N_HIDDEN = 64


# ---------------------------------------------------------------------------
# Session-scoped fixtures: build + compile once per wg_n
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session", params=_WG_N, ids=[f"wgN{w}" for w in _WG_N])
def k3_kernel(request):
    """Build, assemble, and compile the K3 RstdScale kernel once per wg_n."""
    sys.path.insert(0, TENSILE_ROOT)
    from gemm_partialrms_colv2_helpers import (
        setup_tensile,
        build_k3_solution,
        generate_asm,
    )

    wg_n    = request.param
    chip    = amdgpu_exec.get_chip()
    N_out   = _N_HIDDEN * wg_n   # MacroTile1 = 64 * wg_n

    assembler, isaInfoMap, debugConfig = setup_tensile(chip)
    solution = build_k3_solution(chip, assembler, isaInfoMap,
                                 N_hidden=_N_HIDDEN, N_out=N_out, wg_n=wg_n)
    asm_str, kernel_name = generate_asm(solution, assembler, debugConfig)
    hsaco = amdgpu_exec.compile_asm_to_hsaco(asm_str, chip)
    return solution, kernel_name, hsaco, chip


# ---------------------------------------------------------------------------
# Helper: run one shape and return comparison data
# ---------------------------------------------------------------------------

def _run_shape(solution, kernel_name, hsaco, chip, M, N_hidden):
    from gemm_partialrms_colv2_helpers import compute_sk3_dp_args

    MT0   = solution["MacroTile0"]
    N_out = solution["MacroTile1"]   # row-containment invariant

    M_padded = math.ceil(M / MT0) * MT0
    numWG    = math.ceil(M / MT0) * math.ceil(N_out / N_out)  # == ceil(M/MT0)

    rng = np.random.default_rng(seed=M * 10000 + N_hidden)

    h2_f32 = np.asfortranarray(rng.random((N_hidden, M), dtype=np.float32) * 0.1)
    w1_f32 = np.asfortranarray(rng.random((N_hidden, N_out), dtype=np.float32) * 0.1)
    h2_bf16 = np.asfortranarray(h2_f32.astype(ml_dtypes.bfloat16))
    w1_bf16 = np.asfortranarray(w1_f32.astype(ml_dtypes.bfloat16))

    c_bf16 = np.zeros((M, N_out), dtype=ml_dtypes.bfloat16, order='F')
    y_bf16 = np.zeros((M, N_out), dtype=ml_dtypes.bfloat16, order='F')

    # Generate synthetic rstd (uniform positive values).
    rstd_ref = rng.random(M, dtype=np.float32) * 0.5 + 0.5

    # Pad rstd to M_padded.
    rstd_padded = np.zeros(M_padded, dtype=np.float32)
    rstd_padded[:M] = rstd_ref

    # Numpy reference.
    h2_ref = np.asarray(h2_bf16).astype(np.float32)   # N_hidden x M col-major
    w1_ref = np.asarray(w1_bf16).astype(np.float32)   # N_hidden x N_out
    h3     = h2_ref.T @ w1_ref                         # M x N_out, fp32
    y_ref  = (h3 * rstd_ref[:, np.newaxis]).astype(ml_dtypes.bfloat16)

    sk_args      = compute_sk3_dp_args(M, N_out, N_hidden, solution)
    stagger_u    = solution.get("StaggerU", 0)
    su_map       = solution.get("StaggerUMapping", 0)
    ss_shift     = solution.get("_staggerStrideShift", 0)
    su_word      = (su_map << 13) | ((ss_shift << 8) & 0x1F00) | (stagger_u & 0xFF)
    kernel_info0 = np.uint32((su_word << 16) | (solution["GlobalSplitU"] & 0x3FFF))
    wgmxcc       = solution.get("WorkGroupMappingXCC", 1)
    kernel_info1 = np.uint32((wgmxcc << 16) | (solution["WorkGroupMapping"] & 0xFFFF))

    ws_dummy    = np.zeros(4, dtype=np.float32)
    flags_dummy = np.zeros(4, dtype=np.float32)

    args = [
        np.uint32(1), kernel_info0, kernel_info1, np.uint32(numWG),
        np.uint32(M), np.uint32(N_out), np.uint32(1), np.uint32(N_hidden),
        amdgpu_exec.InOutArray(y_bf16),
        amdgpu_exec.InputArray(c_bf16),
        amdgpu_exec.InputArray(h2_bf16),
        amdgpu_exec.InputArray(w1_bf16),
        amdgpu_exec.InputArray(ws_dummy),
        amdgpu_exec.InputArray(flags_dummy),
        np.uint32(M), np.uint32(0),
        np.uint32(M), np.uint32(0),
        np.uint32(N_hidden), np.uint32(0),
        np.uint32(N_hidden), np.uint32(0),
        np.float32(1.0), np.float32(0.0),
        sk_args["iters_per_tile"],
        sk_args["magic_iters_per_tile"],
        sk_args["shift_iters_per_tile"],
        sk_args["sk_iters_per_wg"],
        sk_args["sk_grid"],
        sk_args["sk_tiles"],
        amdgpu_exec.InputArray(rstd_padded),
    ]

    result_holder = {}

    def capture(arguments):
        result_holder["y_gpu"] = np.asarray(arguments[8].array).copy()

    amdgpu_exec.execute_hsaco(
        hsaco=hsaco,
        kernel_name=kernel_name,
        arguments=args,
        grid_dim=(numWG, 1, 1),
        block_dim=(solution["NumThreads"], 1, 1),
        num_iterations=1,
        verify_fn=capture,
    )

    y_gpu_f32 = result_holder["y_gpu"].astype(np.float32)
    y_ref_f32 = np.asarray(y_ref).astype(np.float32)

    return y_gpu_f32, y_ref_f32, M


# ---------------------------------------------------------------------------
# Parametrised test
# ---------------------------------------------------------------------------

@requires_gfx950
@pytest.mark.parametrize("M,label", _SHAPES, ids=[s[1] for s in _SHAPES])
def test_k3_shape(k3_kernel, M, label):
    """Verify K3 (RstdScale) output y for shape M x N_out x N_hidden."""
    solution, kernel_name, hsaco, chip = k3_kernel
    N_out    = solution["MacroTile1"]
    N_hidden = _N_HIDDEN

    y_gpu_f32, y_ref_f32, M_actual = \
        _run_shape(solution, kernel_name, hsaco, chip, M, N_hidden)

    # Check y (bf16 output, always upcast to fp32 before comparing).
    rtol_y, atol_y = 2e-2, 2e-2
    bad_y = np.where(
        ~np.isfinite(y_gpu_f32[:M]) |
        (np.abs(y_gpu_f32[:M] - y_ref_f32[:M]) > atol_y + rtol_y * np.abs(y_ref_f32[:M]))
    )
    n_bad_y = len(bad_y[0])
    assert n_bad_y == 0, (
        f"y mismatch: M={M} N_out={N_out} N_hidden={N_hidden} ({label}): "
        f"{n_bad_y} elements out of tolerance. "
        f"max_abs={np.nanmax(np.abs(y_gpu_f32[:M] - y_ref_f32[:M])):.3e}, "
        f"first bad row={bad_y[0][0]}, col={bad_y[1][0]}: "
        f"gpu={y_gpu_f32[bad_y[0][0], bad_y[1][0]]:.6f} "
        f"ref={y_ref_f32[bad_y[0][0], bad_y[1][0]]:.6f}"
    )
