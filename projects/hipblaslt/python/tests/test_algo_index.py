# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Explicit solution-index selection.

The heuristic path hands back an opaque algo and the caller takes what it is
given. These tests cover the other direction: naming a Tensile solution index
and getting a usable algo back. That is the capability a tuning loop needs in
order to execute one specific kernel and measure it, rather than whatever the
selector currently prefers.
"""
import numpy as np
import pytest
import hipblaslt

c = hipblaslt._core
requires_gpu = pytest.mark.skipif(not c.hip_available(), reason="no HIP device")
pytestmark = pytest.mark.gpu

M = N = K = 32


def _problem():
    """Build a column-major f32 GEMM problem and return descriptors plus buffers."""
    A = np.random.rand(M, K).astype(np.float32)
    B = np.random.rand(K, N).astype(np.float32)
    ref = A @ B
    desc = c.MatmulDesc(c.ComputeType.COMPUTE_32F, c.DataType.R_32F)
    dA = c.DeviceArray.from_numpy(np.ascontiguousarray(A.T), c.DataType.R_32F)
    dB = c.DeviceArray.from_numpy(np.ascontiguousarray(B.T), c.DataType.R_32F)
    dC = c.DeviceArray.from_numpy(np.zeros((N, M), np.float32), c.DataType.R_32F)
    dD = c.DeviceArray.from_numpy(np.zeros((N, M), np.float32), c.DataType.R_32F)
    la = c.MatrixLayout(c.DataType.R_32F, M, K, M)
    lb = c.MatrixLayout(c.DataType.R_32F, K, N, K)
    lc = c.MatrixLayout(c.DataType.R_32F, M, N, M)
    ld = c.MatrixLayout(c.DataType.R_32F, M, N, M)
    pref = c.Preference()
    pref.set_max_workspace(64 * 1024 * 1024)
    return desc, (dA, dB, dC, dD), (la, lb, lc, ld), pref, ref


@requires_gpu
def test_heuristic_algo_exposes_solution_index():
    """A heuristic algo carries a real Tensile solution index, not just a rank."""
    with c.Handle() as h:
        desc, _, (la, lb, lc, ld), pref, _ = _problem()
        results = c.heuristic(h, desc, la, lb, lc, ld, pref, 32)
        assert results
        # index is the rank within this result list; solution_index is the
        # library-wide identity and is what can be round-tripped.
        assert results[0].algo.index == 0
        assert results[0].algo.solution_index >= 0


@requires_gpu
def test_solution_index_round_trips():
    """Resolving a heuristic algo's index yields an algo with the same index."""
    with c.Handle() as h:
        desc, _, (la, lb, lc, ld), pref, _ = _problem()
        results = c.heuristic(h, desc, la, lb, lc, ld, pref, 32)
        assert results
        want = results[0].algo.solution_index

        resolved = c.get_algos_from_index(h, [want])
        assert resolved, f"solution index {want} did not resolve"
        assert resolved[0].algo.solution_index == want


@requires_gpu
def test_pinned_index_matches_heuristic_numerically():
    """An algo built from an index computes the same result as the heuristic algo."""
    with c.Handle() as h:
        desc, (dA, dB, dC, dD), (la, lb, lc, ld), pref, ref = _problem()
        results = c.heuristic(h, desc, la, lb, lc, ld, pref, 32)
        assert results
        want = results[0].algo.solution_index

        resolved = c.get_algos_from_index(h, [want])
        assert resolved
        r = resolved[0]

        ws = c.DeviceArray.from_numpy(
            np.zeros(max(1, r.workspace_size), np.uint8), c.DataType.R_8I)
        c.matmul(h, desc, 1.0, dA, la, dB, lb, 0.0, dC, lc, dD, ld, r.algo, ws)
        out = dD.to_numpy().reshape(N, M).T
        np.testing.assert_allclose(out, ref, rtol=1e-3, atol=1e-3)


@requires_gpu
def test_unknown_index_is_skipped_not_fatal():
    """A bogus index is dropped rather than raising, so sweeps can be sloppy."""
    with c.Handle() as h:
        # Far above any shipped solution count, and below the 2^30 user-kernel
        # range, so this is simply absent from the solution map.
        resolved = c.get_algos_from_index(h, [999_999_999])
        assert resolved == []


@requires_gpu
def test_get_all_algos_enumerates_library():
    """Enumeration returns many candidates for a plain f32 NN problem type."""
    with c.Handle() as h:
        algos = c.get_all_algos(
            h, c.GemmType.GEMM,
            c.Operation.OP_N, c.Operation.OP_N,
            c.DataType.R_32F, c.DataType.R_32F,
            c.DataType.R_32F, c.DataType.R_32F,
            c.ComputeType.COMPUTE_32F)
        assert len(algos) > 1
        assert all(a.algo.solution_index >= 0 for a in algos)


@requires_gpu
def test_is_algo_supported_reports_workspace():
    """A heuristic algo is supported for the problem it was selected for."""
    with c.Handle() as h:
        desc, _, (la, lb, lc, ld), pref, _ = _problem()
        results = c.heuristic(h, desc, la, lb, lc, ld, pref, 32)
        assert results
        ws = c.is_algo_supported(
            h, desc, 1.0, la, lb, 0.0, lc, ld, results[0].algo)
        assert ws is not None
        assert ws >= 0


@requires_gpu
def test_names_resolve_for_heuristic_algo():
    """Solution and kernel names are non-empty, which Phase 5 needs to confirm
    that a registered Geko kernel is the one actually dispatched."""
    with c.Handle() as h:
        desc, _, (la, lb, lc, ld), pref, _ = _problem()
        results = c.heuristic(h, desc, la, lb, lc, ld, pref, 32)
        assert results
        assert c.solution_name(h, results[0].algo)
        assert c.kernel_name(h, results[0].algo)
