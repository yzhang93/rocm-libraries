# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""User kernel registration through the Python bindings.

The tier is process-global, so the registration sequence is one ordered test
rather than several. Split into separate tests they would each assume an empty
tier and interfere with one another depending on collection order, which is the
sort of flake that only shows up under -p no:randomly.
"""
import os
from pathlib import Path

import numpy as np
import pytest

import hipblaslt

c = hipblaslt._core
requires_gpu = pytest.mark.skipif(not c.hip_available(), reason="no HIP device")
pytestmark = pytest.mark.gpu

ml_dtypes = pytest.importorskip("ml_dtypes")

M, N, K = 16, 6144, 2048
USER_INDEX_BASE = 1 << 30

# A shipped bf16 TN shard doubles as a payload: it is a real compiled
# (.dat, .co) pair with the stem pairing the contract wants, so these tests do
# not depend on any tuning tool having been run. Discovered by glob rather than
# named, so the tests follow whatever GPU this build targets.
_LIBRARY_RELATIVE = "hipblaslt-install/lib/hipblaslt/library"
_PAYLOAD_GLOB = "TensileLibrary_BB_BB_*_Alik_Bljk_*.dat*"


def _device_pci_id():
    """PCI device id of GPU 0, e.g. "75a0", or None if it cannot be read.

    Shard filenames carry the device ids they were built for. Filtering on it
    keeps the test from registering kernels that belong to other GPUs, which
    would be inert here and would inflate the counts the assertions check.
    """
    try:
        import ctypes

        lib = ctypes.CDLL("libamdhip64.so")
        buf = ctypes.create_string_buffer(64)
        if lib.hipDeviceGetPCIBusId(buf, 64, 0) != 0:
            return None
        bus = buf.value.decode().lower()
        return (Path("/sys/bus/pci/devices") / bus / "device").read_text().strip().removeprefix("0x") or None
    except Exception:
        return None


def _payloads():
    """Candidate payloads for this GPU, biggest first.

    Shards for other device ids are filtered out by the id their filename
    carries, so the test never registers kernels that could not run here. A
    CU-count variant can still miss, so the caller walks what remains.

    Resolved against the source tree rather than the working directory, since a
    cwd-relative path makes these tests skip or run depending on where pytest
    was invoked from, which is a silent loss of coverage.
    """
    override = os.environ.get("HIPBLASLT_USER_KERNEL_PAYLOAD")
    if override:
        return [(override + ".dat", override + ".co")]

    pci_id = _device_pci_id()
    found = []
    for parent in Path(__file__).resolve().parents:
        library = parent / _LIBRARY_RELATIVE
        if not library.is_dir():
            continue
        for arch_dir in sorted(library.iterdir()):
            if not arch_dir.is_dir():
                continue
            for shard in arch_dir.glob(_PAYLOAD_GLOB):
                if pci_id and pci_id not in shard.name.lower():
                    continue
                stem = str(shard)
                for suffix in (".zlib", ".dat"):
                    if stem.endswith(suffix):
                        stem = stem[: -len(suffix)]
                if Path(stem + ".co").exists():
                    found.append((shard.stat().st_size, stem))
        if found:
            break

    if not found:
        pytest.skip("no registration payload in this build's installed library")
    return [(s + ".dat", s + ".co") for _, s in sorted(found, reverse=True)]


def _problem(handle, m=M, n=N, k=K):
    desc = c.MatmulDesc(c.ComputeType.COMPUTE_32F, c.DataType.R_32F)
    desc.set_attribute_int(c.MatmulDescAttr.TRANSA, c.Operation.OP_T)
    desc.set_attribute_int(c.MatmulDescAttr.TRANSB, c.Operation.OP_N)
    bf = c.DataType.R_16BF
    a = np.ones((k, m), np.float32).astype(ml_dtypes.bfloat16)
    b = np.ones((k, n), np.float32).astype(ml_dtypes.bfloat16)
    bufs = (
        hipblaslt.from_numpy(np.ascontiguousarray(a.T), bf),
        hipblaslt.from_numpy(np.ascontiguousarray(b.T), bf),
        hipblaslt.from_numpy(np.zeros((n, m), ml_dtypes.bfloat16), bf),
        hipblaslt.from_numpy(np.zeros((n, m), ml_dtypes.bfloat16), bf),
    )
    lay = (
        c.MatrixLayout(bf, k, m, k),
        c.MatrixLayout(bf, k, n, k),
        c.MatrixLayout(bf, m, n, m),
        c.MatrixLayout(bf, m, n, m),
    )
    pref = c.Preference()
    pref.set_max_workspace(64 * 1024 * 1024)
    return desc, bufs, lay, pref


def _top(handle, desc, lay, pref):
    got = c.heuristic(handle, desc, lay[0], lay[1], lay[2], lay[3], pref, 1)
    assert got, "no algo for this shape"
    return got[0]


@requires_gpu
def test_counts_and_provenance_without_registering():
    """Provenance is answerable for a shipped algo, with nothing registered."""
    with c.Handle() as h:
        desc, _, lay, pref = _problem(h)
        top = _top(h, desc, lay, pref)
        assert not c.is_user_kernel(top.algo)
        assert top.algo.solution_index < USER_INDEX_BASE

        registered, selectable = c.user_kernel_counts(h)
        assert registered >= 0 and selectable >= 0


@requires_gpu
def test_register_run_then_select():
    """The whole registration story, in the order it has to happen.

    Calling the heuristic before registering is load-bearing: it puts a
    selection result for the shape in the cache, so the final step proves the
    cached answer was discarded rather than merely absent.
    """
    payloads = _payloads()

    with c.Handle() as h:
        desc, bufs, lay, pref = _problem(h)
        dA, dB, dC, dD = bufs

        # Warm the cached selection result for this shape.
        shipped = _top(h, desc, lay, pref)
        assert not c.is_user_kernel(shipped.algo)

        # Walk candidates until one yields a kernel that serves this problem.
        # A shard built for another device id or CU count registers fine but
        # every kernel in it is rejected when validated against this GPU.
        chosen = None
        for dat, co in payloads:
            before_registered, _ = c.user_kernel_counts(h)

            indices = c.user_kernel_register(h, dat, co)
            assert indices, "payload produced no kernels"
            assert all(i >= USER_INDEX_BASE for i in indices), \
                "every assigned index must be in the user range"

            after_registered, after_selectable = c.user_kernel_counts(h)
            assert after_registered == before_registered + len(indices), \
                "registering must add exactly the kernels it reports"
            assert after_selectable == 0, "registering alone makes nothing selectable"

            # Registration must not disturb selection.
            assert _top(h, desc, lay, pref).algo.solution_index \
                == shipped.algo.solution_index, "registration changed selection"

            for idx in indices:
                got = c.get_algos_from_index(h, [idx])
                if not got:
                    continue
                need = c.is_algo_supported(
                    h, desc, 1.0, lay[0], lay[1], 0.0, lay[2], lay[3], got[0].algo)
                if need is not None and need <= 64 * 1024 * 1024:
                    chosen = (idx, got[0], need)
                    break
            if chosen:
                break

        assert chosen is not None, "no shipped shard has a kernel for this shape"
        idx, result, need = chosen

        assert c.is_user_kernel(result.algo)
        assert result.algo.solution_index == idx, "algo must round-trip its index"

        ws = hipblaslt.from_numpy(np.zeros(max(1, need), np.uint8), c.DataType.R_8I)
        c.matmul(h, desc, 1.0, dA, lay[0], dB, lay[1], 0.0,
                 dC, lay[2], dD, lay[3], result.algo, ws)
        out = dD.to_numpy().reshape(N, M).T.astype(np.float32)
        # Inputs are exactly 1.0, so every element is exactly K and bf16
        # represents that without rounding.
        assert np.all(out == float(K)), "registered kernel computed the wrong result"

        # Mapping the shape is what moves selection.
        c.user_kernel_set_exact_match(h, idx, M, N, 1, K)
        _, selectable = c.user_kernel_counts(h)
        assert selectable == 1

        picked = _top(h, desc, lay, pref)
        assert c.is_user_kernel(picked.algo), \
            "selection did not move to the mapped kernel, so the cached result was kept"
        assert picked.algo.solution_index == idx

        # An adjacent shape must be untouched: matching is exact.
        desc2, _, lay2, pref2 = _problem(h, m=32)
        other = _top(h, desc2, lay2, pref2)
        assert not c.is_user_kernel(other.algo), \
            "the mapping leaked to a shape nobody mapped"


@requires_gpu
def test_set_exact_match_rejects_unknown_index():
    """A non-user index cannot be mapped, so a typo fails loudly."""
    with c.Handle() as h:
        with pytest.raises(c.HipblasLtError):
            c.user_kernel_set_exact_match(h, 12345, M, N, 1, K)


@requires_gpu
def test_refresh_without_open_library_fails():
    """Replay is meaningless with no library open, and says so."""
    with c.Handle() as h:
        with pytest.raises(c.HipblasLtError):
            c.user_kernel_refresh(h)
