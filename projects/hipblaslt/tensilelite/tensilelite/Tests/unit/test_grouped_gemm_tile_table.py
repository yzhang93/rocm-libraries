# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Grouped GEMM workgroups find their gemm in a tile table instead of searching.

hipBLASLt's grouped GEMM C API can follow the device user args with a table of one
(gemm, tile within the gemm) pair per tile and sets bit 29 of gemm_count
(GROUPED_TILE_TABLE_FLAG). A workgroup then loads the total tile count and its own entry,
ends if it is past the total, and otherwise loads its gemm's sizes and joins the path the
search leaves at label_FOUND. Every grouped kernel clears the flag from the gemm count, so
kernels without the lookup ignore it.

The policy is checked on kernel dictionaries; the emitted code on the kernels of the common
GroupedGemm config (Tests/common/groupedgemm/grouped_gemm.yaml).
"""

import re
from types import SimpleNamespace

import pytest

from Tensile.KernelWriterAssembly import KernelWriterAssembly
from test_grouped_gemm_surplus_exit import _emit_grouped_kernels, _kernel

pytestmark = pytest.mark.unit

_LOOKUP = re.compile(r"Grouped Gemm:: tile table lookup.*?\n(label_GG_TileTableSkip\w*):", re.S)
_COUNT_MASK = "0x1fffffff"


def _grouped(**overrides):
    kernel = _kernel(**overrides)
    kernel["ProblemType"] = {"GroupedGemm": True, "SupportUserArgs": True}
    return kernel


def _supported(kernel, gsuItems=()):
    writer = SimpleNamespace(
        groupedSurplusEarlyExit=lambda k: KernelWriterAssembly.groupedSurplusEarlyExit(None, k))
    gsuModule = SimpleNamespace(items=lambda: list(gsuItems))
    return KernelWriterAssembly.groupedTileTableSupported(writer, kernel, gsuModule)


def test_flag_is_below_the_arg_type():
    assert KernelWriterAssembly.GROUPED_TILE_TABLE_FLAG == 1 << 29
    assert KernelWriterAssembly.GROUPED_GEMM_COUNT_MASK == (1 << 29) - 1


@pytest.mark.parametrize("overrides, gsuItems, expected", [
    ({}, (), True),
    ({"GlobalSplitU": -1}, (), True),
    ({}, ("per-gemm GSU",), False),
    ({"GlobalSplitU": -1, "GlobalSplitUAlgorithm": "MultipleBufferSingleKernel"}, (), False),
    ({"TileProcessingStrategy": "StreamK"}, (), False),
    ({"ClusterDim": [2, 1]}, (), False),
])
def test_tile_table_policy(overrides, gsuItems, expected):
    assert _supported(_grouped(**overrides), gsuItems) is expected


def test_tile_table_needs_device_user_args():
    kernel = _grouped()
    kernel["ProblemType"]["SupportUserArgs"] = False
    assert _supported(kernel) is False


def _lookup_lines(src):
    m = _LOOKUP.search(src)
    if not m:
        return None
    lines = [l.split("//")[0].strip() for l in m.group(0).splitlines()[1:]]
    return [l for l in lines if l and not l.startswith("/*")]


def _check_lookup(lines):
    text = "\n".join(lines)
    assert "0x20000000" in lines[0] and _COUNT_MASK in lines[1], lines[:2]
    assert re.search(r"s_cmp_eq_u32 s\w+, 2\n", text), "ArgType == 2 check"
    loads = [l for l in lines if l.startswith("s_load")]
    assert len(loads) == 3, loads
    total, entry, sizes = loads
    assert total.startswith("s_load_dword ") and entry.startswith("s_load_dwordx2 ")
    assert total.split(",")[1].strip() == entry.split(",")[1].strip(), "same table address"
    assert sizes.startswith("s_load_dwordx4 ") \
        and sizes.split(",")[1].strip() == "s[sgprKernArgAddress:sgprKernArgAddress+1]", sizes
    wait = lines.index("s_waitcnt lgkmcnt(0)")
    assert lines.index(entry) < wait < lines.index("s_endpgm") < lines.index(sizes)
    assert lines[-2].startswith("s_branch label_FOUND"), lines[-2:]


def _check_kernels(results):
    assert results and all(err == 0 for _k, _s, err in results)
    with_table = 0
    for kernel, src, _err in results:
        lines = _lookup_lines(src)
        if lines is not None:
            with_table += 1
            _check_lookup(lines)
            assert "gemm sizes and args" in src
        else:
            assert "gemm count without the tile table flag" in src
            assert not KernelWriterAssembly.groupedSurplusEarlyExit(None, kernel)
    return with_table


def test_grouped_kernels_look_up_the_tile_table():
    """Kernels that may end surplus workgroups look the gemm up in the table when the flag
    is set; the others clear the flag and search."""
    assert _check_kernels(_emit_grouped_kernels()), "no kernel of the config uses the tile table"


def test_tile_table_with_the_linear_search(monkeypatch):
    """The lookup does not depend on the wave-parallel search."""
    monkeypatch.setattr(KernelWriterAssembly, "groupedGemmWaveSearchSupported",
                        lambda self, kernel, gsuModule: False)
    assert _check_kernels(_emit_grouped_kernels()), "no kernel of the config uses the tile table"
