# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Generate a LibraryLogic YAML for the RstdScale K3 (GEMM+RstdScale) kernel.

Builds the solution from gemm_partialrms_colv2_helpers.build_k3_solution,
converts its state to YAML-safe primitives, and writes a 12-element
LibraryLogic list file that TensileCreateLibrary can compile into a device
library (.hsaco/.co/.dat).

Usage:
    python tools/gen_rstdscale_logic.py [--chip gfx950] [--out-dir /path/to/logic]
"""

import argparse
import os
import sys
import yaml

_TENSILELITE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_TOOLS_DIR   = os.path.join(_TENSILELITE, "tools")
for _d in (_TOOLS_DIR, _TENSILELITE):
    if _d not in sys.path:
        sys.path.insert(0, _d)

from gemm_partialrms_colv2_helpers import setup_tensile, build_k3_solution
from Tensile.SolutionStructs.Naming import getKernelNameMin

# All gfx950 device IDs (see Tensile/Common/Architectures.py).
_GFX950_DEVICE_IDS = [
    "Device 74a1",
    "Device 75a0", "Device 75b0",
    "Device 75a2", "Device 75b2",
    "Device 75a3", "Device 75b3",
    "Device 75a8", "Device 75b8",
]


def _to_primitive(v):
    """Recursively convert Tensile-typed objects to YAML-safe Python primitives."""
    if isinstance(v, (bool, int, float, str)) or v is None:
        return v
    if isinstance(v, list):
        return [_to_primitive(x) for x in v]
    if isinstance(v, dict):
        return {k: _to_primitive(val) for k, val in v.items()}
    # DataType and similar objects expose .value as an int.
    if hasattr(v, "value") and isinstance(v.value, int):
        return v.value
    # ActivationType and other Tensile enums: stringify (yields e.g. "None").
    if hasattr(v, "__module__") and "Tensile" in str(getattr(v, "__module__", "")):
        return str(v)
    # SemanticVersion / ISA namedtuple: convert to [major, minor, patch].
    if hasattr(v, "major"):
        return [v.major, v.minor, v.patch]
    return str(v)


def _problem_type_state(pt):
    return {k: _to_primitive(pt[k]) for k in sorted(pt.keys())}


def _solution_state(sol, kernelName):
    d = {k: _to_primitive(v) for k, v in dict(sol).items()}

    # ISA namedtuple → [major, minor, patch].
    if hasattr(d.get("ISA"), "major"):
        d["ISA"] = [d["ISA"].major, d["ISA"].minor, d["ISA"].patch]

    # ProblemType lives in the outer YAML element; drop it from the solution dict.
    d.pop("ProblemType", None)

    # InternalSupportParams namedtuple → plain dict.
    isp = d.get("InternalSupportParams")
    if isp is not None and not isinstance(isp, dict):
        try:
            d["InternalSupportParams"] = dict(isp._asdict())
        except AttributeError:
            d["InternalSupportParams"] = dict(vars(isp))

    # Mandatory bookkeeping fields.
    d.update({"SolutionIndex": 0, "Valid": True, "KernelNameMin": kernelName,
              "BaseName": kernelName, "CustomKernelName": ""})

    # Type coercions required to pass TensileCreateLibrary validation.
    if "BufferStore" in d:
        d["BufferStore"] = bool(d["BufferStore"])
    if "GlobalReadPerMfma" in d:
        d["GlobalReadPerMfma"] = float(d["GlobalReadPerMfma"])
    if "StaggerUStride" in d:
        d["StaggerUStride"] = int(d["StaggerUStride"])

    # The C++ runtime computes the GPU workgroup dispatch size as
    # WorkGroup[0] * WorkGroup[1] * WorkGroup[2].  For subtile MFMA kernels
    # WorkGroup stores the MFMA tile shape [mfma_m, mfma_n, 1] which is
    # unrelated to NumThreads.  Override it so the dispatch equals NumThreads,
    # preventing hipErrorLaunchFailure when NumThreads < WorkGroup product.
    num_threads = d.get("NumThreads", 0)
    if num_threads > 0:
        d["WorkGroup"] = [num_threads, 1, 1]

    return d


def generate(chip: str, outDir: str) -> str:
    """Build K3 solutions for wg_n in [1, 2] and write LibraryLogic YAMLs to outDir.

    Returns the path to the last written file.
    """
    logicDir = os.path.join(outDir, chip, "Equality")
    os.makedirs(logicDir, exist_ok=True)

    print(f"Setting up TensileLite for {chip} ...")
    assembler, isaInfoMap, _ = setup_tensile(chip)

    canonicalLogic = None
    for wg_n in [1, 2]:
        print(f"Building K3 (GEMM+RstdScale) solution wg_n={wg_n} ...")
        sol = build_k3_solution(chip, assembler, isaInfoMap,
                                N_hidden=64, N_out=64, wg_n=wg_n)
        assert sol["Valid"], "solution validation failed"

        kernel = sol.getKernels()[0]
        kernel.duplicate = False
        kernelName = getKernelNameMin(kernel, splitGSU=False)
        print(f"  KernelNameMin : {kernelName}")
        print(f"  MacroTile     : {sol['MacroTile0']}x{sol['MacroTile1']}")
        print(f"  NumThreads    : {sol['NumThreads']}")

        ptState  = _problem_type_state(sol["ProblemType"])
        solState = _solution_state(sol, kernelName)

        # 12-element LibraryLogic list expected by TensileCreateLibrary.
        logic = [
            {"MinimumRequiredVersion": "5.0.0"},
            chip,                   # ScheduleName
            chip,                   # ArchitectureName
            _GFX950_DEVICE_IDS,
            ptState,
            [solState],
            [0],                    # solution index ordering
            [[[4096, 64, 1, 64], [0, 0]]],  # ExactLogic (placeholder size)
            None,                   # RangeLogic
            None,
            "DeviceEfficiency",
            "Equality",             # LibraryType
        ]

        suffix = f"_wgn{wg_n}"
        outPath = os.path.join(logicDir, f"RstdScale_BF16_TN{suffix}.yaml")
        with open(outPath, "w") as fh:
            yaml.dump(logic, fh, default_flow_style=None, sort_keys=True,
                      allow_unicode=True)

        # Round-trip sanity check.
        with open(outPath) as fh:
            check = yaml.safe_load(fh)
        assert isinstance(check, list) and len(check) == 12, "YAML round-trip failed"
        assert check[5][0]["KernelNameMin"] == kernelName, \
            "KernelNameMin mismatch after round-trip"

        print(f"\nWrote LibraryLogic YAML → {outPath}")

        if wg_n == 1:
            canonicalLogic = (logic, kernelName)

    # Also write the canonical name (wg_n=1) without suffix for the test script.
    logic1, kernelName1 = canonicalLogic
    canonicalPath = os.path.join(logicDir, "RstdScale_BF16_TN.yaml")
    with open(canonicalPath, "w") as fh:
        yaml.dump(logic1, fh, default_flow_style=None, sort_keys=True,
                  allow_unicode=True)
    with open(canonicalPath) as fh:
        check1 = yaml.safe_load(fh)
    assert isinstance(check1, list) and len(check1) == 12, "YAML round-trip failed"
    assert check1[5][0]["KernelNameMin"] == kernelName1, \
        "KernelNameMin mismatch after round-trip"
    print(f"Wrote canonical LibraryLogic YAML → {canonicalPath}")
    return canonicalPath


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--chip", default="gfx950",
                        help="GPU architecture (default: gfx950)")
    parser.add_argument("--out-dir", default="/tmp/rstdscale_logic",
                        help="Root output directory for logic YAMLs "
                             "(default: /tmp/rstdscale_logic)")
    args = parser.parse_args()
    generate(args.chip, args.out_dir)


if __name__ == "__main__":
    main()
