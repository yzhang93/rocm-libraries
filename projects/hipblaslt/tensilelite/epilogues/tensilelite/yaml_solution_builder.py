# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Build TensileLite Solution objects from benchmark YAML files.

Uses BenchmarkProcess -> constructForkPermutations -> _generate_single_solution
to drive the same pipeline that BenchmarkProblems uses, but serially and without
GPU benchmarking. This avoids the need to hand-construct Solution dicts.
"""

import os

_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
_YAML_DIR = os.path.join(os.path.dirname(_PKG_DIR), "yaml")
_DEFAULT_K1_YAML = os.path.join(_YAML_DIR, "gemm_partial_rms_k1_rowmajor.yaml")
_DEFAULT_K3_YAML = os.path.join(_YAML_DIR, "gemm_rstdscale_k3.yaml")


def _parseBenchmarkGroup(yamlPath, problemIdx=0, groupIdx=0):
    """Parse one BenchmarkProblems group into (process, step) via BenchmarkProcess."""
    from Tensile import LibraryIO
    from Tensile.BenchmarkStructs import BenchmarkProcess

    data = LibraryIO.readYAML(yamlPath)
    problems = data["BenchmarkProblems"][problemIdx]
    process = BenchmarkProcess(problems[0], problems[1 + groupIdx], False)
    return process, process[0]


def buildSolutionsFromYaml(yamlPath, assembler, isaInfoMap, debugConfig,
                           problemIdx=0, groupIdx=0, forkOverrides=None):
    """Build every forked Solution for a benchmark YAML group via the normal pipeline.

    forkOverrides: optional dict mapping a parameter name to a list of values to
    fork over; each key replaces that parameter's fork/constant value from the
    YAML (removed from constant params, added to fork params). Returns a list of
    valid Solution objects (None results and duplicates removed).
    """
    from Tensile.BenchmarkStructs import constructForkPermutations
    from Tensile.BenchmarkProblems import _generate_single_solution

    process, step = _parseBenchmarkGroup(yamlPath, problemIdx, groupIdx)

    forkParams = dict(step.forkParams)
    constantParams = dict(step.constantParams)
    if forkOverrides:
        for key, values in forkOverrides.items():
            constantParams.pop(key, None)
            forkParams[key] = values

    solutions = []
    seen = set()
    for perm in constructForkPermutations(forkParams, step.paramGroups):
        solution = _generate_single_solution(
            perm, process.problemType, constantParams,
            assembler, debugConfig, isaInfoMap,
        )
        if solution is None or solution in seen:
            continue
        seen.add(solution)
        solutions.append(solution)
    return solutions


def _baseMatrixInstruction(yamlPath, problemIdx=0, groupIdx=0):
    """Return the first 9-element MatrixInstruction defined in the YAML group."""
    _process, step = _parseBenchmarkGroup(yamlPath, problemIdx, groupIdx)
    mi = step.constantParams.get("MatrixInstruction")
    if mi is None:
        mi = step.forkParams["MatrixInstruction"][0]
    return list(mi)


def problemSizesFromYaml(yamlPath, problemIdx=0, groupIdx=0):
    """Return the ProblemSizes from a benchmark YAML as a list of raw size tuples.

    Each entry is the Exact size tuple (free0, free1, batch, bound, ...).
    The caller interprets the leading four elements according to the problem type.
    """
    _process, step = _parseBenchmarkGroup(yamlPath, problemIdx, groupIdx)
    return [tuple(int(x) for x in p.sizes) for p in step.problemSizes.problems]


def readTestAxes(yamlPath, section, mt1=None):
    """Read test axis lists from the TestAxes section of a benchmark YAML.

    section: key under TestAxes (e.g. "K1", "Pipeline").

    Returns a dict with keys present in that section:
      "M":       [(m, label), ...] — expanded from MMultipliers/MFractions/MOffsets/MFixed
                 against mt1 (required when any M* key is present).
      "NHidden": [...]
      "K":       [...]
    """
    from Tensile import LibraryIO
    data = LibraryIO.readYAML(yamlPath)
    cfg = data.get("TestAxes", {}).get(section, {})
    if not cfg:
        raise KeyError(f"TestAxes.{section} not found in {yamlPath}")

    result = {}

    if "NHidden" in cfg:
        result["NHidden"] = list(cfg["NHidden"])
    if "K" in cfg:
        result["K"] = list(cfg["K"])

    mKeys = {"MMultipliers", "MFractions", "MOffsets", "MFixed"}
    if mKeys & set(cfg):
        if mt1 is None:
            raise ValueError(f"mt1 required to expand M shapes in TestAxes.{section}")
        seen = set()
        mShapes = []

        def addM(m, label):
            m = max(1, m)
            if m not in seen:
                seen.add(m)
                mShapes.append((m, label))

        for mult in cfg.get("MMultipliers", []):
            addM(mt1 * mult, f"{mult}xMT1")
        for num, den in cfg.get("MFractions", []):
            addM(mt1 * num // den, f"MT1_{num}d{den}")
        for baseMult, delta in cfg.get("MOffsets", []):
            m = mt1 * abs(baseMult) + delta
            sign = "p" if delta >= 0 else "m"
            addM(m, f"{abs(baseMult)}MT1{sign}{abs(delta)}")
        for m in cfg.get("MFixed", []):
            addM(m, f"M{m}")

        result["M"] = mShapes

    return result


def solutionId(solution):
    """Return a short stable string identifying a solution by its tile dimensions and flags."""
    mt0 = solution["MacroTile0"]
    mt1 = solution["MacroTile1"]
    parts = [f"MT{mt0}x{mt1}"]
    if solution.get("PartialRMSResidualAdd"):
        parts.append("res")
    elif solution.get("PartialRMS"):
        parts.append("nores")
    if solution.get("RstdScale"):
        parts.append("rstd")
    return "_".join(parts)


def solutionsFromYaml(yamlPath, assembler, isaInfoMap, debugConfig,
                      problemIdx=0, groupIdx=0):
    """Return all (solution, id) pairs produced by a benchmark YAML group.

    Solutions are enumerated exactly as the YAML specifies — no overrides.
    The id string is derived from tile dimensions and epilogue flags.
    """
    solutions = buildSolutionsFromYaml(
        yamlPath, assembler, isaInfoMap, debugConfig, problemIdx, groupIdx,
    )
    return [(s, solutionId(s)) for s in solutions]


def buildK1SolutionFromYaml(assembler, isaInfoMap, debugConfig,
                            wgN=2, yamlPath=_DEFAULT_K1_YAML,
                            miOverride=None, residualAdd=False):
    """Build one K1 (PartialRMS) Solution by MI override.

    Used by bench_gemm_rms.py and bench_gemm_rmsnorm.py to build a single
    tile configuration. Tests and library builds should use solutionsFromYaml instead.
    """
    if miOverride is not None:
        mi9 = list(miOverride)
    else:
        mi9 = _baseMatrixInstruction(yamlPath)
        mi9[-1] = wgN

    solutions = buildSolutionsFromYaml(
        yamlPath, assembler, isaInfoMap, debugConfig,
        forkOverrides={
            "MatrixInstruction": [mi9],
            "PartialRMSResidualAdd": [residualAdd],
        },
    )
    if len(solutions) != 1:
        raise RuntimeError(
            f"expected exactly 1 K1 solution, got {len(solutions)} "
            f"(mi={mi9}, residualAdd={residualAdd})"
        )
    return solutions[0]


def buildK3SolutionFromYaml(assembler, isaInfoMap, debugConfig,
                            N_out, wg_n=1, yamlPath=_DEFAULT_K3_YAML):
    """Build one K3 (RstdScale) Solution by MacroTile1.

    Used by bench_gemm_rms.py to build a single tile configuration.
    Tests and library builds should use solutionsFromYaml instead.
    """
    assert N_out == 16 * 4 * wg_n, "n_out must equal 16 * 4 * wg_n for the k3 rstdscale tile"
    solutions = buildSolutionsFromYaml(yamlPath, assembler, isaInfoMap, debugConfig)
    matches = [s for s in solutions if s["MacroTile1"] == N_out]
    if len(matches) != 1:
        raise RuntimeError(
            f"expected exactly 1 K3 solution with MacroTile1={N_out}, "
            f"got {len(matches)} of {len(solutions)} total"
        )
    return matches[0]
