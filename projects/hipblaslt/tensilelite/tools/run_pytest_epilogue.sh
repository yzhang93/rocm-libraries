#!/usr/bin/env bash
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# Run the two-kernel pipeline pytest suite (K1 + partial_rms_epilogue).
#
# Validates the full GEMM+RMSNorm pipeline:
#   K1  (SubtilePartialRMS): D = h1 * gamma,  partialBuf = per-tile Σx²
#   K2  (PartialRmsEpilogueGenerator): D_out = D / sqrt(mean(Σx²) + eps)
#
# Parametrised over wg_n ∈ {1, 2} (MacroTile1 = 64, 128), three K values,
# and a large set of (M, N_hidden) shapes including boundary and prime sizes.
#
# Prerequisites:
#   * Python env with: amdgpu_exec, ml_dtypes, pytest, numpy.
#   * A gfx950 GPU present.  Tests are auto-skipped on other chips.
#
# Usage:
#   tools/run_pytest_epilogue.sh [pytest options]
#
# Examples:
#   tools/run_pytest_epilogue.sh                   # full suite
#   tools/run_pytest_epilogue.sh -x                # stop on first failure
#   tools/run_pytest_epilogue.sh -k "wg_n1 and K64"

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TENSILE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
TEST_FILE="${TENSILE_ROOT}/Tensile/Tests/unit/test_gemm_partial_rms_epilogue.py"

export PYTHONPATH="${SCRIPT_DIR}:${TENSILE_ROOT}:${PYTHONPATH:-}"

exec python3 -m pytest "$TEST_FILE" -v --tb=short "$@"
