#!/usr/bin/env bash
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# Run the K3 (fused GEMM+RstdScale) pytest suite.
#
# The suite exercises SubtileRstdScaleEmit across two wg_n tile configs
# and 29 (M, N_hidden) shapes. Kernels are compiled once per fixture
# config (session scope) via amdgpu_exec.
#
# Prerequisites:
#   * Python env with: amdgpu_exec, ml_dtypes, pytest, numpy (install rocisa first).
#   * A gfx950 GPU present.  Tests are auto-skipped on other chips.
#
# Usage:
#   tools/run_pytest_rstd_scale.sh [pytest options]
#
# Examples:
#   tools/run_pytest_rstd_scale.sh                      # full suite
#   tools/run_pytest_rstd_scale.sh -x                   # stop on first failure
#   tools/run_pytest_rstd_scale.sh -k "wgN1 and K64"   # single config + K
#   tools/run_pytest_rstd_scale.sh -n 4                 # parallel workers (pytest-xdist)
#   tools/run_pytest_rstd_scale.sh -s --tb=short        # verbose output

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TENSILE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
TEST_FILE="${TENSILE_ROOT}/Tensile/Tests/unit/test_gemm_rstd_scale.py"

export PYTHONPATH="${TOOLS_DIR:-${SCRIPT_DIR}}:${TENSILE_ROOT}:${PYTHONPATH:-}"

exec python3 -m pytest "$TEST_FILE" -v --tb=short "$@"
