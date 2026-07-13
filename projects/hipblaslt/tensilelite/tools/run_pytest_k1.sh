#!/usr/bin/env bash
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# Run the K1 (fused GEMM+PartialRMS) pytest suite.
#
# The suite exercises SubtilePartialRMSEmit across five (wg0_waves, wg1_waves)
# tile configs, two residual-add modes, 14 K values, and 7 N_hidden values.
# Kernels are compiled once per fixture config (session scope) via amdgpu_exec.
#
# Prerequisites:
#   * Python env with: amdgpu_exec, ml_dtypes, pytest, numpy (install rocisa first).
#   * A gfx950 GPU present.  Tests are auto-skipped on other chips.
#
# Usage:
#   tools/run_pytest_k1.sh [pytest options]
#
# Examples:
#   tools/run_pytest_k1.sh                        # full suite
#   tools/run_pytest_k1.sh -x                     # stop on first failure
#   tools/run_pytest_k1.sh -k "wg4x2 and K256"   # single config + K
#   tools/run_pytest_k1.sh -n 4                   # parallel workers (pytest-xdist)
#   tools/run_pytest_k1.sh -s --tb=short          # verbose output

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TENSILE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
TEST_FILE="${TENSILE_ROOT}/Tensile/Tests/unit/test_gemm_partial_rms.py"

# Add the tools dir and tensile root to PYTHONPATH so helpers and amdgpu_exec
# are importable without an editable install.
export PYTHONPATH="${TOOLS_DIR:-${SCRIPT_DIR}}:${TENSILE_ROOT}:${PYTHONPATH:-}"

exec python3 -m pytest "$TEST_FILE" -v --tb=short "$@"
