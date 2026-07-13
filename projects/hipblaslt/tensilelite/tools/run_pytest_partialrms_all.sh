#!/usr/bin/env bash
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# Run both PartialRMS pytest suites together:
#   1. test_gemm_partial_rms.py        — K1 subtile epilogue (D + partialBuf)
#   2. test_gemm_partial_rms_epilogue.py — full two-kernel pipeline (final D)
#
# Also runs the max-M guard unit test (no GPU required).
#
# Requires a gfx950 GPU; GPU tests are auto-skipped on other chips.
#
# Usage:
#   tools/run_pytest_partialrms_all.sh [OPTIONS] [-- PYTEST_ARGS]
#
# Options:
#   --venv PATH      Activate a virtualenv at PATH before running.
#   --workers N      Pass -n N to pytest-xdist (default: 1, serial).
#   --suite k1|epi   Run only the K1 suite ("k1") or the epilogue suite ("epi").
#
# Examples:
#   tools/run_pytest_partialrms_all.sh
#   tools/run_pytest_partialrms_all.sh --workers 2
#   tools/run_pytest_partialrms_all.sh --suite k1 -- -k "wg4x2"
#   tools/run_pytest_partialrms_all.sh --venv /path/to/venv

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TENSILE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

VENV_PATH=""
WORKERS="1"
SUITE="all"
EXTRA_PYTEST_ARGS=()

# Parse arguments.
while [[ $# -gt 0 ]]; do
    case "$1" in
        --venv)    VENV_PATH="$2"; shift 2 ;;
        --workers) WORKERS="$2";   shift 2 ;;
        --suite)   SUITE="$2";     shift 2 ;;
        --)        shift; EXTRA_PYTEST_ARGS+=("$@"); break ;;
        *)         EXTRA_PYTEST_ARGS+=("$1"); shift ;;
    esac
done

# Activate venv if requested.
if [[ -n "$VENV_PATH" ]]; then
    # shellcheck disable=SC1091
    source "${VENV_PATH}/bin/activate"
fi

# Extend PYTHONPATH so helper modules and test files are importable without an
# editable install.
export PYTHONPATH="${SCRIPT_DIR}:${TENSILE_ROOT}:${PYTHONPATH:-}"

K1_TEST="${TENSILE_ROOT}/Tensile/Tests/unit/test_gemm_partial_rms.py"
EPI_TEST="${TENSILE_ROOT}/Tensile/Tests/unit/test_gemm_partial_rms_epilogue.py"

# Determine parallel flag.
if [[ "$WORKERS" -gt 1 ]] 2>/dev/null; then
    PAR_FLAG=(-n "$WORKERS")
else
    PAR_FLAG=()
fi

run_suite() {
    local label="$1"
    local file="$2"
    echo ""
    echo "======================================================================"
    echo " Running: $label"
    echo " File:    $file"
    echo "======================================================================"
    python3 -m pytest "$file" -v --tb=short "${PAR_FLAG[@]}" "${EXTRA_PYTEST_ARGS[@]}"
}

OVERALL_RC=0

case "$SUITE" in
    k1)
        run_suite "K1 PartialRMS epilogue" "$K1_TEST" || OVERALL_RC=$?
        ;;
    epi|epilogue)
        run_suite "Two-kernel pipeline (K1 + PartialRmsEpilogue)" "$EPI_TEST" || OVERALL_RC=$?
        ;;
    all)
        run_suite "K1 PartialRMS epilogue" "$K1_TEST"  || OVERALL_RC=$?
        run_suite "Two-kernel pipeline (K1 + PartialRmsEpilogue)" "$EPI_TEST" || { rc=$?; [[ $OVERALL_RC -eq 0 ]] && OVERALL_RC=$rc; }
        ;;
    *)
        echo "error: unknown --suite value '$SUITE' (choices: k1, epi, all)" >&2
        exit 1
        ;;
esac

echo ""
if [[ $OVERALL_RC -eq 0 ]]; then
    echo "All selected suites PASSED."
else
    echo "One or more suites FAILED (exit code $OVERALL_RC)."
fi
exit "$OVERALL_RC"
