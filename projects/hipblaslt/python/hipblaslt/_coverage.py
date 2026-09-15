# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Parse hipBLASLt header enums to drive API-surface coverage tests."""
import os
import re
from pathlib import Path

# header enum typedef -> set of integer values intentionally not bound.
# Extend here rather than silently skipping new gaps.
ALLOWED_MISSING = {
    # Epilogue variants bound here are the basic set (DEFAULT, RELU, BIAS,
    # RELU_BIAS, GELU, GELU_BIAS, SIGMOID_EXT). The remaining values are
    # gradient epilogues (DGELU*), auxiliary-output variants (*_AUX*), bias-
    # gradient variants (BGRADA/BGRADB), and extended activation epilogues
    # (SWISH_EXT, CLAMP_EXT, SIGMOID_BIAS_EXT). These are not yet surfaced by
    # the public Python API; add them here so the harness stays green until
    # enums.cpp is extended.
    "hipblasLtEpilogue_t": {
        130,    # HIPBLASLT_EPILOGUE_RELU_AUX
        134,    # HIPBLASLT_EPILOGUE_RELU_AUX_BIAS
        136,    # HIPBLASLT_EPILOGUE_DRELU
        152,    # HIPBLASLT_EPILOGUE_DRELU_BGRAD
        160,    # HIPBLASLT_EPILOGUE_GELU_AUX
        164,    # HIPBLASLT_EPILOGUE_GELU_AUX_BIAS
        192,    # HIPBLASLT_EPILOGUE_DGELU
        208,    # HIPBLASLT_EPILOGUE_DGELU_BGRAD
        256,    # HIPBLASLT_EPILOGUE_BGRADA
        512,    # HIPBLASLT_EPILOGUE_BGRADB
        65536,  # HIPBLASLT_EPILOGUE_SWISH_EXT
        65540,  # HIPBLASLT_EPILOGUE_SWISH_BIAS_EXT
        131072, # HIPBLASLT_EPILOGUE_CLAMP_EXT
        131076, # HIPBLASLT_EPILOGUE_CLAMP_BIAS_EXT
        131200, # HIPBLASLT_EPILOGUE_CLAMP_AUX_EXT
        131204, # HIPBLASLT_EPILOGUE_CLAMP_AUX_BIAS_EXT
        262148, # HIPBLASLT_EPILOGUE_SIGMOID_BIAS_EXT
    },
    # ScaleMode: the bound set is SCALAR_32F, VEC32_UE8M0, OUTER_VEC_32F and
    # (version-gated) BLK32_UE8M0_32_8_EXT. Everything else is either marked
    # "not supported yet" in the header or is the END sentinel.
    "hipblasLtMatmulMatrixScale_t": {
        1,     # HIPBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3 — not supported yet
        4,     # HIPBLASLT_MATMUL_MATRIX_SCALE_VEC128_32F — not supported yet
        5,     # HIPBLASLT_MATMUL_MATRIX_SCALE_BLK128x128_32F — not supported yet
        6,     # END sentinel in <=1.2 headers, where the EXT block did not exist
        1002,  # HIPBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE8M0_EXT — not supported yet
        1003,  # HIPBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE4M3_EXT — not supported yet
        1004,  # HIPBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE5M3_EXT — not supported yet
        1005,  # HIPBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE5M3_EXT — not supported yet
        1006,  # HIPBLASLT_MATMUL_MATRIX_SCALE_END — sentinel, not a real mode
    },
    # MatmulDescAttr: only the subset needed for basic GEMM dispatch is bound;
    # the rest are auxiliary-output, scaling, and tuning-hint attributes that
    # the Python API does not surface yet. Note the MAX sentinel moved from 104
    # to 106 as the _EXT block grew, so both values are listed.
    "hipblasLtMatmulDescAttributes_t": {
        4,   # HIPBLASLT_MATMUL_DESC_BIAS_DATA_TYPE
        7,   # HIPBLASLT_MATMUL_DESC_C_SCALE_POINTER
        9,   # HIPBLASLT_MATMUL_DESC_EPILOGUE_AUX_SCALE_POINTER
        10,  # HIPBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER
        11,  # HIPBLASLT_MATMUL_DESC_EPILOGUE_AUX_LD
        12,  # HIPBLASLT_MATMUL_DESC_EPILOGUE_AUX_BATCH_STRIDE
        13,  # HIPBLASLT_MATMUL_DESC_POINTER_MODE
        14,  # HIPBLASLT_MATMUL_DESC_AMAX_D_POINTER
        22,  # HIPBLASLT_MATMUL_DESC_EPILOGUE_AUX_DATA_TYPE
        23,  # HIPBLASLT_MATMUL_DESC_BIAS_BATCH_STRIDE
        33,  # HIPBLASLT_MATMUL_DESC_SM_COUNT_TARGET
        100, # HIPBLASLT_MATMUL_DESC_COMPUTE_INPUT_TYPE_A_EXT
        101, # HIPBLASLT_MATMUL_DESC_COMPUTE_INPUT_TYPE_B_EXT
        102, # HIPBLASLT_MATMUL_DESC_EPILOGUE_ACT_ARG0_EXT
        103, # HIPBLASLT_MATMUL_DESC_EPILOGUE_ACT_ARG1_EXT
        104, # STREAMK_TILE_SCHEDULING_EXT in >=1.4; the MAX sentinel in <=1.2
        105, # HIPBLASLT_MATMUL_DESC_UNIFORM_SUMMATION_ORDER_EXT
        106, # HIPBLASLT_MATMUL_DESC_MAX — sentinel
    },
}


def find_header():
    """Locate the hipblaslt.h that the extension was compiled against.

    The build records its hipBLASLt include directory on the extension module,
    which is the only reliable answer: a developer build links a freshly built
    hipBLASLt while an older copy may still sit under ``$ROCM_PATH``, and
    parsing the wrong one reports bound values as missing from the header.

    Search order:
    1. The include directory recorded at build time by ``_core``.
    2. ``$ROCM_PATH/include/hipblaslt/hipblaslt.h`` (or ``/opt/rocm``).
    3. Walk up from this file's location to find the in-tree header at
       ``library/include/hipblaslt/hipblaslt.h`` (developer build fallback).

    Returns a :class:`pathlib.Path` to the first found header.
    Raises :class:`FileNotFoundError` if none exist.
    """
    candidates = []

    # Build-time include dir: matches the headers that produced the bindings.
    try:
        from . import _core

        build_dir = getattr(_core, "_hipblaslt_include_dir", None)
        if build_dir:
            candidates.append(Path(build_dir) / "hipblaslt" / "hipblaslt.h")
    except ImportError:
        # Header-only checks can run without a built extension.
        pass

    rocm = os.environ.get("ROCM_PATH", "/opt/rocm")
    candidates.append(Path(rocm) / "include" / "hipblaslt" / "hipblaslt.h")

    # In-tree fallback: walk up from python/hipblaslt/_coverage.py.
    here = Path(__file__).resolve()
    for parent in here.parents:
        p = parent / "library" / "include" / "hipblaslt" / "hipblaslt.h"
        if p.exists():
            candidates.append(p)
            break

    for c in candidates:
        if c.exists():
            return c

    raise FileNotFoundError(
        f"hipblaslt.h not found; searched: {[str(c) for c in candidates]}"
    )


def header_enum_values(header_path, enum_type):
    """Parse a ``typedef enum { ... } enum_type;`` block from *header_path*.

    Returns a ``{member_name: int_value}`` dict.  Simple ``= N`` and
    ``= 0xHEX`` assignments are resolved; members without an explicit value
    receive the previous value plus one (C auto-increment semantics).

    Block comments (``/* ... */``) and line comments (``// ...``) are stripped
    before name/value extraction.

    Parameters
    ----------
    header_path : str or pathlib.Path
        Path to ``hipblaslt.h``.
    enum_type : str
        The C typedef name, e.g. ``"hipblasLtEpilogue_t"``.

    Returns
    -------
    dict[str, int]
    """
    text = Path(header_path).read_text()

    # Strip comments before locating the enum, not after: doc comments contain
    # braces (e.g. "values outside ``{0, 1, 2}`` are rejected"), and the body
    # pattern below stops at the first '}' it sees, so a braced comment would
    # otherwise hide the whole enum.
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    text = re.sub(r"//[^\n]*", "", text)

    # Match: typedef enum { ... } enum_type;
    # Use [^}]* instead of .*? to avoid spanning across multiple enum blocks
    # when the file contains several consecutive typedef enums (re.DOTALL would
    # allow .*? to skip over the closing brace of an earlier enum).
    pattern = re.compile(
        r"typedef\s+enum\s*\{(?P<body>[^}]*)\}\s*"
        + re.escape(enum_type)
        + r"\s*;",
        re.DOTALL,
    )
    match = pattern.search(text)
    if not match:
        raise ValueError(f"enum {enum_type!r} not found in {header_path}")

    body = match.group("body")

    values: dict[str, int] = {}
    current = -1

    for raw_token in body.split(","):
        token = raw_token.strip()
        if not token:
            continue

        # Match "NAME" or "NAME = VALUE" (decimal or hex, optional sign).
        m = re.match(
            r"([A-Za-z_][A-Za-z0-9_]*)\s*(?:=\s*(0[xX][0-9a-fA-F]+|-?\d+))?",
            token,
        )
        if not m:
            continue

        name = m.group(1)
        if m.group(2) is not None:
            current = int(m.group(2), 0)
        else:
            current += 1

        values[name] = current

    return values
