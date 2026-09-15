// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once
#include <hipblaslt/hipblaslt.h>
#include <cstddef>

namespace hipblaslt_py
{

struct Algo
{
    hipblasLtMatmulAlgo_t algo{};
    // Rank within the heuristic result list this algo came from (0 = fastest).
    // -1 when the algo was not produced by a heuristic query, e.g. when built
    // from an explicit solution index. Use the solution_index property for the
    // stable Tensile identity.
    int index = -1;
};

struct HeuristicResult
{
    Algo   algo;
    size_t workspace_size = 0;
    float  waves_count    = 0.0f;
};

} // namespace hipblaslt_py
