// Thin C wrapper over the hipBLASLt grouped GEMM C API (PR 13580 + 13581) for MoE.
// One plan = one grouped MXFP8 (or MXFP4) TN GEMM with every group's weight shape fixed
// (M x K) and per-group token counts read on the device:
//   D_g[n_g, M] = B_g[n_g, K] . A_g[M, K]^T     (row-major PyTorch view)
// A_g is the expert weight, B_g the expert's token slice, scales pre-swizzled for gfx950
// (HIPBLASLT_MATMUL_MATRIX_SCALE_BLK32_UE8M0_32_8_EXT), one scale tensor per group.
#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt-ext.hpp>
#include <hipblaslt/hipblaslt.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>

namespace
{
    hipblasLtHandle_t g_handle = nullptr;

    // HIPBLASLT_MATMUL_DESC_MOE_*_POINTER_EXT (users/vivizhan/moe-sorted-layout), spelled as
    // values so the shim still builds against hipBLASLt headers without them.
    constexpr auto kMoeSortedExpertIds = static_cast<hipblasLtMatmulDescAttributes_t>(108);
    constexpr auto kMoeNumValidIds     = static_cast<hipblasLtMatmulDescAttributes_t>(109);
    constexpr auto kMoeSortedIds       = static_cast<hipblasLtMatmulDescAttributes_t>(110);
    constexpr auto kMoeSortedIdsDiv    = static_cast<hipblasLtMatmulDescAttributes_t>(111);

    struct Plan
    {
        hipblasLtMatmulDesc_t            desc = nullptr;
        hipblasLtMatrixLayout_t          la = nullptr, lb = nullptr, lc = nullptr, ld = nullptr;
        hipblasLtMatmulHeuristicResult_t result{};
        std::string                      kernelName;
    };

#define CHECK(expr)                                                                     \
    do                                                                                  \
    {                                                                                   \
        hipblasStatus_t s_ = (expr);                                                    \
        if(s_ != HIPBLAS_STATUS_SUCCESS)                                                \
        {                                                                               \
            std::fprintf(stderr, "gg_shim: %s failed (%d) at line %d\n", #expr, int(s_), \
                         __LINE__);                                                     \
            return int(s_);                                                             \
        }                                                                               \
    } while(0)

    int makeLayout(hipblasLtMatrixLayout_t* l,
                   hipDataType              type,
                   int                      groups,
                   uint64_t                 rows,
                   uint64_t                 cols,
                   const void*              colsArray,
                   int64_t                  ldv)
    {
        CHECK(hipblasLtGroupedMatrixLayoutCreate(l, type, groups, nullptr, colsArray, nullptr));
        int32_t width = HIPBLASLT_INTEGER_WIDTH_32;
        CHECK(hipblasLtMatrixLayoutSetAttribute(
            *l, HIPBLASLT_GROUPED_MATRIX_LAYOUT_ROWS_COLS_ARRAY_INTEGER_WIDTH, &width, sizeof(width)));
        CHECK(hipblasLtMatrixLayoutSetAttribute(*l, HIPBLASLT_MATRIX_LAYOUT_ROWS, &rows, sizeof(rows)));
        if(!colsArray)
            CHECK(hipblasLtMatrixLayoutSetAttribute(
                *l, HIPBLASLT_MATRIX_LAYOUT_COLS, &cols, sizeof(cols)));
        CHECK(hipblasLtMatrixLayoutSetAttribute(*l, HIPBLASLT_MATRIX_LAYOUT_LD, &ldv, sizeof(ldv)));
        return 0;
    }
}

extern "C" {

// ab: 0 = MXFP8 E4M3, 1 = MXFP4 E2M1, 3 = MXFP4 A x MXFP8 B (all with bf16 D),
// 2 = fp16 with fp16 D (no scales).
// nArray: device int32[groups] of tokens per group. Scale pointer arrays: device
// arrays of per-group pointers. maxN / totalN bound any group's and the sum of tokens.
int gg_create(void**      planOut,
              int         groups,
              int64_t     M,
              int64_t     K,
              int64_t     maxN,
              int64_t     totalN,
              int         ab,
              const void* nArray,
              const void* aScalePtrs,
              const void* bScalePtrs,
              uint64_t    wsBytes)
{
    if(!g_handle)
        CHECK(hipblasLtCreate(&g_handle));
    auto*       p    = new Plan;
    hipDataType tA   = ab == 0 ? HIP_R_8F_E4M3 : ab == 2 ? HIP_R_16F : HIP_R_4F_E2M1;
    hipDataType tB   = ab == 1 ? HIP_R_4F_E2M1 : ab == 2 ? HIP_R_16F : HIP_R_8F_E4M3;
    hipDataType tOut = ab == 2 ? HIP_R_16F : HIP_R_16BF;

    // Column-major: A is K x M (OP_T), B is K x n_g (OP_N), D is M x n_g.
    if(int r = makeLayout(&p->la, tA, groups, K, M, nullptr, K))
        return r;
    if(int r = makeLayout(&p->lb, tB, groups, K, 0, nArray, K))
        return r;
    if(int r = makeLayout(&p->lc, tOut, groups, M, 0, nArray, M))
        return r;
    if(int r = makeLayout(&p->ld, tOut, groups, M, 0, nArray, M))
        return r;

    CHECK(hipblasLtMatmulDescCreate(&p->desc, HIPBLAS_COMPUTE_32F, HIP_R_32F));
    hipblasOperation_t opA = HIPBLAS_OP_T, opB = HIPBLAS_OP_N;
    CHECK(hipblasLtMatmulDescSetAttribute(p->desc, HIPBLASLT_MATMUL_DESC_TRANSA, &opA, sizeof(opA)));
    CHECK(hipblasLtMatmulDescSetAttribute(p->desc, HIPBLASLT_MATMUL_DESC_TRANSB, &opB, sizeof(opB)));
    if(ab != 2)
    {
        hipblasLtMatmulMatrixScale_t mode = HIPBLASLT_MATMUL_MATRIX_SCALE_BLK32_UE8M0_32_8_EXT;
        for(auto attr : {HIPBLASLT_MATMUL_DESC_A_SCALE_MODE, HIPBLASLT_MATMUL_DESC_B_SCALE_MODE})
            CHECK(hipblasLtMatmulDescSetAttribute(p->desc, attr, &mode, sizeof(mode)));
        CHECK(hipblasLtMatmulDescSetAttribute(
            p->desc, HIPBLASLT_MATMUL_DESC_A_SCALE_POINTER, &aScalePtrs, sizeof(aScalePtrs)));
        CHECK(hipblasLtMatmulDescSetAttribute(
            p->desc, HIPBLASLT_MATMUL_DESC_B_SCALE_POINTER, &bScalePtrs, sizeof(bScalePtrs)));
    }

    hipblasLtMatmulPreference_t pref = nullptr;
    CHECK(hipblasLtMatmulPreferenceCreate(&pref));
    CHECK(hipblasLtMatmulPreferenceSetAttribute(
        pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &wsBytes, sizeof(wsBytes)));
    auto hint = [&](hipblasLtMatmulPreferenceAttributes_t a, int64_t v) {
        return hipblasLtMatmulPreferenceSetAttribute(pref, a, &v, sizeof(v));
    };
    CHECK(hint(HIPBLASLT_MATMUL_PREF_GROUPED_DESC_D_MAX_COLS_EXT, maxN));
    CHECK(hint(HIPBLASLT_MATMUL_PREF_GROUPED_DESC_D_MAX_TOTAL_COLS_EXT, totalN));
    CHECK(hint(HIPBLASLT_MATMUL_PREF_GROUPED_DESC_D_AVERAGE_COLS,
               std::max<int64_t>(1, (totalN + groups - 1) / groups)));
    CHECK(hint(HIPBLASLT_MATMUL_PREF_GROUPED_AVERAGE_REDUCTION_DIM, K));

    // GG_KERNEL_MATCH=<substring> picks the first returned algo whose kernel name contains it
    // (e.g. MT128x32x256); GG_LIST_ALGOS=1 prints every returned kernel name.
    constexpr int                    kMaxAlgos = 32;
    hipblasLtMatmulHeuristicResult_t results[kMaxAlgos]{};
    const char*                      match = std::getenv("GG_KERNEL_MATCH");
    const bool                       list  = std::getenv("GG_LIST_ALGOS") != nullptr;
    const int                        want  = (match || list) ? kMaxAlgos : 1;
    int                              count = 0;
    CHECK(hipblasLtMatmulAlgoGetHeuristic(
        g_handle, p->desc, p->la, p->lb, p->lc, p->ld, pref, want, results, &count));
    hipblasLtMatmulPreferenceDestroy(pref);
    int pick = count > 0 ? 0 : -1;
    if(match)
    {
        pick = -1;
        for(int i = 0; i < count && pick < 0; ++i)
            if(hipblaslt_ext::getKernelNameFromAlgo(g_handle, results[i].algo).find(match)
               != std::string::npos)
                pick = i;
    }
    if(list)
        for(int i = 0; i < count; ++i)
            std::fprintf(stderr,
                         "gg_shim: algo %d%s %s\n",
                         i,
                         i == pick ? "*" : " ",
                         hipblaslt_ext::getKernelNameFromAlgo(g_handle, results[i].algo).c_str());
    if(pick < 0)
    {
        std::fprintf(stderr,
                     "gg_shim: no grouped solution (M=%ld K=%ld%s%s)\n",
                     long(M),
                     long(K),
                     match ? " matching " : "",
                     match ? match : "");
        return -1;
    }
    p->result     = results[pick];
    p->kernelName = hipblaslt_ext::getKernelNameFromAlgo(g_handle, p->result.algo);
    *planOut      = p;
    return 0;
}

const char* gg_kernel_name(void* plan)
{
    return static_cast<Plan*>(plan)->kernelName.c_str();
}

// MoE expert GEMM on aiter's sorted token layout (HIPBLASLT_MATMUL_DESC_MOE_*): one
// non-grouped MXFP4 A x MXFP8 B TN GEMM over P sorted, block-padded rows,
//   D[P, M] = B[rows, K] . A_e[M, K]^T   with e = sorted_expert_ids[row / MT1]
// A holds all experts at aStride elements apart. rowsB > 0 gathers B rows through
// sorted_ids (B has rowsB token rows): row = (sorted_ids[r] & 0xFFFFFF) / divisor, with
// divisor 1 for aiter's ids (token) and topk for vLLM's (token * topk + slot).
// rowsB = 0 reads B in sorted order (P rows).
// The MoE pointers set the heuristic's mode here and are reset on every run.
int moe_create(void**      planOut,
               int64_t     M,
               int64_t     K,
               int64_t     P,
               int64_t     rowsB,
               int64_t     aStride,
               const void* sortedExpertIds,
               const void* numValidIds,
               const void* sortedIds,
               int32_t     divisor,
               uint64_t    wsBytes)
{
    if(!g_handle)
        CHECK(hipblasLtCreate(&g_handle));
    auto*         p     = new Plan;
    const int64_t bRows = rowsB > 0 ? rowsB : P;
    const int64_t bStr  = bRows * K;
    CHECK(hipblasLtMatrixLayoutCreate(&p->la, HIP_R_4F_E2M1, K, M, K));
    CHECK(hipblasLtMatrixLayoutCreate(&p->lb, HIP_R_8F_E4M3, K, P, K));
    CHECK(hipblasLtMatrixLayoutCreate(&p->lc, HIP_R_16BF, M, P, M));
    CHECK(hipblasLtMatrixLayoutCreate(&p->ld, HIP_R_16BF, M, P, M));
    CHECK(hipblasLtMatrixLayoutSetAttribute(
        p->la, HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &aStride, sizeof(aStride)));
    CHECK(hipblasLtMatrixLayoutSetAttribute(
        p->lb, HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &bStr, sizeof(bStr)));
    const int64_t dStr = M * P;
    for(auto l : {p->lc, p->ld})
        CHECK(hipblasLtMatrixLayoutSetAttribute(
            l, HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &dStr, sizeof(dStr)));

    CHECK(hipblasLtMatmulDescCreate(&p->desc, HIPBLAS_COMPUTE_32F, HIP_R_32F));
    hipblasOperation_t opA = HIPBLAS_OP_T, opB = HIPBLAS_OP_N;
    CHECK(hipblasLtMatmulDescSetAttribute(p->desc, HIPBLASLT_MATMUL_DESC_TRANSA, &opA, sizeof(opA)));
    CHECK(hipblasLtMatmulDescSetAttribute(p->desc, HIPBLASLT_MATMUL_DESC_TRANSB, &opB, sizeof(opB)));
    hipblasLtMatmulMatrixScale_t mode = HIPBLASLT_MATMUL_MATRIX_SCALE_BLK32_UE8M0_32_8_EXT;
    for(auto attr : {HIPBLASLT_MATMUL_DESC_A_SCALE_MODE, HIPBLASLT_MATMUL_DESC_B_SCALE_MODE})
        CHECK(hipblasLtMatmulDescSetAttribute(p->desc, attr, &mode, sizeof(mode)));
    CHECK(hipblasLtMatmulDescSetAttribute(p->desc,
                                          kMoeSortedExpertIds,
                                          &sortedExpertIds,
                                          sizeof(sortedExpertIds)));
    CHECK(hipblasLtMatmulDescSetAttribute(
        p->desc, kMoeNumValidIds, &numValidIds, sizeof(numValidIds)));
    if(rowsB > 0)
    {
        CHECK(hipblasLtMatmulDescSetAttribute(
            p->desc, kMoeSortedIds, &sortedIds, sizeof(sortedIds)));
        CHECK(hipblasLtMatmulDescSetAttribute(p->desc, kMoeSortedIdsDiv, &divisor, sizeof(divisor)));
    }

    hipblasLtMatmulPreference_t pref = nullptr;
    CHECK(hipblasLtMatmulPreferenceCreate(&pref));
    CHECK(hipblasLtMatmulPreferenceSetAttribute(
        pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &wsBytes, sizeof(wsBytes)));

    constexpr int                    kMaxAlgos = 32;
    hipblasLtMatmulHeuristicResult_t results[kMaxAlgos]{};
    const char*                      match = std::getenv("GG_KERNEL_MATCH");
    const bool                       list  = std::getenv("GG_LIST_ALGOS") != nullptr;
    int                              count = 0;
    CHECK(hipblasLtMatmulAlgoGetHeuristic(
        g_handle, p->desc, p->la, p->lb, p->lc, p->ld, pref, kMaxAlgos, results, &count));
    hipblasLtMatmulPreferenceDestroy(pref);
    int pick = count > 0 ? 0 : -1;
    if(match)
    {
        pick = -1;
        for(int i = 0; i < count && pick < 0; ++i)
            if(hipblaslt_ext::getKernelNameFromAlgo(g_handle, results[i].algo).find(match)
               != std::string::npos)
                pick = i;
    }
    if(list)
        for(int i = 0; i < count; ++i)
            std::fprintf(stderr,
                         "gg_shim: moe algo %d%s %s\n",
                         i,
                         i == pick ? "*" : " ",
                         hipblaslt_ext::getKernelNameFromAlgo(g_handle, results[i].algo).c_str());
    if(pick < 0)
    {
        std::fprintf(stderr,
                     "gg_shim: no MoE solution (M=%ld K=%ld P=%ld%s%s)\n",
                     long(M),
                     long(K),
                     long(P),
                     match ? " matching " : "",
                     match ? match : "");
        return -1;
    }
    p->result     = results[pick];
    p->kernelName = hipblaslt_ext::getKernelNameFromAlgo(g_handle, p->result.algo);
    *planOut      = p;
    return 0;
}

int moe_run(void*       plan,
            const void* a,
            const void* aScale,
            const void* b,
            const void* bScale,
            void*       d,
            const void* sortedExpertIds,
            const void* numValidIds,
            const void* sortedIds,
            void*       ws,
            uint64_t    wsBytes,
            void*       stream)
{
    auto* p = static_cast<Plan*>(plan);
    CHECK(hipblasLtMatmulDescSetAttribute(
        p->desc, HIPBLASLT_MATMUL_DESC_A_SCALE_POINTER, &aScale, sizeof(aScale)));
    CHECK(hipblasLtMatmulDescSetAttribute(
        p->desc, HIPBLASLT_MATMUL_DESC_B_SCALE_POINTER, &bScale, sizeof(bScale)));
    CHECK(hipblasLtMatmulDescSetAttribute(p->desc,
                                          kMoeSortedExpertIds,
                                          &sortedExpertIds,
                                          sizeof(sortedExpertIds)));
    CHECK(hipblasLtMatmulDescSetAttribute(
        p->desc, kMoeNumValidIds, &numValidIds, sizeof(numValidIds)));
    if(sortedIds)
        CHECK(hipblasLtMatmulDescSetAttribute(
            p->desc, kMoeSortedIds, &sortedIds, sizeof(sortedIds)));
    const float alpha = 1.f, beta = 0.f;
    CHECK(hipblasLtMatmul(g_handle,
                          p->desc,
                          &alpha,
                          a,
                          p->la,
                          b,
                          p->lb,
                          &beta,
                          d,
                          p->lc,
                          d,
                          p->ld,
                          &p->result.algo,
                          ws,
                          wsBytes,
                          static_cast<hipStream_t>(stream)));
    return 0;
}

// aPtrs, bPtrs, dPtrs: device arrays of per-group pointers. C = D with beta = 0.
int gg_run(void*       plan,
           const void* aPtrs,
           const void* bPtrs,
           void*       dPtrs,
           void*       ws,
           uint64_t    wsBytes,
           void*       stream)
{
    auto*       p     = static_cast<Plan*>(plan);
    const float alpha = 1.f, beta = 0.f;
    CHECK(hipblasLtMatmul(g_handle,
                          p->desc,
                          &alpha,
                          aPtrs,
                          p->la,
                          bPtrs,
                          p->lb,
                          &beta,
                          dPtrs,
                          p->lc,
                          dPtrs,
                          p->ld,
                          &p->result.algo,
                          ws,
                          wsBytes,
                          static_cast<hipStream_t>(stream)));
    return 0;
}
}
