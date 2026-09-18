// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// End-to-end cover for user kernel registration, driving the public C API in a
// single process the way the demo has to.
//
// The three claims being checked are the ones the design rests on, and two of
// them are about what must *not* happen:
//
//   1. Registering makes a kernel executable by index immediately.
//   2. Registering does NOT change what heuristic selection returns.
//   3. Mapping an exact shape DOES change it, even for a shape this process has
//      already been running -- which only works if the cached selection result
//      is discarded.
//
// Claim 3 is why the heuristic is called before registering. Against a cold
// cache the test would pass even with cache invalidation completely broken.
//
// The payload is a shipped bf16 TN shard. Registering a kernel the library
// already owns is a legitimate payload -- it is a real compiled (.dat, .co)
// pair with the stem pairing the contract requires -- and it keeps this test
// independent of any tuning tool.

#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt-ext.hpp>
#include <hipblaslt/hipblaslt.h>

#include <algorithm>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

namespace
{
    constexpr int64_t M = 16, N = 6144, K = 2048;

    // bf16 bit patterns. A and B are filled with exactly 1.0 so every output
    // element is exactly K, which bf16 represents without rounding; any
    // mismatch is then a real dispatch or launch fault rather than precision.
    constexpr uint16_t BF16_ONE = 0x3F80;

#define CHECK_HIP(expr)                                               \
    do                                                                \
    {                                                                 \
        hipError_t e_ = (expr);                                       \
        if(e_ != hipSuccess)                                          \
        {                                                             \
            std::printf("FAIL %s:%d  %s -> %s\n",                     \
                        __FILE__, __LINE__, #expr, hipGetErrorString(e_)); \
            return 1;                                                 \
        }                                                             \
    } while(0)

#define CHECK_BLAS(expr)                                                       \
    do                                                                         \
    {                                                                          \
        hipblasStatus_t s_ = (expr);                                           \
        if(s_ != HIPBLAS_STATUS_SUCCESS)                                       \
        {                                                                      \
            std::printf("FAIL %s:%d  %s -> status %d\n",                       \
                        __FILE__, __LINE__, #expr, (int)s_);                   \
            return 1;                                                          \
        }                                                                      \
    } while(0)

#define EXPECT(cond, msg)                                     \
    do                                                        \
    {                                                         \
        if(!(cond))                                           \
        {                                                     \
            std::printf("FAIL: %s\n", msg);                   \
            return 1;                                         \
        }                                                     \
        std::printf("  ok: %s\n", msg);                        \
    } while(0)

    // Solution index carried at the head of an algo's opaque data.
    int algoIndex(hipblasLtMatmulAlgo_t const& algo)
    {
        int index = -1;
        std::memcpy(&index, algo.data, sizeof(index));
        return index;
    }

    // PCI device id of the current GPU, e.g. "75a0", or empty if unreadable.
    //
    // Shard filenames carry the device ids they were built for. Filtering on
    // it keeps the test from registering kernels that belong to other GPUs,
    // which would be inert here and would inflate the counts asserted below.
    std::string devicePciId()
    {
        int device = 0;
        if(hipGetDevice(&device) != hipSuccess)
            return {};

        char bus[64] = {};
        if(hipDeviceGetPCIBusId(bus, sizeof(bus), device) != hipSuccess)
            return {};

        std::string lower(bus);
        std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);

        std::ifstream in("/sys/bus/pci/devices/" + lower + "/device");
        std::string   id;
        if(!in || !(in >> id))
            return {};
        if(id.rfind("0x", 0) == 0)
            id = id.substr(2);
        return id;
    }

    // Candidate payloads in this build's installed library, biggest first.
    //
    // Globbed rather than named so the test follows whatever GPU this build
    // targets. Several shards match and which is usable cannot be known
    // without trying: one built for a different device id or CU count
    // deserializes fine, but every kernel in it is rejected when validated
    // against this GPU. Bigger shards come first only because they hold more
    // kernels and are likelier to contain a fit.
    std::vector<std::string> candidatePayloads()
    {
        const char* env = std::getenv("HIPBLASLT_USER_KERNEL_TEST_LIB_DIR");
        const std::string root
            = env ? env : "hipblaslt-install/lib/hipblaslt/library";

        const std::string pciId = devicePciId();

        std::vector<std::pair<uintmax_t, std::string>> found;
        std::error_code ec;
        if(!std::filesystem::is_directory(root, ec))
            return {};

        for(auto const& arch : std::filesystem::directory_iterator(root, ec))
        {
            if(!arch.is_directory())
                continue;
            for(auto const& entry : std::filesystem::directory_iterator(arch.path(), ec))
            {
                std::string name = entry.path().filename().string();
                if(name.rfind("TensileLibrary_BB_BB_", 0) != 0)
                    continue;
                if(name.find("_Alik_Bljk_") == std::string::npos)
                    continue;

                // Skip shards built for other GPUs rather than registering
                // kernels that can never run here.
                std::string lowerName(name);
                std::transform(lowerName.begin(), lowerName.end(), lowerName.begin(),
                               ::tolower);
                if(!pciId.empty() && lowerName.find(pciId) == std::string::npos)
                    continue;

                // Recover the stem: the loader appends the ".zlib" probe
                // itself, so the bare ".dat" name is what gets passed even
                // when only the compressed form exists.
                std::string stem = entry.path().string();
                for(std::string suffix : {std::string(".zlib"), std::string(".dat")})
                    if(stem.size() > suffix.size()
                       && stem.compare(stem.size() - suffix.size(), suffix.size(), suffix) == 0)
                        stem.resize(stem.size() - suffix.size());
                if(stem == entry.path().string())
                    continue; // not a .dat/.dat.zlib
                if(!std::filesystem::exists(stem + ".co"))
                    continue;

                found.emplace_back(entry.file_size(ec), stem);
            }
        }

        std::sort(found.begin(), found.end(), [](auto const& a, auto const& b) {
            return a.first > b.first;
        });
        std::vector<std::string> stems;
        for(auto const& f : found)
            if(std::find(stems.begin(), stems.end(), f.second) == stems.end())
                stems.push_back(f.second);
        return stems;
    }
} // namespace

int main()
{
    const std::vector<std::string> candidates = candidatePayloads();
    if(candidates.empty())
    {
        std::printf("SKIP: no payload shard in this build's installed library\n");
        return 0;
    }

    hipblasLtHandle_t handle;
    CHECK_BLAS(hipblasLtCreate(&handle));

    // ---- problem: M=16, N=6144, K=2048, bf16 TN, no epilogue -------------
    hipblasLtMatmulDesc_t desc;
    CHECK_BLAS(hipblasLtMatmulDescCreate(&desc, HIPBLAS_COMPUTE_32F, HIP_R_32F));
    hipblasOperation_t opT = HIPBLAS_OP_T, opN = HIPBLAS_OP_N;
    CHECK_BLAS(hipblasLtMatmulDescSetAttribute(
        desc, HIPBLASLT_MATMUL_DESC_TRANSA, &opT, sizeof(opT)));
    CHECK_BLAS(hipblasLtMatmulDescSetAttribute(
        desc, HIPBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(opN)));

    hipblasLtMatrixLayout_t la, lb, lc, ld;
    CHECK_BLAS(hipblasLtMatrixLayoutCreate(&la, HIP_R_16BF, K, M, K));
    CHECK_BLAS(hipblasLtMatrixLayoutCreate(&lb, HIP_R_16BF, K, N, K));
    CHECK_BLAS(hipblasLtMatrixLayoutCreate(&lc, HIP_R_16BF, M, N, M));
    CHECK_BLAS(hipblasLtMatrixLayoutCreate(&ld, HIP_R_16BF, M, N, M));

    hipblasLtMatmulPreference_t pref;
    CHECK_BLAS(hipblasLtMatmulPreferenceCreate(&pref));
    uint64_t maxWs = 64ull * 1024 * 1024;
    CHECK_BLAS(hipblasLtMatmulPreferenceSetAttribute(
        pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &maxWs, sizeof(maxWs)));

    std::vector<uint16_t> hA(K * M, BF16_ONE), hB(K * N, BF16_ONE);
    void *dA, *dB, *dC, *dD, *dWs;
    CHECK_HIP(hipMalloc(&dA, hA.size() * 2));
    CHECK_HIP(hipMalloc(&dB, hB.size() * 2));
    CHECK_HIP(hipMalloc(&dC, M * N * 2));
    CHECK_HIP(hipMalloc(&dD, M * N * 2));
    CHECK_HIP(hipMalloc(&dWs, maxWs));
    CHECK_HIP(hipMemcpy(dA, hA.data(), hA.size() * 2, hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(dB, hB.data(), hB.size() * 2, hipMemcpyHostToDevice));
    CHECK_HIP(hipMemset(dC, 0, M * N * 2));
    CHECK_HIP(hipMemset(dD, 0, M * N * 2));

    auto heuristicTop = [&](int* outIndex) -> hipblasStatus_t {
        hipblasLtMatmulHeuristicResult_t results[8];
        int                              returned = 0;
        auto st = hipblasLtMatmulAlgoGetHeuristic(
            handle, desc, la, lb, lc, ld, pref, 8, results, &returned);
        if(st != HIPBLAS_STATUS_SUCCESS || returned <= 0)
            return st == HIPBLAS_STATUS_SUCCESS ? HIPBLAS_STATUS_NOT_SUPPORTED : st;
        *outIndex = algoIndex(results[0].algo);
        return HIPBLAS_STATUS_SUCCESS;
    };

    // ---- 1. warm the selection cache BEFORE registering -------------------
    int shippedTop = -1;
    CHECK_BLAS(heuristicTop(&shippedTop));
    std::printf("shipped top solution index: %d\n", shippedTop);
    EXPECT(shippedTop >= 0, "heuristic returns a shipped kernel before registration");

    int isUser = -1;
    {
        hipblasLtMatmulHeuristicResult_t r[1];
        int                              n = 0;
        CHECK_BLAS(hipblasLtMatmulAlgoGetHeuristic(handle, desc, la, lb, lc, ld, pref, 1, r, &n));
        CHECK_BLAS(hipblasLtMatmulAlgoIsUserKernel(&r[0].algo, &isUser));
    }
    EXPECT(isUser == 0, "a shipped kernel is not reported as a user kernel");

    size_t registered = 123, selectable = 123;
    CHECK_BLAS(hipblasLtUserKernelGetCounts(handle, &registered, &selectable));
    EXPECT(registered == 0 && selectable == 0, "nothing registered or selectable at start");

    // ---- 2. register ------------------------------------------------------
    // Walk candidates until one yields a kernel that serves this problem. A
    // shard built for another device id or CU count registers fine but every
    // kernel in it is rejected when validated against this GPU, so which
    // payload is usable cannot be known without trying.
    int                              chosen   = -1;
    size_t                           chosenWs = 0;
    hipblasLtMatmulHeuristicResult_t chosenResult{};
    size_t                           totalRegistered = 0;

    for(auto const& stem : candidates)
    {
        const std::string dat = stem + ".dat", co = stem + ".co";

        std::vector<int> indices(4096);
        int              numIndices = 0;
        CHECK_BLAS(hipblasLtUserKernelRegister(
            handle, dat.c_str(), co.c_str(), indices.data(), (int)indices.size(), &numIndices));
        EXPECT(numIndices > 0, "the payload produced at least one kernel");
        indices.resize(std::min<size_t>(numIndices, indices.size()));
        totalRegistered += (size_t)numIndices;

        bool allInUserRange = true;
        for(int idx : indices)
            if(idx < (1 << 30))
                allInUserRange = false;
        EXPECT(allInUserRange, "every assigned index is in the user range [2^30, 2^31)");

        CHECK_BLAS(hipblasLtUserKernelGetCounts(handle, &registered, &selectable));
        EXPECT(registered == totalRegistered, "registered count matches");
        EXPECT(selectable == 0, "registering alone makes nothing selectable");

        // ---- 3. registering must not change selection ---------------------
        int afterRegisterTop = -1;
        CHECK_BLAS(heuristicTop(&afterRegisterTop));
        EXPECT(afterRegisterTop == shippedTop,
               "selection is unchanged by registration (register != select)");

        // ---- 4. a registered kernel is executable by index right away -----
        // Finding one is what a tuning loop does before committing.
        for(int idx : indices)
        {
            std::vector<int>                              one{idx};
            std::vector<hipblasLtMatmulHeuristicResult_t> got;
            if(hipblaslt_ext::getAlgosFromIndex(handle, one, got) != HIPBLAS_STATUS_SUCCESS
               || got.empty())
                continue;

            size_t ws    = 0;
            float  alpha = 1.0f, beta = 0.0f;
            if(hipblaslt_ext::matmulIsAlgoSupported(
                   handle, desc, &alpha, la, lb, &beta, lc, ld, got[0].algo, ws)
               != HIPBLAS_STATUS_SUCCESS)
                continue;
            if(ws > maxWs)
                continue;

            chosen       = idx;
            chosenWs     = ws;
            chosenResult = got[0];
            break;
        }
        if(chosen >= 0)
        {
            std::printf("registered %d kernels from %s\n",
                        numIndices, std::filesystem::path(stem).filename().string().c_str());
            break;
        }
    }
    EXPECT(chosen >= 0, "a registered kernel resolves by index and supports the problem");

    CHECK_BLAS(hipblasLtMatmulAlgoIsUserKernel(&chosenResult.algo, &isUser));
    EXPECT(isUser == 1, "the registered kernel is reported as a user kernel");
    EXPECT(algoIndex(chosenResult.algo) == chosen, "the algo round-trips its assigned index");

    {
        float alpha = 1.0f, beta = 0.0f;
        CHECK_BLAS(hipblasLtMatmul(handle, desc, &alpha, dA, la, dB, lb, &beta,
                                   dC, lc, dD, ld, &chosenResult.algo, dWs, chosenWs, nullptr));
        CHECK_HIP(hipDeviceSynchronize());

        std::vector<uint16_t> hD(M * N, 0);
        CHECK_HIP(hipMemcpy(hD.data(), dD, hD.size() * 2, hipMemcpyDeviceToHost));
        // bf16 -> float by placing the pattern in the high half.
        auto toFloat = [](uint16_t bits) {
            uint32_t w = (uint32_t)bits << 16;
            float    f;
            std::memcpy(&f, &w, sizeof(f));
            return f;
        };
        bool exact = true;
        for(size_t i = 0; i < hD.size(); ++i)
            if(toFloat(hD[i]) != (float)K)
                exact = false;
        std::printf("D[0] = %g (expected %g)\n", toFloat(hD[0]), (double)K);
        EXPECT(exact, "the registered kernel computes the exact expected result");
    }

    // ---- 5. mapping the shape DOES change selection, on a warm cache ------
    CHECK_BLAS(hipblasLtUserKernelSetExactMatch(handle, chosen, M, N, 1, K));
    CHECK_BLAS(hipblasLtUserKernelGetCounts(handle, &registered, &selectable));
    EXPECT(selectable == 1, "one kernel is now selectable");

    int afterMapTop = -1;
    CHECK_BLAS(heuristicTop(&afterMapTop));
    std::printf("top solution index after mapping: %d (was %d)\n", afterMapTop, shippedTop);
    EXPECT(afterMapTop == chosen,
           "selection now returns the mapped kernel, so the cached result was discarded");

    // ---- 6. an unmapped shape is untouched --------------------------------
    hipblasLtMatrixLayout_t la2, lb2, lc2, ld2;
    CHECK_BLAS(hipblasLtMatrixLayoutCreate(&la2, HIP_R_16BF, K, 32, K));
    CHECK_BLAS(hipblasLtMatrixLayoutCreate(&lb2, HIP_R_16BF, K, N, K));
    CHECK_BLAS(hipblasLtMatrixLayoutCreate(&lc2, HIP_R_16BF, 32, N, 32));
    CHECK_BLAS(hipblasLtMatrixLayoutCreate(&ld2, HIP_R_16BF, 32, N, 32));
    {
        hipblasLtMatmulHeuristicResult_t r[1];
        int                              n = 0;
        CHECK_BLAS(
            hipblasLtMatmulAlgoGetHeuristic(handle, desc, la2, lb2, lc2, ld2, pref, 1, r, &n));
        EXPECT(n > 0, "a neighbouring shape still resolves");
        CHECK_BLAS(hipblasLtMatmulAlgoIsUserKernel(&r[0].algo, &isUser));
        EXPECT(isUser == 0, "a shape one step away (M=32) still gets a shipped kernel");
    }

    std::printf("\nPASS\n");
    return 0;
}
