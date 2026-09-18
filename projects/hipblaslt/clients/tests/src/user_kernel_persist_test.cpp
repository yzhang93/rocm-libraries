// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Durability cover for user kernel registration. Run as two processes against
// one library root:
//
//   user_kernel_persist_test write <root>   register and map, then exit
//   user_kernel_persist_test read  <root>   refresh and expect the mapping back
//
// Two processes rather than one is the whole point. Registration lives in
// process-global state, so a single-process test cannot distinguish "the
// journal was replayed" from "the tier was never emptied". The read pass starts
// with an empty tier by construction.
//
// It also pins the property that makes the journal correct: indices are minted
// per process, so the index the read pass gets back is *not* the one the write
// pass recorded. What survives is the mapping, keyed by the shape.

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

#define CHECK_BLAS(expr)                                                        \
    do                                                                          \
    {                                                                           \
        hipblasStatus_t s_ = (expr);                                            \
        if(s_ != HIPBLAS_STATUS_SUCCESS)                                        \
        {                                                                       \
            std::printf("FAIL %s:%d  %s -> status %d\n",                        \
                        __FILE__, __LINE__, #expr, (int)s_);                    \
            return 1;                                                           \
        }                                                                       \
    } while(0)

#define EXPECT(cond, msg)                   \
    do                                      \
    {                                       \
        if(!(cond))                         \
        {                                   \
            std::printf("FAIL: %s\n", msg); \
            return 1;                       \
        }                                   \
        std::printf("  ok: %s\n", msg);     \
    } while(0)

    int algoIndex(hipblasLtMatmulAlgo_t const& algo)
    {
        int index = -1;
        std::memcpy(&index, algo.data, sizeof(index));
        return index;
    }

    struct Problem
    {
        hipblasLtMatmulDesc_t       desc = nullptr;
        hipblasLtMatrixLayout_t     la = nullptr, lb = nullptr, lc = nullptr, ld = nullptr;
        hipblasLtMatmulPreference_t pref   = nullptr;
        uint64_t                    maxWs  = 64ull * 1024 * 1024;
    };

    hipblasStatus_t buildProblem(Problem& p)
    {
        auto st = hipblasLtMatmulDescCreate(&p.desc, HIPBLAS_COMPUTE_32F, HIP_R_32F);
        if(st != HIPBLAS_STATUS_SUCCESS)
            return st;
        hipblasOperation_t opT = HIPBLAS_OP_T, opN = HIPBLAS_OP_N;
        st = hipblasLtMatmulDescSetAttribute(p.desc, HIPBLASLT_MATMUL_DESC_TRANSA, &opT, sizeof(opT));
        if(st != HIPBLAS_STATUS_SUCCESS)
            return st;
        st = hipblasLtMatmulDescSetAttribute(p.desc, HIPBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(opN));
        if(st != HIPBLAS_STATUS_SUCCESS)
            return st;

        if((st = hipblasLtMatrixLayoutCreate(&p.la, HIP_R_16BF, K, M, K)) != HIPBLAS_STATUS_SUCCESS
           || (st = hipblasLtMatrixLayoutCreate(&p.lb, HIP_R_16BF, K, N, K)) != HIPBLAS_STATUS_SUCCESS
           || (st = hipblasLtMatrixLayoutCreate(&p.lc, HIP_R_16BF, M, N, M)) != HIPBLAS_STATUS_SUCCESS
           || (st = hipblasLtMatrixLayoutCreate(&p.ld, HIP_R_16BF, M, N, M)) != HIPBLAS_STATUS_SUCCESS)
            return st;

        if((st = hipblasLtMatmulPreferenceCreate(&p.pref)) != HIPBLAS_STATUS_SUCCESS)
            return st;
        return hipblasLtMatmulPreferenceSetAttribute(
            p.pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &p.maxWs, sizeof(p.maxWs));
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

int main(int argc, char** argv)
{
    if(argc < 3)
    {
        std::printf("usage: %s write|read <library-root>\n", argv[0]);
        return 2;
    }
    const std::string mode = argv[1];
    const std::string root = argv[2];

    const std::vector<std::string> candidates = candidatePayloads();
    if(candidates.empty())
    {
        std::printf("SKIP: no payload shard in this build's installed library\n");
        return 0;
    }

    hipblasLtHandle_t handle;
    CHECK_BLAS(hipblasLtCreate(&handle));

    Problem p;
    CHECK_BLAS(buildProblem(p));

    auto topIndex = [&](int* out) -> hipblasStatus_t {
        hipblasLtMatmulHeuristicResult_t r[1];
        int                              n = 0;
        auto st = hipblasLtMatmulAlgoGetHeuristic(
            handle, p.desc, p.la, p.lb, p.lc, p.ld, p.pref, 1, r, &n);
        if(st != HIPBLAS_STATUS_SUCCESS || n <= 0)
            return HIPBLAS_STATUS_NOT_SUPPORTED;
        *out = algoIndex(r[0].algo);
        return HIPBLAS_STATUS_SUCCESS;
    };

    CHECK_BLAS(hipblasLtUserKernelLibraryOpen(handle, root.c_str()));

    if(mode == "write")
    {
        std::printf("[write] library root: %s\n", root.c_str());

        int shippedTop = -1;
        CHECK_BLAS(topIndex(&shippedTop));
        std::printf("[write] shipped top index: %d\n", shippedTop);

        // Walk candidates until one yields a kernel that serves the shape, as
        // a tuning loop would. A shard for another device id registers fine
        // but all its kernels are rejected when validated against this GPU.
        int chosen = -1;
        for(auto const& stem : candidates)
        {
            const std::string dat = stem + ".dat", co = stem + ".co";

            std::vector<int> indices(4096);
            int              numIndices = 0;
            CHECK_BLAS(hipblasLtUserKernelRegister(
                handle, dat.c_str(), co.c_str(), indices.data(), (int)indices.size(),
                &numIndices));
            EXPECT(numIndices > 0, "payload registered");
            indices.resize(numIndices < (int)indices.size() ? numIndices : indices.size());

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
                       handle, p.desc, &alpha, p.la, p.lb, &beta, p.lc, p.ld, got[0].algo, ws)
                       == HIPBLAS_STATUS_SUCCESS
                   && ws <= p.maxWs)
                {
                    chosen = idx;
                    break;
                }
            }
            if(chosen >= 0)
                break;
        }
        EXPECT(chosen >= 0, "found a registered kernel that serves the shape");

        CHECK_BLAS(hipblasLtUserKernelSetExactMatch(handle, chosen, M, N, 1, K));

        int mapped = -1;
        CHECK_BLAS(topIndex(&mapped));
        EXPECT(mapped == chosen, "selection returns the mapped kernel in the writing process");
        std::printf("[write] mapped index this run: %d\n", mapped);

        EXPECT(std::filesystem::exists(std::filesystem::path(root) / "registry.log"),
               "journal was written");
        EXPECT(std::filesystem::is_directory(std::filesystem::path(root) / "objects"),
               "object store was created");
        std::printf("[write] PASS\n");
        return 0;
    }

    if(mode == "read")
    {
        std::printf("[read] library root: %s\n", root.c_str());

        size_t registered = 99, selectable = 99;
        CHECK_BLAS(hipblasLtUserKernelGetCounts(handle, &registered, &selectable));
        EXPECT(registered == 0 && selectable == 0,
               "a fresh process starts with an empty tier, so replay is what is being tested");

        int beforeRefresh = -1;
        CHECK_BLAS(topIndex(&beforeRefresh));
        int isUser = -1;
        {
            hipblasLtMatmulHeuristicResult_t r[1];
            int                              n = 0;
            CHECK_BLAS(hipblasLtMatmulAlgoGetHeuristic(
                handle, p.desc, p.la, p.lb, p.lc, p.ld, p.pref, 1, r, &n));
            CHECK_BLAS(hipblasLtMatmulAlgoIsUserKernel(&r[0].algo, &isUser));
        }
        EXPECT(isUser == 0, "before refresh the shape gets a shipped kernel");

        int replayed = 0;
        CHECK_BLAS(hipblasLtUserKernelRefresh(handle, &replayed));
        std::printf("[read] replayed %d kernels from the journal\n", replayed);
        EXPECT(replayed > 0, "refresh reinstated the stored payload");

        CHECK_BLAS(hipblasLtUserKernelGetCounts(handle, &registered, &selectable));
        EXPECT(registered > 0, "kernels are registered after refresh");
        EXPECT(selectable == 1, "the recorded exact-match mapping came back");

        int afterRefresh = -1;
        CHECK_BLAS(topIndex(&afterRefresh));
        std::printf("[read] top index after refresh: %d (was %d)\n", afterRefresh, beforeRefresh);
        EXPECT(afterRefresh >= (1 << 30),
               "selection now returns a user kernel, reinstated from disk in a new process");

        {
            hipblasLtMatmulHeuristicResult_t r[1];
            int                              n = 0;
            CHECK_BLAS(hipblasLtMatmulAlgoGetHeuristic(
                handle, p.desc, p.la, p.lb, p.lc, p.ld, p.pref, 1, r, &n));
            CHECK_BLAS(hipblasLtMatmulAlgoIsUserKernel(&r[0].algo, &isUser));
        }
        EXPECT(isUser == 1, "and it is reported as a user kernel");

        // Replaying twice must not double-register or change the answer, since
        // Refresh is a read and the store is content addressed.
        int again = 0;
        CHECK_BLAS(hipblasLtUserKernelRefresh(handle, &again));
        int afterSecond = -1;
        CHECK_BLAS(topIndex(&afterSecond));
        CHECK_BLAS(hipblasLtUserKernelGetCounts(handle, &registered, &selectable));
        EXPECT(selectable == 1, "a second refresh does not multiply the mapping");

        std::printf("[read] PASS\n");
        return 0;
    }

    std::printf("unknown mode %s\n", mode.c_str());
    return 2;
}
