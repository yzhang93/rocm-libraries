// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Shared scaffolding for the two user kernel registration tests.
//
// Both need the same payload discovery and the same bf16 TN problem, and both
// have to run in a child process. The user kernel tier is process-global and
// both tests assert it is empty before they start, which cannot hold inside a
// binary that has already run other cases -- and registering a few thousand
// kernels would leak into whatever runs next. So the visible test case spawns
// this same executable with only its DISABLED_ worker enabled, and the worker
// gets the fresh process it needs.
//
// Included by relative path rather than through a target include directory,
// for the reason recorded at the bottom of this directory's CMakeLists.txt.

#pragma once

#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt-ext.hpp>
#include <hipblaslt/hipblaslt.h>

#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

extern char** environ;

namespace user_kernel_test
{
    // A tall, skinny GEMM: small M against large N and K. Memory-bound, and
    // small M means tile choice matters, so registering a different kernel has
    // something to prove.
    inline constexpr int64_t M = 16, N = 6144, K = 2048;

    inline constexpr uint64_t MAX_WORKSPACE = 64ull * 1024 * 1024;

    // User kernels are numbered from 2^30 up, so they cannot collide with the
    // system range [0, 2^30).
    inline constexpr int USER_INDEX_BASE = 1 << 30;

    // bf16 1.0. Filling A and B with it makes every output element exactly K,
    // which bf16 represents without rounding, so a mismatch is a dispatch or
    // launch fault rather than precision.
    inline constexpr uint16_t BF16_ONE = 0x3F80;

    // How the parent hands a library root to the persist workers.
    inline constexpr char const* ROOT_ENV = "HIPBLASLT_USER_KERNEL_TEST_ROOT";

    // Solution index carried at the head of an algo's opaque data.
    inline int algoIndex(hipblasLtMatmulAlgo_t const& algo)
    {
        int index = -1;
        std::memcpy(&index, algo.data, sizeof(index));
        return index;
    }

    inline std::string selfExe()
    {
        std::error_code ec;
        auto            path = std::filesystem::read_symlink("/proc/self/exe", ec);
        return ec ? std::string{} : path.string();
    }

    // PCI device id of the current GPU, e.g. "75a0", or empty if unreadable.
    //
    // Shard filenames carry the device ids they were built for. Filtering on it
    // keeps the test from registering kernels that belong to other GPUs, which
    // would be inert here and would inflate the counts asserted below.
    inline std::string devicePciId()
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

    // Directories that may hold compiled kernel shards.
    //
    // Derived from the executable rather than the working directory, so the
    // test works whether it is run from the build tree, from an install, or by
    // ctest from somewhere else entirely.
    inline std::vector<std::string> libraryRoots()
    {
        std::vector<std::string> roots;
        if(char const* env = std::getenv("HIPBLASLT_USER_KERNEL_TEST_LIB_DIR"))
            roots.emplace_back(env);

        static char const* const relative[] = {
            "Tensile/library", // build tree
            "lib/hipblaslt/library", // install tree
            "hipblaslt-install/lib/hipblaslt/library", // in-repo install
        };

        std::error_code ec;
        for(std::filesystem::path dir = std::filesystem::path(selfExe()).parent_path();
            !dir.empty() && dir != dir.root_path();
            dir = dir.parent_path())
        {
            for(auto const* suffix : relative)
            {
                auto candidate = dir / suffix;
                if(std::filesystem::is_directory(candidate, ec))
                    roots.push_back(candidate.string());
            }
        }
        return roots;
    }

    // Candidate payloads, biggest first.
    //
    // Globbed rather than named so the test follows whatever GPU this build
    // targets. Several shards match and which is usable cannot be known without
    // trying: one built for a different device id or CU count deserializes fine
    // but every kernel in it is rejected when validated against this GPU.
    // Bigger shards come first only because they hold more kernels and are
    // likelier to contain a fit.
    inline std::vector<std::string> candidatePayloads()
    {
        const std::string pciId = devicePciId();

        std::vector<std::pair<uintmax_t, std::string>> found;
        std::error_code                                ec;

        for(auto const& root : libraryRoots())
        {
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
                    std::transform(
                        lowerName.begin(), lowerName.end(), lowerName.begin(), ::tolower);
                    if(!pciId.empty() && lowerName.find(pciId) == std::string::npos)
                        continue;

                    // Recover the stem: the loader appends the ".zlib" probe
                    // itself, so the bare ".dat" name is what gets passed even
                    // when only the compressed form exists.
                    std::string stem = entry.path().string();
                    for(std::string suffix : {std::string(".zlib"), std::string(".dat")})
                        if(stem.size() > suffix.size()
                           && stem.compare(
                                  stem.size() - suffix.size(), suffix.size(), suffix)
                                  == 0)
                            stem.resize(stem.size() - suffix.size());
                    if(stem == entry.path().string())
                        continue; // not a .dat/.dat.zlib
                    if(!std::filesystem::exists(stem + ".co"))
                        continue;

                    found.emplace_back(entry.file_size(ec), stem);
                }
            }
            if(!found.empty())
                break;
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

    struct Problem
    {
        hipblasLtMatmulDesc_t       desc = nullptr;
        hipblasLtMatrixLayout_t     la = nullptr, lb = nullptr, lc = nullptr, ld = nullptr;
        hipblasLtMatmulPreference_t pref = nullptr;
    };

    // A bf16 TN GEMM at m x n x K, no epilogue.
    inline hipblasStatus_t buildProblem(Problem& p, int64_t m = M, int64_t n = N)
    {
        auto st = hipblasLtMatmulDescCreate(&p.desc, HIPBLAS_COMPUTE_32F, HIP_R_32F);
        if(st != HIPBLAS_STATUS_SUCCESS)
            return st;

        hipblasOperation_t opT = HIPBLAS_OP_T, opN = HIPBLAS_OP_N;
        if((st = hipblasLtMatmulDescSetAttribute(
                p.desc, HIPBLASLT_MATMUL_DESC_TRANSA, &opT, sizeof(opT)))
           != HIPBLAS_STATUS_SUCCESS)
            return st;
        if((st = hipblasLtMatmulDescSetAttribute(
                p.desc, HIPBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(opN)))
           != HIPBLAS_STATUS_SUCCESS)
            return st;

        if((st = hipblasLtMatrixLayoutCreate(&p.la, HIP_R_16BF, K, m, K))
               != HIPBLAS_STATUS_SUCCESS
           || (st = hipblasLtMatrixLayoutCreate(&p.lb, HIP_R_16BF, K, n, K))
                  != HIPBLAS_STATUS_SUCCESS
           || (st = hipblasLtMatrixLayoutCreate(&p.lc, HIP_R_16BF, m, n, m))
                  != HIPBLAS_STATUS_SUCCESS
           || (st = hipblasLtMatrixLayoutCreate(&p.ld, HIP_R_16BF, m, n, m))
                  != HIPBLAS_STATUS_SUCCESS)
            return st;

        if((st = hipblasLtMatmulPreferenceCreate(&p.pref)) != HIPBLAS_STATUS_SUCCESS)
            return st;

        uint64_t maxWs = MAX_WORKSPACE;
        return hipblasLtMatmulPreferenceSetAttribute(
            p.pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &maxWs, sizeof(maxWs));
    }

    inline hipblasStatus_t
        topResult(hipblasLtHandle_t handle, Problem const& p, hipblasLtMatmulHeuristicResult_t& out)
    {
        hipblasLtMatmulHeuristicResult_t results[1];
        int                              returned = 0;
        auto                             st       = hipblasLtMatmulAlgoGetHeuristic(
            handle, p.desc, p.la, p.lb, p.lc, p.ld, p.pref, 1, results, &returned);
        if(st != HIPBLAS_STATUS_SUCCESS)
            return st;
        if(returned <= 0)
            return HIPBLAS_STATUS_NOT_SUPPORTED;
        out = results[0];
        return HIPBLAS_STATUS_SUCCESS;
    }

    // First index that resolves and can serve the problem within the workspace
    // budget, or -1. This is the lookup a tuning loop does before committing.
    inline int findUsableKernel(hipblasLtHandle_t                 handle,
                                Problem const&                    p,
                                std::vector<int> const&           indices,
                                hipblasLtMatmulHeuristicResult_t& out,
                                size_t&                           workspace)
    {
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
               != HIPBLAS_STATUS_SUCCESS)
                continue;
            if(ws > MAX_WORKSPACE)
                continue;

            out       = got[0];
            workspace = ws;
            return idx;
        }
        return -1;
    }

    // Re-run this executable with only the named DISABLED_ case enabled, and
    // return its exit status (-1 if it could not be run).
    //
    // posix_spawn rather than fork, because HIP is already initialised in the
    // parent by the time a test calls this and forking a process with live
    // driver threads is a hazard the spawn path avoids.
    inline int runWorker(std::string const& filter)
    {
        std::string exe = selfExe();
        if(exe.empty())
            return -1;

        std::string alsoDisabled = "--gtest_also_run_disabled_tests";
        std::string filterArg    = "--gtest_filter=" + filter;
        char*       argv[]
            = {exe.data(), alsoDisabled.data(), filterArg.data(), nullptr};

        pid_t pid = 0;
        if(posix_spawn(&pid, exe.c_str(), nullptr, nullptr, argv, environ) != 0)
            return -1;

        int status = 0;
        if(waitpid(pid, &status, 0) < 0)
            return -1;
        return WIFEXITED(status) ? WEXITSTATUS(status) : -1;
    }
} // namespace user_kernel_test
