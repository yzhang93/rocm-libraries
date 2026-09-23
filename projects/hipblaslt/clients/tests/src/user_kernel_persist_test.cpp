// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Durability cover for user kernel registration, run as two child processes
// against one library root:
//
//   DISABLED_WriteWorker   register and map, then exit
//   DISABLED_ReadWorker    refresh and expect the mapping back
//
// Two processes rather than one is the whole point. Registration lives in
// process-global state, so a single-process test cannot distinguish "the
// journal was replayed" from "the tier was never emptied". The read pass starts
// with an empty tier by construction, and asserts as much before refreshing.
//
// It also pins the property that makes the journal correct: indices are minted
// per process, so the index the read pass gets back is *not* the one the write
// pass recorded. What survives is the mapping, keyed by the shape.

#include <gtest/gtest.h>

#include "user_kernel_test_common.hpp"

namespace
{
    using namespace user_kernel_test;

    std::string workerRoot()
    {
        char const* root = std::getenv(ROOT_ENV);
        return root ? root : std::string{};
    }
} // namespace

TEST(UserKernelPersist, DISABLED_WriteWorker)
{
    const std::string root = workerRoot();
    ASSERT_FALSE(root.empty()) << ROOT_ENV << " is not set";

    const std::vector<std::string> candidates = candidatePayloads();
    ASSERT_FALSE(candidates.empty()) << "no payload shard in this build's library";

    hipblasLtHandle_t handle;
    ASSERT_EQ(hipblasLtCreate(&handle), HIPBLAS_STATUS_SUCCESS);

    Problem p;
    ASSERT_EQ(buildProblem(p), HIPBLAS_STATUS_SUCCESS);

    ASSERT_EQ(hipblasLtUserKernelLibraryOpen(handle, root.c_str()), HIPBLAS_STATUS_SUCCESS);

    hipblasLtMatmulHeuristicResult_t shipped{};
    ASSERT_EQ(topResult(handle, p, shipped), HIPBLAS_STATUS_SUCCESS);
    std::printf("[write] shipped top index: %d\n", algoIndex(shipped.algo));

    // Walk candidates until one yields a kernel that serves the shape, as a
    // tuning loop would. A shard for another device id registers fine but all
    // its kernels are rejected when validated against this GPU.
    int                              chosen = -1;
    size_t                           ws     = 0;
    hipblasLtMatmulHeuristicResult_t result{};

    for(auto const& stem : candidates)
    {
        const std::string dat = stem + ".dat", co = stem + ".co";

        std::vector<int> indices(4096);
        int              numIndices = 0;
        ASSERT_EQ(hipblasLtUserKernelRegister(handle,
                                              dat.c_str(),
                                              co.c_str(),
                                              indices.data(),
                                              static_cast<int>(indices.size()),
                                              &numIndices),
                  HIPBLAS_STATUS_SUCCESS);
        ASSERT_GT(numIndices, 0) << "payload registered";
        indices.resize(std::min<size_t>(numIndices, indices.size()));

        chosen = findUsableKernel(handle, p, indices, result, ws);
        if(chosen >= 0)
            break;
    }
    ASSERT_GE(chosen, 0) << "found a registered kernel that serves the shape";

    ASSERT_EQ(hipblasLtUserKernelSetExactMatch(handle, chosen, M, N, 1, K),
              HIPBLAS_STATUS_SUCCESS);

    hipblasLtMatmulHeuristicResult_t mapped{};
    ASSERT_EQ(topResult(handle, p, mapped), HIPBLAS_STATUS_SUCCESS);
    EXPECT_EQ(algoIndex(mapped.algo), chosen)
        << "selection returns the mapped kernel in the writing process";
    std::printf("[write] mapped index this run: %d\n", algoIndex(mapped.algo));

    const std::filesystem::path rootPath(root);
    EXPECT_TRUE(std::filesystem::exists(rootPath / "registry.log")) << "journal was written";
    EXPECT_TRUE(std::filesystem::is_directory(rootPath / "objects"))
        << "object store was created";
}

TEST(UserKernelPersist, DISABLED_ReadWorker)
{
    const std::string root = workerRoot();
    ASSERT_FALSE(root.empty()) << ROOT_ENV << " is not set";

    hipblasLtHandle_t handle;
    ASSERT_EQ(hipblasLtCreate(&handle), HIPBLAS_STATUS_SUCCESS);

    Problem p;
    ASSERT_EQ(buildProblem(p), HIPBLAS_STATUS_SUCCESS);

    ASSERT_EQ(hipblasLtUserKernelLibraryOpen(handle, root.c_str()), HIPBLAS_STATUS_SUCCESS);

    size_t registered = 99, selectable = 99;
    ASSERT_EQ(hipblasLtUserKernelGetCounts(handle, &registered, &selectable),
              HIPBLAS_STATUS_SUCCESS);
    ASSERT_EQ(registered, 0u)
        << "a fresh process starts with an empty tier, so replay is what is being tested";
    ASSERT_EQ(selectable, 0u) << "and nothing is selectable before refresh";

    hipblasLtMatmulHeuristicResult_t before{};
    ASSERT_EQ(topResult(handle, p, before), HIPBLAS_STATUS_SUCCESS);
    int isUser = -1;
    ASSERT_EQ(hipblasLtMatmulAlgoIsUserKernel(&before.algo, &isUser), HIPBLAS_STATUS_SUCCESS);
    EXPECT_EQ(isUser, 0) << "before refresh the shape gets a shipped kernel";

    int replayed = 0;
    ASSERT_EQ(hipblasLtUserKernelRefresh(handle, &replayed), HIPBLAS_STATUS_SUCCESS);
    std::printf("[read] replayed %d kernels from the journal\n", replayed);
    ASSERT_GT(replayed, 0) << "refresh reinstated the stored payload";

    ASSERT_EQ(hipblasLtUserKernelGetCounts(handle, &registered, &selectable),
              HIPBLAS_STATUS_SUCCESS);
    EXPECT_GT(registered, 0u) << "kernels are registered after refresh";
    EXPECT_EQ(selectable, 1u) << "the recorded exact-match mapping came back";

    hipblasLtMatmulHeuristicResult_t after{};
    ASSERT_EQ(topResult(handle, p, after), HIPBLAS_STATUS_SUCCESS);
    std::printf("[read] top index after refresh: %d (was %d)\n",
                algoIndex(after.algo),
                algoIndex(before.algo));
    EXPECT_GE(algoIndex(after.algo), USER_INDEX_BASE)
        << "selection now returns a user kernel, reinstated from disk in a new process";

    ASSERT_EQ(hipblasLtMatmulAlgoIsUserKernel(&after.algo, &isUser), HIPBLAS_STATUS_SUCCESS);
    EXPECT_EQ(isUser, 1) << "and it is reported as a user kernel";

    // Replaying twice must not double-register or change the answer, since
    // Refresh is a read and the store is content addressed.
    int again = 0;
    ASSERT_EQ(hipblasLtUserKernelRefresh(handle, &again), HIPBLAS_STATUS_SUCCESS);
    ASSERT_EQ(hipblasLtUserKernelGetCounts(handle, &registered, &selectable),
              HIPBLAS_STATUS_SUCCESS);
    EXPECT_EQ(selectable, 1u) << "a second refresh does not multiply the mapping";
}

TEST(UserKernelPersist, SurvivesRestart)
{
    if(candidatePayloads().empty())
        GTEST_SKIP() << "no payload shard in this build's library";

    // A path of our own under the temp directory. The registry refuses a root
    // that is group- or world-writable, since anything able to write there
    // chooses the code the process loads, so the mode is set explicitly rather
    // than left to the caller's umask.
    const auto root = std::filesystem::temp_directory_path()
                      / ("hipblaslt_user_kernel_persist_" + std::to_string(getpid()));

    std::error_code ec;
    std::filesystem::remove_all(root, ec);
    ASSERT_TRUE(std::filesystem::create_directories(root, ec)) << "could not create " << root;
    std::filesystem::permissions(root, std::filesystem::perms::owner_all, ec);
    ASSERT_FALSE(ec) << "could not set permissions on " << root;

    ASSERT_EQ(setenv(ROOT_ENV, root.string().c_str(), 1), 0);

    const int writeStatus = runWorker("UserKernelPersist.DISABLED_WriteWorker");
    const int readStatus
        = writeStatus == 0 ? runWorker("UserKernelPersist.DISABLED_ReadWorker") : -1;

    std::filesystem::remove_all(root, ec);
    static_cast<void>(unsetenv(ROOT_ENV));

    EXPECT_EQ(writeStatus, 0) << "the write pass failed; its output is above";
    EXPECT_EQ(readStatus, 0) << "the read pass failed; its output is above";
}
