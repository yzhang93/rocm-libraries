/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

// Regression test for ROCM-25647.
//
// hipBLASLt PR #7754 ("Reduce CachingLibrary map lookup/write overhead") changed
// CachingLibrary's solution caches to be keyed on a bare size_t problem hash
// (std::hash<MyProblem>) instead of the full MyProblem object. Keying on the hash
// alone means two DISTINCT problems whose hashes collide land in the same cache
// slot, so a lookup for problem B can return a Tensile solution that was cached for
// a different problem A. Executing the wrong solution yields numerically wrong GEMM
// results -- the gfx12 f16 NN gemm_512 (M=K=511) failures reported in ROCM-25647.
//
// PR #8356 reverted #7754, restoring the full-MyProblem cache key (an
// std::unordered_map keyed on MyProblem distinguishes hash-colliding entries via
// operator==). This test pins that behavior so the defect cannot silently return:
//
//   * It is GPU-free and deterministic: it instantiates CachingLibrary with a small
//     mock problem/solution and a mock sub-library, and FORCES a hash collision by
//     giving MockProblem a std::hash that always returns the same value while
//     operator== still distinguishes instances.
//   * On the buggy (size_t-keyed) code the cache returns the WRONG problem's
//     solution -> the EXPECT below fails.
//   * On the fixed (MyProblem-keyed) code the cache correctly misses for the second,
//     distinct problem and the sub-library supplies the right solution -> it passes.
//
// ---------------------------------------------------------------------------
// Why this CachingLibrary (TensileLite-host) unit test lives in the hipBLASLt
// *client* test binary (hipblaslt-test), not in tensilelite/tests:
//
// CachingLibrary is TensileLite-host code, so tensilelite/tests/ is its natural
// conceptual home -- but that C++ gtest suite (the `tensilelite-tests` target) is
// NOT built or run by TheRock GitHub Actions CI: TENSILELITE_BUILD_TESTING defaults
// OFF (projects/hipblaslt/CMakeLists.txt) and the TheRock superproject build leaves
// it off and does not package the binary, and there is no CI job that runs it. The
// only C++ test binary TheRock builds, ships, and runs for hipBLASLt is this client
// `hipblaslt-test`
// (HIPBLASLT_BUILD_TESTING), executed by test/therock/test_hipblaslt.py. Because
// hipblaslt-test already links roc::tensilelite-host (via
// hipblaslt-clients-common), this white-box unit test can include
// <Tensile/CachingLibrary.hpp> directly and run with no new build dependency.
// Placing the regression here is what makes it actually execute in CI and guard
// ROCM-25647. (If the tensilelite-tests suite is ever CI-enabled, consider moving
// this back alongside its siblings.)
//
// Smoke tier: PR CI runs hipBLASLt with TEST_TYPE=quick, which selects
// `--gtest_filter=*smoke*` (test/therock/test_hipblaslt.py). The test names below
// therefore carry the `smoke` category token (the hipBLASLt convention of encoding
// the test category in the test name) so this fast, host-only guard runs on the PR
// gate -- not only in the full/nightly lane.
// ---------------------------------------------------------------------------
//
// Defect category (why these are unit, not integration, tests):
// This is a "lossy memoization key" defect -- a cache keyed on a hash (lossy)
// instead of the value (lossless) can serve a wrong cached result whenever two
// distinct inputs collide. Such a collision cannot be reliably reproduced by an
// end-to-end GEMM test: a real 64-bit hash collision between two specific configs
// is rare, only manifests once BOTH colliding configs run in one process against
// the shared cache, and is invisible to any suite that doesn't happen to run the
// exact colliding pair (which is why this surfaced in the large, fixed rocBLAS
// pre-checkin set but not in isolated hipBLASLt tests). The robust guard is to
// FORCE a collision at the unit level and assert collision-safety for EVERY cache
// CachingLibrary owns:
//   * findBestSolution (m_cache)                         -- #7754 made it size_t-keyed; this
//   * findTopSolutions (m_caches)                        -- test FAILS on #7754, PASSES on the fix.
//   * findTopSolutionsGroupedGemm (m_cachesGroupedGemm)  -- this cache was NOT changed by #7754
//        (it stayed keyed on the full std::vector<MyProblem>); the test below is therefore a
//        PREVENTIVE guard that PASSES on both, ensuring this last cache is never regressed into
//        the same lossy-key class. This is the "category, not just the instance" coverage.
// findAllSolutions / findAllSolutionsGroupedGemm are not cached (they delegate
// straight to the sub-library) and so carry no collision risk.

#include <gtest/gtest.h>

#include <cstdint>

#include <Tensile/AMDGPU.hpp>
#include <Tensile/CachingLibrary.hpp>
#include <Tensile/LibraryGeneration.hpp>
#include <Tensile/SolutionLibrary.hpp>
#include <Tensile/UserKernelIndex.hpp>

using namespace TensileLite;

namespace
{
    // Minimal stand-in for a Tensile problem. `id` is its identity.
    struct MockProblem
    {
        int id = 0;

        bool operator==(MockProblem const& rhs) const
        {
            return id == rhs.id;
        }
    };

    // Minimal stand-in for a Tensile solution; carries the id of the problem it
    // was generated for so the test can detect a wrong-problem cache hit.
    struct MockSolution
    {
        int id = 0;
    };
}

namespace std
{
    // Force a hash collision for EVERY MockProblem. Combined with the distinguishing
    // operator== above, this is exactly the situation #7754 mishandled: distinct
    // problems that share a hash bucket. A correct cache must still tell them apart.
    template <>
    struct hash<MockProblem>
    {
        size_t operator()(MockProblem const&) const noexcept
        {
            return 0xC0FFEEu;
        }
    };

    // CachingLibrary also instantiates a grouped-GEMM cache keyed on
    // std::vector<MyProblem>, so a hash for the vector is required to compile.
    // (The real ContractionProblemGemm provides the analogous specialization.)
    template <>
    struct hash<std::vector<MockProblem>>
    {
        size_t operator()(std::vector<MockProblem> const&) const noexcept
        {
            return 0xC0FFEEu;
        }
    };
}

namespace
{
    // Sub-library that always "finds" a solution whose id matches the queried
    // problem, and counts how many times it was consulted (to prove caching works).
    struct MockSubLibrary : public SolutionLibrary<MockProblem, MockSolution>
    {
        mutable int findBestCalls     = 0;
        mutable int findTopCalls      = 0;
        mutable int findTopGroupCalls = 0;

        // What findTopSolutions reports about completeness. The SolutionLibrary
        // base returns true, so this default preserves that.
        bool retAll = true;

        bool lastFindTopAlreadyRetAll() const override
        {
            return retAll;
        }

        std::shared_ptr<MockSolution> makeSolution(int id) const
        {
            auto s = std::make_shared<MockSolution>();
            s->id  = id;
            return s;
        }

        std::shared_ptr<MockSolution>
            getSolutionByIndex(MockProblem const&, Hardware const&, const int) const override
        {
            return nullptr;
        }

        std::shared_ptr<MockSolution> findBestSolution(MockProblem const& problem,
                                                       Hardware const&,
                                                       double* fitness) const override
        {
            ++findBestCalls;
            if(fitness)
                *fitness = 1.0;
            return makeSolution(problem.id);
        }

        SolutionSet<MockSolution> findAllSolutions(MockProblem const&,
                                                   Hardware const&,
                                                   SolutionLibrarySearchType) const override
        {
            return {};
        }

        SolutionSet<MockSolution>
            findAllSolutionsGroupedGemm(std::vector<MockProblem> const&,
                                        Hardware const&,
                                        SolutionLibrarySearchType) const override
        {
            return {};
        }

        SolutionVector<MockSolution>
            findTopSolutions(MockProblem const& problem, Hardware const&, int) const override
        {
            ++findTopCalls;
            return {makeSolution(problem.id)};
        }

        SolutionVector<MockSolution> findTopSolutionsGroupedGemm(
            std::vector<MockProblem> const& problems, Hardware const&, int) const override
        {
            ++findTopGroupCalls;
            return {makeSolution(problems.empty() ? 0 : problems.front().id)};
        }

        std::string type() const override
        {
            return "MockSubLibrary";
        }
        std::string description() const override
        {
            return "MockSubLibrary";
        }
    };

    AMDGPU makeGpu()
    {
        return AMDGPU(AMDGPU::Processor::gfx1201, 64, "gfx1201");
    }
}

// findBestSolution path: a hash collision must not return a different problem's
// cached solution.
TEST(CachingLibraryCollision, smoke_FindBestSolutionDistinguishesCollidingProblems)
{
    MockProblem a{1};
    MockProblem b{2};

    // Precondition: the two problems are distinct but share a hash bucket. This is
    // the collision that #7754 mishandled.
    ASSERT_FALSE(a == b);
    ASSERT_EQ(std::hash<MockProblem>{}(a), std::hash<MockProblem>{}(b));

    auto                                      sub = std::make_shared<MockSubLibrary>();
    CachingLibrary<MockProblem, MockSolution> library(sub);
    auto                                      gpu = makeGpu();

    auto solA = library.findBestSolution(a, gpu);
    ASSERT_NE(solA, nullptr);
    EXPECT_EQ(solA->id, 1);

    auto solB = library.findBestSolution(b, gpu);
    ASSERT_NE(solB, nullptr);
    EXPECT_EQ(solB->id, 2)
        << "ROCM-25647: CachingLibrary returned a solution cached for a DIFFERENT problem "
           "that shares a hash bucket. A size_t-hash-keyed cache (PR #7754) cannot "
           "distinguish hash-colliding problems and serves the wrong Tensile solution, "
           "producing numerically wrong GEMM results.";
}

// The cache must still actually cache: a repeated lookup of the same problem must
// not re-consult the sub-library. (Guards against a "fix" that simply disables
// caching, which would make the collision test pass vacuously.)
TEST(CachingLibraryCollision, smoke_RepeatedLookupIsCached)
{
    MockProblem a{7};

    auto                                      sub = std::make_shared<MockSubLibrary>();
    CachingLibrary<MockProblem, MockSolution> library(sub);
    auto                                      gpu = makeGpu();

    (void)library.findBestSolution(a, gpu);
    (void)library.findBestSolution(a, gpu);

    EXPECT_EQ(sub->findBestCalls, 1)
        << "CachingLibrary should consult the sub-library once per distinct problem and "
           "serve the cached result thereafter.";
}

// findTopSolutions path: #7754 also merged the top-solutions caches into a single
// size_t-keyed map, so exercise that path too.
TEST(CachingLibraryCollision, smoke_FindTopSolutionsDistinguishesCollidingProblems)
{
    MockProblem a{1};
    MockProblem b{2};

    ASSERT_FALSE(a == b);
    ASSERT_EQ(std::hash<MockProblem>{}(a), std::hash<MockProblem>{}(b));

    auto                                      sub = std::make_shared<MockSubLibrary>();
    CachingLibrary<MockProblem, MockSolution> library(sub);
    auto                                      gpu = makeGpu();

    auto topA = library.findTopSolutions(a, gpu, 1);
    ASSERT_EQ(topA.size(), 1u);
    ASSERT_NE(topA[0], nullptr);
    EXPECT_EQ(topA[0]->id, 1);

    auto topB = library.findTopSolutions(b, gpu, 1);
    ASSERT_EQ(topB.size(), 1u);
    ASSERT_NE(topB[0], nullptr);
    EXPECT_EQ(topB[0]->id, 2)
        << "ROCM-25647: CachingLibrary::findTopSolutions returned the wrong, hash-colliding "
           "problem's cached solutions (size_t-hash-keyed cache from PR #7754).";
}

// findTopSolutionsGroupedGemm path: the grouped-GEMM cache is keyed on the full
// std::vector<MyProblem>. Unlike the other caches, #7754 did NOT make this one
// size_t-keyed, so this test PASSES on both the buggy and fixed code. It is a
// PREVENTIVE guard: it documents and enforces that this cache stays value-keyed
// (collision-safe), so a future "optimization" cannot quietly reintroduce the
// ROCM-25647 lossy-key class here. (To confirm it has teeth, temporarily change
// the key to a size_t hash and it fails like the others.)
TEST(CachingLibraryCollision, smoke_FindTopSolutionsGroupedGemmDistinguishesCollidingProblems)
{
    std::vector<MockProblem> groupA{MockProblem{1}};
    std::vector<MockProblem> groupB{MockProblem{2}};

    // Distinct problem groups that share a hash bucket.
    ASSERT_FALSE(groupA == groupB);
    ASSERT_EQ(std::hash<std::vector<MockProblem>>{}(groupA),
              std::hash<std::vector<MockProblem>>{}(groupB));

    auto                                      sub = std::make_shared<MockSubLibrary>();
    CachingLibrary<MockProblem, MockSolution> library(sub);
    auto                                      gpu = makeGpu();

    auto topA = library.findTopSolutionsGroupedGemm(groupA, gpu, 1);
    ASSERT_EQ(topA.size(), 1u);
    ASSERT_NE(topA[0], nullptr);
    EXPECT_EQ(topA[0]->id, 1);

    auto topB = library.findTopSolutionsGroupedGemm(groupB, gpu, 1);
    ASSERT_EQ(topB.size(), 1u);
    ASSERT_NE(topB[0], nullptr);
    EXPECT_EQ(topB[0]->id, 2)
        << "ROCM-25647 (preventive): CachingLibrary::findTopSolutionsGroupedGemm returned the "
           "wrong, hash-colliding problem group's cached solutions. The grouped-GEMM cache must "
           "stay keyed on the full std::vector<MyProblem>, not a lossy hash.";
}

// ---------------------------------------------------------------------------
// Cache coherence across a user kernel registration.
//
// CachingLibrary sits above the row ladder and answers without consulting it,
// so registering a kernel for a problem that is already memoized would leave
// the new kernel invisible to the one problem it was built for -- and that
// problem is by construction the shape a tuning loop has been hammering, so it
// is the entry most certain to be cached.
//
// The mechanism is a generation counter shared with the enclosing
// MasterSolutionLibrary. Cached values carry the generation they were resolved
// under; a mismatch is treated as a miss and re-resolved. Entries are never
// erased, only rejected and overwritten.
// ---------------------------------------------------------------------------

// findBestSolution must re-resolve after the generation advances.
TEST(CachingLibraryGeneration, smoke_FindBestSolutionRejectsStaleEntry)
{
    MockProblem a{11};

    auto sub        = std::make_shared<MockSubLibrary>();
    auto generation = std::make_shared<LibraryGeneration>();
    CachingLibrary<MockProblem, MockSolution> library(sub, generation);
    auto                                      gpu = makeGpu();

    (void)library.findBestSolution(a, gpu);
    (void)library.findBestSolution(a, gpu);
    ASSERT_EQ(sub->findBestCalls, 1) << "second lookup should have been a cache hit";

    // Stands in for a registration publishing a new user tier snapshot.
    generation->value.fetch_add(1, std::memory_order_release);

    (void)library.findBestSolution(a, gpu);
    EXPECT_EQ(sub->findBestCalls, 2)
        << "after the generation advanced, the memoized solution predates the new user "
           "tier and must be re-resolved rather than served from the cache.";

    // And the refreshed entry is itself cacheable under the new generation.
    (void)library.findBestSolution(a, gpu);
    EXPECT_EQ(sub->findBestCalls, 2)
        << "the re-resolved entry should be stamped with the current generation and hit.";
}

// findTopSolutions must re-resolve after the generation advances. This is the
// path that matters most for the demo, since hipblasLtMatmulAlgoGetHeuristic
// reaches Tensile through it.
TEST(CachingLibraryGeneration, smoke_FindTopSolutionsRejectsStaleEntry)
{
    MockProblem a{12};

    auto sub        = std::make_shared<MockSubLibrary>();
    auto generation = std::make_shared<LibraryGeneration>();
    CachingLibrary<MockProblem, MockSolution> library(sub, generation);
    auto                                      gpu = makeGpu();

    (void)library.findTopSolutions(a, gpu, 1);
    (void)library.findTopSolutions(a, gpu, 1);
    ASSERT_EQ(sub->findTopCalls, 1) << "second lookup should have been a cache hit";

    generation->value.fetch_add(1, std::memory_order_release);

    (void)library.findTopSolutions(a, gpu, 1);
    EXPECT_EQ(sub->findTopCalls, 2)
        << "a registered kernel must be reachable through findTopSolutions for the problem "
           "it was registered for, which requires rejecting the stale memo entry.";
}

// The reason m_caches and m_cachesAllSolutions had to be merged rather than
// stamped separately. The hit condition is satisfied by the completeness flag
// alone, so a stale "cache holds everything" answer short-circuits the lookup
// and returns a stale vector no matter how many solutions were requested. With
// the flag living in the same stamped entry as the vector, one comparison
// governs both.
TEST(CachingLibraryGeneration, smoke_StaleContainsAllFlagDoesNotShortCircuit)
{
    MockProblem a{13};

    auto sub = std::make_shared<MockSubLibrary>();
    // The sub-library reports that one solution is all there is, so the cached
    // entry records containsAll = true.
    sub->retAll     = true;
    auto generation = std::make_shared<LibraryGeneration>();
    CachingLibrary<MockProblem, MockSolution> library(sub, generation);
    auto                                      gpu = makeGpu();

    (void)library.findTopSolutions(a, gpu, 1);
    ASSERT_EQ(sub->findTopCalls, 1);

    // Asking for more than was cached is still a hit, because containsAll says
    // there is nothing more to find. That is correct while the library is
    // unchanged.
    (void)library.findTopSolutions(a, gpu, 8);
    ASSERT_EQ(sub->findTopCalls, 1) << "containsAll should suppress a re-query on its own";

    generation->value.fetch_add(1, std::memory_order_release);

    (void)library.findTopSolutions(a, gpu, 8);
    EXPECT_EQ(sub->findTopCalls, 2)
        << "a stale containsAll flag must not short-circuit the lookup after the generation "
           "advanced; the flag and the vector have to share one staleness decision.";
}

// The hit path has to keep publishing the completeness flag, because
// getBestSolutions consumes lastFindTopAlreadyRetAll() to decide whether to
// fall back to getAllSolutions.
TEST(CachingLibraryGeneration, smoke_LastFindTopRetAllSurvivesCacheHit)
{
    MockProblem a{14};

    auto sub    = std::make_shared<MockSubLibrary>();
    sub->retAll = true;
    auto generation = std::make_shared<LibraryGeneration>();
    CachingLibrary<MockProblem, MockSolution> library(sub, generation);
    auto                                      gpu = makeGpu();

    // The miss path reports whether the sub-library came up short of the
    // request, which for an exact fill is false -- distinct from the
    // completeness flag it stores in the cache. That difference is what makes
    // the assertion below meaningful: only the hit path can turn this true.
    (void)library.findTopSolutions(a, gpu, 1);
    ASSERT_FALSE(library.lastFindTopAlreadyRetAll());

    // Served from the cache this time, so the flag must be republished from the
    // cached entry rather than left where the previous call happened to leave it.
    (void)library.findTopSolutions(a, gpu, 1);
    ASSERT_EQ(sub->findTopCalls, 1) << "expected a cache hit";
    EXPECT_TRUE(library.lastFindTopAlreadyRetAll())
        << "the cache-hit path must publish the cached completeness flag, or callers lose the "
           "signal that drives the getAllSolutions fallback.";
}

// Grouped GEMM shares the same staleness rule.
TEST(CachingLibraryGeneration, smoke_GroupedGemmRejectsStaleEntry)
{
    std::vector<MockProblem> group{MockProblem{15}};

    auto sub        = std::make_shared<MockSubLibrary>();
    auto generation = std::make_shared<LibraryGeneration>();
    CachingLibrary<MockProblem, MockSolution> library(sub, generation);
    auto                                      gpu = makeGpu();

    (void)library.findTopSolutionsGroupedGemm(group, gpu, 1);
    (void)library.findTopSolutionsGroupedGemm(group, gpu, 1);
    ASSERT_EQ(sub->findTopGroupCalls, 1) << "second lookup should have been a cache hit";

    generation->value.fetch_add(1, std::memory_order_release);

    (void)library.findTopSolutionsGroupedGemm(group, gpu, 1);
    EXPECT_EQ(sub->findTopGroupCalls, 2)
        << "the grouped-GEMM memo must honour the generation stamp too.";
}

// A cache constructed without a generation keeps a private counter that never
// advances, so libraries built outside a MasterSolutionLibrary behave exactly
// as they did before generation stamping was introduced.
TEST(CachingLibraryGeneration, smoke_CacheWithoutGenerationStillCaches)
{
    MockProblem a{16};

    auto                                      sub = std::make_shared<MockSubLibrary>();
    CachingLibrary<MockProblem, MockSolution> library(sub);
    auto                                      gpu = makeGpu();

    (void)library.findBestSolution(a, gpu);
    (void)library.findBestSolution(a, gpu);

    EXPECT_EQ(sub->findBestCalls, 1)
        << "the generation argument is optional; omitting it must not disable caching.";
}

// ---------------------------------------------------------------------------
// Index space partition. The range a solution index falls in is what records
// whether it came from the shipped library or from a runtime registration, so
// the boundary is a contract rather than an implementation detail.
// ---------------------------------------------------------------------------
TEST(UserKernelIndexSpace, smoke_PartitionBoundary)
{
    EXPECT_EQ(UserKernelIndexBase, 1 << 30);

    EXPECT_FALSE(isUserKernelIndex(0));
    EXPECT_FALSE(isUserKernelIndex(1));
    EXPECT_FALSE(isUserKernelIndex(UserKernelIndexBase - 1))
        << "the last system index must not be mistaken for a user kernel.";

    EXPECT_TRUE(isUserKernelIndex(UserKernelIndexBase));
    EXPECT_TRUE(isUserKernelIndex(UserKernelIndexBase + 1));

    // Indices are int, so every index in the user range [2^30, 2^31) has to be
    // representable as a positive int. The range ends exactly at INT32_MAX,
    // which leaves 2^30 usable slots.
    EXPECT_GT(UserKernelIndexBase, 0);
    EXPECT_LE(static_cast<int64_t>(UserKernelIndexBase) * 2 - 1, INT32_MAX);
}
