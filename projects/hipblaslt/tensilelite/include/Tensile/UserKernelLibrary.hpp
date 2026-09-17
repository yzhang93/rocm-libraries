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

#pragma once

#include <atomic>
#include <cstddef>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <unordered_map>
#include <vector>

#include <Tensile/LibraryGeneration.hpp>
#include <Tensile/SolutionLibrary.hpp>
#include <Tensile/Task.hpp>
#include <Tensile/UserKernelIndex.hpp>
#include <Tensile/Utils.hpp>

namespace TensileLite
{
    /**
     * Fine match key for the user tier: the same [M, N, batch, K] tuple the
     * shipped Equality tables are keyed on, which is
     * (FreeSizeA, FreeSizeB, BatchSize, BoundSize) of the problem.
     */
    struct UserExactKey
    {
        size_t m     = 0;
        size_t n     = 0;
        size_t batch = 0;
        size_t k     = 0;

        bool operator==(UserExactKey const& rhs) const
        {
            return m == rhs.m && n == rhs.n && batch == rhs.batch && k == rhs.k;
        }
        bool operator!=(UserExactKey const& rhs) const
        {
            return !(*this == rhs);
        }
    };

    struct UserExactKeyHash
    {
        size_t operator()(UserExactKey const& key) const noexcept
        {
            size_t h = 0;
            for(size_t v : {key.m, key.n, key.batch, key.k})
                h ^= std::hash<size_t>{}(v) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
            return h;
        }
    };

    template <typename MyProblem>
    inline UserExactKey userExactKeyFor(MyProblem const& problem)
    {
        return UserExactKey{problem.freeSizeA(0),
                            problem.freeSizeB(0),
                            problem.batchSize(0),
                            problem.boundSize(0)};
    }

    /**
     * Immutable view of the user tier. A registration builds a new one and
     * publishes it by atomic store, so a reader either sees the whole previous
     * state or the whole new one and nothing in between. Readers hold a
     * shared_ptr, so a snapshot displaced mid-read stays alive until they are
     * done with it.
     *
     * The two members answer different questions, and keeping them apart is
     * what makes "registration does not change selection" true:
     *
     *  - `solutions` is everything registered. It feeds enumeration and
     *    dispatch by explicit index, so a freshly registered kernel is
     *    immediately executable.
     *  - `exactTable` holds only kernels made selectable by SetExactMatch. It
     *    feeds findBestSolution and findTopSolutions, so nothing a tuning loop
     *    registers can displace a shipped kernel until it is asked for.
     */
    template <typename MySolution>
    struct UserTierSnapshot
    {
        std::map<int, std::shared_ptr<MySolution>>                                  solutions;
        std::unordered_map<UserExactKey, std::shared_ptr<MySolution>, UserExactKeyHash> exactTable;
    };

    /**
     * The mutable tier at the head of the selection ladder.
     *
     * The row carrying this library is inserted once, at deserialization, and
     * never removed: ProblemSelectionLibrary::rows is a bare vector walked
     * without a lock, so touching it while GEMMs are in flight would be a data
     * race. All mutability lives here instead, behind the snapshot swap.
     *
     * Because the row is always present it is also always consulted, so the
     * empty case has to be close to free. The two atomic flags below are the
     * whole fast path: with nothing registered a lookup is one relaxed load and
     * a branch, and the snapshot shared_ptr is never touched.
     */
    template <typename MyProblem, typename MySolution = typename MyProblem::Solution>
    struct UserKernelLibrary : public SolutionLibrary<MyProblem, MySolution>
    {
        using Snapshot = UserTierSnapshot<MySolution>;

        UserKernelLibrary(std::shared_ptr<LibraryGeneration> generation = nullptr)
            : m_generation(generation ? generation : std::make_shared<LibraryGeneration>())
            , m_snapshot(std::make_shared<const Snapshot>())
        {
        }

        static std::string Type()
        {
            return "UserKernel";
        }
        virtual std::string type() const override
        {
            return Type();
        }
        virtual std::string description() const override
        {
            auto snap = current();
            return concatenate(type(),
                               " (",
                               snap->solutions.size(),
                               " registered, ",
                               snap->exactTable.size(),
                               " selectable)");
        }

        // -------------------------------------------------------------------
        // Selection. Only kernels with an exact mapping participate.
        // -------------------------------------------------------------------

        virtual std::shared_ptr<MySolution> findBestSolution(MyProblem const& problem,
                                                             Hardware const&  hardware,
                                                             double*          fitness
                                                             = nullptr) const override
        {
            auto solution = findExact(problem, hardware);
            if(solution && fitness)
                *fitness = 0.0; // an exact match is distance zero, as Equality reports
            return solution;
        }

        virtual SolutionVector<MySolution> findTopSolutions(MyProblem const& problem,
                                                            Hardware const&  hardware,
                                                            int numSolutions) const override
        {
            SolutionVector<MySolution> rv;
            if(numSolutions <= 0)
            {
                m_lastFindTopRetAll = true;
                return rv;
            }

            if(auto solution = findExact(problem, hardware))
                rv.push_back(solution);

            // At most one kernel can be mapped to a given shape, so this tier
            // has nothing further to offer and the rest of the ladder supplies
            // the remainder.
            m_lastFindTopRetAll = (static_cast<int>(rv.size()) < numSolutions);
            return rv;
        }

        virtual bool lastFindTopAlreadyRetAll() const override
        {
            return m_lastFindTopRetAll;
        }

        // -------------------------------------------------------------------
        // Enumeration. Everything registered and applicable is reported,
        // mapped or not, because enumeration is not selection.
        // -------------------------------------------------------------------

        virtual SolutionSet<MySolution>
            findAllSolutions(MyProblem const&          problem,
                             Hardware const&           hardware,
                             SolutionLibrarySearchType searchType
                             = SolutionLibrarySearchType::DEFAULT) const override
        {
            SolutionSet<MySolution> rv;
            if(!m_hasAny.load(std::memory_order_acquire))
                return rv;

            auto snap = current();
            for(auto const& entry : snap->solutions)
            {
                if(canServe(*entry.second, problem, hardware))
                    rv.insert(entry.second);
            }
            return rv;
        }

        // Grouped GEMM registration is out of scope; the tier reports nothing
        // rather than guessing, and selection falls through to the system rows.
        virtual SolutionSet<MySolution>
            findAllSolutionsGroupedGemm(std::vector<MyProblem> const&,
                                        Hardware const&,
                                        SolutionLibrarySearchType
                                        = SolutionLibrarySearchType::DEFAULT) const override
        {
            return SolutionSet<MySolution>();
        }

        // -------------------------------------------------------------------
        // Dispatch by explicit index, which is how a just-registered kernel is
        // executed before (or without) being made selectable.
        // -------------------------------------------------------------------

        virtual std::shared_ptr<MySolution> getSolutionByIndex(MyProblem const& problem,
                                                               Hardware const&  hardware,
                                                               const int index) const override
        {
            // A system index is none of this tier's business; saying so lets
            // ExactLogicLibrary keep walking rather than treating the miss as
            // an error.
            if(!isUserKernelIndex(index))
                return std::shared_ptr<MySolution>();

            return lookupIndex(index);
        }

        virtual std::shared_ptr<MySolution> getSolutionByIndex(const int index) const override
        {
            if(!isUserKernelIndex(index))
                return std::shared_ptr<MySolution>();
            return lookupIndex(index);
        }

        virtual std::shared_ptr<MySolution> getSolutionByIndex(Hardware const&,
                                                               const int index) const override
        {
            if(!isUserKernelIndex(index))
                return std::shared_ptr<MySolution>();
            return lookupIndex(index);
        }

        // -------------------------------------------------------------------
        // Publication. Writers are serialized by m_writerMutex; readers never
        // block on it.
        // -------------------------------------------------------------------

        /**
         * Adds solutions to the tier and returns the index assigned to each, in
         * order. Indices from the payload are discarded and re-minted into the
         * user range, so they cannot collide with the shipped library, and
         * `solution->index` is rewritten to the assigned value so that an algo
         * handle built from it round-trips.
         *
         * The assigned index is only meaningful in this process. Nothing may
         * persist it and replay it in a later run.
         */
        std::vector<int> addSolutions(std::vector<std::shared_ptr<MySolution>> const& incoming)
        {
            std::vector<int> assigned;
            assigned.reserve(incoming.size());

            std::unique_lock<std::shared_mutex> lock(m_snapshotMutex);
            auto                                next = std::make_shared<Snapshot>(*m_snapshot);

            for(auto const& solution : incoming)
            {
                if(!solution)
                    continue;
                const int index        = m_nextIndex++;
                solution->index        = index;
                next->solutions[index] = solution;
                assigned.push_back(index);
            }

            publishLocked(std::move(next));
            return assigned;
        }

        /**
         * Makes an already registered kernel selectable for one exact shape.
         * Returns false if the index is unknown, which keeps a typo from
         * silently doing nothing observable.
         *
         * This is the only call that can change what selection returns, which
         * is what lets a tuning loop register and benchmark a candidate through
         * the real dispatch path and commit nothing if it loses.
         */
        bool setExactMatch(UserExactKey const& key, int index)
        {
            std::unique_lock<std::shared_mutex> lock(m_snapshotMutex);

            auto it = m_snapshot->solutions.find(index);
            if(it == m_snapshot->solutions.end())
                return false;

            auto next             = std::make_shared<Snapshot>(*m_snapshot);
            next->exactTable[key] = it->second;

            publishLocked(std::move(next));
            return true;
        }

        size_t registeredCount() const
        {
            return current()->solutions.size();
        }

        size_t selectableCount() const
        {
            return current()->exactTable.size();
        }

    private:
        std::shared_ptr<const Snapshot> current() const
        {
            std::shared_lock<std::shared_mutex> lock(m_snapshotMutex);
            return m_snapshot;
        }

        /**
         * Stores the snapshot, then advances the generation. The order is not
         * interchangeable: the counter is what tells the selection memo its
         * entries are stale, so it must not advance while the tier it describes
         * is still the old one.
         *
         * Caller must hold m_snapshotMutex exclusively, which is also what
         * serializes writers against each other across their
         * read-copy-modify sequence.
         */
        void publishLocked(std::shared_ptr<const Snapshot> next)
        {
            const bool hasAny   = !next->solutions.empty();
            const bool hasExact = !next->exactTable.empty();

            m_snapshot = std::move(next);
            m_hasAny.store(hasAny, std::memory_order_release);
            m_hasExact.store(hasExact, std::memory_order_release);

            m_generation->value.fetch_add(1, std::memory_order_release);
        }

        std::shared_ptr<MySolution> lookupIndex(int index) const
        {
            if(!m_hasAny.load(std::memory_order_acquire))
                return std::shared_ptr<MySolution>();

            auto snap = current();
            auto it   = snap->solutions.find(index);
            return it == snap->solutions.end() ? std::shared_ptr<MySolution>() : it->second;
        }

        std::shared_ptr<MySolution> findExact(MyProblem const& problem,
                                              Hardware const&  hardware) const
        {
            if(!m_hasExact.load(std::memory_order_acquire))
                return std::shared_ptr<MySolution>();

            auto snap = current();
            auto it   = snap->exactTable.find(userExactKeyFor(problem));
            if(it == snap->exactTable.end())
                return std::shared_ptr<MySolution>();

            if(!canServe(*it->second, problem, hardware))
                return std::shared_ptr<MySolution>();

            return it->second;
        }

        /**
         * Whether a registered kernel may serve this problem at all.
         *
         * The fine key having matched is not sufficient. A registered library
         * covers every problem type from a single row, whereas the shipped
         * tiers reach an Equality table only after the problem type has already
         * selected the subtree, so the type comparison that is structural for
         * them has to be explicit here. This is what keeps a bf16 TN kernel
         * from being handed an fp32 NN problem that happens to share [M, N,
         * batch, K].
         *
         * The predicate triple afterwards is the same validation
         * MasterSolutionLibrary applies to a hand-picked TENSILE_SOLUTION_INDEX.
         * The warning printed there -- that it "will only work for a particular
         * transpose and data type" -- is precisely why the type check above it
         * is needed and cannot be replaced by the predicates.
         */
        static bool canServe(MySolution const& solution,
                             MyProblem const&  problem,
                             Hardware const&   hardware)
        {
            auto const& pt = solution.problemType;

            if(pt.transA != problem.transA() || pt.transB != problem.transB())
                return false;

            if(pt.aType != problem.a().dataType() || pt.bType != problem.b().dataType()
               || pt.cType != problem.c().dataType() || pt.dType != problem.d().dataType())
                return false;

            if(pt.computeType != problem.computeType())
                return false;

            if(pt.highPrecisionAccumulate != problem.highPrecisionAccumulate())
                return false;

            if(solution.problemPredicate && !(*solution.problemPredicate)(problem))
                return false;

            Task task(hardware, problem, solution);
            if(solution.taskPredicate && !(*solution.taskPredicate)(task))
                return false;

            if(solution.hardwarePredicate && !(*solution.hardwarePredicate)(hardware))
                return false;

            return true;
        }

        std::shared_ptr<LibraryGeneration> m_generation;

        // Guards the snapshot pointer only, never a lookup through it: a reader
        // copies the shared_ptr and releases the lock, so the snapshot it is
        // walking stays alive even once a registration has displaced it.
        // std::atomic<std::shared_ptr<>> would express this more directly but is
        // not usable here -- libstdc++ selects the primary std::atomic template
        // in some translation units and rejects the non-trivially-copyable
        // shared_ptr, so the lock is the portable spelling.
        mutable std::shared_mutex       m_snapshotMutex;
        std::shared_ptr<const Snapshot> m_snapshot;

        // Read before the lock on every lookup, so an untouched tier costs one
        // relaxed load and a branch rather than any lock traffic. This is what
        // keeps the always-present row nearly free in the common case where
        // nothing has been registered.
        std::atomic<bool> m_hasAny{false};
        std::atomic<bool> m_hasExact{false};

        int m_nextIndex = UserKernelIndexBase;

        mutable std::atomic<bool> m_lastFindTopRetAll{false};
    };

    /**
     * The one user kernel tier for the process, shared by every row ladder in
     * every library.
     *
     * A single instance rather than one per library is what makes lazy loading
     * work. Shards are deserialized on demand through their own
     * LoadLibraryFile calls, long after the top-level library was read, so a
     * tier scoped to a library would leave a registered kernel invisible to
     * every ladder loaded afterwards -- which is most of them, and exactly the
     * ones that serve real GEMMs.
     *
     * Sharing one instance across ladders that serve different problem types is
     * safe because the tier re-checks the full problem type before returning
     * anything. A ladder is reached only after the problem type has already
     * selected it, so that check is structural for the shipped tiers and has to
     * be explicit here.
     */
    template <typename MyProblem, typename MySolution = typename MyProblem::Solution>
    inline std::shared_ptr<UserKernelLibrary<MyProblem, MySolution>> globalUserKernelLibrary()
    {
        static std::shared_ptr<UserKernelLibrary<MyProblem, MySolution>> instance
            = std::make_shared<UserKernelLibrary<MyProblem, MySolution>>(
                globalLibraryGeneration());
        return instance;
    }
} // namespace TensileLite
