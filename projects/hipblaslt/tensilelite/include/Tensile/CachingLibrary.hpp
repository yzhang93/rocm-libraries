/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2022-2023 Advanced Micro Devices, Inc. All rights reserved.
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
#include <shared_mutex>
#include <unordered_map>

#include <Tensile/ContractionProblem.hpp>
#include <Tensile/LibraryGeneration.hpp>
#include <Tensile/SolutionLibrary.hpp>

#include <Tensile/AMDGPU_Detail.hpp>
#include <Tensile/ContractionProblem_Detail.hpp>
#include <Tensile/TensorDescriptor_Detail.hpp>

#include <tensilelitehost/export.h>

namespace TensileLite
{
    template <typename Value, typename Key, typename... Keys>
    struct MultiLevelMap
    {
        using type = typename MultiLevelMap<std::unordered_map<Key, Value>, Keys...>::type;
    };

    template <typename Value, typename Key>
    struct MultiLevelMap<Value, Key>
    {
        using type = std::unordered_map<Key, Value>;
    };

    /**
     * Thread-safe multi-valued cache.
     *
     * Note that due to a quirk with templates, the order of the keys in find() and add() is *opposite* of that in the type.
     *
     * e.g.
     *
     *     CacheMap<int, float, std::string> myCache
     *     myCache.find("foo", 1.4); // great
     *     myCache.find(1.4, "foo"); // error!
     */
    template <typename Value, typename... Keys>
    class CacheMap
    {
        using Map = typename MultiLevelMap<Value, Keys...>::type;

    public:
        CacheMap(Value const& nullValue)
            : m_nullValue(nullValue)
            , m_lookupEfficiency(Debug::Instance().printLookupEfficiency())
            , m_lookups(0)
            , m_hits(0)

        {
        }

        ~CacheMap()
        {
            if(m_lookupEfficiency)
                std::cout << "CacheMap: " << m_hits << "/" << m_lookups << " cache hits"
                          << std::endl;
        }

        template <typename... Ks>
        Value find(Ks const&... keys)
        {
            std::shared_lock<std::shared_timed_mutex> lock(m_mutex);

            auto rv = find_impl(m_map, keys...);

            if(m_lookupEfficiency)
            {
                m_lookups++;
                if(rv != m_nullValue)
                    m_hits++;
            }

            return rv;
        }

        template <typename... Ks>
        void add(Value const& value, Ks const&... ks)
        {
            std::lock_guard<std::shared_timed_mutex> lock(m_mutex);

            add_impl(m_map, value, ks...);
        }

    private:
        template <typename SubMap, typename K>
        Value find_impl(SubMap const& map, K const& key)
        {
            auto iter = map.find(key);

            if(iter == map.end())
                return m_nullValue;

            return iter->second;
        }

        template <typename SubMap, typename K, typename... Ks>
        Value find_impl(SubMap const& map, K const& key, Ks const&... ks)
        {
            auto iter = map.find(key);

            if(iter == map.end())
                return m_nullValue;

            return find_impl(iter->second, ks...);
        }

        template <typename SubMap, typename K>
        void add_impl(SubMap& map, Value const& value, K const& key)
        {
            map.insert_or_assign(key, value);
        }

        template <typename SubMap, typename K, typename... Ks>
        void add_impl(SubMap& map, Value const& value, K const& key, Ks const&... ks)
        {
            add_impl(map[key], value, ks...);
        }

        Map                     m_map;
        std::shared_timed_mutex m_mutex;
        Value                   m_nullValue;

        bool                 m_lookupEfficiency;
        std::atomic<int64_t> m_lookups;
        std::atomic<int64_t> m_hits;
    };

    template <typename MyProblem, typename MySolution = typename MyProblem::Solution>
    class CachingLibrary : public SolutionLibrary<MyProblem, MySolution>
    {
    public:
        using Library = SolutionLibrary<MyProblem, MySolution>;
        // Every memoized value carries the library generation it was resolved
        // under, so an entry that predates a user kernel registration can be
        // recognised as stale. See LibraryGeneration.hpp.
        using Cache
            = CacheMap<std::tuple<std::shared_ptr<MySolution>, double, uint64_t>, AMDGPU, MyProblem>;
        // The solution vector and the "cache already holds every solution" flag
        // were two maps keyed identically. They cannot be stamped
        // independently: the hit condition below is satisfied by the flag
        // alone, so a stale flag would return a stale vector even after the
        // generation advances. Worse, the flag's map was CacheMap<bool> with a
        // null value of false, making "absent" indistinguishable from "cached
        // false". Merged here so one lookup yields one staleness decision.
        using Caches
            = CacheMap<std::tuple<SolutionVector<MySolution>, bool, uint64_t>, AMDGPU, MyProblem>;
        using CachesGroupedGemm = CacheMap<std::tuple<SolutionVector<MySolution>, uint64_t>,
                                           AMDGPU,
                                           std::vector<MyProblem>>;

        // The generation is optional so that callers which construct a cache
        // outside a MasterSolutionLibrary keep working; a private counter never
        // advances, which reproduces the previous always-fresh behaviour.
        CachingLibrary(std::shared_ptr<Library>           subLibrary,
                       std::shared_ptr<LibraryGeneration> generation = nullptr)
            : m_subLibrary(subLibrary)
            , m_generation(generation ? generation : std::make_shared<LibraryGeneration>())
            , m_cache(std::make_tuple(nullptr, std::numeric_limits<double>::max(), uint64_t{0}))
            , m_caches(std::make_tuple(SolutionVector<MySolution>{}, false, uint64_t{0}))
            , m_cachesGroupedGemm(std::make_tuple(SolutionVector<MySolution>{}, uint64_t{0}))
        {
        }

        virtual std::shared_ptr<MySolution> getSolutionByIndex(MyProblem const& problem,
                                                               Hardware const&  hardware,
                                                               const int index) const override
        {
            return m_subLibrary->getSolutionByIndex(problem, hardware, index);
        }

        virtual std::shared_ptr<MySolution> findBestSolution(MyProblem const& problem,
                                                             Hardware const&  hardware,
                                                             double*          fitness
                                                             = nullptr) const override
        {
            try
            {
                double cachedFitness = std::numeric_limits<double>::max();
                fitness              = (fitness) ? fitness : &cachedFitness;

                auto const& amdgpu = dynamic_cast<AMDGPU const&>(hardware);

                // Read the generation before resolving, so a registration that
                // lands during resolution stamps the result as the older
                // generation and is re-resolved on the next call rather than
                // being cached as current.
                const uint64_t gen = m_generation->value.load(std::memory_order_acquire);

                std::shared_ptr<MySolution> solution;
                uint64_t                    entryGen = 0;
                std::tie(solution, *fitness, entryGen) = m_cache.find(problem, amdgpu);

                if(solution && entryGen == gen)
                    return solution;

                // Rejecting a stale entry must also discard its fitness, which
                // the tie above has already written through, so the sub-library
                // sees the same starting value it would on a plain miss.
                if(solution)
                    *fitness = std::numeric_limits<double>::max();

                solution = m_subLibrary->findBestSolution(problem, hardware, fitness);
                if(solution)
                    m_cache.add(std::make_tuple(solution, *fitness, gen), problem, amdgpu);

                return solution;
            }
            catch(std::bad_cast const& exc)
            {
                return m_subLibrary->findBestSolution(problem, hardware, fitness);
            }
        }

        virtual SolutionSet<MySolution>
            findAllSolutions(MyProblem const&          problem,
                             Hardware const&           hardware,
                             SolutionLibrarySearchType searchType
                             = SolutionLibrarySearchType::DEFAULT) const override
        {
            return m_subLibrary->findAllSolutions(problem, hardware, searchType);
        }

        virtual SolutionSet<MySolution>
            findAllSolutionsGroupedGemm(std::vector<MyProblem> const& problems,
                                        Hardware const&               hardware,
                                        SolutionLibrarySearchType     searchType
                                        = SolutionLibrarySearchType::DEFAULT) const override
        {
            return m_subLibrary->findAllSolutionsGroupedGemm(problems, hardware, searchType);
        }

        std::shared_ptr<MySolution> findSolutionInCache(MyProblem const& problem,
                                                        Hardware const&  hardware) const
        {
            auto const& amdgpu = dynamic_cast<AMDGPU const&>(hardware);

            auto entry = m_cache.find(problem, amdgpu);
            if(std::get<uint64_t>(entry) != m_generation->value.load(std::memory_order_acquire))
                return nullptr;

            return std::get<std::shared_ptr<MySolution>>(entry);
        }

        virtual std::string type() const override
        {
            return "Caching Library";
        }
        virtual std::string description() const override
        {
            return "Caching Library";
        }

        std::shared_ptr<Library> library() const
        {
            return m_subLibrary;
        }

        virtual SolutionVector<MySolution> findTopSolutions(MyProblem const& problem,
                                                            Hardware const&  hardware,
                                                            int numSolutions) const override
        {
            try
            {
                auto const&    amdgpu = dynamic_cast<AMDGPU const&>(hardware);
                const uint64_t gen    = m_generation->value.load(std::memory_order_acquire);

                SolutionVector<MySolution> solutions;
                bool                       cacheAlreadyContainAll = false;
                uint64_t                   entryGen               = 0;
                std::tie(solutions, cacheAlreadyContainAll, entryGen)
                    = m_caches.find(problem, amdgpu);

                if(entryGen == gen
                   && (solutions.size() >= numSolutions || cacheAlreadyContainAll))
                {
                    // getBestSolutions consumes lastFindTopAlreadyRetAll() to
                    // decide whether to fall back to getAllSolutions, so the
                    // hit path must publish the flag before returning.
                    lastFindTopRetAll = cacheAlreadyContainAll;
                    return solutions;
                }

                solutions = m_subLibrary->findTopSolutions(problem, hardware, numSolutions);
                if(solutions.size() != 0)
                {
                    bool alreadyRetAll = m_subLibrary->lastFindTopAlreadyRetAll();
                    m_caches.add(std::make_tuple(solutions, alreadyRetAll, gen), problem, amdgpu);
                }

                // can't reach the requested number, means findTop already done its best
                lastFindTopRetAll = (solutions.size() < numSolutions);
                return solutions;
            }
            catch(std::bad_cast const& exc)
            {
                return m_subLibrary->findTopSolutions(problem, hardware, numSolutions);
            }
            // TODO- redundant ??
            return m_subLibrary->findTopSolutions(problem, hardware, numSolutions);
        }

        virtual bool lastFindTopAlreadyRetAll() const override
        {
            return lastFindTopRetAll;
        }

        virtual SolutionVector<MySolution>
            findTopSolutionsGroupedGemm(std::vector<MyProblem> const& problems,
                                        Hardware const&               hardware,
                                        int                           numSolutions) const override
        {
            try
            {
                auto const&    amdgpu = dynamic_cast<AMDGPU const&>(hardware);
                const uint64_t gen    = m_generation->value.load(std::memory_order_acquire);

                SolutionVector<MySolution> solutions;
                uint64_t                   entryGen = 0;
                std::tie(solutions, entryGen)       = m_cachesGroupedGemm.find(problems, amdgpu);

                if(solutions.size() != 0 && entryGen == gen)
                    return solutions;

                solutions
                    = m_subLibrary->findTopSolutionsGroupedGemm(problems, hardware, numSolutions);
                if(solutions.size() != 0)
                    m_cachesGroupedGemm.add(std::make_tuple(solutions, gen), problems, amdgpu);

                return solutions;
            }
            catch(std::bad_cast const& exc)
            {
                return m_subLibrary->findTopSolutionsGroupedGemm(problems, hardware, numSolutions);
            }
            return m_subLibrary->findTopSolutionsGroupedGemm(problems, hardware, numSolutions);
        }

    private:
        std::shared_ptr<Library>           m_subLibrary;
        std::shared_ptr<LibraryGeneration> m_generation;
        mutable Cache                      m_cache;
        mutable Caches                     m_caches;
        mutable CachesGroupedGemm          m_cachesGroupedGemm;
        mutable std::atomic<bool>          lastFindTopRetAll = false;
    };

#if 0
    struct ContractionCachingLibrary: public CachingLibrary<ContractionProblemGemm>
    {
        using Super = CachingLibrary<ContractionProblemGemm>;
        using Library = typename Super::Library;
        using Key = typename Super::Key;

        ContractionCachingLibrary(std::shared_ptr<Library> subLibrary)
            : CachingLibrary<ContractionProblemGemm>(subLibrary)
        {}

    };
#endif

} // namespace TensileLite

