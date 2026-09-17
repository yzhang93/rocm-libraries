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
#include <cstdint>
#include <memory>

namespace TensileLite
{
    /**
     * Monotonic counter incremented whenever a mutable tier below the selection
     * memo changes, which today means a user kernel registration.
     *
     * The memo in CachingLibrary sits above the row ladder and answers without
     * consulting it, so registering a kernel for a problem that is already
     * cached would otherwise leave the new kernel invisible to the very problem
     * it was built for. Cached values carry the counter value they were
     * computed under; a reader compares stamps and treats a mismatch as a miss.
     *
     * Stale entries are rejected rather than deleted, because CacheMap has no
     * notion of a miss beyond its null value and no way to erase. The
     * re-resolved answer overwrites in place via insert_or_assign.
     *
     * Scope is the enclosing MasterSolutionLibrary, which is process-global.
     */
    struct LibraryGeneration
    {
        std::atomic<uint64_t> value{0};
    };

    /**
     * The one counter for the process.
     *
     * It has to be process-global rather than per-MasterSolutionLibrary,
     * because a lazily loaded library is not one MasterSolutionLibrary but
     * many: PlaceholderLibrary deserializes each shard through its own
     * LoadLibraryFile call, and each shard brings its own CachingLibrary memo
     * nested below the top-level one. A registration must invalidate all of
     * them, and a per-library counter would leave every memo but one serving
     * pre-registration answers.
     *
     * This matches the scope the design gives the user kernel library: one per
     * process, because the library and its caches are reached through a
     * function-local static in get_library_and_adapter.
     */
    inline std::shared_ptr<LibraryGeneration> globalLibraryGeneration()
    {
        static std::shared_ptr<LibraryGeneration> instance
            = std::make_shared<LibraryGeneration>();
        return instance;
    }
} // namespace TensileLite
