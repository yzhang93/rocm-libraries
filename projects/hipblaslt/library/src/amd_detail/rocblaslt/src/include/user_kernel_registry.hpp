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

#include <cstdint>
#include <string>
#include <vector>

/*******************************************************************************
 * Durable side of the user kernel library.
 *
 * On-disk shape, under a root the application names explicitly:
 *
 *   <root>/objects/<hash>.dat     payload shards, content addressed
 *   <root>/objects/<hash>.co      payload code objects, content addressed
 *   <root>/registry.log           append-only journal of what to replay
 *
 * The journal is a text file, one record per line, so a registration can be
 * inspected and repaired with ordinary tools:
 *
 *   R <datHash> <coHash>
 *   E <datHash> <shardIndex> <m> <n> <batch> <k>
 *
 * An E record names its kernel by the index the *payload* gave it, not by the
 * index this process assigned. Assigned indices are minted per process and
 * would be meaningless on replay; the shard index is a property of the file and
 * survives.
 *
 * Content addressing is what lets the store be append-only. Registering the
 * same payload twice writes the same object path, so a repeat is idempotent and
 * nothing ever has to be deleted or rewritten in place.
 *
 * Scope note: this is the single-writer journal the plan recommends for the
 * demo, not the full cross-process design. There is no lock file, so two
 * processes registering into one root concurrently can interleave journal
 * records. Reads are safe; concurrent writes are not, and that is the gap to
 * close if this outlives the demo.
 ******************************************************************************/

namespace rocblaslt
{
    namespace user_kernel
    {
        struct ExactRecord
        {
            std::string datHash;
            int         shardIndex = -1;
            uint64_t    m = 0, n = 0, batch = 0, k = 0;
        };

        struct PayloadRecord
        {
            std::string datHash;
            std::string coHash;
        };

        struct Journal
        {
            std::vector<PayloadRecord> payloads;
            std::vector<ExactRecord>   exacts;
        };

        /**
         * Binds the process to a library root, creating it if absent.
         *
         * Rejects a root that is group- or world-writable: anything able to
         * write there can choose which code object this process loads, which
         * makes the directory's permissions part of the trust boundary rather
         * than an operational detail. Deliberately takes the path as an
         * argument and consults no environment variable, so the choice of
         * library cannot be redirected from outside the application.
         */
        bool open(std::string const& root, std::string* error);

        /** Whether open() has succeeded, i.e. whether persistence is active. */
        bool isOpen();

        std::string root();

        /**
         * Copies a payload into the object store and appends its journal
         * record. Returns the content hashes through datHash and coHash, and
         * the paths of the stored copies through storedDat and storedCo.
         *
         * Objects are written and fsynced, and the directory entry fsynced,
         * *before* the journal record that references them. A crash in between
         * therefore leaves an unreferenced object, which is harmless, rather
         * than a journal pointing at a file that is not there.
         */
        bool storePayload(std::string const& datPath,
                          std::string const& coPath,
                          std::string*       datHash,
                          std::string*       coHash,
                          std::string*       storedDat,
                          std::string*       storedCo,
                          std::string*       error);

        /** Appends one exact-match record. */
        bool recordExactMatch(ExactRecord const& record, std::string* error);

        /** Reads the journal back. Missing journal reads as empty, not an error. */
        bool readJournal(Journal* journal, std::string* error);

        /** Paths of a stored object pair, whether or not they exist. */
        std::string objectPath(std::string const& hash, std::string const& extension);
    } // namespace user_kernel
} // namespace rocblaslt
