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

#include "include/user_kernel_registry.hpp"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <sstream>
#include <vector>

namespace fs = std::filesystem;

namespace rocblaslt
{
    namespace user_kernel
    {
        namespace
        {
            std::mutex  g_mutex;
            std::string g_root;
            bool        g_open = false;

            constexpr const char* kJournalName = "registry.log";
            constexpr const char* kObjectsDir  = "objects";

            /**
             * Identity hash over file contents, FNV-1a widened to 128 bits by
             * running two independently seeded lanes.
             *
             * This names objects; it is not a security boundary. A store that
             * must resist deliberate tampering needs a cryptographic digest,
             * which is a dependency decision rather than a code change here.
             * Collision resistance against accident is what content addressing
             * needs, and 128 bits gives that comfortably.
             */
            bool hashFile(fs::path const& path, std::string* out, std::string* error)
            {
                std::ifstream in(path, std::ios::binary);
                if(!in)
                {
                    if(error)
                        *error = "cannot read " + path.string();
                    return false;
                }

                uint64_t a = 0xcbf29ce484222325ull, b = 0x9e3779b97f4a7c15ull;
                char     buffer[64 * 1024];
                while(in.read(buffer, sizeof(buffer)) || in.gcount() > 0)
                {
                    const std::streamsize got = in.gcount();
                    for(std::streamsize i = 0; i < got; ++i)
                    {
                        const auto byte = static_cast<unsigned char>(buffer[i]);
                        a = (a ^ byte) * 0x100000001b3ull;
                        b = (b ^ byte) * 0xff51afd7ed558ccdull;
                    }
                }

                char text[33];
                std::snprintf(text, sizeof(text), "%016llx%016llx",
                              (unsigned long long)a, (unsigned long long)b);
                *out = text;
                return true;
            }

            /** fsync a directory so a newly created entry in it is durable. */
            void syncDirectory(fs::path const& dir)
            {
                const int fd = ::open(dir.c_str(), O_RDONLY | O_DIRECTORY);
                if(fd < 0)
                    return;
                static_cast<void>(::fsync(fd));
                static_cast<void>(::close(fd));
            }

            /**
             * Copies to a temporary name in the destination directory, fsyncs
             * the contents, then renames into place. The rename is atomic, so a
             * reader never sees a partially written object under its final
             * name.
             */
            bool copyDurable(fs::path const& from, fs::path const& to, std::string* error)
            {
                if(fs::exists(to))
                    return true; // content addressed: same name means same bytes

                const fs::path temp = to.string() + ".partial";

                {
                    std::ifstream in(from, std::ios::binary);
                    std::ofstream out(temp, std::ios::binary | std::ios::trunc);
                    if(!in || !out)
                    {
                        if(error)
                            *error = "cannot copy " + from.string() + " to " + temp.string();
                        return false;
                    }
                    out << in.rdbuf();
                    out.flush();
                    if(!out)
                    {
                        if(error)
                            *error = "short write to " + temp.string();
                        return false;
                    }
                }

                const int fd = ::open(temp.c_str(), O_RDONLY);
                if(fd >= 0)
                {
                    static_cast<void>(::fsync(fd));
                    static_cast<void>(::close(fd));
                }

                std::error_code ec;
                fs::rename(temp, to, ec);
                if(ec)
                {
                    fs::remove(temp, ec);
                    if(error)
                        *error = "cannot publish object " + to.string();
                    return false;
                }

                syncDirectory(to.parent_path());
                return true;
            }

            /**
             * Clears the group and other write bits, leaving read and execute
             * alone.
             *
             * Applied only to directories this code creates. A umask of 0002 is
             * common, which would otherwise hand us a group-writable store that
             * the check in open() would immediately refuse -- a directory we
             * made ourselves failing our own trust test. A directory the caller
             * already had is never modified, only accepted or refused, since
             * silently tightening someone else's directory is not ours to do.
             */
            void tightenWritePermissions(fs::path const& dir)
            {
                struct stat st{};
                if(::stat(dir.c_str(), &st) != 0)
                    return;
                const mode_t safe = st.st_mode & ~static_cast<mode_t>(S_IWGRP | S_IWOTH);
                if(safe != st.st_mode)
                    static_cast<void>(::chmod(dir.c_str(), safe));
            }

            bool isWritableByOthers(fs::path const& dir, std::string* error)
            {
                struct stat st{};
                if(::stat(dir.c_str(), &st) != 0)
                {
                    if(error)
                        *error = "cannot stat " + dir.string();
                    return true;
                }
                if(st.st_mode & (S_IWGRP | S_IWOTH))
                {
                    if(error)
                        *error = dir.string()
                                 + " is group- or world-writable; refusing to load kernels from it";
                    return true;
                }
                return false;
            }

            bool appendJournalLocked(std::string const& line, std::string* error)
            {
                const fs::path path = fs::path(g_root) / kJournalName;

                std::ofstream out(path, std::ios::app);
                if(!out)
                {
                    if(error)
                        *error = "cannot append to " + path.string();
                    return false;
                }
                out << line << '\n';
                out.flush();
                if(!out)
                {
                    if(error)
                        *error = "short write to " + path.string();
                    return false;
                }
                out.close();

                const int fd = ::open(path.c_str(), O_WRONLY | O_APPEND);
                if(fd >= 0)
                {
                    static_cast<void>(::fsync(fd));
                    static_cast<void>(::close(fd));
                }
                return true;
            }
        } // namespace

        std::string objectPath(std::string const& hash, std::string const& extension)
        {
            std::lock_guard<std::mutex> lock(g_mutex);
            return (fs::path(g_root) / kObjectsDir / (hash + extension)).string();
        }

        bool open(std::string const& rootIn, std::string* error)
        {
            std::lock_guard<std::mutex> lock(g_mutex);

            if(rootIn.empty())
            {
                if(error)
                    *error = "library path is empty";
                return false;
            }

            std::error_code ec;
            const fs::path  root = fs::absolute(rootIn, ec);
            if(ec)
            {
                if(error)
                    *error = "cannot resolve " + rootIn;
                return false;
            }

            const bool rootExisted = fs::exists(root);

            fs::create_directories(root / kObjectsDir, ec);
            if(ec && !fs::is_directory(root / kObjectsDir))
            {
                if(error)
                    *error = "cannot create " + (root / kObjectsDir).string();
                return false;
            }

            if(!rootExisted)
            {
                tightenWritePermissions(root);
                tightenWritePermissions(root / kObjectsDir);
            }

            // Whoever can write here chooses which code object this process
            // loads, so the permissions are part of the trust boundary. Both
            // levels matter: the objects directory holds the code, and the root
            // holds the journal naming which objects to load.
            if(isWritableByOthers(root, error) || isWritableByOthers(root / kObjectsDir, error))
                return false;

            g_root = root.string();
            g_open = true;
            return true;
        }

        bool isOpen()
        {
            std::lock_guard<std::mutex> lock(g_mutex);
            return g_open;
        }

        std::string root()
        {
            std::lock_guard<std::mutex> lock(g_mutex);
            return g_root;
        }

        bool storePayload(std::string const& datPath,
                          std::string const& coPath,
                          std::string*       datHash,
                          std::string*       coHash,
                          std::string*       storedDat,
                          std::string*       storedCo,
                          std::string*       error)
        {
            // Callers name the shard by its bare ".dat" name even when only the
            // compressed form is on disk, because that is the invariant
            // LoadLibraryFile works to: it appends the ".zlib" probe suffix
            // itself. Resolve to the file that actually exists so it can be
            // hashed and copied, and keep the suffix so the stored copy stays
            // loadable the same way.
            fs::path          source     = datPath;
            const std::string zlibSuffix = ".zlib";
            bool              compressed = false;
            if(!fs::exists(source) && fs::exists(datPath + zlibSuffix))
            {
                source     = datPath + zlibSuffix;
                compressed = true;
            }

            std::string dh, ch;
            if(!hashFile(source, &dh, error) || !hashFile(coPath, &ch, error))
                return false;

            std::lock_guard<std::mutex> lock(g_mutex);
            if(!g_open)
            {
                if(error)
                    *error = "no user kernel library is open";
                return false;
            }

            const fs::path objects = fs::path(g_root) / kObjectsDir;
            // The stored name keeps the compression suffix, while the path
            // handed back is always the bare ".dat" the loader expects.
            const fs::path datStored = objects / (dh + ".dat" + (compressed ? zlibSuffix : ""));
            const fs::path datLogical = objects / (dh + ".dat");
            const fs::path co         = objects / (ch + ".co");

            // Objects become durable before the journal record naming them, so
            // an interrupted registration leaves an unreferenced object rather
            // than a dangling reference.
            if(!copyDurable(source, datStored, error) || !copyDurable(coPath, co, error))
                return false;

            std::ostringstream line;
            line << "R " << dh << ' ' << ch;
            if(!appendJournalLocked(line.str(), error))
                return false;

            if(datHash)
                *datHash = dh;
            if(coHash)
                *coHash = ch;
            if(storedDat)
                *storedDat = datLogical.string();
            if(storedCo)
                *storedCo = co.string();
            return true;
        }

        bool recordExactMatch(ExactRecord const& record, std::string* error)
        {
            std::lock_guard<std::mutex> lock(g_mutex);
            if(!g_open)
            {
                if(error)
                    *error = "no user kernel library is open";
                return false;
            }

            std::ostringstream line;
            line << "E " << record.datHash << ' ' << record.shardIndex << ' ' << record.m << ' '
                 << record.n << ' ' << record.batch << ' ' << record.k;
            return appendJournalLocked(line.str(), error);
        }

        bool readJournal(Journal* journal, std::string* error)
        {
            if(!journal)
                return false;
            journal->payloads.clear();
            journal->exacts.clear();

            std::lock_guard<std::mutex> lock(g_mutex);
            if(!g_open)
            {
                if(error)
                    *error = "no user kernel library is open";
                return false;
            }

            const fs::path path = fs::path(g_root) / kJournalName;
            std::ifstream  in(path);
            if(!in)
                return true; // nothing registered yet is not a failure

            std::string line;
            while(std::getline(in, line))
            {
                std::istringstream fields(line);
                std::string        kind;
                if(!(fields >> kind))
                    continue;

                if(kind == "R")
                {
                    PayloadRecord record;
                    if(fields >> record.datHash >> record.coHash)
                        journal->payloads.push_back(record);
                }
                else if(kind == "E")
                {
                    ExactRecord record;
                    if(fields >> record.datHash >> record.shardIndex >> record.m >> record.n
                       >> record.batch >> record.k)
                        journal->exacts.push_back(record);
                }
                // An unrecognised record is skipped rather than fatal, so a
                // newer library writing a record this build does not know about
                // degrades to ignoring it.
            }
            return true;
        }
    } // namespace user_kernel
} // namespace rocblaslt
