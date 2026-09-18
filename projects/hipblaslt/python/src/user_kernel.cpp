// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <hipblaslt/hipblaslt.h>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include "algo.hpp"
#include "descriptors.hpp"
#include "init.hpp"
#include "status.hpp"

namespace nb = nanobind;
using namespace hipblaslt_py;

namespace {

// Registration happens exactly once per call. There is deliberately no
// size-query pass first: passing a null buffer does not make the call a dry
// run, it registers the payload and declines to report the indices, so probing
// for a size would register everything twice.
//
// The buffer is therefore sized up front. A Tensile shard holds on the order of
// a thousand solutions, and a whole shipped architecture is on the order of
// 100,000 across every shard, so this ceiling is far above anything real, and
// exceeding it raises rather than silently dropping indices the caller can
// never recover.
constexpr int kMaxRegisteredPerPayload = 1 << 20;

std::vector<int> user_kernel_register(Handle&            handle,
                                      const std::string& library_path,
                                      const std::string& code_object_path)
{
    std::vector<int> indices(static_cast<size_t>(kMaxRegisteredPerPayload));
    int              count = 0;
    HIPBLASLT_CHECK(hipblasLtUserKernelRegister(handle.raw(),
                                                library_path.c_str(),
                                                code_object_path.c_str(),
                                                indices.data(),
                                                kMaxRegisteredPerPayload,
                                                &count));

    if(count > kMaxRegisteredPerPayload)
        throw std::runtime_error(
            "payload holds " + std::to_string(count) + " solutions, more than the "
            + std::to_string(kMaxRegisteredPerPayload)
            + " this binding can report; the kernels are registered but their "
              "indices past the limit were not returned");

    indices.resize(static_cast<size_t>(count < 0 ? 0 : count));
    return indices;
}

} // namespace

void init_user_kernel(nb::module_& m)
{
    m.def("user_kernel_library_open",
          [](Handle& handle, const std::string& path) {
              HIPBLASLT_CHECK(hipblasLtUserKernelLibraryOpen(handle.raw(), path.c_str()));
          },
          nb::arg("handle"), nb::arg("path"),
          "Bind this process to a durable user kernel library rooted at path.\n"
          "Without this, registration lasts only as long as the process.");

    m.def("user_kernel_register", &user_kernel_register,
          nb::arg("handle"), nb::arg("library_path"), nb::arg("code_object_path"),
          "Register a (.dat shard, .co) payload and return the assigned indices.\n"
          "The kernels are executable immediately via get_algos_from_index, but\n"
          "nothing becomes selectable until user_kernel_set_exact_match is called.\n"
          "An index is valid only in this process and must not be persisted.");

    m.def("user_kernel_set_exact_match",
          [](Handle& handle, int kernel_index, size_t m_, size_t n, size_t batch, size_t k) {
              HIPBLASLT_CHECK(hipblasLtUserKernelSetExactMatch(
                  handle.raw(), kernel_index, m_, n, batch, k));
          },
          nb::arg("handle"), nb::arg("kernel_index"),
          nb::arg("m"), nb::arg("n"), nb::arg("batch"), nb::arg("k"),
          "Make a registered kernel selectable for exactly this shape.\n"
          "Any other shape is unaffected, and a near miss falls through to\n"
          "ordinary selection: matching is exact, with no tolerance.");

    m.def("user_kernel_refresh",
          [](Handle& handle) {
              int count = 0;
              HIPBLASLT_CHECK(hipblasLtUserKernelRefresh(handle.raw(), &count));
              return count;
          },
          nb::arg("handle"),
          "Reinstate everything recorded in the open library and return how many\n"
          "kernels came back. Indices are assigned afresh, so those from an\n"
          "earlier run do not reappear; the shape mappings are what survive.");

    m.def("user_kernel_counts",
          [](Handle& handle) {
              size_t registered = 0, selectable = 0;
              HIPBLASLT_CHECK(
                  hipblasLtUserKernelGetCounts(handle.raw(), &registered, &selectable));
              return std::make_pair(registered, selectable);
          },
          nb::arg("handle"),
          "Return (registered, selectable). The two differ because registering a\n"
          "kernel does not make it selectable.");

    m.def("is_user_kernel",
          [](Algo& algo) {
              int is_user = 0;
              HIPBLASLT_CHECK(hipblasLtMatmulAlgoIsUserKernel(&algo.algo, &is_user));
              return is_user != 0;
          },
          nb::arg("algo"),
          "Whether this algo refers to a registered kernel rather than a shipped\n"
          "one. Answered from the index range alone, with no library lookup.");
}
