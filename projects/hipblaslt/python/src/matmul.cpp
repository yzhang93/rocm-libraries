// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt.h>
#include <hipblaslt/hipblaslt-ext.hpp>
#include <vector>
#include <optional>
#include <string>
#include <cstdint>
#include "status.hpp"
#include "descriptors.hpp"
#include "device_array.hpp"
#include "algo.hpp"
#include "init.hpp"

namespace nb = nanobind;
using namespace hipblaslt_py;

namespace {

// Drops entries whose state is not SUCCESS: the heuristic and enumeration APIs
// both report per-entry failures in-band rather than via the return status.
std::vector<HeuristicResult>
    convert(const std::vector<hipblasLtMatmulHeuristicResult_t>& raw, int count)
{
    std::vector<HeuristicResult> out;
    for(int i = 0; i < count; ++i)
    {
        if(raw[i].state != HIPBLAS_STATUS_SUCCESS)
            continue;
        HeuristicResult hr;
        hr.algo.algo      = raw[i].algo;
        hr.algo.index     = i;
        hr.workspace_size = raw[i].workspaceSize;
        hr.waves_count    = raw[i].wavesCount;
        out.push_back(hr);
    }
    return out;
}

std::vector<HeuristicResult> heuristic(
    Handle& handle, MatmulDesc& desc,
    MatrixLayout& a, MatrixLayout& b, MatrixLayout& c, MatrixLayout& d,
    Preference& pref, int max_results)
{
    std::vector<hipblasLtMatmulHeuristicResult_t> raw(max_results);
    int returned = 0;
    HIPBLASLT_CHECK(hipblasLtMatmulAlgoGetHeuristic(
        handle.raw(), desc.raw(), a.raw(), b.raw(), c.raw(), d.raw(),
        pref.raw(), max_results, raw.data(), &returned));

    return convert(raw, returned);
}

// Resolve specific Tensile solution indices into usable algos. Indices that are
// not present in the loaded solution map are skipped, so the result may be
// shorter than the request; read solution_index to identify what came back.
//
// INVALID_VALUE is the API's way of saying every requested index was out of
// bounds. That is reported as an empty list rather than an exception, so the
// all-unknown case behaves like the partially-unknown one and callers can sweep
// candidate indices without guarding each call.
std::vector<HeuristicResult> get_algos_from_index(Handle& handle, std::vector<int> indices)
{
    std::vector<hipblasLtMatmulHeuristicResult_t> raw;
    hipblasStatus_t status = hipblaslt_ext::getAlgosFromIndex(handle.raw(), indices, raw);
    if(status == HIPBLAS_STATUS_INVALID_VALUE)
        return {};
    HIPBLASLT_CHECK(status);
    return convert(raw, static_cast<int>(raw.size()));
}

// Enumerate every solution in the library for a problem type. Uses a dummy
// problem internally, so the results are not filtered by any concrete size and
// must still be checked with is_algo_supported before use.
std::vector<HeuristicResult> get_all_algos(
    Handle& handle, hipblaslt_ext::GemmType gemm_type,
    hipblasOperation_t op_a, hipblasOperation_t op_b,
    hipDataType type_a, hipDataType type_b, hipDataType type_c, hipDataType type_d,
    hipblasComputeType_t type_compute)
{
    std::vector<hipblasLtMatmulHeuristicResult_t> raw;
    HIPBLASLT_CHECK(hipblaslt_ext::getAllAlgos(
        handle.raw(), gemm_type, op_a, op_b,
        type_a, type_b, type_c, type_d, type_compute, raw));
    return convert(raw, static_cast<int>(raw.size()));
}

// Returns the required workspace in bytes, or None when this algo cannot serve
// the problem. "Unsupported" is an ordinary answer here rather than an error,
// since callers sweep candidate algos and expect some to be rejected.
std::optional<size_t> is_algo_supported(
    Handle& handle, MatmulDesc& desc,
    double alpha, MatrixLayout& a, MatrixLayout& b,
    double beta, MatrixLayout& c, MatrixLayout& d,
    Algo& algo)
{
    float  alpha_f = static_cast<float>(alpha);
    float  beta_f  = static_cast<float>(beta);
    size_t workspace_size = 0;
    hipblasStatus_t status = hipblaslt_ext::matmulIsAlgoSupported(
        handle.raw(), desc.raw(), &alpha_f, a.raw(), b.raw(), &beta_f,
        c.raw(), d.raw(), algo.algo, workspace_size);

    if(status == HIPBLAS_STATUS_SUCCESS)
        return workspace_size;
    if(status == HIPBLAS_STATUS_NOT_SUPPORTED || status == HIPBLAS_STATUS_INVALID_VALUE)
        return std::nullopt;
    HIPBLASLT_CHECK(status);
    return std::nullopt;
}

// alpha/beta are passed as double from Python and cast to float here because
// the common compute type is COMPUTE_32F (fp32 accumulate). If compute type
// is HIPBLAS_COMPUTE_32I the scalars must be int32_t — that path is a known
// limitation and can be added as a follow-up by branching on compute type.
void matmul(Handle& handle, MatmulDesc& desc,
            double alpha, DeviceArray& A, MatrixLayout& la,
            DeviceArray& B, MatrixLayout& lb,
            double beta, DeviceArray& C, MatrixLayout& lc,
            DeviceArray& D, MatrixLayout& ld,
            Algo& algo, DeviceArray& workspace,
            std::uintptr_t stream_ptr)
{
    float alpha_f = static_cast<float>(alpha);
    float beta_f  = static_cast<float>(beta);
    hipStream_t stream = reinterpret_cast<hipStream_t>(stream_ptr);
    HIPBLASLT_CHECK(hipblasLtMatmul(
        handle.raw(), desc.raw(), &alpha_f,
        A.raw(), la.raw(), B.raw(), lb.raw(), &beta_f,
        C.raw(), lc.raw(), D.raw(), ld.raw(),
        &algo.algo, workspace.raw(), workspace.nbytes(), stream));
    // Synchronize so results are visible to the host when this function returns.
    HIP_CHECK(hipStreamSynchronize(stream));
}

} // namespace

void init_matmul(nb::module_& m)
{
    nb::enum_<hipblaslt_ext::GemmType>(m, "GemmType", nb::is_arithmetic())
        .value("GEMM", hipblaslt_ext::GemmType::HIPBLASLT_GEMM)
        .value("GROUPED_GEMM", hipblaslt_ext::GemmType::HIPBLASLT_GROUPED_GEMM);

    nb::class_<Algo>(m, "Algo")
        .def_ro("index", &Algo::index)
        .def_prop_ro("solution_index",
                     [](Algo& a) { return hipblaslt_ext::getIndexFromAlgo(a.algo); },
                     "Tensile solution index encoded in this algo, or -1 if unset.");

    nb::class_<HeuristicResult>(m, "HeuristicResult")
        .def_ro("algo", &HeuristicResult::algo)
        .def_ro("workspace_size", &HeuristicResult::workspace_size)
        .def_ro("waves_count", &HeuristicResult::waves_count);

    m.def("heuristic", &heuristic,
          nb::arg("handle"), nb::arg("desc"),
          nb::arg("a_layout"), nb::arg("b_layout"),
          nb::arg("c_layout"), nb::arg("d_layout"),
          nb::arg("preference"), nb::arg("max_results") = 32);

    m.def("get_algos_from_index", &get_algos_from_index,
          nb::arg("handle"), nb::arg("indices"),
          "Resolve Tensile solution indices into algos, skipping unknown indices.");

    m.def("get_all_algos", &get_all_algos,
          nb::arg("handle"), nb::arg("gemm_type"),
          nb::arg("op_a"), nb::arg("op_b"),
          nb::arg("type_a"), nb::arg("type_b"),
          nb::arg("type_c"), nb::arg("type_d"),
          nb::arg("type_compute"),
          "Enumerate every solution in the library for a problem type.");

    m.def("is_algo_supported", &is_algo_supported,
          nb::arg("handle"), nb::arg("desc"), nb::arg("alpha"),
          nb::arg("a_layout"), nb::arg("b_layout"), nb::arg("beta"),
          nb::arg("c_layout"), nb::arg("d_layout"), nb::arg("algo"),
          "Required workspace bytes for this algo, or None if unsupported.");

    m.def("solution_name",
          [](Handle& handle, Algo& a) {
              return hipblaslt_ext::getSolutionNameFromAlgo(handle.raw(), a.algo);
          },
          nb::arg("handle"), nb::arg("algo"));

    m.def("kernel_name",
          [](Handle& handle, Algo& a) {
              return hipblaslt_ext::getKernelNameFromAlgo(handle.raw(), a.algo);
          },
          nb::arg("handle"), nb::arg("algo"));

    m.def("matmul", &matmul,
          nb::arg("handle"), nb::arg("desc"), nb::arg("alpha"),
          nb::arg("A"), nb::arg("a_layout"), nb::arg("B"), nb::arg("b_layout"),
          nb::arg("beta"), nb::arg("C"), nb::arg("c_layout"),
          nb::arg("D"), nb::arg("d_layout"),
          nb::arg("algo"), nb::arg("workspace"), nb::arg("stream_ptr") = 0);
}
