# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
cmake_minimum_required(VERSION 3.25)
project(hipblaslt_python LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# scikit-build-core pre-installs nanobind and injects its CMake path.
find_package(Python 3.10 COMPONENTS Interpreter Development.Module REQUIRED)
find_package(nanobind REQUIRED CONFIG)

# Locate HIP and hipBLASLt. ROCM_PATH may be passed by tasks.py; fall back to /opt/rocm.
if(ROCM_PATH)
    list(APPEND CMAKE_PREFIX_PATH "${ROCM_PATH}")
endif()
list(APPEND CMAKE_PREFIX_PATH /opt/rocm/hip /opt/rocm)
find_package(hip REQUIRED)
find_package(hipblaslt REQUIRED)

# Enable HIP language so .hip sources are compiled with hipcc.
# Must come after find_package(hip) so the HIP CMake toolchain files are on the path.
enable_language(HIP)
set(CMAKE_HIP_STANDARD 17)
set(CMAKE_HIP_STANDARD_REQUIRED ON)

set_source_files_properties(
    "${CMAKE_CURRENT_SOURCE_DIR}/src/convert.hip"
    PROPERTIES LANGUAGE HIP)

nanobind_add_module(_core NB_SUPPRESS_WARNINGS
    "${CMAKE_CURRENT_SOURCE_DIR}/src/module.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/enums.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/device_array.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/descriptors.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/matmul.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/user_kernel.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/convert.hip"
)
target_include_directories(_core PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_link_libraries(_core PRIVATE hip::host roc::hipblaslt)

# find_package(hip) contributes ${ROCM_PATH}/include as a system include, which
# on a machine with hipBLASLt installed under ROCm shadows the hipblaslt_DIR we
# were pointed at. The extension then compiles against the installed version's
# headers while linking the one we built, so enum values and newly added
# entry points silently disagree with the library actually loaded at runtime.
# Re-add hipBLASLt's own include dirs as non-system -I so they are searched
# first, since -I precedes -isystem in the header search order.
# Keep the runtime resolution pointed at the same hipBLASLt this was built
# against. CMake strips the build rpath on install, so without this the
# extension in site-packages resolves libhipblaslt.so.1 through the default
# search path and silently loads whatever ROCm has installed -- which fails on
# any entry point newer than that copy, and would otherwise run against a
# different library version than it was compiled for. This is the runtime
# counterpart to forcing the include order below.
get_target_property(hipblaslt_lib_release roc::hipblaslt IMPORTED_LOCATION_RELEASE)
get_target_property(hipblaslt_lib_plain roc::hipblaslt IMPORTED_LOCATION)
set(hipblaslt_lib "${hipblaslt_lib_release}")
if(NOT hipblaslt_lib)
    set(hipblaslt_lib "${hipblaslt_lib_plain}")
endif()
if(hipblaslt_lib)
    get_filename_component(hipblaslt_libdir "${hipblaslt_lib}" DIRECTORY)
    set_target_properties(_core PROPERTIES
        INSTALL_RPATH "${hipblaslt_libdir}"
        BUILD_WITH_INSTALL_RPATH TRUE)
    message(STATUS "hipblaslt python: embedding rpath ${hipblaslt_libdir}")
endif()

get_target_property(hipblaslt_interface_includes
    roc::hipblaslt INTERFACE_INCLUDE_DIRECTORIES)
if(hipblaslt_interface_includes)
    target_include_directories(_core BEFORE PRIVATE ${hipblaslt_interface_includes})
    # Recorded so the coverage tests can parse the same headers the extension
    # was compiled against instead of guessing at ${ROCM_PATH}.
    list(GET hipblaslt_interface_includes 0 hipblaslt_primary_include)
    target_compile_definitions(_core PRIVATE
        HIPBLASLT_PY_INCLUDE_DIR="${hipblaslt_primary_include}")
endif()

install(TARGETS _core DESTINATION hipblaslt)
