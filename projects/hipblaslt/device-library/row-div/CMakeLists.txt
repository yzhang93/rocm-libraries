# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT
#
# Builds and ships the fused-RMSNorm "Kernel 2" reduce-and-apply code object as
# Fabian's row-major `row_div` kernel. This is the host-launched companion to the
# TensileLite PartialRMS "K1" GEMM under the row-major (free0 = N_hidden)
# convention: K1 writes per-token partial sums-of-squares indexed by WorkGroup0,
# then row_div reduces them across tiles and divides D in place by the RMS scale.
#
# row_div.s is a committed, hand-written wave64 (CDNA) kernel with the gfx950
# target baked into its .amdgcn_target directive, so we assemble it verbatim
# (no code generation) and install one .co per matching arch.

set(row_div_src "${CMAKE_CURRENT_SOURCE_DIR}/../../tensilelite/epilogues/kernels/row_div.s")
string(REGEX MATCHALL "gfx[a-z0-9]+" archs "${GPU_TARGETS}")
list(REMOVE_DUPLICATES archs)

set(previous_target "")

foreach(arch IN LISTS archs)
    # row_div is a wave64 (CDNA) kernel authored for gfx950; skip other arches.
    if(NOT arch STREQUAL "gfx950")
        message(STATUS "hipblaslt device library row-div: skipping non-gfx950 arch ${arch}")
        continue()
    endif()

    message(STATUS "hipblaslt device library row-div: configuring target for ${arch}")
    tensilelite_offload_target(offload_arch "${arch}")

    add_library(row-div-obj-${arch} OBJECT)
    target_compile_options(row-div-obj-${arch}
        PRIVATE
            -Wno-unused-command-line-argument -x assembler -target amdgcn-amd-amdhsa -mcode-object-version=4 -mcpu=${offload_arch} -mwavefrontsize64 -c
    )
    # Stage the committed .s into the build dir so the object rule has a local source.
    add_custom_command(
        OUTPUT
            "${CMAKE_CURRENT_BINARY_DIR}/row_div_${arch}.s"
        COMMAND ${CMAKE_COMMAND} -E copy "${row_div_src}" "${CMAKE_CURRENT_BINARY_DIR}/row_div_${arch}.s"
        DEPENDS "${row_div_src}"
        COMMENT "Staging row_div assembly for ${arch}"
    )
    target_sources(row-div-obj-${arch}
        PRIVATE
            "${CMAKE_CURRENT_BINARY_DIR}/row_div_${arch}.s"
    )

    set(output_code_object_file "${CMAKE_CURRENT_BINARY_DIR}/row_div_${arch}.co")
    add_custom_command(
        DEPENDS row-div-obj-${arch}
        OUTPUT "${output_code_object_file}"
        COMMAND ${CMAKE_CXX_COMPILER};-target;amdgcn-amdhsa;-Xlinker;$<TARGET_OBJECTS:row-div-obj-${arch}>;-o;${output_code_object_file}
        COMMENT "Creating row_div_${arch}.co"
        COMMAND_EXPAND_LISTS
    )

    add_custom_target(row-div-library-${arch} ALL
        DEPENDS
            "${output_code_object_file}"
            ${previous_target}
        COMMAND ${CMAKE_COMMAND} -E make_directory "${HIPBLASLT_TENSILE_LIBPATH}/library/${arch}"
        COMMAND ${CMAKE_COMMAND} -E copy "${output_code_object_file}" "${HIPBLASLT_TENSILE_LIBPATH}/library/${arch}"
        COMMENT "Installing row_div_${arch}.co"
    )
    set(previous_target "row-div-library-${arch}")
endforeach()
