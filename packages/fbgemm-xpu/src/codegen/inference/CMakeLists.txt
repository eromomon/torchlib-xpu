# Copyright (c) 2026 Intel Corporation. All Rights Reserved.
# SPDX-License-Identifier: BSD-3-Clause

set(INFERENCE_GEN_DIR ${CMAKE_CURRENT_BINARY_DIR}/generated)
set(INFERENCE_HEADERS
    ${INFERENCE_GEN_DIR}/sycl_kernels/gen_embedding_forward_int4_nobag.h
    ${INFERENCE_GEN_DIR}/sycl_kernels/gen_embedding_forward_int8_nobag.h
)

add_custom_command(
    OUTPUT ${INFERENCE_HEADERS}
    COMMAND ${Python3_EXECUTABLE}
        ${GENSCRIPT_DIR}/generate_forward_quantized.py
        --opensource --install_dir ${INFERENCE_GEN_DIR}
    DEPENDS
        ${GENSCRIPT_DIR}/generate_forward_quantized.py
        ${GENSCRIPT_DIR}/common.py
        ${GENSCRIPT_DIR}/jinja_environment.py
        ${GENSCRIPT_DIR}/scripts_argsparse.py
        ${GENSCRIPT_DIR}/torch_type_utils.py
        ${CMAKE_CURRENT_SOURCE_DIR}/embedding_forward_quantized_kernel_template.h
    VERBATIM
)

add_custom_target(generate_inference_kernels DEPENDS ${INFERENCE_HEADERS})
add_dependencies(_C generate_inference_kernels)
target_sources(_C PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/embedding_forward_quantized.cpp)
target_include_directories(_C PRIVATE ${INFERENCE_GEN_DIR}/sycl_kernels)
