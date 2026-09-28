/*
 * Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
 * Copyright (c) 2026 Intel Corporation. All Rights Reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */

////////////////////////////////////////////////////////////////////////////////
// SYCL PORT MAPPING TO FBGEMM CUDA SOURCE - DENSE TO JAGGED (FORWARD)
////////////////////////////////////////////////////////////////////////////////
//
// ORIGINAL CUDA SOURCE:
//   File: fbgemm_gpu/src/jagged_tensor_ops/dense_to_jagged_forward.cu
//   Function: fbgemm_gpu::dense_to_jagged_forward (lines 15-102 at v1.8.0)
//
// See dense_to_jagged_forward.h for the full mapping notes.
//
////////////////////////////////////////////////////////////////////////////////

#include <limits>

#include <ATen/Dispatch.h>

#include "dense_to_jagged_forward.h"
#include "fbgemm_utils/utils.h"
#include "jagged_common.h"

namespace fbgemm_xpu {

////////////////////////////////////////////////////////////////////////////////
// dense_to_jagged_forward_xpu - Host Function
////////////////////////////////////////////////////////////////////////////////
//
// CUDA SOURCE MAPPING:
//   CUDA Function: dense_to_jagged_forward
//   CUDA File: fbgemm_gpu/src/jagged_tensor_ops/dense_to_jagged_forward.cu
//
// DESCRIPTION:
//   Resolves total_L, validates that everything the kernels index with 32-bit
//   accessors actually fits in int32, allocates the jagged output, and launches
//   the copy-from-dense operation: the Half fast path for Half, the generic
//   jagged-output kernel for all other supported dtypes.
//
////////////////////////////////////////////////////////////////////////////////
at::Tensor dense_to_jagged_forward_xpu(
    const at::Tensor& dense,
    const std::vector<at::Tensor>& offsets,
    std::optional<at::SymInt> total_L) {
    // D is the embedding dimension
    auto D = dense.size(-1);
    TORCH_CHECK(D >= 0, "D must be >= 0, but got ", D);

    // If total_L is not given then compute it
    int64_t total_L_computed;
    if (total_L.has_value()) {
        total_L_computed = total_L.value().expect_int();
        TORCH_CHECK_VALUE(
            total_L_computed >= 0,
            "total_L passed to dense_to_jagged_forward must be >= 0, but got ",
            total_L_computed,
            ". This indicates total_L is corrupted somewhere prior to dense_to_jagged.");
    } else {
        total_L_computed = offsets.back().max().item<int64_t>();
        TORCH_CHECK_VALUE(
            total_L_computed >= 0,
            "total_L must be >= 0, but got ",
            total_L_computed,
            ". This indicates corrupted offsets (offsets.back() contains a garbage/negative value).",
            " offsets.size() = ",
            offsets.size(),
            " offsets.back().size(-1) = ",
            offsets.back().size(-1),
            " offsets.back()[-1] = ",
            offsets.back()[offsets.back().size(-1) - 1].item<int64_t>());
    }
    constexpr int64_t kInt32Max = std::numeric_limits<int32_t>::max();
    TORCH_CHECK_VALUE(
        D == 0 || total_L_computed <= kInt32Max / D,
        "total_L_computed * D overflows int32 max. total_L_computed = ",
        total_L_computed,
        " D = ",
        D,
        ". `values` is defined as PTA32. Contact FBGEMM team for int64 support.");
    TORCH_CHECK_VALUE(
        dense.numel() <= kInt32Max,
        "Expect dense.numel() <= int32 max, but got ",
        dense.numel(),
        ". y_0/y_1/y_reshaped is defined as PTA32. Contact FBGEMM team for int64 support.");
    // offsets are int32-indexed in the binary search (the non-opt kernel handles
    // num_jagged_dim up to kStackArrayMaxDims), so each offsets tensor's numel
    // must fit int32.
    for (const auto& off : offsets) {
        TORCH_CHECK_VALUE(
            off.numel() <= kInt32Max,
            "offsets numel must be <= int32 max, but got ",
            off.numel(),
            ". offsets are int32-indexed. Contact FBGEMM team for int64 support.");
    }
    // `values` is the jagged operand. JaggedOpCopyY never reads it, so it is
    // deliberately left uninitialized, as in the CUDA source.
    auto values = at::empty_symint({total_L_computed, D}, dense.options());
    auto output = dense.numel() == 0 ? at::zeros_like(values)
                                     : at::empty_like(values);

    SYCL_DEVICE_GUARD(dense);

    AT_DISPATCH_SWITCH(
        values.scalar_type(),
        "dense_to_jagged_gpu_op_forward",
        AT_DISPATCH_CASE(
            at::ScalarType::Half,
            [&] {
                jagged_dense_elementwise_jagged_output_opt_<scalar_t>(
                    values, offsets, dense, output, JaggedOpCopyY());
            })

            FBGEMM_DISPATCH_ALL_TYPES_BUT_HALF_CASE([&] {
                jagged_dense_elementwise_jagged_output_<scalar_t>(
                    values, offsets, dense, output, JaggedOpCopyY());
            }));

    return output;
}

/**
 * Register XPU implementation with PyTorch dispatch system.
 *
 * The schema is already declared by the upstream fbgemm-gpu-cpu package
 * (jagged_tensor_ops_cpu.cpp), which fbgemm_xpu/__init__.py imports before _C
 * loads, so only the XPU dispatch key is registered here.
 */
TORCH_LIBRARY_IMPL(fbgemm, XPU, m) {
    m.impl("dense_to_jagged_forward", &dense_to_jagged_forward_xpu);
}

}  // namespace fbgemm_xpu
