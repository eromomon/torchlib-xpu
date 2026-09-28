/*
 * Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
 * Copyright (c) 2026 Intel Corporation. All Rights Reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */

////////////////////////////////////////////////////////////////////////////////
// SYCL PORT MAPPING TO FBGEMM CUDA SOURCE - JAGGED TO PADDED DENSE (BACKWARD)
////////////////////////////////////////////////////////////////////////////////
//
// ORIGINAL CUDA SOURCE:
//   File: fbgemm_gpu/src/jagged_tensor_ops/jagged_to_padded_dense_backward.cu
//
// KERNEL MAPPING:
//   None of its own. The operator is expressed entirely through the shared
//   jagged-output launcher in jagged_common.h:
//
//     JaggedDenseDenseElementwiseJaggedOutputKernel<...>
//       → jagged_dense_dense_elementwise_jagged_output_kernel_ (CUDA)
//       CUDA File: fbgemm_gpu/src/jagged_tensor_ops/common.cuh
//
// HOST FUNCTION MAPPING:
//   jagged_to_padded_dense_backward_xpu (SYCL)
//     → jagged_to_padded_dense_backward (CUDA)
//     CUDA File: fbgemm_gpu/src/jagged_tensor_ops/jagged_to_padded_dense_backward.cu
//
////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <vector>

#include <ATen/ATen.h>
#include <c10/core/SymInt.h>

namespace fbgemm_xpu {

////////////////////////////////////////////////////////////////////////////////
// jagged_to_padded_dense_backward_xpu - Host Function
////////////////////////////////////////////////////////////////////////////////
//
// CUDA SOURCE MAPPING:
//   CUDA Function: jagged_to_padded_dense_backward
//   CUDA File: fbgemm_gpu/src/jagged_tensor_ops/jagged_to_padded_dense_backward.cu
//
////////////////////////////////////////////////////////////////////////////////

at::Tensor jagged_to_padded_dense_backward_xpu(
    const at::Tensor& grad_output,
    const std::vector<at::Tensor>& offsets,
    at::SymInt total_L);

}  // namespace fbgemm_xpu
