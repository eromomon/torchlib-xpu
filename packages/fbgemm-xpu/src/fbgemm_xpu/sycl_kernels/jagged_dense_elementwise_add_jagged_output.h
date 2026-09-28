/*
 * Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
 * Copyright (c) 2026 Intel Corporation. All Rights Reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */

////////////////////////////////////////////////////////////////////////////////
// SYCL PORT MAPPING TO FBGEMM CUDA SOURCE - JAGGED DENSE ELEMENTWISE ADD
// (JAGGED OUTPUT)
////////////////////////////////////////////////////////////////////////////////
//
// This file contains the SYCL/XPU host implementation of the
// jagged_dense_elementwise_add_jagged_output operator, ported from FBGEMM CUDA.
//
// ORIGINAL CUDA SOURCE:
//   File: fbgemm_gpu/src/jagged_tensor_ops/jagged_to_padded_dense_forward.cu
//
// KERNEL MAPPING:
//   None of its own. The operator is expressed entirely through the shared
//   jagged-output launchers in jagged_common.h:
//
//     JaggedDenseDenseElementwiseJaggedOutputKernel<...>
//       → jagged_dense_dense_elementwise_jagged_output_kernel_ (CUDA)
//     JaggedDenseDenseElementwiseJaggedOutputOptSearchKernel<...>
//       → jagged_dense_dense_elementwise_jagged_output_opt_search_kernel_ (CUDA)
//     JaggedDenseDenseElementwiseJaggedOutputOptGatherKernel<...>
//       → jagged_dense_dense_elementwise_jagged_output_opt_gather_kernel_ (CUDA)
//     CUDA File: fbgemm_gpu/src/jagged_tensor_ops/common.cuh
//
// HOST FUNCTION MAPPING:
//   jagged_dense_elementwise_add_jagged_output_xpu (SYCL)
//     → jagged_dense_elementwise_add_jagged_output_cuda (CUDA)
//     CUDA File: fbgemm_gpu/src/jagged_tensor_ops/jagged_to_padded_dense_forward.cu
//
// AUTOGRAD FUNCTION MAPPING:
//   JaggedDenseAddJaggedOutputXPUOp (SYCL)
//     → JaggedDenseAddJaggedOutputGPUOp (CUDA)
//     CUDA File: fbgemm_gpu/src/jagged_tensor_ops/jagged_to_padded_dense_forward.cu
//
////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <tuple>
#include <vector>

#include <ATen/ATen.h>

namespace fbgemm_xpu {

////////////////////////////////////////////////////////////////////////////////
// jagged_dense_elementwise_add_jagged_output_xpu - Host Function
////////////////////////////////////////////////////////////////////////////////
//
// CUDA SOURCE MAPPING:
//   CUDA Function: jagged_dense_elementwise_add_jagged_output_cuda
//   CUDA File: fbgemm_gpu/src/jagged_tensor_ops/jagged_to_padded_dense_forward.cu
//
////////////////////////////////////////////////////////////////////////////////

std::tuple<at::Tensor, std::vector<at::Tensor>>
jagged_dense_elementwise_add_jagged_output_xpu(
    const at::Tensor& x_values,
    const std::vector<at::Tensor>& x_offsets,
    const at::Tensor& y);

}  // namespace fbgemm_xpu
