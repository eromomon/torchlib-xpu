/*
 * Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
 * Copyright (c) 2026 Intel Corporation. All Rights Reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */

////////////////////////////////////////////////////////////////////////////////
// SYCL PORT MAPPING TO FBGEMM CUDA SOURCE - DENSE TO JAGGED (FORWARD)
////////////////////////////////////////////////////////////////////////////////
//
// This file contains the SYCL/XPU host implementation of the
// dense_to_jagged_forward operator, ported from FBGEMM CUDA v1.8.0.
//
// ORIGINAL CUDA SOURCE:
//   File: fbgemm_gpu/src/jagged_tensor_ops/dense_to_jagged_forward.cu
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
//   dense_to_jagged_forward_xpu (SYCL)
//     → dense_to_jagged_forward (CUDA)
//     CUDA File: fbgemm_gpu/src/jagged_tensor_ops/dense_to_jagged_forward.cu
//
// DESCRIPTION:
//   Extracts from a dense tensor exactly the positions that a jagged tensor of
//   the given offsets would occupy, producing [total_L, D] jagged values. The
//   element-wise operation is the identity on the dense operand
//   (JaggedOpCopyY), so the jagged operand supplies shape only and is never
//   read for value - which is why the CUDA source allocates it with at::empty
//   and leaves it uninitialized.
//
//   Half inputs take the vectorized fast path
//   (jagged_dense_elementwise_jagged_output_opt_); every other dtype takes the
//   generic kernel. This split, and the dtype sets on each side, match CUDA
//   exactly.
//
// OPERATOR COVERAGE:
//   Registering the XPU key for this one backend operator also brings up
//   fbgemm::dense_to_jagged, which is CompositeImplicitAutograd upstream and
//   redispatches here. Its backward is jagged_to_padded_dense_forward.
//   CUDA File: fbgemm_gpu/src/jagged_tensor_ops/jagged_tensor_ops_autograd.cpp
//
////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <optional>
#include <vector>

#include <ATen/ATen.h>
#include <c10/core/SymInt.h>

namespace fbgemm_xpu {

////////////////////////////////////////////////////////////////////////////////
// dense_to_jagged_forward_xpu - Host Function
////////////////////////////////////////////////////////////////////////////////
//
// CUDA SOURCE MAPPING:
//   CUDA Function: dense_to_jagged_forward
//   CUDA File: fbgemm_gpu/src/jagged_tensor_ops/dense_to_jagged_forward.cu
//
// Schema (declared upstream by fbgemm-gpu-cpu, so this repo only supplies the
// XPU dispatch):
//   dense_to_jagged_forward(
//       Tensor dense, Tensor[] x_offsets, SymInt? total_L=None) -> Tensor
//
////////////////////////////////////////////////////////////////////////////////

/**
 * @brief XPU implementation of dense_to_jagged_forward
 *
 * Gathers the elements of `dense` that fall inside the jagged region described
 * by `offsets` into a [total_L, D] jagged values tensor.
 *
 * @param dense Dense tensor, [B, max_lengths..., D]
 * @param offsets One offsets tensor per jagged dimension; offsets[0] has B+1
 *        entries. At most 5 jagged dimensions are supported.
 * @param total_L Number of jagged rows to produce. Computed from
 *        offsets.back().max() when not supplied, which costs a device-to-host
 *        synchronization.
 * @return Jagged values, [total_L, D]
 */
at::Tensor dense_to_jagged_forward_xpu(
    const at::Tensor& dense,
    const std::vector<at::Tensor>& offsets,
    std::optional<at::SymInt> total_L);

}  // namespace fbgemm_xpu
