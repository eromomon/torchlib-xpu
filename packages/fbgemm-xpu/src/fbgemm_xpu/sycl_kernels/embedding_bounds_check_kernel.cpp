/*
 * Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
 * Copyright (c) 2026 Intel Corporation. All Rights Reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */

////////////////////////////////////////////////////////////////////////////////
// SYCL PORT MAPPING TO FBGEMM CUDA SOURCE - EMBEDDING BOUNDS CHECK V1 HOST
////////////////////////////////////////////////////////////////////////////////
//
// ORIGINAL CUDA SOURCES:
//   Host:   fbgemm_gpu/codegen/utils/embedding_bounds_check_host.cpp
//   Kernel: fbgemm_gpu/codegen/utils/embedding_bounds_check_v1.cu
//
// HOST FUNCTION MAPPING:
//   bounds_check_indices_xpu
//     -> bounds_check_indices_cuda
//     -> _bounds_check_indices_cuda_v1
//
// LAUNCH MAPPING:
//   launch_bounds_check_indices_v1<index_t, vbe>
//     -> FBGEMM_LAUNCH_DSA_KERNEL(bounds_check_indices_kernel_v1<...>)
//
// XPU DEVIATIONS:
//   - Offset validation and repair are split out and ordered before the index
//     kernel; see embedding_bounds_check_kernel.h for the kernel-level map.
//   - Grid size is capped for DPC++ and paired with grid-stride coverage.
//   - FATAL reports through a device flag and host TORCH_CHECK rather than a
//     device-side assert.
//
////////////////////////////////////////////////////////////////////////////////

#include "embedding_bounds_check_kernel.h"

#include <limits>

#include <c10/xpu/XPUStream.h>
#include <torch/library.h>

#include "fbgemm_utils/tensor_utils.h"

namespace fbgemm_xpu {
namespace {

constexpr int64_t kWarpSize = 32;
constexpr int64_t kWarpsPerGroup = 8;
constexpr int64_t kThreadsPerGroup = kWarpSize * kWarpsPerGroup;

template <typename index_t>
sycl::event launch_bounds_check_offsets(sycl::queue &queue, at::Tensor &offsets,
                                        at::Tensor &warning,
                                        int64_t *offsets_invalid,
                                        int64_t *fatal_error, int64_t total_B,
                                        int64_t num_indices,
                                        BoundsCheckMode bounds_check_mode) {
  constexpr int64_t threads = 256;
  const int64_t requested_groups =
      total_B == 0 ? 1 : (total_B - 1) / threads + 1;
  const uint32_t groups = xpu_cap_grid_dim_x(requested_groups, threads);
  auto kernel = BoundsCheckOffsetsKernel<index_t>(
      offsets.mutable_data_ptr<index_t>(), offsets.stride(0), total_B,
      num_indices, warning.mutable_data_ptr<int64_t>(), offsets_invalid,
      fatal_error, bounds_check_mode);
  return queue.submit([&](sycl::handler &cgh) {
    cgh.parallel_for<BoundsCheckOffsetsKernel<index_t>>(
        sycl::nd_range<1>(sycl::range<1>(static_cast<size_t>(groups * threads)),
                          sycl::range<1>(static_cast<size_t>(threads))),
        kernel);
  });
}

template <typename index_t>
sycl::event launch_repair_bounds_check_offsets(
    sycl::queue &queue, at::Tensor &offsets, const int64_t *offsets_invalid,
    int64_t total_B, int64_t num_indices, BoundsCheckMode bounds_check_mode,
    const sycl::event &check_offsets_event) {
  auto kernel = RepairBoundsCheckOffsetsKernel<index_t>(
      offsets.mutable_data_ptr<index_t>(), offsets.stride(0), total_B,
      num_indices, offsets_invalid, bounds_check_mode);
  return queue.submit([&](sycl::handler &cgh) {
    cgh.depends_on(check_offsets_event);
    cgh.parallel_for<RepairBoundsCheckOffsetsKernel<index_t>>(sycl::range<1>(1),
                                                              kernel);
  });
}

sycl::event validate_and_repair_bounds_check_offsets(
    sycl::queue &queue, at::Tensor &offsets, at::Tensor &warning,
    int64_t *offsets_invalid, int64_t *fatal_error, int64_t total_B,
    int64_t num_indices, BoundsCheckMode bounds_check_mode) {
  sycl::event repair_offsets_event;
  AT_DISPATCH_INDEX_TYPES(
      offsets.scalar_type(), "bounds_check_offsets_xpu_v1", [&] {
        const sycl::event check_offsets_event =
            launch_bounds_check_offsets<index_t>(
                queue, offsets, warning, offsets_invalid, fatal_error, total_B,
                num_indices, bounds_check_mode);
        repair_offsets_event = launch_repair_bounds_check_offsets<index_t>(
            queue, offsets, offsets_invalid, total_B, num_indices,
            bounds_check_mode, check_offsets_event);
      });
  return repair_offsets_event;
}

template <typename index_t, bool vbe>
void launch_bounds_check_indices_v1(
    sycl::queue &queue, const at::Tensor &rows_per_table, at::Tensor &indices,
    at::Tensor &offsets, const std::optional<at::Tensor> &B_offsets,
    at::Tensor &warning, int64_t T, int64_t total_B, int64_t max_B,
    const int64_t *offsets_invalid, int64_t *fatal_error,
    BoundsCheckMode bounds_check_mode,
    const sycl::event &repair_offsets_event) {
  const int64_t logical_warps = max_B * T;
  const int64_t requested_groups = (logical_warps - 1) / kWarpsPerGroup + 1;
  const uint32_t groups =
      xpu_cap_grid_dim_x(requested_groups, kThreadsPerGroup);

  auto kernel = BoundsCheckIndicesKernelV1<index_t, vbe>(
      rows_per_table.const_data_ptr<int64_t>(), rows_per_table.stride(0),
      indices.mutable_data_ptr<index_t>(), indices.stride(0),
      offsets.mutable_data_ptr<index_t>(), offsets.stride(0),
      vbe ? B_offsets->const_data_ptr<int32_t>() : nullptr,
      vbe ? B_offsets->stride(0) : 0, warning.mutable_data_ptr<int64_t>(),
      offsets_invalid, fatal_error, T, total_B, max_B, bounds_check_mode);

  queue.submit([&](sycl::handler &cgh) {
    cgh.depends_on(repair_offsets_event);
    cgh.parallel_for<BoundsCheckIndicesKernelV1<index_t, vbe>>(
        sycl::nd_range<2>(
            sycl::range<2>(static_cast<size_t>(groups * kWarpsPerGroup),
                           static_cast<size_t>(kWarpSize)),
            sycl::range<2>(static_cast<size_t>(kWarpsPerGroup),
                           static_cast<size_t>(kWarpSize))),
        kernel);
  });
}

} // namespace

void bounds_check_indices_xpu(
    at::Tensor &rows_per_table, at::Tensor &indices, at::Tensor &offsets,
    int64_t bounds_check_mode, at::Tensor &warning,
    const std::optional<at::Tensor> &weights,
    const std::optional<at::Tensor> &B_offsets, int64_t max_B,
    const std::optional<at::Tensor> &b_t_map, int64_t info_B_num_bits,
    int64_t info_B_mask, int8_t bounds_check_version, bool prefetch_pipeline) {
  TORCH_CHECK(bounds_check_version == 1,
              "bounds_check_indices: XPU supports bounds_check_version=1, got ",
              bounds_check_version);
  TORCH_CHECK(!prefetch_pipeline,
              "bounds_check_indices: XPU version 1 does not support "
              "prefetch_pipeline=true");
  TORCH_CHECK(
      bounds_check_mode == static_cast<int64_t>(BoundsCheckMode::FATAL) ||
          bounds_check_mode == static_cast<int64_t>(BoundsCheckMode::WARNING) ||
          bounds_check_mode == static_cast<int64_t>(BoundsCheckMode::IGNORE),
      "bounds_check_indices: bounds_check_mode=", bounds_check_mode,
      " is not supported");

  TENSORS_ON_SAME_SYCL_XPU_IF_NOT_OPTIONAL(rows_per_table, indices, offsets,
                                           warning, B_offsets, b_t_map);
  if (weights.has_value() && weights->defined()) {
    TENSORS_EMPTY_OR_ON_SAME_DEVICE(weights.value(), rows_per_table);
  }

  TORCH_CHECK(rows_per_table.dim() == 1,
              "bounds_check_indices: rows_per_table must be 1-dimensional");
  TORCH_CHECK(indices.dim() == 1,
              "bounds_check_indices: indices must be 1-dimensional");
  TORCH_CHECK(offsets.dim() == 1,
              "bounds_check_indices: offsets must be 1-dimensional");
  TORCH_CHECK(warning.dim() == 1 && warning.numel() >= 1,
              "bounds_check_indices: warning must be a non-empty 1D tensor");
  TORCH_CHECK(rows_per_table.scalar_type() == at::kLong,
              "bounds_check_indices: rows_per_table must have dtype int64");
  TORCH_CHECK(indices.scalar_type() == at::kInt ||
                  indices.scalar_type() == at::kLong,
              "bounds_check_indices: indices must have dtype int32 or int64");
  TORCH_CHECK(
      offsets.scalar_type() == indices.scalar_type(),
      "bounds_check_indices: offsets and indices must have the same dtype");
  TORCH_CHECK(warning.scalar_type() == at::kLong,
              "bounds_check_indices: warning must have dtype int64");
  TORCH_CHECK(
      offsets.numel() >= 1,
      "bounds_check_indices: offsets must contain at least one element");

  if (weights.has_value() && weights->defined() && weights->numel() != 0) {
    TORCH_CHECK(weights->dim() == 1,
                "bounds_check_indices: weights must be 1-dimensional");
    TORCH_CHECK(weights->numel() == indices.numel(),
                "bounds_check_indices: weights size ", weights->numel(),
                " is not equal to indices size ", indices.numel());
  }

  const int64_t T = rows_per_table.numel();
  const int64_t total_B = offsets.numel() - 1;
  SYCL_DEVICE_GUARD(rows_per_table);
  if (static_cast<BoundsCheckMode>(bounds_check_mode) ==
      BoundsCheckMode::WARNING) {
    warning.zero_();
  }
  if (T == 0) {
    return;
  }

  if (indices.scalar_type() == at::kInt) {
    TORCH_CHECK(indices.numel() <= std::numeric_limits<int32_t>::max(),
                "bounds_check_indices: int32 indices cannot address ",
                indices.numel(), " elements");
  }

  const bool vbe = B_offsets.has_value() && B_offsets->defined();
  int64_t launch_max_B = 0;
  if (total_B > 0) {
    if (vbe) {
      TORCH_CHECK(B_offsets->dim() == 1,
                  "bounds_check_indices: B_offsets must be 1-dimensional");
      TORCH_CHECK(B_offsets->scalar_type() == at::kInt,
                  "bounds_check_indices: B_offsets must have dtype int32");
      TORCH_CHECK(B_offsets->numel() == T + 1,
                  "bounds_check_indices: B_offsets must have T + 1 elements");
      TORCH_CHECK(
          max_B > 0,
          "bounds_check_indices: max_B must be positive for VBE inputs");
      launch_max_B = max_B;
    } else {
      const int64_t B = total_B / T;
      TORCH_CHECK(total_B == B * T, "bounds_check_indices: offsets size ",
                  offsets.numel(), " is not equal to B * T + 1 for T=", T);
      launch_max_B = B;
    }

    TORCH_CHECK(launch_max_B <= std::numeric_limits<int64_t>::max() / T,
                "bounds_check_indices: max_B * T overflows int64");
  }

  // These v1 arguments are accepted to preserve the upstream schema but are
  // only interpreted by the v2 implementation.
  static_cast<void>(info_B_num_bits);
  static_cast<void>(info_B_mask);

  sycl::queue &queue = c10::xpu::getCurrentXPUStream().queue();
  at::Tensor offsets_invalid = at::zeros({1}, warning.options());
  int64_t *offsets_invalid_ptr = offsets_invalid.mutable_data_ptr<int64_t>();
  at::Tensor fatal_error;
  int64_t *fatal_error_ptr = nullptr;
  if (static_cast<BoundsCheckMode>(bounds_check_mode) ==
      BoundsCheckMode::FATAL) {
    fatal_error = at::zeros({1}, warning.options());
    fatal_error_ptr = fatal_error.mutable_data_ptr<int64_t>();
  }

  const sycl::event repair_offsets_event =
      validate_and_repair_bounds_check_offsets(
          queue, offsets, warning, offsets_invalid_ptr, fatal_error_ptr,
          total_B, indices.numel(),
          static_cast<BoundsCheckMode>(bounds_check_mode));

  if (total_B == 0) {
    if (fatal_error.defined()) {
      TORCH_CHECK(
          fatal_error.item<int64_t>() == 0,
          "bounds_check_indices: out-of-bounds indices or offsets detected");
    }
    return;
  }

  AT_DISPATCH_INDEX_TYPES(
      indices.scalar_type(), "bounds_check_indices_xpu_v1", [&] {
        if (vbe) {
          launch_bounds_check_indices_v1<index_t, true>(
              queue, rows_per_table, indices, offsets, B_offsets, warning, T,
              total_B, launch_max_B, offsets_invalid_ptr, fatal_error_ptr,
              static_cast<BoundsCheckMode>(bounds_check_mode),
              repair_offsets_event);
        } else {
          launch_bounds_check_indices_v1<index_t, false>(
              queue, rows_per_table, indices, offsets, B_offsets, warning, T,
              total_B, launch_max_B, offsets_invalid_ptr, fatal_error_ptr,
              static_cast<BoundsCheckMode>(bounds_check_mode),
              repair_offsets_event);
        }
      });
  if (fatal_error.defined()) {
    TORCH_CHECK(
        fatal_error.item<int64_t>() == 0,
        "bounds_check_indices: out-of-bounds indices or offsets detected");
  }
}

TORCH_LIBRARY_IMPL(fbgemm, XPU, m) {
  m.impl("bounds_check_indices", &bounds_check_indices_xpu);
}

} // namespace fbgemm_xpu
