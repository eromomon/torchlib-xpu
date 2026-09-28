/*
 * Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
 * Copyright (c) 2026 Intel Corporation. All Rights Reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */

////////////////////////////////////////////////////////////////////////////////
// SYCL PORT MAPPING TO FBGEMM CUDA SOURCE - QUANTIZED INFERENCE LOOKUP HOST
////////////////////////////////////////////////////////////////////////////////
//
// ORIGINAL CUDA SOURCES:
//   Public dispatcher:
//     fbgemm_gpu/codegen/inference/embedding_forward_quantized_host.cpp
//   Generated no-bag host launcher:
//     fbgemm_gpu/codegen/inference/
//       embedding_forward_quantized_split_nbit_host_template.cu
//   Generated CUDA kernel:
//     fbgemm_gpu/codegen/inference/
//       embedding_forward_quantized_split_nbit_kernel_template.cu
//
// HOST FUNCTION MAPPING:
//   int_nbit_lookup
//     -> int_nbit_split_embedding_codegen_lookup_function (CUDA dispatcher)
//     -> int_nbit_split_embedding_nobag_codegen_forward_unweighted_cuda_impl
//        (generated CUDA no-bag launcher)
//
// INTENTIONAL XPU SUBSET AND STRUCTURAL DIFFERENCES:
//   - Supports eager, unweighted, no-bag INT4/INT8 DEVICE lookup with a
//     uniform embedding dimension and FP32/FP16/BF16 output.
//   - Preserves int32 and int64 indices instead of converting the no-bag path
//     to int32 as the pinned CUDA dispatcher does.
//   - Packs device table metadata into one host transfer for validation and
//     launch setup, then reduces physical-storage errors to one device flag.
//   - Submits one named lookup functor per table. The CUDA launcher includes
//     the table dimension in its grid; multi-table fusion is tracked as a
//     separate performance optimization and measured in the package docs.
//
////////////////////////////////////////////////////////////////////////////////

#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <c10/core/DeviceGuard.h>
#include <c10/xpu/XPUStream.h>
#include <torch/library.h>

#include <algorithm>
#include <limits>
#include <string_view>
#include <vector>

#include "gen_embedding_forward_int4_nobag.h"
#include "gen_embedding_forward_int8_nobag.h"

namespace fbgemm_xpu {
namespace {

constexpr std::string_view op_name = "int_nbit_split_embedding_codegen_lookup_function: ";

void check_tensor_layout(const at::Tensor& tensor, const at::Device& device,
                         std::string_view name) {
    TORCH_CHECK(tensor.device() == device, op_name, name, " must be on ", device);
    TORCH_CHECK(tensor.dim() == 1 && tensor.is_contiguous(), op_name, name,
                " must be contiguous and one-dimensional");
}

void check_tensor(const at::Tensor& tensor, at::ScalarType dtype,
                  const at::Device& device, std::string_view name) {
    check_tensor_layout(tensor, device, name);
    TORCH_CHECK(tensor.scalar_type() == dtype, op_name, name, " has unsupported dtype");
}

void check_empty(const std::optional<at::Tensor>& tensor, std::string_view name) {
    TORCH_CHECK(!tensor || tensor->numel() == 0, op_name, name, " is unsupported");
}

template <typename index_t, typename output_t>
void launch(sycl::queue& queue, int64_t weight_type, const uint8_t* weights,
            const index_t* indices, output_t* output, int64_t begin, int64_t end,
            int64_t dimension, int64_t row_stride, int64_t storage_rows, int32_t* error) {
    if (weight_type == 3) {
        lookup_int4_nobag(queue, weights, indices, output, begin, end, dimension, row_stride,
                          storage_rows, error);
    } else {
        lookup_int8_nobag(queue, weights, indices, output, begin, end, dimension, row_stride,
                          storage_rows, error);
    }
}

at::Tensor int_nbit_lookup(
    at::Tensor dev_weights, at::Tensor uvm_weights, at::Tensor weights_placements,
    at::Tensor weights_offsets, at::Tensor weights_tys, at::Tensor D_offsets,
    c10::SymInt total_D, int64_t max_int2_D, int64_t max_int4_D,
    int64_t max_int8_D, int64_t max_float16_D, int64_t max_float32_D,
    at::Tensor indices, at::Tensor offsets, int64_t pooling_mode,
    std::optional<at::Tensor> indice_weights, int64_t output_dtype,
    std::optional<at::Tensor> lxu_cache_weights,
    std::optional<at::Tensor> lxu_cache_locations,
    std::optional<int64_t> row_alignment, std::optional<int64_t> max_float8_D,
    std::optional<int64_t> fp8_exponent_bits, std::optional<int64_t> fp8_exponent_bias) {
    TORCH_CHECK(dev_weights.is_xpu(), op_name, "weights must be on XPU");
    const auto device = dev_weights.device();
    const c10::OptionalDeviceGuard guard(device);
    TORCH_CHECK(pooling_mode == 2, op_name, "only PoolingMode.NONE is supported");
    check_empty(indice_weights, "weighted lookup");
    check_empty(lxu_cache_weights, "cache weights");
    check_empty(lxu_cache_locations, "cache locations");
    TORCH_CHECK(uvm_weights.numel() == 0, op_name, "UVM/cache is unsupported");
    TORCH_CHECK(max_int2_D == 0 && max_float16_D == 0 && max_float32_D == 0 &&
                    max_float8_D.value_or(0) == 0 &&
                    fp8_exponent_bits.value_or(-1) == -1 &&
                    fp8_exponent_bias.value_or(-1) == -1,
                op_name, "only INT4 and INT8 weights are supported");
    const int64_t alignment = row_alignment.value_or(16);
    TORCH_CHECK(alignment > 0 && alignment <= 128 && (alignment & (alignment - 1)) == 0,
                op_name, "row_alignment must be a power of two between 1 and 128");
    at::ScalarType output_type;
    switch (output_dtype) {
        case 0: output_type = at::kFloat; break;
        case 1: output_type = at::kHalf; break;
        case 5: output_type = at::kBFloat16; break;
        default: TORCH_CHECK(false, op_name, "output dtype must be FP32, FP16 or BF16");
    }
    check_tensor(dev_weights, at::kByte, device, "dev_weights");
    check_tensor(weights_placements, at::kInt, device, "weights_placements");
    check_tensor(weights_offsets, at::kLong, device, "weights_offsets");
    check_tensor(weights_tys, at::kByte, device, "weights_tys");
    check_tensor(D_offsets, at::kInt, device, "D_offsets");
    TORCH_CHECK(indices.scalar_type() == at::kInt || indices.scalar_type() == at::kLong,
                op_name, "indices must have int32 or int64 dtype");
    TORCH_CHECK(offsets.scalar_type() == at::kInt || offsets.scalar_type() == at::kLong,
                op_name, "offsets must have int32 or int64 dtype");
    check_tensor_layout(indices, device, "indices");
    check_tensor_layout(offsets, device, "offsets");
    const int64_t tables = weights_offsets.numel();
    TORCH_CHECK(tables > 0 && weights_tys.numel() == tables &&
                    weights_placements.numel() == tables && D_offsets.numel() == tables + 1,
                op_name, "inconsistent table metadata");
    TORCH_CHECK(offsets.numel() > 0 && (offsets.numel() - 1) % tables == 0,
                op_name, "offsets must have B * T + 1 elements");

    auto& queue = c10::xpu::getCurrentXPUStream(device.index()).queue();
    const int64_t metadata_size = 4 * tables + 1 + offsets.numel();
    auto metadata = at::empty({metadata_size}, dev_weights.options().dtype(at::kLong));
    auto* metadata_ptr = metadata.mutable_data_ptr<int64_t>();
    const auto* dims_ptr = D_offsets.const_data_ptr<int32_t>();
    const auto* starts_ptr = weights_offsets.const_data_ptr<int64_t>();
    const auto* types_ptr = weights_tys.const_data_ptr<uint8_t>();
    const auto* placements_ptr = weights_placements.const_data_ptr<int32_t>();
    const int64_t metadata_work_items = std::min<int64_t>(metadata_size, 4096 * 256);
    AT_DISPATCH_INDEX_TYPES(offsets.scalar_type(), "int_nbit_metadata_xpu", [&] {
        const auto* offsets_ptr = offsets.const_data_ptr<index_t>();
        queue.parallel_for(sycl::range<1>(metadata_work_items), [=](sycl::id<1> item) {
            for (int64_t position = item[0]; position < metadata_size; position += metadata_work_items) {
                if (position < tables + 1) {
                    metadata_ptr[position] = dims_ptr[position];
                } else if (position < 2 * tables + 1) {
                    metadata_ptr[position] = starts_ptr[position - tables - 1];
                } else if (position < 3 * tables + 1) {
                    metadata_ptr[position] = types_ptr[position - 2 * tables - 1];
                } else if (position < 4 * tables + 1) {
                    metadata_ptr[position] = placements_ptr[position - 3 * tables - 1];
                } else {
                    metadata_ptr[position] = offsets_ptr[position - 4 * tables - 1];
                }
            }
        });
    });
    const auto host_metadata = metadata.cpu();
    const auto* dims = host_metadata.const_data_ptr<int64_t>();
    const auto* starts = dims + tables + 1;
    const auto* types = starts + tables;
    const auto* placements = types + tables;
    const auto* boundaries = placements + tables;
    const int64_t dimension = int64_t(dims[1]) - dims[0];
    // The pinned FBGEMM IntNBit frontend documents embedding dimensions as
    // multiples of four, and its inference tests align every generated D to at
    // least that boundary. Keep the direct XPU entry point within the same
    // supported contract even though this scalar kernel and row padding could
    // represent some narrower byte-packed layouts.
    TORCH_CHECK(dims[0] == 0 && dimension > 0 && dimension % 4 == 0,
                op_name, "D must be positive and divisible by four");
    TORCH_CHECK(total_D.expect_int() == dims[tables], op_name, "total_D disagrees with D_offsets");
    TORCH_CHECK(indices.numel() <= (std::numeric_limits<int64_t>::max() - 255) / dimension,
                op_name, "output size overflow");
    TORCH_CHECK(boundaries[0] == 0 && boundaries[offsets.numel() - 1] == indices.numel(),
                op_name, "offsets must span all indices");
    for (int64_t boundary = 1; boundary < offsets.numel(); ++boundary) {
        TORCH_CHECK(boundaries[boundary] >= boundaries[boundary - 1] &&
                        boundaries[boundary] <= indices.numel(),
                    op_name, "offsets must be monotonic and within indices");
    }
    std::vector<int64_t> physical_offsets(starts, starts + tables);
    std::sort(physical_offsets.begin(), physical_offsets.end());
    TORCH_CHECK(physical_offsets.front() >= 0 && physical_offsets.back() <= dev_weights.numel(),
                op_name, "weights_offsets outside packed storage");
    std::vector<int64_t> strides(tables);
    std::vector<int64_t> storage_rows(tables);
    const int64_t batches = (offsets.numel() - 1) / tables;
    bool has_int4 = false;
    bool has_int8 = false;
    for (int64_t table = 0; table < tables; ++table) {
        TORCH_CHECK(int64_t(dims[table + 1]) - dims[table] == dimension,
                    op_name, "mixed dimensions are unsupported");
        TORCH_CHECK(placements[table] == 0, op_name, "only DEVICE placement is supported; no cache/UVM");
        TORCH_CHECK(types[table] == 2 || types[table] == 3,
                    op_name, "only INT4 and INT8 weights are supported");
        has_int4 |= types[table] == 3;
        has_int8 |= types[table] == 2;
        const int64_t row_bytes = dimension / (types[table] == 3 ? 2 : 1) + 4;
        strides[table] = ((row_bytes + alignment - 1) / alignment) * alignment;
        const auto next = std::upper_bound(physical_offsets.begin(), physical_offsets.end(), starts[table]);
        const int64_t storage_end = next == physical_offsets.end() ? dev_weights.numel() : *next;
        storage_rows[table] = (storage_end - starts[table]) / strides[table];
    }
    TORCH_CHECK(max_int4_D == (has_int4 ? dimension : 0) &&
                    max_int8_D == (has_int8 ? dimension : 0),
                op_name, "max_INT4/INT8_D disagrees with table dimensions");
    auto output = at::empty({indices.numel(), dimension}, dev_weights.options().dtype(output_type));
    if (indices.numel() == 0) {
        return output;
    }
    auto error = at::zeros({1}, dev_weights.options().dtype(at::kInt));
    auto* error_ptr = error.mutable_data_ptr<int32_t>();
    AT_DISPATCH_INDEX_TYPES(indices.scalar_type(), "int_nbit_lookup_xpu", [&] {
        for (int64_t table = 0; table < tables; ++table) {
            const auto* weights = dev_weights.const_data_ptr<uint8_t>() + starts[table];
            const auto* input = indices.const_data_ptr<index_t>();
            const int64_t begin = boundaries[table * batches];
            const int64_t end = boundaries[(table + 1) * batches];
            if (output_type == at::kFloat) {
                launch(queue, types[table], weights, input, output.mutable_data_ptr<float>(),
                       begin, end, dimension, strides[table], storage_rows[table], error_ptr);
            } else if (output_type == at::kHalf) {
                launch(queue, types[table], weights, input, output.mutable_data_ptr<at::Half>(),
                       begin, end, dimension, strides[table], storage_rows[table], error_ptr);
            } else {
                launch(queue, types[table], weights, input, output.mutable_data_ptr<at::BFloat16>(),
                       begin, end, dimension, strides[table], storage_rows[table], error_ptr);
            }
        }
    });
    const int32_t status = error.item<int32_t>();
    TORCH_CHECK(!(status & 1), op_name,
                "negative indices/pruning are unsupported; run bounds_check_indices first");
    TORCH_CHECK(!(status & 2), op_name,
                "index exceeds packed storage; run bounds_check_indices first");
    return output;
}

}

TORCH_LIBRARY_IMPL(fbgemm, XPU, library) {
    library.impl("int_nbit_split_embedding_codegen_lookup_function", TORCH_FN(int_nbit_lookup));
}

}
