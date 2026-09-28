/*
 * Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
 * Copyright (c) 2026 Intel Corporation. All Rights Reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */

////////////////////////////////////////////////////////////////////////////////
// SYCL PORT MAPPING TO FBGEMM CUDA SOURCE - INT4/INT8 NO-BAG LOOKUP
////////////////////////////////////////////////////////////////////////////////
//
// ORIGINAL CUDA SOURCE:
//   Template:
//     fbgemm_gpu/codegen/inference/
//       embedding_forward_quantized_split_nbit_kernel_template.cu
//   Generated kernels:
//     INT4_split_embedding_nobag_codegen_forward_unweighted_kernel_small_L
//     INT8_split_embedding_nobag_codegen_forward_unweighted_kernel_small_L
//
// KERNEL MAPPING:
//   LookupInt4NobagKernel
//     -> INT4_split_embedding_nobag_codegen_forward_unweighted_kernel_small_L
//   LookupInt8NobagKernel
//     -> INT8_split_embedding_nobag_codegen_forward_unweighted_kernel_small_L
//
// INTENTIONAL XPU SUBSET AND STRUCTURAL DIFFERENCES:
//   - Uses one work-item per output element and scalar packed-value loads,
//     rather than the CUDA warp-per-bag vectorized kernel.
//   - Supports DEVICE placement, unweighted no-bag lookup, prefix FP16
//     scale/bias parameters, and uniform dimensions only.
//   - Uses a capped one-dimensional launch with a grid-stride loop.
//   - Reports negative or physical out-of-storage rows through a device error
//     bitmask instead of a device assertion.
//
////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <algorithm>
#include <cstdint>
#include <sycl/sycl.hpp>

namespace fbgemm_xpu {

template <typename index_t, typename output_t>
class LookupInt{{ bit_rate }}NobagKernel {
public:
    LookupInt{{ bit_rate }}NobagKernel(
        const uint8_t* weights,
        const index_t* indices,
        output_t* output,
        int64_t begin,
        int64_t dimension,
        int64_t row_stride,
        int64_t storage_rows,
        int32_t* error,
        int64_t elements,
        int64_t work_items)
        : weights_(weights),
          indices_(indices),
          output_(output),
          begin_(begin),
          dimension_(dimension),
          row_stride_(row_stride),
          storage_rows_(storage_rows),
          error_(error),
          elements_(elements),
          work_items_(work_items) {}

    void operator()(const sycl::nd_item<1>& item) const {
        for (int64_t element = item.get_global_linear_id();
             element < elements_;
             element += work_items_) {
            const int64_t position = begin_ + element / dimension_;
            const int64_t column = element % dimension_;
            const int64_t row_index = indices_[position];
            if (row_index < 0 || row_index >= storage_rows_) {
                if (column == 0) {
                    sycl::atomic_ref<int32_t, sycl::memory_order::relaxed,
                                     sycl::memory_scope::device,
                                     sycl::access::address_space::global_space>
                        error_ref(*error_);
                    error_ref.fetch_or(row_index < 0 ? 1 : 2);
                }
                continue;
            }
            const uint8_t* row = weights_ + row_index * row_stride_;
            const uint16_t scale_bits = uint16_t(row[0]) | (uint16_t(row[1]) << 8);
            const uint16_t bias_bits = uint16_t(row[2]) | (uint16_t(row[3]) << 8);
            const float scale = static_cast<float>(sycl::bit_cast<sycl::half>(scale_bits));
            const float bias = static_cast<float>(sycl::bit_cast<sycl::half>(bias_bits));
            const uint8_t packed = row[4 + column / {{ 8 // bit_rate }}];
            const int quantized = (packed >> ((column % {{ 8 // bit_rate }}) * {{ bit_rate }})) & {{ (2 ** bit_rate) - 1 }};
            output_[position * dimension_ + column] =
                static_cast<output_t>(sycl::fma(static_cast<float>(quantized), scale, bias));
        }
    }

private:
    const uint8_t* weights_;
    const index_t* indices_;
    output_t* output_;
    int64_t begin_;
    int64_t dimension_;
    int64_t row_stride_;
    int64_t storage_rows_;
    int32_t* error_;
    int64_t elements_;
    int64_t work_items_;
};

template <typename index_t, typename output_t>
void lookup_int{{ bit_rate }}_nobag(
    sycl::queue& queue,
    const uint8_t* weights,
    const index_t* indices,
    output_t* output,
    int64_t begin,
    int64_t end,
    int64_t dimension,
    int64_t row_stride,
    int64_t storage_rows,
    int32_t* error) {
    const int64_t elements = (end - begin) * dimension;
    if (elements == 0) {
        return;
    }
    const int64_t work_items = std::min<int64_t>(
        ((elements + 255) / 256) * 256, 4096 * 256);
    queue.parallel_for<LookupInt{{ bit_rate }}NobagKernel<index_t, output_t>>(
        sycl::nd_range<1>(sycl::range<1>(work_items), sycl::range<1>(256)),
        LookupInt{{ bit_rate }}NobagKernel<index_t, output_t>(
            weights, indices, output, begin, dimension, row_stride,
            storage_rows, error, elements, work_items));
}

}
