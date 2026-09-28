/*
 * Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
 * Copyright (c) 2026 Intel Corporation. All Rights Reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <functional>
#include <numeric>

#include <ATen/Dispatch.h>

#include "fbgemm_utils/utils.h"
#include "jagged_common.h"
#include "jagged_to_padded_dense_forward.h"

namespace fbgemm_xpu {

at::Tensor jagged_to_padded_dense_forward_xpu(
    const at::Tensor& values,
    const std::vector<at::Tensor>& offsets,
    c10::SymIntArrayRef max_lengths,
    const double padding_value) {
    const size_t num_jagged_dim = offsets.size();
    TORCH_CHECK(
        max_lengths.size() == num_jagged_dim,
        "max_lengths.size(), ",
        max_lengths.size(),
        " != num_jagged_dim, ",
        num_jagged_dim);
    SYCL_DEVICE_GUARD(values);

    const at::Tensor values_canonicalized = values.view(
        {values.size(0),
         std::accumulate(
             values.sizes().begin() + 1,
             values.sizes().end(),
             1,
             std::multiplies<size_t>())});

    at::SymDimVector padded_values_shape({at::SymInt(offsets[0].size(0) - 1)});
    padded_values_shape.insert(
        padded_values_shape.end(), max_lengths.begin(), max_lengths.end());

    // Canonicalize padded_values by unsqueezing the last dim if the inner dense
    // dimension is 1 and folded.
    const bool D_folded = values.dim() == 1;
    if (!D_folded) {
        padded_values_shape.push_back(values.size(-1));
    }
    at::Tensor padded_values =
        at::empty_symint(padded_values_shape, values.options());
    at::Tensor padded_values_view =
        D_folded ? padded_values.unsqueeze(-1) : padded_values;

    FBGEMM_DISPATCH_ALL_TYPES(
        values.scalar_type(), "jagged_to_padded_dense", [&] {
            jagged_dense_elementwise_dense_output_<scalar_t>(
                values_canonicalized,
                offsets,
                padded_values_view,
                padded_values_view,
                JaggedOpCopyX(),
                static_cast<scalar_t>(padding_value));
        });

    return padded_values;
}

TORCH_LIBRARY_IMPL(fbgemm, XPU, m) {
    m.impl(
        "jagged_to_padded_dense_forward",
        &jagged_to_padded_dense_forward_xpu);
}

}  // namespace fbgemm_xpu
