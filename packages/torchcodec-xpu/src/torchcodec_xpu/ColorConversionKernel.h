// Copyright (c) 2026 Intel Corporation. All Rights Reserved.

#pragma once

#ifdef WITH_SYCL_KERNELS

#include <sycl/sycl.hpp>
#include <cstdint>

extern "C" {
#include <libavutil/pixfmt.h>
}

namespace facebook::torchcodec {

void convertNV12ToRGB(
    sycl::queue& queue,
    const uint8_t* y_plane,
    const uint8_t* uv_plane,
    uint8_t* rgb_output,
    int width,
    int height,
    int stride,
    enum AVColorRange color_range,
    enum AVColorSpace colorspace);

// Encoding: NCHW uint8 RGB tensor (on XPU) -> NV12 VAAPI surface.
// is_tiled: true for Intel Tile-Y surfaces (drm_format_modifier != 0), false for linear.
void convertRGBToNV12(
    sycl::queue& queue,
    const uint8_t* rgb_nchw,
    int64_t ch_stride,
    int64_t row_stride,
    int64_t pixel_stride,
    uint8_t* dst_y,
    uint8_t* dst_uv,
    int width,
    int height,
    int y_pitch,
    int uv_pitch,
    bool is_tiled,
    enum AVColorRange color_range,
    enum AVColorSpace colorspace);

// Anchor function to force kernel registration
void registerColorConversionKernel();

} // namespace facebook::torchcodec

#endif // WITH_SYCL_KERNELS
