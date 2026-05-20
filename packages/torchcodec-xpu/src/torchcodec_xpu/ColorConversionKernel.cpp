// Copyright (c) 2026 Intel Corporation. All Rights Reserved.

#ifdef WITH_SYCL_KERNELS

#include "ColorConversionKernel.h"
#include <algorithm> // For std::clamp

namespace facebook::torchcodec {

using float3x3 = std::array<sycl::float3, 3>;

// ============================================================
// Decoding matrices: YCbCr -> RGB  (used by NV12toRGBKernel)
// ============================================================
struct rgb_matrix {
  static constexpr float3x3 BT709 = {
    sycl::float3{ 1.0, 0.0, 1.5748 },
    sycl::float3{ 1.0, -0.187324, -0.468124 },
    sycl::float3{ 1.0, 1.8556, 0.0 }
  };

  static constexpr float3x3 BT601 = {
    sycl::float3{ 1.0, 0.0, 1.402 },
    sycl::float3{ 1.0, -0.344136, -0.714136 },
    sycl::float3{ 1.0, 1.772, 0.0}
  };
};

// ============================================================
// Encoding matrices: RGB -> YCbCr  (used by RGB24toNV12Kernel)
// Inverse of rgb_matrix above.
// Row 0: Y  coefficients
// Row 1: Cb coefficients
// Row 2: Cr coefficients
// ============================================================
struct yuv_matrix {
  static constexpr float3x3 BT709 = {
    sycl::float3{  0.2126f,    0.7152f,    0.0722f  },  // Y
    sycl::float3{ -0.1146f,   -0.3854f,    0.5f     },  // Cb
    sycl::float3{  0.5f,      -0.4542f,   -0.0458f  }   // Cr
  };
  static constexpr float3x3 BT601 = {
    sycl::float3{  0.299f,     0.587f,     0.114f   },  // Y
    sycl::float3{ -0.168736f, -0.331264f,  0.5f     },  // Cb
    sycl::float3{  0.5f,      -0.418688f, -0.081312f}   // Cr
  };
};

// Helper function for the Intel Tile-Y offset calculation
// Intel Y-Tiling uses COLUMN-MAJOR OWord (16 bytes) organization
// Tile: 128 bytes wide × 32 rows = 4KB
// Within tile: 8 OWords (16-byte columns) arranged column-by-column
// Each OWord covers all 32 rows before moving to next OWord
size_t get_tile_offset(int x, int y, int stride) {
  const int TileW = 128;  // Tile width in bytes
  const int TileH = 32;   // Tile height in rows
  const int OWordSize = 16; // OWord = 16 bytes
  const int TileSize = TileW * TileH;  // 4096 bytes per tile

  // Which tile does this pixel belong to?
  int tile_x = x / TileW;
  int tile_y = y / TileH;

  // Position within the tile
  int x_in_tile = x % TileW;
  int y_in_tile = y % TileH;

  // Block position added to remove swap of 64-byte blocks in the tile (TileY XOR pattern)
  int block_x = x_in_tile / 64;  // width of pixel blocks
  int block_y = y_in_tile / 4;   // heigh of pixel blocks

  // Y-Tiling: Column-major OWord layout
  // OWord index (0-7): which 16-byte column within the tile
  int oword_idx = x_in_tile / OWordSize;
  // Offset within OWord (0-15)
  int offset_in_oword = x_in_tile % OWordSize;

  int sub_tile_size = OWordSize * 4;
  int sub_tile_y = y_in_tile / 4;
  int y_in_sub_tile = y_in_tile % 4;

  // conditional to remove swap of 64-byte blocks in the tile (TileY XOR pattern)
  if ((block_x ^ block_y ) & 0x1){
    block_x ^= 1;
    block_y ^= 1;

    x_in_tile = block_x * 64 + (x_in_tile % 64);
    y_in_tile = block_y * 4 + (y_in_tile % 4);

    sub_tile_y = block_y;
    y_in_sub_tile = y_in_tile % 4;

    oword_idx = x_in_tile / OWordSize;
    offset_in_oword = x_in_tile % 16;
  }

  int offset_in_tile = (sub_tile_y * TileW/OWordSize + oword_idx) * sub_tile_size + y_in_sub_tile * OWordSize + offset_in_oword;

  // Number of tiles per row
  int stride_in_tiles = stride / TileW;

  // Final tiled offset
  size_t tile_offset = (size_t)(tile_y * stride_in_tiles + tile_x) * TileSize;
  return tile_offset + offset_in_tile;
}

sycl::uchar3 yuv2rgb(uint8_t y, uint8_t u, uint8_t v, bool fullrange, const float3x3 &rgb_matrix) {
  sycl::float3 src;
  if (fullrange) {
    src = sycl::float3(y/255.0f, (u-128.0f)/255.0f - 0.5f, (v-128.0f)/255.0f - 0.5f);
  } else {
    src = sycl::float3((y-16.0f)/219.0f, (u-128.0f)/224.0f, (v-128.0f)/224.0f);
  }

  sycl::float3 fdst;
  fdst.x() = sycl::dot(src, rgb_matrix[0]);
  fdst.y() = sycl::dot(src, rgb_matrix[1]);
  fdst.z() = sycl::dot(src, rgb_matrix[2]);

  sycl::uchar3 dst;
  dst.x() = (uint8_t)std::clamp(fdst[0] * 255.0f, 0.0f, 255.0f);
  dst.y() = (uint8_t)std::clamp(fdst[1] * 255.0f, 0.0f, 255.0f);
  dst.z() = (uint8_t)std::clamp(fdst[2] * 255.0f, 0.0f, 255.0f);
  return dst;
}

struct NV12toRGBKernel {
  const uint8_t* y_plane;
  const uint8_t* uv_plane;
  uint8_t* rgb_output;
  int width;
  int height;
  int stride;
  bool fullrange;
  const float3x3 rgb_matrix;

  NV12toRGBKernel(
      const uint8_t* y_plane,
      const uint8_t* uv_plane,
      uint8_t* rgb_output,
      int width,
      int height,
      int stride,
      bool fullrange,
      const float3x3 &rgb_matrix):
    y_plane(y_plane),
    uv_plane(uv_plane),
    rgb_output(rgb_output),
    width(width),
    height(height),
    stride(stride),
    fullrange(fullrange),
    rgb_matrix(rgb_matrix)
  {}

  void operator()(sycl::id<2> idx) const {
    int yx = idx[1];
    int yy = idx[0];

    if (yx >= width || yy >= height) {
      return;
    }

    int ux = sycl::floor(yx/2.0);
    int uy = sycl::floor(yy/2.0);

    size_t tiled_idx_y = get_tile_offset(yx, yy, stride);
    size_t tiled_idx_u = get_tile_offset(2*ux, uy, stride);
    size_t tiled_idx_v = get_tile_offset(2*ux+1, uy, stride);

    uint8_t y = y_plane[tiled_idx_y];
    uint8_t u = uv_plane[tiled_idx_u];
    uint8_t v = uv_plane[tiled_idx_v];

    sycl::uchar3 rgb = yuv2rgb(y, u, v, fullrange, rgb_matrix);

    int rgb_idx = 3 * (yy * width + yx);

    rgb_output[rgb_idx + 0] = rgb.x();
    rgb_output[rgb_idx + 1] = rgb.y();
    rgb_output[rgb_idx + 2] = rgb.z();
  }
};

const float3x3 getColorConversionMatrix(enum AVColorSpace colorspace) {
  if (colorspace == AVCOL_SPC_BT709) {
      return rgb_matrix::BT709;
  }
  return rgb_matrix::BT601;
}

const float3x3 getYUVConversionMatrix(enum AVColorSpace colorspace) {
  if (colorspace == AVCOL_SPC_BT709) {
    return yuv_matrix::BT709;
  }
  return yuv_matrix::BT601;
}

void convertNV12ToRGB(
    sycl::queue& queue,
    const uint8_t* y_plane,
    const uint8_t* uv_plane,
    uint8_t* rgb_output,
    int width,
    int height,
    int stride,
    enum AVColorRange color_range,
    enum AVColorSpace colorspace) {
  bool fullrange = (color_range == AVCOL_RANGE_JPEG);
  queue.submit([&](sycl::handler& cgh) {
    NV12toRGBKernel kernel(
      y_plane, uv_plane, rgb_output,
      width, height, stride,
      fullrange, getColorConversionMatrix(colorspace));

    cgh.parallel_for(
        sycl::range<2>(height, width),
        kernel);
  });

  queue.wait();
}

// This function is called during library initialization to ensure
// the SYCL runtime registers the kernel associated with this type.
void registerColorConversionKernel() {
  // Creating a dummy pointer to the kernel type is often enough
  // to force the compiler to emit the necessary RTTI/integration info.
  // We use volatile to prevent optimization.
  volatile size_t s = sizeof(NV12toRGBKernel);
  (void)s;
}

// ============================================================
// Encoding kernel: NCHW RGB tensor -> NV12 VAAPI surface
// ============================================================
struct RGB24toNV12Kernel {
  const uint8_t* rgb_nchw;  // CHW uint8 device pointer (R, G, B planes)
  int64_t ch_stride;        // stride between channel planes
  int64_t row_stride;       // stride between rows within a plane
  int64_t pixel_stride;     // stride between adjacent pixels (1 for NCHW, 3 for HWC-permuted)
  uint8_t* y_plane;
  uint8_t* uv_plane;
  int width;
  int height;
  int y_pitch;              // surface Y-plane row pitch in bytes
  int uv_pitch;             // surface UV-plane row pitch in bytes
  bool is_tiled;            // true → Tile-Y; false → linear
  bool fullrange;
  float3x3 yuv_mat;

  RGB24toNV12Kernel(
      const uint8_t* rgb_nchw_,
      int64_t ch_stride_,
      int64_t row_stride_,
      int64_t pixel_stride_,
      uint8_t* y_plane_,
      uint8_t* uv_plane_,
      int width_,
      int height_,
      int y_pitch_,
      int uv_pitch_,
      bool is_tiled_,
      bool fullrange_,
      const float3x3& yuv_mat_)
    : rgb_nchw(rgb_nchw_),
      ch_stride(ch_stride_),
      row_stride(row_stride_),
      pixel_stride(pixel_stride_),
      y_plane(y_plane_),
      uv_plane(uv_plane_),
      width(width_),
      height(height_),
      y_pitch(y_pitch_),
      uv_pitch(uv_pitch_),
      is_tiled(is_tiled_),
      fullrange(fullrange_),
      yuv_mat(yuv_mat_)
  {}

  void operator()(sycl::id<2> idx) const {
    int x = idx[1];
    int y = idx[0];

    if (x >= width || y >= height) {
      return;
    }

    // Read RGB from NCHW tensor.
    float r = rgb_nchw[0 * ch_stride + y * row_stride + x * pixel_stride] / 255.0f;
    float g = rgb_nchw[1 * ch_stride + y * row_stride + x * pixel_stride] / 255.0f;
    float b = rgb_nchw[2 * ch_stride + y * row_stride + x * pixel_stride] / 255.0f;
    sycl::float3 src{r, g, b};

    // Luma Y — write to Tile-Y or linear destination
    float Y_norm = sycl::dot(src, yuv_mat[0]);
    float Y = fullrange ? Y_norm * 255.0f : 16.0f + Y_norm * 219.0f;
    size_t y_dst = is_tiled ? get_tile_offset(x, y, y_pitch)
                            : (size_t)y * y_pitch + x;
    y_plane[y_dst] = (uint8_t)std::clamp(Y, 0.0f, 255.0f);

    // Chroma UV: one pair per 2x2 block (NV12 4:2:0 subsampling).
    if ((x % 2 == 0) && (y % 2 == 0)) {
      float Cb_norm = sycl::dot(src, yuv_mat[1]);
      float Cr_norm = sycl::dot(src, yuv_mat[2]);
      float U = fullrange ? Cb_norm * 255.0f + 128.0f : 128.0f + Cb_norm * 224.0f;
      float V = fullrange ? Cr_norm * 255.0f + 128.0f : 128.0f + Cr_norm * 224.0f;
      size_t u_dst = is_tiled ? get_tile_offset(x,     y / 2, uv_pitch)
                              : (size_t)(y / 2) * uv_pitch + x;
      size_t v_dst = is_tiled ? get_tile_offset(x + 1, y / 2, uv_pitch)
                              : (size_t)(y / 2) * uv_pitch + x + 1;
      uv_plane[u_dst] = (uint8_t)std::clamp(U, 0.0f, 255.0f);
      uv_plane[v_dst] = (uint8_t)std::clamp(V, 0.0f, 255.0f);
    }
  }
};

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
    enum AVColorSpace colorspace) {
  bool fullrange = (color_range == AVCOL_RANGE_JPEG);
  queue.submit([&](sycl::handler& cgh) {
    RGB24toNV12Kernel kernel(
        rgb_nchw, ch_stride, row_stride, pixel_stride,
        dst_y, dst_uv,
        width, height,
        y_pitch, uv_pitch,
        is_tiled,
        fullrange, getYUVConversionMatrix(colorspace));
    cgh.parallel_for(sycl::range<2>(height, width), kernel);
  });
  queue.wait();
}

} // namespace facebook::torchcodec
#endif // WITH_SYCL_KERNELS
