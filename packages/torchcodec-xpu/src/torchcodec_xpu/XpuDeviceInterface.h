// Copyright (c) 2025 Dmitry Rogozhkin.
// Copyright (c) 2026 Intel Corporation. All Rights Reserved.

#pragma once

#include "DeviceInterface.h"
#include "FilterGraph.h"

namespace facebook::torchcodec {

class XpuDeviceInterface : public DeviceInterface {
 public:
  XpuDeviceInterface(const StableDevice& device);

  virtual ~XpuDeviceInterface();

  std::optional<const AVCodec*> find_codec(
      const AVCodecID& codec_id,
      bool is_decoder = true) override;

  void initialize(const SharedAVCodecContext& codec_context) override;

  void initialize_video(
      const AVStream* av_stream,
      const UniqueDecodingAVFormatContext& av_format_ctx,
      const VideoStreamOptions& video_stream_options,
      const std::vector<std::unique_ptr<Transform>>&
          transforms,
      const std::optional<FrameDims>& resized_output_dims)
      override;

  void register_hardware_device_with_codec(
      AVCodecContext* codec_context) override;

  // ---- Decoding overrides ----
  void convert_av_frame_to_frame_output(
      UniqueAVFrame& av_frame,
      FrameOutput& frame_output,
      std::optional<torch::stable::Tensor> pre_allocated_output_tensor =
          std::nullopt) override;

  // ---- Encoding overrides ----
  UniqueAVFrame convert_tensor_to_av_frame_for_encoding(
      const torch::stable::Tensor& tensor,
      int frame_index,
      AVCodecContext* codec_context) override;

  AVPixelFormat get_encoding_pixel_format(
      const AVCodec& av_codec,
      const std::optional<std::string>& user_pixel_format) const override;

  void setup_hardware_frame_context_for_encoding(
      AVCodecContext* codec_context) override;

 private:
  // We sometimes encounter frames that cannot be decoded on the XPU device.
  // Rather than erroring out, we decode them on the CPU.
  std::unique_ptr<DeviceInterface> cpu_interface_;

  VideoStreamOptions video_stream_options_;
  AVRational time_base_;
  bool has_fp64_;

  UniqueAVBufferRef ctx_;

  std::unique_ptr<FilterGraph> filter_graph_;

  // Used to know whether a new FilterGraphContext should
  // be created before decoding a new frame.
  FiltersConfig prev_filters_config_;

  // Optimized conversion. Return value indicates if conversion was
  // successfull.
  bool convert_av_frame_to_frame_output_sycl(
      UniqueAVFrame& av_frame,
      torch::stable::Tensor& dst);
  // Fallback conversion if optimized path is not available.
  void convert_av_frame_to_frame_output_filter_graph(
      UniqueAVFrame& av_frame,
      torch::stable::Tensor& dst);

  // ---- Encoding helpers ----
  // SYCL path: exports VAAPI surface as DMA-BUF, imports via Level Zero USM,
  // runs convertRGBToNV12 directly on the surface. Returns null when SYCL
  // is unavailable.
  UniqueAVFrame convert_tensor_to_av_frame_for_encoding_sycl(
      const torch::stable::Tensor& tensor,
      int frame_index,
      AVCodecContext* codec_context);
  // CPU fallback: moves tensor to CPU, uses libswscale GBRP->NV12,
  // then av_hwframe_transfer_data to upload into the VAAPI surface.
  UniqueAVFrame convert_tensor_to_av_frame_for_encoding_cpu(
      const torch::stable::Tensor& tensor,
      int frame_index,
      AVCodecContext* codec_context);
};

} // namespace facebook::torchcodec
