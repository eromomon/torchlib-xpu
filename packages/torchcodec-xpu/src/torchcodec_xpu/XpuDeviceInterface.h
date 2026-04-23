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

  std::optional<const AVCodec*> findCodec(
      const AVCodecID& codecId,
      bool isDecoder = true) override;

  void initialize(
      const AVStream* avStream,
      const UniqueDecodingAVFormatContext& avFormatCtx,
      const SharedAVCodecContext& codecContext) override;

  void initializeVideo(
      const VideoStreamOptions& videoStreamOptions,
      [[maybe_unused]] const std::vector<std::unique_ptr<Transform>>&
          transforms,
      [[maybe_unused]] const std::optional<FrameDims>& resizedOutputDims)
      override;

  void registerHardwareDeviceWithCodec(AVCodecContext* codecContext) override;

  void convertAVFrameToFrameOutput(
      UniqueAVFrame& avFrame,
      FrameOutput& frameOutput,
      std::optional<torch::stable::Tensor> preAllocatedOutputTensor =
          std::nullopt) override;

 private:
  VideoStreamOptions videoStreamOptions_;
  AVRational timeBase_;
  bool has_fp64_;

  // VAAPI render-node path used for this XPU device (e.g. /dev/dri/renderD128
  // or a PCI-BDF-indexed path). Captured at construction so it can appear in
  // diagnostic logs emitted later by the CPU-fallback path.
  std::string renderD_;

  // Per-stream flag set by registerHardwareDeviceWithCodec(). Stays false if
  // VAAPI is unavailable, if the codec has no VAAPI decode support, or if the
  // first decoded frame reveals a silent FFmpeg SW fallback.
  bool hwDecodeActiveForCurrentStream_ = false;

  UniqueAVBufferRef ctx_;

  std::unique_ptr<FilterGraph> filterGraph_;

  // Used to know whether a new FilterGraphContext should
  // be created before decoding a new frame.
  FiltersConfig prevFiltersConfig_;

  // Optimized conversion. Return value indicates if conversion was
  // successfull.
  bool convertAVFrameToFrameOutput_SYCL(
      UniqueAVFrame& avFrame,
      torch::stable::Tensor& dst);
  // Fallback conversion if optimized path is not available.
  void convertAVFrameToFrameOutput_FilterGraph(
      UniqueAVFrame& avFrame,
      torch::stable::Tensor& dst);
};

} // namespace facebook::torchcodec
