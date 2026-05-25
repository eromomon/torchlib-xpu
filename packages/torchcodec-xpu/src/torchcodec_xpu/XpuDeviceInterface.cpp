// Copyright (c) 2025 Dmitry Rogozhkin.
// Copyright (c) 2026 Intel Corporation. All Rights Reserved.

#include <unistd.h>
#include <stdlib.h>
#include <string>
#include <unordered_map>

#include <level_zero/ze_api.h>
#include <va/va_drmcommon.h>

#include <ATen/DLConvertor.h>
#include <c10/xpu/XPUStream.h>

#include "ColorConversionKernel.h"
#include "Cache.h"
#include "FFMPEGCommon.h"
#include "XpuDeviceInterface.h"

extern "C" {
#include <libavutil/hwcontext_vaapi.h>
#include <libavutil/pixdesc.h>
#include <libswscale/swscale.h>
}

namespace facebook::torchcodec {

namespace xpu {

const char* USE_SYCL_KERNELS = std::getenv("USE_SYCL_KERNELS");
const char* FORCE_CPU_FALLBACK = std::getenv("FORCE_CPU_FALLBACK");

static bool g_xpu = registerDeviceInterface(
    DeviceInterfaceKey(StableDeviceType::XPU),
    [](const StableDevice& device) { return new XpuDeviceInterface(device); });

const int MAX_XPU_GPUS = 128;
// Set to -1 to have an infinitely sized cache. Set it to 0 to disable caching.
// Set to a positive number to have a cache of that size.
const int MAX_CONTEXTS_PER_GPU_IN_CACHE = -1;
PerGpuCache<AVBufferRef, Deleterp<AVBufferRef, void, av_buffer_unref>>
    g_cached_hw_device_ctxs(MAX_XPU_GPUS, MAX_CONTEXTS_PER_GPU_IN_CACHE);

inline bool to_bool(std::string str) {
    static const std::unordered_map<std::string, bool> bool_map = {
        {"1", true},  {"0", false},
        {"on", true}, {"off", false},
        {"true", true}, {"false", false}
    };

    auto it = bool_map.find(str);
    if (it != bool_map.end()) {
        return it->second;
    }
    return false;
}

inline bool use_sycl_color_conversion_kernel() {
#ifndef WITH_SYCL_KERNELS
  return false;
#else
  if (!USE_SYCL_KERNELS) {
    return true;
  }
  return to_bool(USE_SYCL_KERNELS);
#endif
}

inline bool force_cpu_fallback() {
  if (!FORCE_CPU_FALLBACK) {
    return false;
  }
  return to_bool(FORCE_CPU_FALLBACK);
}

bool has_fp64(const StableDevice& device) {
  int deviceIndex = getDeviceIndex(device);
  sycl::device syclDevice = c10::xpu::get_raw_device(deviceIndex);
  return syclDevice.has(sycl::aspect::fp64);
}

sycl::ext::oneapi::experimental::architecture getArchitecture(
    const StableDevice& device) {
  sycl::queue queue = c10::xpu::getCurrentXPUStream(device.index());
  return queue.get_device().get_info<
      sycl::ext::oneapi::experimental::info::device::architecture>();
}

// Resolves the VAAPI render-node path this XPU device should open.
std::string resolveRenderD(const StableDevice& device) {
  std::string renderD = "/dev/dri/renderD128";
  int deviceIndex = getDeviceIndex(device);
  sycl::device syclDevice = c10::xpu::get_raw_device(deviceIndex);
  if (syclDevice.has(sycl::aspect::ext_intel_pci_address)) {
    auto BDF =
        syclDevice.get_info<sycl::ext::intel::info::device::pci_address>();
    renderD = "/dev/dri/by-path/pci-" + BDF + "-render";
  }
  return renderD;
}

UniqueAVBufferRef getVaapiContext(const StableDevice& device) {
  enum AVHWDeviceType type = av_hwdevice_find_type_by_name("vaapi");
  TORCH_CHECK(type != AV_HWDEVICE_TYPE_NONE, "Failed to find vaapi device");

  UniqueAVBufferRef hw_device_ctx = g_cached_hw_device_ctxs.get(device);
  if (hw_device_ctx) {
    return hw_device_ctx;
  }

  std::string renderD = resolveRenderD(device);

  AVBufferRef* ctx = nullptr;
  int err = av_hwdevice_ctx_create(&ctx, type, renderD.c_str(), nullptr, 0);
  if (err < 0) {
    TORCH_CHECK(
        false,
        "Failed to create specified HW device: ",
        getFFMPEGErrorStringFromErrorCode(err));
  }
  return UniqueAVBufferRef(ctx);
}

torch::stable::Tensor allocateEmptyHWCTensor(
    const FrameDims& frameDims,
    const StableDevice& device) {
  STD_TORCH_CHECK(
      frameDims.height > 0, "height must be > 0, got: ", frameDims.height);
  STD_TORCH_CHECK(
      frameDims.width > 0, "width must be > 0, got: ", frameDims.width);
  return torch::stable::empty(
      {frameDims.height, frameDims.width, 3},
      kStableUInt8,
      std::nullopt,
      device);
}

} // namespace xpu

int getDeviceIndex(const StableDevice& device) {
  // PyTorch uses int8_t as its torch::DeviceIndex, but FFmpeg and XPU
  // libraries use int. So we use int, too.
  int deviceIndex = static_cast<int>(device.index());
  TORCH_CHECK(
      deviceIndex >= -1 && deviceIndex < xpu::MAX_XPU_GPUS,
      "Invalid device index = ",
      deviceIndex);

  return (deviceIndex == -1)? 0: deviceIndex;
}

XpuDeviceInterface::XpuDeviceInterface(const StableDevice& device)
    : DeviceInterface(device) {
  TORCH_CHECK(xpu::g_xpu, "XpuDeviceInterface was not registered!");
  TORCH_CHECK(
      device_.type() == kStableXPU, "Unsupported device: must be XPU");

  // It is important for pytorch itself to create the xpu context. If ffmpeg
  // creates the context it may not be compatible with pytorch.
  // This is a dummy tensor to initialize the xpu context.
  torch::stable::Tensor dummyTensorForXpuInitialization = torch::stable::empty(
      {1}, kStableUInt8, std::nullopt, StableDevice(device));

  auto arch = xpu::getArchitecture(device);
  if (!xpu::force_cpu_fallback()) {
    // Checking for devices which don't have HW media engines so we can skip
    // initialization of VAAPI context.
    if (arch != sycl::ext::oneapi::experimental::architecture::intel_gpu_pvc &&
      arch != sycl::ext::oneapi::experimental::architecture::intel_gpu_pvc_vg) {
      ctx_ = xpu::getVaapiContext(device_);
    }
  }

  if (xpu::use_sycl_color_conversion_kernel()) {
    VLOG(1) << "XpuDeviceInterface initialized with SYCL kernel backend";
    VLOG(1) << "Backend: SYCL_KERNEL (Direct NV12→RGB)";
  } else {
    VLOG(1) << "XpuDeviceInterface initialized with VAAPI filter graph backend";
    VLOG(1) << "Backend: VAAPI_FILTER (Flexible, with scaling)";
  }

  has_fp64_ = xpu::has_fp64(device);
  VLOG(1) << "Device supports FP64: " << has_fp64_;
}

XpuDeviceInterface::~XpuDeviceInterface() {
  if (ctx_) {
    xpu::g_cached_hw_device_ctxs.addIfCacheHasCapacity(device_, std::move(ctx_));
  }
}

void XpuDeviceInterface::initialize(const SharedAVCodecContext& codecContext) {
  codecContext_ = codecContext;
}

void XpuDeviceInterface::initializeVideo(
    const AVStream* avStream,
    const UniqueDecodingAVFormatContext& avFormatCtx,
    const VideoStreamOptions& videoStreamOptions,
    [[maybe_unused]] const std::vector<std::unique_ptr<Transform>>& transforms,
    [[maybe_unused]] const std::optional<FrameDims>& resizedOutputDims) {
  TORCH_CHECK(avStream != nullptr, "avStream is null");
  timeBase_ = avStream->time_base;
  videoStreamOptions_ = videoStreamOptions;

  cpuInterface_ = createDeviceInterface(kStableCPU);
  STD_TORCH_CHECK(
      cpuInterface_ != nullptr, "Failed to create CPU device interface");
  cpuInterface_->initialize(codecContext_);
  cpuInterface_->initializeVideo(
      avStream,
      avFormatCtx,
      VideoStreamOptions(),
      {},
      /*resizedOutputDims=*/std::nullopt);
}

void XpuDeviceInterface::registerHardwareDeviceWithCodec(
    AVCodecContext* codecContext) {
  if (!ctx_) {
    VLOG(1) << "HW context not initialized, falling back to CPU";
    return;
  }
  TORCH_CHECK(codecContext != nullptr, "codecContext is null");
  codecContext->hw_device_ctx = av_buffer_ref(ctx_.get());
}

VADisplay getVaDisplayFromAV(AVFrame* avFrame) {
  AVHWFramesContext* hwfc = (AVHWFramesContext*)avFrame->hw_frames_ctx->data;
  AVHWDeviceContext* hwdc = hwfc->device_ctx;
  AVVAAPIDeviceContext* vactx = (AVVAAPIDeviceContext*)hwdc->hwctx;
  return vactx->display;
}

struct xpuManagerCtx {
  UniqueAVFrame avFrame;
  ze_context_handle_t zeCtx = nullptr;
};

void deleter(DLManagedTensor* self) {
  std::unique_ptr<DLManagedTensor> tensor(self);
  std::unique_ptr<xpuManagerCtx> context((xpuManagerCtx*)self->manager_ctx);
  zeMemFree(context->zeCtx, self->dl_tensor.data);
  free(self->dl_tensor.shape);
  free(self->dl_tensor.strides);
}

torch::stable::Tensor AVFrameToTensor(
    const StableDevice& device,
    const UniqueAVFrame& frame) {
  TORCH_CHECK_EQ(frame->format, AV_PIX_FMT_VAAPI);

  VADRMPRIMESurfaceDescriptor desc{};

  VAStatus sts = vaExportSurfaceHandle(
      getVaDisplayFromAV(frame.get()),
      (VASurfaceID)(uintptr_t)frame->data[3],
      VA_SURFACE_ATTRIB_MEM_TYPE_DRM_PRIME_2,
      VA_EXPORT_SURFACE_READ_ONLY,
      &desc);
  TORCH_CHECK(
      sts == VA_STATUS_SUCCESS,
      "vaExportSurfaceHandle failed: ",
      vaErrorStr(sts));

  TORCH_CHECK(desc.num_objects == 1, "Expected 1 fd, got ", desc.num_objects);
  TORCH_CHECK(desc.num_layers == 1, "Expected 1 layer, got ", desc.num_layers);
  TORCH_CHECK(
      desc.layers[0].num_planes == 1,
      "Expected 1 plane, got ",
      desc.layers[0].num_planes);

  std::unique_ptr<xpuManagerCtx> context = std::make_unique<xpuManagerCtx>();
  ze_device_handle_t ze_device{};
  sycl::queue queue = c10::xpu::getCurrentXPUStream(device.index());

  queue
      .submit([&](sycl::handler& cgh) {
        cgh.host_task([&](const sycl::interop_handle& ih) {
          context->zeCtx =
              ih.get_native_context<sycl::backend::ext_oneapi_level_zero>();
          ze_device =
              ih.get_native_device<sycl::backend::ext_oneapi_level_zero>();
        });
      })
      .wait();

  ze_external_memory_import_fd_t import_fd_desc{};
  import_fd_desc.stype = ZE_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMPORT_FD;
  import_fd_desc.flags = ZE_EXTERNAL_MEMORY_TYPE_FLAG_DMA_BUF;
  import_fd_desc.fd = desc.objects[0].fd;

  ze_device_mem_alloc_desc_t alloc_desc{};
  alloc_desc.pNext = &import_fd_desc;
  void* usm_ptr = nullptr;

  ze_result_t res = zeMemAllocDevice(
      context->zeCtx, &alloc_desc, desc.objects[0].size, 0, ze_device, &usm_ptr);
  TORCH_CHECK(
      res == ZE_RESULT_SUCCESS, "Failed to import fd=", desc.objects[0].fd);

  close(desc.objects[0].fd);

  std::unique_ptr<DLManagedTensor> dl_dst = std::make_unique<DLManagedTensor>();
  int64_t* shape = (int64_t*)malloc(3*sizeof(int64_t));

  shape[0] = frame->height;
  shape[1] = frame->width;
  shape[2] = 4;

  context->avFrame.reset(av_frame_alloc());
  TORCH_CHECK(context->avFrame.get(), "Failed to allocate AVFrame");

  int status = av_frame_ref(context->avFrame.get(), frame.get());
  TORCH_CHECK(
      status >= 0,
      "Failed to reference AVFrame: ",
      getFFMPEGErrorStringFromErrorCode(status));

  dl_dst->manager_ctx = context.release();
  dl_dst->deleter = deleter;
  dl_dst->dl_tensor.data = usm_ptr;
  dl_dst->dl_tensor.device.device_type = kDLOneAPI;
  dl_dst->dl_tensor.device.device_id = device.index();
  dl_dst->dl_tensor.ndim = 3;
  dl_dst->dl_tensor.dtype.code = kDLUInt;
  dl_dst->dl_tensor.dtype.bits = 8;
  dl_dst->dl_tensor.dtype.lanes = 1;
  dl_dst->dl_tensor.shape = shape;
  dl_dst->dl_tensor.strides = nullptr;
  dl_dst->dl_tensor.byte_offset = desc.layers[0].offset[0];

  // torch::stable::Tensor(AtenTensorHandle) constructor steals the ownership, so
  // we need to release at::Tensor after getting its handle.
  auto dst = std::make_unique<at::Tensor>(at::fromDLPack(dl_dst.release()));
  // From: https://github.com/pytorch/pytorch/blob/v2.11.0/torch/csrc/inductor/aoti_torch/utils.h#L32
  // NOTE: this conversion migth get broken in the new PyTorch release which will
  // likely result in the segmentation fault. If this will happen - update the conversion.
  AtenTensorHandle dst_handle = reinterpret_cast<AtenTensorHandle>(dst.release());
  return torch::stable::Tensor(dst_handle);
}

VADisplay getVaDisplayFromAV(UniqueAVFrame& avFrame) {
  AVHWFramesContext* hwfc = (AVHWFramesContext*)avFrame->hw_frames_ctx->data;
  AVHWDeviceContext* hwdc = hwfc->device_ctx;
  AVVAAPIDeviceContext* vactx = (AVVAAPIDeviceContext*)hwdc->hwctx;
  return vactx->display;
}

void XpuDeviceInterface::convertAVFrameToFrameOutput(
    UniqueAVFrame& avFrame,
    FrameOutput& frameOutput,
    std::optional<torch::stable::Tensor> preAllocatedOutputTensor) {
  if (avFrame->format != AV_PIX_FMT_VAAPI) {
    // The frame's format is AV_PIX_FMT_VAAPI if and only if its content is on
    // the GPU. In this branch, the frame is on the CPU. This is what FFmpeg VAAPI
    // decoder gives us if it wasn't able to decode a frame, for whatever reason.
    // Typically that happens if the video's decoder isn't supported by VAAPI in
    // general or on this particular device. In this case we have a frame on the
    // CPU. We send the frame back to the XPU device when we're done.

    FrameOutput cpuFrameOutput;
    cpuInterface_->convertAVFrameToFrameOutput(avFrame, cpuFrameOutput);

    // Finally, we need to send the frame back to the GPU. Note that the
    // pre-allocated tensor is on the GPU, so we can't send that to the CPU
    // device interface. We copy it over here.
    if (preAllocatedOutputTensor.has_value()) {
      torch::stable::copy_(preAllocatedOutputTensor.value(), cpuFrameOutput.data);
      frameOutput.data = preAllocatedOutputTensor.value();
    } else {
      frameOutput.data = torch::stable::to(cpuFrameOutput.data, device_);
    }
    return;
  }

  TORCH_CHECK(
      avFrame->format == AV_PIX_FMT_VAAPI,
      "Expected format to be AV_PIX_FMT_VAAPI, got " +
          std::string(av_get_pix_fmt_name((AVPixelFormat)avFrame->format)));
  auto frameDims = FrameDims(avFrame->height, avFrame->width);
  torch::stable::Tensor& dst = frameOutput.data;
  if (preAllocatedOutputTensor.has_value()) {
    auto shape = preAllocatedOutputTensor.value().sizes();
    TORCH_CHECK(
        (shape.size() == 3) && (shape[0] == frameDims.height) &&
	    (shape[1] == frameDims.width) && (shape[2] == 3),
        "Expected tensor of shape ",
        frameDims.height,
        "x",
        frameDims.width,
        "x3, got ",
        intArrayRefToString(shape));
    dst = preAllocatedOutputTensor.value();
  } else {
    // Explicitly load the version defined in facebook::torchcodec::xpu
    // namespace as facebook::torchcodec defines the same but with the linkage
    // type which we can't use.
    dst = xpu::allocateEmptyHWCTensor(frameDims, device_);
  }

  auto start = std::chrono::high_resolution_clock::now();
  if (!convertAVFrameToFrameOutput_SYCL(avFrame, dst)) {
    convertAVFrameToFrameOutput_FilterGraph(avFrame, dst);
  }

  auto end = std::chrono::high_resolution_clock::now();

  std::chrono::duration<double, std::micro> duration = end - start;
  VLOG(9) << "Conversion of frame height=" << frameDims.height << " width=" << frameDims.width
          << " took: " << duration.count() << "us" << std::endl;
}

void XpuDeviceInterface::convertAVFrameToFrameOutput_FilterGraph(
    UniqueAVFrame& avFrame,
    torch::stable::Tensor& dst) {
  VLOG(1) << "Using VAAPI filter graph backend for conversion";
  auto frameDims = FrameDims(avFrame->height, avFrame->width);

  // We need to compare the current frame context with our previous frame
  // context. If they are different, then we need to re-create our colorspace
  // conversion objects. We create our colorspace conversion objects late so
  // that we don't have to depend on the unreliable metadata in the header.
  // And we sometimes re-create them because it's possible for frame
  // resolution to change mid-stream. Finally, we want to reuse the colorspace
  // conversion objects as much as possible for performance reasons.
  enum AVPixelFormat frameFormat =
      static_cast<enum AVPixelFormat>(avFrame->format);
  FiltersConfig filtersConfig;

  filtersConfig.inputWidth = avFrame->width;
  filtersConfig.inputHeight = avFrame->height;
  filtersConfig.inputFormat = frameFormat;
  filtersConfig.inputAspectRatio = avFrame->sample_aspect_ratio;
  // Actual output color format will be set via filter options
  filtersConfig.outputFormat = AV_PIX_FMT_VAAPI;
  filtersConfig.timeBase = timeBase_;
  filtersConfig.hwFramesCtx.reset(av_buffer_ref(avFrame->hw_frames_ctx));

  std::stringstream filters;
  filters << "scale_vaapi=" << frameDims.width << ":" << frameDims.height;
  // CPU device interface outputs RGB in full (pc) color range.
  // We are doing the same to match.
  filters << ":format=rgba:out_range=pc";

  filtersConfig.filtergraphStr = filters.str();

  if (!filterGraph_ || prevFiltersConfig_ != filtersConfig) {
    filterGraph_ =
        std::make_unique<FilterGraph>(filtersConfig, videoStreamOptions_);
    prevFiltersConfig_ = std::move(filtersConfig);
  }

  // We convert input to the RGBX color format with VAAPI getting WxHx4
  // tensor on the output.
  UniqueAVFrame filteredAVFrame = filterGraph_->convert(avFrame);

  TORCH_CHECK_EQ(filteredAVFrame->format, AV_PIX_FMT_VAAPI);

  torch::stable::Tensor dst_rgb4 = AVFrameToTensor(device_, filteredAVFrame);
  torch::stable::copy_(dst, torch::stable::narrow(dst_rgb4, 2, 0, 3));
}

bool XpuDeviceInterface::convertAVFrameToFrameOutput_SYCL(
    [[maybe_unused]] UniqueAVFrame& frame,
    [[maybe_unused]] torch::stable::Tensor& dst) {
  bool converted = false;
  if (!xpu::use_sycl_color_conversion_kernel()) {
    return converted;
  }
  if (!has_fp64_) {
    return converted;
  }

#ifdef WITH_SYCL_KERNELS
  VLOG(1) << "Using SYCL kernel backend for conversion";
  TORCH_CHECK_EQ(frame->format, AV_PIX_FMT_VAAPI);
  VADRMPRIMESurfaceDescriptor desc{};
  VAStatus sts = vaExportSurfaceHandle(
      getVaDisplayFromAV(frame.get()),
      (VASurfaceID)(uintptr_t)frame->data[3],
      VA_SURFACE_ATTRIB_MEM_TYPE_DRM_PRIME_2,
      VA_EXPORT_SURFACE_READ_ONLY,
      &desc);
  TORCH_CHECK(
      sts == VA_STATUS_SUCCESS,
      "vaExportSurfaceHandle failed: ",
      vaErrorStr(sts));

  TORCH_CHECK(desc.num_objects == 1, "Expected 1 fd, got ", desc.num_objects);
  std::unique_ptr<xpuManagerCtx> context = std::make_unique<xpuManagerCtx>();
  ze_device_handle_t ze_device{};
  sycl::queue queue = c10::xpu::getCurrentXPUStream(device_.index());
  queue
      .submit([&](sycl::handler& cgh) {
        cgh.host_task([&](const sycl::interop_handle& ih) {
          context->zeCtx =
              ih.get_native_context<sycl::backend::ext_oneapi_level_zero>();
          ze_device =
              ih.get_native_device<sycl::backend::ext_oneapi_level_zero>();
        });
      })
      .wait();

  ze_external_memory_import_fd_t import_fd_desc{};
  import_fd_desc.stype = ZE_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMPORT_FD;
  import_fd_desc.flags = ZE_EXTERNAL_MEMORY_TYPE_FLAG_DMA_BUF;
  import_fd_desc.fd = desc.objects[0].fd;

  ze_device_mem_alloc_desc_t alloc_desc{};
  alloc_desc.pNext = &import_fd_desc;
  void* usm_ptr = nullptr;

  ze_result_t res = zeMemAllocDevice(
      context->zeCtx,
      &alloc_desc,
      desc.objects[0].size,
      0,
      ze_device,
      &usm_ptr);
  TORCH_CHECK(
      res == ZE_RESULT_SUCCESS, "Failed to import fd=", desc.objects[0].fd);

  close(desc.objects[0].fd);

  convertNV12ToRGB(
      queue,
      (uint8_t*)usm_ptr + desc.layers[0].offset[0],
      (uint8_t*)usm_ptr + desc.layers[1].offset[0],
      (uint8_t*)dst.data_ptr(),
      frame->width,
      frame->height,
      desc.layers[0].pitch[0],
      frame->color_range,
      frame->colorspace);

  zeMemFree(context->zeCtx, usm_ptr);
  converted = true;
#endif
  return converted;
}

// inspired by https://github.com/FFmpeg/FFmpeg/commit/ad67ea9
// we have to do this because of an FFmpeg bug where hardware decoding is not
// appropriately set, so we just go off and find the matching codec for the XPU
// device
std::optional<const AVCodec*> XpuDeviceInterface::findCodec(
    const AVCodecID& codecId,
    bool isDecoder) {
  void* i = nullptr;
  const AVCodec* codec = nullptr;
  while ((codec = av_codec_iterate(&i)) != nullptr) {
    if (isDecoder) {
      if (codec->id != codecId || !av_codec_is_decoder(codec)) {
        continue;
      }
    } else {
      if (codec->id != codecId || !av_codec_is_encoder(codec)) {
        continue;
      }
    }

    const AVCodecHWConfig* config = nullptr;
    for (int j = 0; (config = avcodec_get_hw_config(codec, j)) != nullptr;
         ++j) {
      if (config->device_type == AV_HWDEVICE_TYPE_VAAPI) {
        return codec;
      }
    }
  }

  return std::nullopt;
}

// ============================================================
// Encoding: setupHardwareFrameContextForEncoding
// ============================================================
// Allocates a VAAPI hw_frames_ctx on the codec context so the encoder
// can write directly into VAAPI surfaces (NV12 layout, VAAPI wrapper).
void XpuDeviceInterface::setupHardwareFrameContextForEncoding(
    AVCodecContext* codecContext) {
  TORCH_CHECK(
      ctx_,
      "VAAPI hw device context is not initialized. "
      "This device may not have a media engine (e.g. PVC/Ponte Vecchio). "
      "Encoding via XPU is only supported on devices with VAAPI.");
  TORCH_CHECK(codecContext != nullptr, "codecContext is null");

  AVBufferRef* hwFramesCtxRef = av_hwframe_ctx_alloc(ctx_.get());
  TORCH_CHECK(
      hwFramesCtxRef != nullptr,
      "Failed to allocate VAAPI hw frames context for codec");

  // sw_pix_fmt: the software (CPU-accessible) format the encoder consumes inside the surface
  // pix_fmt:    the hardware wrapper format the codec sees (must match hw_frames_ctx->format)
  codecContext->sw_pix_fmt = AV_PIX_FMT_NV12;
  codecContext->pix_fmt    = AV_PIX_FMT_VAAPI;

  auto* hwFramesCtx = reinterpret_cast<AVHWFramesContext*>(hwFramesCtxRef->data);
  hwFramesCtx->format    = AV_PIX_FMT_VAAPI;
  hwFramesCtx->sw_format = AV_PIX_FMT_NV12;
  hwFramesCtx->width     = codecContext->width;
  hwFramesCtx->height    = codecContext->height;

  int ret = av_hwframe_ctx_init(hwFramesCtxRef);
  if (ret < 0) {
    av_buffer_unref(&hwFramesCtxRef);
    TORCH_CHECK(
        false,
        "Failed to initialize VAAPI hw frames context: ",
        getFFMPEGErrorStringFromErrorCode(ret));
  }
  codecContext->hw_frames_ctx = hwFramesCtxRef;
}

// ============================================================
// Encoding: convertTensorToAVFrameForEncoding
// ============================================================
UniqueAVFrame XpuDeviceInterface::convertTensorToAVFrameForEncoding(
    const torch::stable::Tensor& tensor,
    int frameIndex,
    AVCodecContext* codecContext) {
  TORCH_CHECK(
      tensor.dim() == 3 && tensor.sizes()[0] == 3,
      "Expected CHW tensor with 3 channels (RGB), got shape: ",
      tensor.sizes()[0], "x", tensor.sizes()[1], "x", tensor.sizes()[2]);
  TORCH_CHECK(codecContext != nullptr, "codecContext is null");
  TORCH_CHECK(
      codecContext->hw_frames_ctx != nullptr,
      "hw_frames_ctx is null: call setupHardwareFrameContextForEncoding first");

  UniqueAVFrame vaFrame(av_frame_alloc());
  TORCH_CHECK(vaFrame != nullptr, "Failed to allocate AVFrame for encoding");
  vaFrame->format = AV_PIX_FMT_VAAPI;
  vaFrame->height = static_cast<int>(tensor.sizes()[1]);
  vaFrame->width  = static_cast<int>(tensor.sizes()[2]);
  vaFrame->pts    = frameIndex;

  // Allocate a VAAPI surface from the hw_frames_ctx pool created in
  // setupHardwareFrameContextForEncoding.
  int ret = av_hwframe_get_buffer(codecContext->hw_frames_ctx, vaFrame.get(), 0);
  TORCH_CHECK(
      ret >= 0,
      "av_hwframe_get_buffer failed: ",
      getFFMPEGErrorStringFromErrorCode(ret));

#ifdef WITH_SYCL_KERNELS
  if (xpu::use_sycl_color_conversion_kernel()) {
    VLOG(9) << "[XPU Encoder] Encoding frame " << frameIndex
            << " via SYCL on device=xpu:" << device_.index();
    return encodeConvert_SYCL(tensor, codecContext, std::move(vaFrame));
  }
#endif
  VLOG(9) << "[XPU Encoder] Encoding frame " << frameIndex << " via CPU fallback";
  return encodeConvert_CPU(tensor, codecContext, std::move(vaFrame));
}

// ============================================================
// Encoding: encodeConvert_SYCL
// ============================================================
UniqueAVFrame XpuDeviceInterface::encodeConvert_SYCL(
    const torch::stable::Tensor& tensor,
    AVCodecContext* codecContext,
    UniqueAVFrame vaFrame) {
#ifdef WITH_SYCL_KERNELS
  VADisplay display = getVaDisplayFromAV(vaFrame.get());
  VASurfaceID surfaceId = (VASurfaceID)(uintptr_t)vaFrame->data[3];

  VADRMPRIMESurfaceDescriptor desc{};
  VAStatus sts = vaExportSurfaceHandle(
      display,
      surfaceId,
      VA_SURFACE_ATTRIB_MEM_TYPE_DRM_PRIME_2,
      VA_EXPORT_SURFACE_WRITE_ONLY,  // write for encoding (vs. READ_ONLY for decoding)
      &desc);
  TORCH_CHECK(
      sts == VA_STATUS_SUCCESS,
      "vaExportSurfaceHandle (WRITE_ONLY) failed: ",
      vaErrorStr(sts));
  TORCH_CHECK(desc.num_objects == 1, "Expected 1 DMA-BUF object, got ", desc.num_objects);
  // NV12 surfaces can be exported in two valid layouts depending on the driver:
  //  Layout A: 1 layer,  2 planes  — layers[0].planes[0]=Y, layers[0].planes[1]=UV
  //  Layout B: 2 layers, 1 plane each — layers[0].planes[0]=Y, layers[1].planes[0]=UV
  const bool layoutA = (desc.num_layers == 1 && desc.layers[0].num_planes == 2);
  const bool layoutB = (desc.num_layers == 2 && desc.layers[0].num_planes == 1
                        && desc.layers[1].num_planes == 1);
  TORCH_CHECK(
      layoutA || layoutB,
      "Unsupported NV12 export layout: num_layers=", desc.num_layers,
      " layers[0].num_planes=", desc.layers[0].num_planes);
  // Get Level Zero context and device handles via SYCL interop.
  sycl::queue queue = c10::xpu::getCurrentXPUStream(device_.index());
  ze_context_handle_t zeCtx  = nullptr;
  ze_device_handle_t  zeDevice = nullptr;
  queue
      .submit([&](sycl::handler& cgh) {
        cgh.host_task([&](const sycl::interop_handle& ih) {
          zeCtx    = ih.get_native_context<sycl::backend::ext_oneapi_level_zero>();
          zeDevice = ih.get_native_device<sycl::backend::ext_oneapi_level_zero>();
        });
      })
      .wait();

  ze_external_memory_import_fd_t import_fd_desc{};
  import_fd_desc.stype = ZE_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMPORT_FD;
  import_fd_desc.flags = ZE_EXTERNAL_MEMORY_TYPE_FLAG_DMA_BUF;
  import_fd_desc.fd    = desc.objects[0].fd;

  ze_device_mem_alloc_desc_t alloc_desc{};
  alloc_desc.pNext = &import_fd_desc;
  void* usm_ptr = nullptr;
  ze_result_t res = zeMemAllocDevice(
      zeCtx, &alloc_desc, desc.objects[0].size, 0, zeDevice, &usm_ptr);
  TORCH_CHECK(
      res == ZE_RESULT_SUCCESS,
      "zeMemAllocDevice failed importing encode surface fd=",
      desc.objects[0].fd);

  // Extract Y and UV plane pointers and pitches for both layouts
  uint8_t* y_ptr;
  uint8_t* uv_ptr;
  int y_pitch, uv_pitch;
  if (layoutA) {
    y_ptr    = static_cast<uint8_t*>(usm_ptr) + desc.layers[0].offset[0];
    uv_ptr   = static_cast<uint8_t*>(usm_ptr) + desc.layers[0].offset[1];
    y_pitch  = static_cast<int>(desc.layers[0].pitch[0]);
    uv_pitch = static_cast<int>(desc.layers[0].pitch[1]);
  } else {
    y_ptr    = static_cast<uint8_t*>(usm_ptr) + desc.layers[0].offset[0];
    uv_ptr   = static_cast<uint8_t*>(usm_ptr) + desc.layers[1].offset[0];
    y_pitch  = static_cast<int>(desc.layers[0].pitch[0]);
    uv_pitch = static_cast<int>(desc.layers[1].pitch[0]);
  }

  // drm_format_modifier != 0 means tiled (e.g. Intel Tile-Y on BMG/Gen12+).
  const bool is_tiled = (desc.objects[0].drm_format_modifier != 0);
  convertRGBToNV12(
      queue,
      static_cast<const uint8_t*>(tensor.data_ptr()),
      tensor.strides()[0],   // ch_stride
      tensor.strides()[1],   // row_stride
      tensor.strides()[2],   // pixel_stride
      y_ptr,
      uv_ptr,
      vaFrame->width,
      vaFrame->height,
      y_pitch,
      uv_pitch,
      is_tiled,
      codecContext->color_range,
      codecContext->colorspace);

  zeMemFree(zeCtx, usm_ptr);
  close(desc.objects[0].fd);

  vaFrame->colorspace  = codecContext->colorspace;
  vaFrame->color_range = codecContext->color_range;
  return vaFrame;
#else
  return encodeConvert_CPU(tensor, codecContext, std::move(vaFrame));
#endif
}

// ============================================================
// Encoding: encodeConvert_CPU  (CPU fallback)
// ============================================================
UniqueAVFrame XpuDeviceInterface::encodeConvert_CPU(
    const torch::stable::Tensor& tensor,
    AVCodecContext* codecContext,
    UniqueAVFrame vaFrame) {
  // Move XPU tensor to CPU (blocking)
  torch::stable::Tensor cpuTensor =
      torch::stable::to(tensor, StableDevice(kStableCPU, 0));

  const uint8_t* data = static_cast<const uint8_t*>(cpuTensor.data_ptr());
  // strides() are in elements (uint8), so they equal byte strides here.
  int64_t ch_stride  = cpuTensor.strides()[0];
  int64_t row_stride = cpuTensor.strides()[1];

  // Allocate an intermediate CPU NV12 frame for sws_scale output
  UniqueAVFrame cpuFrame(av_frame_alloc());
  TORCH_CHECK(cpuFrame != nullptr, "Failed to allocate CPU NV12 AVFrame");
  cpuFrame->format = AV_PIX_FMT_NV12;
  cpuFrame->width  = vaFrame->width;
  cpuFrame->height = vaFrame->height;
  int ret = av_frame_get_buffer(cpuFrame.get(), 0);
  TORCH_CHECK(ret >= 0, "av_frame_get_buffer (NV12) failed: ",
              getFFMPEGErrorStringFromErrorCode(ret));

  // Zero-copy GBRP view of the NCHW tensor (GBRP plane order: G=ch1, B=ch2, R=ch0).
  UniqueAVFrame gbrpFrame(av_frame_alloc());
  TORCH_CHECK(gbrpFrame != nullptr, "Failed to allocate GBRP AVFrame");
  gbrpFrame->format = AV_PIX_FMT_GBRP;
  gbrpFrame->width  = vaFrame->width;
  gbrpFrame->height = vaFrame->height;
  gbrpFrame->data[0] = const_cast<uint8_t*>(data + 1 * ch_stride);  // G
  gbrpFrame->data[1] = const_cast<uint8_t*>(data + 2 * ch_stride);  // B
  gbrpFrame->data[2] = const_cast<uint8_t*>(data + 0 * ch_stride);  // R
  gbrpFrame->linesize[0] = static_cast<int>(row_stride);
  gbrpFrame->linesize[1] = static_cast<int>(row_stride);
  gbrpFrame->linesize[2] = static_cast<int>(row_stride);

  // GBRP -> NV12 via libswscale
  SwsContext* swsCtx = sws_getContext(
      vaFrame->width, vaFrame->height, AV_PIX_FMT_GBRP,
      vaFrame->width, vaFrame->height, AV_PIX_FMT_NV12,
      SWS_BILINEAR, nullptr, nullptr, nullptr);
  TORCH_CHECK(swsCtx != nullptr, "sws_getContext(GBRP->NV12) failed");
  sws_scale(
      swsCtx,
      gbrpFrame->data,
      gbrpFrame->linesize,
      0,
      vaFrame->height,
      cpuFrame->data,
      cpuFrame->linesize);
  sws_freeContext(swsCtx);

  // Upload CPU NV12 -> VAAPI surface
  ret = av_hwframe_transfer_data(vaFrame.get(), cpuFrame.get(), 0);
  TORCH_CHECK(
      ret >= 0,
      "av_hwframe_transfer_data (NV12->VAAPI) failed: ",
      getFFMPEGErrorStringFromErrorCode(ret));

  vaFrame->colorspace  = codecContext->colorspace;
  vaFrame->color_range = codecContext->color_range;
  return vaFrame;
}

} // namespace facebook::torchcodec
