# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright (c) 2025 Dmitry Rogozhkin.
# Copyright (c) 2026 Intel Corporation. All Rights Reserved.

import atexit
import ctypes
import importlib
import traceback

import torch
import torchcodec

try:
    # Note that version.py is generated during install
    from ._version import __version__
except ImportError:
    __version__ = "unknown"


def _get_extension_path(lib_name: str) -> str:
    spec = importlib.util.find_spec(lib_name)
    if spec is None or spec.origin is None:
        raise ImportError(f"No spec found for {lib_name}")
    return spec.origin

def load_torchcodec_xpu_shared_library():
    exceptions = []
    ffmpeg_major_version = torchcodec.ffmpeg_major_version
    xpu_library_name = f"torchcodec_xpu.xpu_ops{ffmpeg_major_version}"
    try:
        ctypes.CDLL(torchcodec.core_library_path)
        torch.ops.load_library(_get_extension_path(xpu_library_name))
        return
    except Exception:
        # Capture the full traceback for this exception
        exc_traceback = traceback.format_exc()
        exceptions.append((ffmpeg_major_version, exc_traceback))

    traceback_info = (
        "\n[start of libtorchcodec_xpu loading traceback]\n"
        + "\n".join(f"FFmpeg version {v}: {str(e)}" for v, e in exceptions)
        + "\n[end of libtorchcodec_xpu loading traceback]."
    )
    raise RuntimeError(
        f"""Could not load libtorchcodec_xpu. Likely causes:
          1. Missing dependencies. Such as FFmpeg, L0 or LibVA libraries.
          1. Intel extension for TorchCodec (libtorchcodec_xpu) is not compatible
             with this version of TorchCodec.
          2. The PyTorch version ({torch.__version__}) is not compatible with
             this version of TorchCodec.
          3. Another runtime dependency; see exceptions below.
        The following exceptions were raised as we tried to load libtorchcodec_xpu:
        """
        f"{traceback_info}"
    )

load_torchcodec_xpu_shared_library()


# Keep the CDLL handle alive at module level so the C function pointer used by
# atexit remains valid until the interpreter finishes shutting down.
_shutdown_lib = None


def _register_shutdown_hook():
    """Drain plugin-owned VAAPI contexts before torch tears down its XPU state.

    torchcodec_xpu keeps a global cache of VAAPI ``AVBufferRef`` handles for
    reuse. Their C++ static destructor would otherwise run at process exit in
    an order undefined relative to torch's XPU/Level Zero/UR teardown, which
    on Intel Arc / Battlemage typically segfaults inside libva /
    ``iHD_drv_video.so``. Because ``torchcodec_xpu`` is imported after
    ``torch``, this atexit callback runs first (Python invokes atexit handlers
    in reverse registration order), so VAAPI cleanup happens while the SYCL
    runtime is still alive.
    """
    global _shutdown_lib
    try:
        ffmpeg_major_version = torchcodec.ffmpeg_major_version
        lib_name = f"torchcodec_xpu.xpu_ops{ffmpeg_major_version}"
        lib_path = _get_extension_path(lib_name)
        _shutdown_lib = ctypes.CDLL(lib_path)
        shutdown = _shutdown_lib.torchcodec_xpu_shutdown
        shutdown.argtypes = []
        shutdown.restype = None
        atexit.register(shutdown)
    except Exception:
        # Best-effort: never let atexit registration break plugin load.
        pass


_register_shutdown_hook()

