"""GPU backends: the kernel set is chosen from the device that holds q / k / v.

    gpu="auto"      SM90 (H100) -> "h100", SM100 (B200) -> "b200", SM120 (RTX 5090) -> "rtx5090"
    gpu="h100"      Triton kernels of `kernels.sm90`; needs compute capability 9.x
    gpu="b200"      tcgen05 / tensor-memory CuTe kernels; needs compute capability 10.x
    gpu="rtx5090"   Triton kernels of `kernels.sm120`; needs compute capability 12.x

The choice is made at the first attention call, so a pipeline built on the CPU (or offloaded) resolves it on the
device it actually runs on.
"""
import importlib

import torch

GPUS = {"h100": 9, "b200": 10, "rtx5090": 12}          # name -> compute capability major
_NAMES = {major: name for name, major in GPUS.items()}
_FA4 = {}


def check_name(gpu):
    if gpu not in ("auto", *GPUS):
        raise ValueError(f"gpu must be 'auto', 'h100', 'b200' or 'rtx5090', not {gpu!r}")


def resolve(gpu, device):
    """Backend name for `gpu` ("auto", "h100", "b200" or "rtx5090") on `device`; raises if they do not match."""
    check_name(gpu)
    device = torch.device(device)
    if device.type != "cuda":
        raise ValueError(f"EchoSparse runs on CUDA tensors; got a tensor on {device}")
    major, minor = torch.cuda.get_device_capability(device)
    found = _NAMES.get(major)
    if gpu == "auto":
        if found is None:
            raise RuntimeError(f"{device} has compute capability {major}.{minor}; EchoSparse supports SM90 (H100), "
                               f"SM100 (B200) and SM120 (RTX 5090) GPUs")
        return found
    if found != gpu:
        hint = f"use gpu='auto' or gpu='{found}'" if found else "no EchoSparse backend supports this GPU"
        raise RuntimeError(f"gpu='{gpu}' needs compute capability {GPUS[gpu]}.x, but {device} has {major}.{minor}; "
                           f"{hint}")
    return gpu


def kernels(name):
    """Kernel module of a backend (imported on first use: the B200 kernels need the SM100 CuTe toolchain)."""
    return importlib.import_module(f"echosparse.kernels.sm{GPUS[name]}0")


def flash_attn_func(device):
    """FlashAttention-4's forward on `device`. On SM120, FA4 4.0.0b15 needs the constructor fix of
    flash-attention PR #2484 (not merged): its SM80-derived SM120 kernel must keep the SM80 code paths."""
    fa4 = _FA4.get(device)
    if fa4 is None:
        from flash_attn.cute.interface import flash_attn_func as fa4
        if torch.cuda.get_device_capability(device)[0] == 12:
            _patch_fa4_sm120()
        _FA4[device] = fa4
    return fa4


def _patch_fa4_sm120():
    from importlib.metadata import PackageNotFoundError, version
    try:
        if version("flash-attn-4") != "4.0.0b15":
            return
    except PackageNotFoundError:
        return
    from cutlass.base_dsl.arch import Arch
    from flash_attn.cute.flash_fwd_sm120 import FlashAttentionForwardSm120 as cls
    if getattr(cls, "_echosparse_patched", False):
        return
    init = cls.__init__

    def patched_init(self, *args, **kwargs):
        init(self, *args, **kwargs)
        self.arch = Arch.sm_80
        self.is_split_kv = False
        self.pack_gqa = False

    cls.__init__ = patched_init
    cls._echosparse_patched = True
