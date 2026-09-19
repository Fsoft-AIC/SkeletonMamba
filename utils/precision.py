"""CUDA autocast policy with FP32 model parameters."""
from contextlib import nullcontext

import torch


AMP_DTYPES = {"float16": torch.float16, "bfloat16": torch.bfloat16}


def amp_dtype(options):
    name = options.get("amp_dtype", "float16")  # Historical checkpoint default.
    if name not in AMP_DTYPES:
        raise ValueError("training.amp_dtype must be float16 or bfloat16")
    return AMP_DTYPES[name]


def check_cuda_precision(device, dtype):
    device = torch.device(device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("CUDA mixed precision requires an available CUDA device")
    if dtype == torch.bfloat16:
        with torch.cuda.device(device):
            if not torch.cuda.is_bf16_supported():
                raise RuntimeError("This GPU does not support bfloat16; select training.amp_dtype: float16")


def autocast_context(device, *, enabled, dtype):
    if not enabled:
        return nullcontext()
    return torch.autocast(device_type=torch.device(device).type, dtype=dtype)
