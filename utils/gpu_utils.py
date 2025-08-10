"""utils/gpu_utils.py
Helpers for GPU memory management and preflight checks.
"""
from typing import Optional
import torch


def check_cuda_available() -> bool:
    return torch.cuda.is_available()


def available_device() -> str:
    return 'cuda' if check_cuda_available() else 'cpu'


def free_cuda_mem():
    if check_cuda_available():
        torch.cuda.empty_cache()


def gpu_mem_info() -> Optional[dict]:
    if not check_cuda_available():
        return None
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return { 'total': info.total, 'free': info.free, 'used': info.used }
    except Exception:
        return None
