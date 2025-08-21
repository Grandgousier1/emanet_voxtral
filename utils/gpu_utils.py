"""utils/gpu_utils.py
Helpers for GPU memory management and preflight checks.
"""
import torch
from typing import Dict


def check_cuda_available() -> bool:
    """Check if CUDA is available for PyTorch."""
    return torch.cuda.is_available()


def available_device() -> str:
    """Return the best available device for computation.
    
    Returns:
        Device string: 'cuda:0' if CUDA available, 'cpu' otherwise.
    """
    if torch.cuda.is_available():
        return f"cuda:{torch.cuda.current_device()}"
    return "cpu"


def free_cuda_mem() -> None:
    """Free CUDA memory cache if CUDA is available."""
    if check_cuda_available():
        torch.cuda.empty_cache()


def gpu_mem_info() -> Dict[str, int]:
    """Return GPU memory information for device 0."""
    if not check_cuda_available():
        return {}
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return { 'total': info.total, 'free': info.free, 'used': info.used }
    except (ImportError, AttributeError, RuntimeError) as e:
        # Log specific error for debugging but don't crash
        import logging
        logging.getLogger(__name__).debug(f"GPU memory info failed: {e}")
        return {}
