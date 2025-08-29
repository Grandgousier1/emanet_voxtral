"""
config package - Configuration management for EMANET VOXTRAL
Contains application configuration and settings.
"""

from .app_config import *

# Import functions from the main config module for backward compatibility
try:
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from config import get_optimal_config, detect_hardware, setup_runpod_environment, get_vllm_args
except ImportError as e:
    # Fallback implementations if main config module is not available
    def get_optimal_config():
        raise ImportError("get_optimal_config not available")
    def detect_hardware():
        raise ImportError("detect_hardware not available")
    def setup_runpod_environment():
        raise ImportError("setup_runpod_environment not available")
    def get_vllm_args(model_name):
        raise ImportError("get_vllm_args not available")

__all__ = [
    'AppConfig',
    'get_config',
    'load_config',
    'get_optimal_config',
    'detect_hardware',
    'setup_runpod_environment',
    'get_vllm_args',
]