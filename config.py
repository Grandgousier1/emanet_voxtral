"""
config.py - RunPod B200 optimized configuration
Hardware: 1x B200 (180GB VRAM), 188GB RAM, 28 vCPU, 80GB disk
"""

import torch
import os
import logging
from typing import Dict, Any, Optional, TypedDict, List, Union
from pathlib import Path
from constants import BYTES_TO_GB

logger = logging.getLogger(__name__)


class HardwareInfo(TypedDict):
    """Hardware configuration information."""
    gpu_count: int
    gpu_memory_gb: List[float]
    total_ram_gb: float
    cpu_count: int
    is_b200: bool


class VLLMConfig(TypedDict):
    """vLLM configuration parameters."""
    gpu_memory_utilization: float
    tensor_parallel_size: int
    max_model_len: int
    trust_remote_code: bool
    dtype: str
    quantization: Optional[str]
    engine_use_ray: bool
    disable_log_stats: bool
    max_num_batched_tokens: int
    max_num_seqs: int
    semaphore_limit: int


class SystemConfig(TypedDict):
    """Complete system configuration."""
    vllm: VLLMConfig
    audio: Dict[str, Union[int, float]]
    memory: Dict[str, Union[int, float, bool]]
    disk: Dict[str, Union[int, float, str, bool]]
    validation: Dict[str, Union[int, float]]
    performance: Dict[str, bool]

from rich.prompt import Prompt, Confirm
from cli_feedback import get_feedback

# Cache for hardware detection
_hardware_cache: Optional[HardwareInfo] = None

def setup_interactive_configuration():
    """Guides the user to create a .env file if it doesn't exist."""
    env_path = Path('.env')
    feedback = get_feedback()
    if env_path.exists():
        return

    feedback.info("Le fichier de configuration .env est manquant.")
    if Confirm.ask("Voulez-vous en créer un maintenant ?", default=True):
        hf_token = Prompt.ask("Entrez votre token Hugging Face (laissez vide si non requis)", password=True, default=None)
        
        with open(env_path, 'w', encoding='utf-8') as f:
            if hf_token:
                # Encrypt token before storing for security
                try:
                    from utils.auth_manager import TokenManager
                    token_manager = TokenManager(feedback)
                    encrypted_token = token_manager._encrypt_token(hf_token)
                    f.write(f'HF_TOKEN="{encrypted_token}"\n')
                except ImportError:
                    # Fallback to plain text if encryption not available
                    feedback.warning("Token encryption not available, storing in plain text")
                    f.write(f'HF_TOKEN="{hf_token}"\n')
        feedback.success(f"Fichier .env créé avec succès.")

# Hardware detection
def detect_hardware() -> HardwareInfo:
    """Detect and return hardware configuration with caching."""
    global _hardware_cache
    
    # Return cached result if available
    if _hardware_cache is not None:
        return _hardware_cache
    
    config = {
        'gpu_count': 0,
        'gpu_memory_gb': [],
        'total_ram_gb': 0,
        'cpu_count': os.cpu_count() or 1,
        'is_b200': False
    }
    
    # Safe CUDA detection with error handling
    try:
        if torch.cuda.is_available():
            config['gpu_count'] = torch.cuda.device_count()
            for i in range(config['gpu_count']):
                props = torch.cuda.get_device_properties(i)
                memory_gb = props.total_memory / BYTES_TO_GB
                config['gpu_memory_gb'].append(memory_gb)
                
                # Detect B200 (roughly 180GB VRAM)
                if memory_gb > 170:
                    config['is_b200'] = True
    except (RuntimeError, AttributeError, OSError) as e:
        import logging
        logging.getLogger(__name__).warning(f"CUDA detection failed: {e}")
    
    # Estimate RAM from system
    try:
        import psutil
        config['total_ram_gb'] = psutil.virtual_memory().total / BYTES_TO_GB
    except ImportError:
        config['total_ram_gb'] = 188  # RunPod default
    
    # Cache the result
    _hardware_cache = config
    return config



# B200 optimized configurations
B200_CONFIG = {
    # vLLM optimization for B200
    'vllm': {
        # Allocate 95% of VRAM to vLLM, leaving a small buffer for other processes.
        'gpu_memory_utilization': 0.95,
        # A single B200 is powerful enough, so we don't need tensor parallelism.
        'tensor_parallel_size': 1,
        # Utilize the full 32k context length of the Voxtral model.
        'max_model_len': 32768,
        'trust_remote_code': True,
        # bfloat16 is the optimal dtype for B200 GPUs, offering a good balance of performance and precision.
        'dtype': 'bfloat16',
        # With 180GB of VRAM, quantization is not necessary.
        'quantization': None,
        # For a single GPU setup, Ray is not needed.
        'engine_use_ray': False,
        'disable_log_stats': False,
        # A large batch size for batched tokens is crucial for maximizing throughput on the B200.
        'max_num_batched_tokens': 8192,
        # Allow a high number of sequences to be processed in parallel.
        'max_num_seqs': 64,  # Process many sequences in parallel
        'semaphore_limit': 2, # Limit concurrent GPU batches to 2 for the B200
    },
    
    # Audio processing optimization
    'audio': {
        # A large batch size for audio processing to leverage the B200's power.
        'batch_size': 32,
        'max_segment_length': 30.0,
        'overlap_length': 2.0,
        'sample_rate': 16000,
        # Dedicate half of the 28 vCPUs to audio processing for parallel execution.
        'parallel_workers': 14,
    },
    
    # Memory management
    'memory': {
        # Allocate a significant amount of RAM for model caching to speed up model loading.
        'cache_size_gb': 20,
        'audio_cache_max_entries': 500,
        # A large buffer for audio data to ensure smooth processing.
        'audio_buffer_gb': 10,
        # With a limited 80GB disk, aggressive cleanup is necessary.
        'work_dir_cleanup': True,
        # Unified GPU memory cleanup intervals for all components
        'gpu_memory_cleanup_interval_segments': 5,  # Cleanup every N segments in main processing
        'gpu_memory_cleanup_interval_batches': 5,   # Cleanup every N batches in parallel processing
        'gpu_memory_cleanup_forced_interval': 60,   # Force cleanup every N seconds
    },
    
    # Disk management (80GB constraint)
    'disk': {
        # Limit the size of the working directory to avoid filling up the disk.
        'max_work_dir_size_gb': 15,
        'temp_audio_cleanup': True,
        # Use the temporary directory for the model cache to avoid filling up the persistent disk.
        'model_cache_dir': '/tmp/hf_cache',
        # Limit concurrent downloads to avoid overwhelming the disk I/O.
        'max_concurrent_downloads': 2,
    },
    
    # File validation limits (security and resource protection)
    'validation': {
        # Maximum file sizes for security and resource protection
        'max_audio_file_size_gb': 5.0,  # Large enough for long videos but prevents abuse
        'max_batch_file_size_mb': 10.0,  # Batch files should be small text files
        'max_srt_output_size_mb': 50.0,  # Reasonable subtitle file size limit
        
        # Timeout limits to prevent resource exhaustion
        'download_timeout_seconds': 1800,  # 30 minutes max for downloads
        'processing_timeout_seconds': 3600,  # 1 hour max for processing
        'model_load_timeout_seconds': 600,   # 10 minutes max for model loading
        'vad_timeout_seconds': 900,          # 15 minutes max for VAD
        'parallel_task_timeout_seconds': 300, # 5 minutes per parallel batch
        
        # Rate limiting
        'max_concurrent_processes': 3,       # Limit concurrent video processing
        'min_processing_interval_seconds': 5, # Minimum time between processing starts
        
        # Content validation
        'min_audio_duration_seconds': 1.0,   # Minimum audio length
        'max_audio_duration_hours': 12.0,    # Maximum audio length (12 hours)
        'max_segments_per_file': 10000,      # Prevent excessive segmentation
    },
    
    # Performance optimization
    'performance': {
        # With 188GB of RAM, we can safely prefetch models to speed up initialization.
        'prefetch_models': True,
        'parallel_vad': True,
        'async_audio_loading': True,
        'batch_transcription': True,
    }
}

# Standard configuration for other hardware
STANDARD_CONFIG = {
    'vllm': {
        'gpu_memory_utilization': 0.8,
        'tensor_parallel_size': 1,
        'max_model_len': 16384,
        'trust_remote_code': True,
        'dtype': 'float16',
        'max_num_batched_tokens': 2048,
        'max_num_seqs': 16,
        'semaphore_limit': 1, # Limit concurrent GPU batches to 1 for standard hardware
    },
    'audio': {
        'batch_size': 8,
        'max_segment_length': 20.0,
        'overlap_length': 1.0,
        'sample_rate': 16000,
        'parallel_workers': 4,
    },
    'memory': {
        'cache_size_gb': 8,
        'audio_cache_max_entries': 100,
        'audio_buffer_gb': 4,
        'work_dir_cleanup': True,
        # Unified GPU memory cleanup intervals for all components
        'gpu_memory_cleanup_interval_segments': 10,  # Cleanup every N segments in main processing
        'gpu_memory_cleanup_interval_batches': 10,   # Cleanup every N batches in parallel processing
        'gpu_memory_cleanup_forced_interval': 120,   # Force cleanup every N seconds
    },
    'disk': {
        'max_work_dir_size_gb': 5,
        'temp_audio_cleanup': True,
        'model_cache_dir': '~/.cache/huggingface',
        'max_concurrent_downloads': 1,
    },
    
    # File validation limits (security and resource protection) - more conservative for standard hardware
    'validation': {
        # Maximum file sizes for security and resource protection
        'max_audio_file_size_gb': 2.0,  # Smaller limit for standard hardware
        'max_batch_file_size_mb': 5.0,  # Batch files should be small text files
        'max_srt_output_size_mb': 25.0,  # Reasonable subtitle file size limit
        
        # Timeout limits to prevent resource exhaustion
        'download_timeout_seconds': 1200,  # 20 minutes max for downloads
        'processing_timeout_seconds': 2400,  # 40 minutes max for processing
        'model_load_timeout_seconds': 300,   # 5 minutes max for model loading
        'vad_timeout_seconds': 600,          # 10 minutes max for VAD
        
        # Rate limiting
        'max_concurrent_processes': 1,       # Limit concurrent video processing
        'min_processing_interval_seconds': 10, # Minimum time between processing starts
        
        # Content validation
        'min_audio_duration_seconds': 1.0,   # Minimum audio length
        'max_audio_duration_hours': 6.0,     # Maximum audio length (6 hours for standard)
        'max_segments_per_file': 5000,       # Prevent excessive segmentation
    },
    
    'performance': {
        'prefetch_models': False,
        'parallel_vad': False,
        'async_audio_loading': False,
        'batch_transcription': False,
    }
}

def get_optimal_config() -> SystemConfig:
    """Return optimal configuration based on detected hardware with validation."""
    hw = detect_hardware()
    
    if hw['is_b200'] and hw['total_ram_gb'] > 150:
        config = B200_CONFIG.copy()
        # Protection contre accès index vide
        vram_gb = hw['gpu_memory_gb'][0] if hw['gpu_memory_gb'] else 0
        logger.info(f"Detected B200 with {vram_gb:.0f}GB VRAM - using optimized config")
    else:
        config = STANDARD_CONFIG.copy() 
        logger.info(f"Using standard config for {hw['gpu_count']} GPU(s)")
    
    # Validate configuration before returning
    if not validate_config(config):
        logger.warning("Configuration validation failed, using safe defaults")
        # Return a minimal safe configuration
        return {
            'vllm': {'gpu_memory_utilization': 0.7, 'max_num_seqs': 8},
            'audio': {'batch_size': 4, 'parallel_workers': 2},
            'memory': {'cache_size_gb': 4, 'gpu_memory_cleanup_interval_segments': 10},
            'disk': {'max_work_dir_size_gb': 2},
            'performance': {'prefetch_models': False}
        }
    
    return config

def get_vllm_args(model_name: str) -> VLLMConfig:
    """Get optimized vLLM arguments for the current hardware."""
    config = get_optimal_config()
    vllm_config = config['vllm'].copy()
    hw = detect_hardware()
    
    # Adjust for specific models
    if 'Small' in model_name:
        # Voxtral Small 24B needs more memory
        if hw['is_b200']:
            vllm_config['gpu_memory_utilization'] = 0.90  # Be conservative
        else:
            vllm_config['gpu_memory_utilization'] = 0.75
    elif 'Mini' in model_name:
        # Voxtral Mini 3B is lighter
        vllm_config['max_num_seqs'] = config['vllm']['max_num_seqs'] * 2
    
    return vllm_config

def validate_config(config: SystemConfig) -> bool:
    """Validate configuration values for safety and consistency."""
    try:
        # Validate GPU memory utilization
        gpu_util = config.get('vllm', {}).get('gpu_memory_utilization', 0)
        if not (0.1 <= gpu_util <= 0.99):
            logger.warning(f"Invalid GPU memory utilization: {gpu_util}")
            return False
        
        # Validate batch sizes
        batch_size = config.get('audio', {}).get('batch_size', 0)
        if not (1 <= batch_size <= 128):
            logger.warning(f"Invalid batch size: {batch_size}")
            return False
        
        # Validate memory limits
        cache_size = config.get('memory', {}).get('cache_size_gb', 0)
        if cache_size > 100:  # Reasonable upper limit
            logger.warning(f"Cache size too large: {cache_size}GB")
            return False
        
        # Validate cleanup intervals
        cleanup_interval = config.get('memory', {}).get('gpu_memory_cleanup_interval_segments', 0)
        if not (1 <= cleanup_interval <= 100):
            logger.warning(f"Invalid cleanup interval: {cleanup_interval}")
            return False
        
        # Validate file size limits
        max_audio_size = config.get('validation', {}).get('max_audio_file_size_gb', 0)
        if not (0.1 <= max_audio_size <= 50.0):  # Reasonable range
            logger.warning(f"Invalid max audio file size: {max_audio_size}GB")
            return False
        
        # Validate timeout limits
        download_timeout = config.get('validation', {}).get('download_timeout_seconds', 0)
        if not (60 <= download_timeout <= 7200):  # 1 minute to 2 hours
            logger.warning(f"Invalid download timeout: {download_timeout}s")
            return False
        
        processing_timeout = config.get('validation', {}).get('processing_timeout_seconds', 0)
        if not (300 <= processing_timeout <= 14400):  # 5 minutes to 4 hours
            logger.warning(f"Invalid processing timeout: {processing_timeout}s")
            return False
        
        return True
        
    except (KeyError, TypeError, ValueError) as e:
        import logging
        logging.getLogger(__name__).error(f"Configuration validation error: {e}")
        return False

# Environment setup for RunPod
def setup_runpod_environment():
    """Setup optimal environment variables for RunPod with path validation."""
    
    def safe_set_env(key: str, value: str, validate_path: bool = False):
        """Safely set environment variable with optional path validation."""
        if validate_path:
            path = Path(value)
            # Create directory if it doesn't exist (for cache directories)
            if not path.exists() and str(path.expanduser()).startswith(('/tmp', str(Path('~/.cache').expanduser()))):
                try:
                    path.mkdir(parents=True, exist_ok=True)
                except (OSError, PermissionError) as e:
                    import logging
                    logging.getLogger(__name__).warning(f"Could not create directory {path}: {e}")
                    return
        
        os.environ[key] = value
    
    try:
        # CUDA optimizations for B200
        safe_set_env('CUDA_LAUNCH_BLOCKING', '0')
        safe_set_env('TORCH_CUDA_ARCH_LIST', '9.0')
        safe_set_env('PYTORCH_CUDA_ALLOC_CONF', 'max_split_size_mb:512,expandable_segments:True')
        
        # Memory optimizations with path validation
        safe_set_env('PYTORCH_KERNEL_CACHE_PATH', '/tmp/pytorch_kernel_cache', validate_path=True)
        safe_set_env('HF_HOME', '/tmp/hf_cache', validate_path=True)
        safe_set_env('TRANSFORMERS_CACHE', '/tmp/hf_cache/transformers', validate_path=True)
        
        # vLLM optimizations
        safe_set_env('VLLM_ATTENTION_BACKEND', 'FLASHINFER')
        # Only set NCCL path if file exists
        nccl_path = '/usr/local/cuda/lib64/libnccl.so'
        if Path(nccl_path).exists():
            safe_set_env('VLLM_NCCL_SO_PATH', nccl_path)
        
        # Disk space management
        safe_set_env('TMPDIR', '/tmp', validate_path=True)
        
        logger.info("RunPod environment optimized for B200")
    except (OSError, KeyError) as e:
        import logging
        logging.getLogger(__name__).warning(f"Environment setup partially failed: {e}")

if __name__ == '__main__':
    # Test configuration detection
    hw = detect_hardware()
    config = get_optimal_config()
    logger.info(f"Hardware: {hw}")
    logger.info(f"Config: {config['vllm']}")