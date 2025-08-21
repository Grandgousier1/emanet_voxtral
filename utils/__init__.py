"""
utils package - Utility modules for EMANET VOXTRAL
Contains various utility functions for audio processing, GPU optimization, and system utilities.
"""

# Core utilities (with optional imports)
try:
    from .audio_utils import *
except ImportError:
    pass

try:
    from .gpu_utils import *
except ImportError:
    pass

from .system_utils import *
from .validation_utils import *

# Audio and processing
try:
    from .srt_utils import *
except ImportError:
    pass

try:
    from .processing_utils import *
except ImportError:
    pass

# Authentication and security
from .auth_manager import *
from .security_utils import *

# Performance and optimization
try:
    from .b200_optimizer import *
except ImportError:
    pass

try:
    from .memory_manager import *
except ImportError:
    pass

try:
    from .performance_profiler import *
except ImportError:
    pass

# Error handling and logging
from .error_messages import *
from .logging_config import *

# Telemetry and monitoring
try:
    from .telemetry import *
except ImportError:
    pass

__all__ = [
    # Audio utilities
    'download_audio_from_youtube',
    'apply_vad_on_audio',
    'get_audio_duration',
    
    # GPU utilities
    'get_gpu_info',
    'free_cuda_mem',
    'check_gpu_memory',
    
    # System utilities
    'check_disk_space',
    'get_system_info',
    
    # Validation utilities
    'validate_file_size',
    'validate_audio_duration',
    'validate_batch_file',
    
    # SRT utilities
    'generate_srt_file',
    'format_timestamp',
    
    # Processing utilities
    'process_local_file',
    'process_youtube_url',
    
    # Auth manager
    'AuthManager',
    
    # B200 optimizer
    'get_b200_optimizer',
    'B200Optimizer',
    
    # Memory manager
    'MemoryManager',
    
    # Error messages
    'ErrorMessageManager',
    
    # Telemetry
    'TelemetryManager',
]