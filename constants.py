"""
This file contains the global constants for the Emanet Voxtral application.
"""
from pathlib import Path

# Audio constants
SAMPLE_RATE = 16000
CHANNELS = 1

# Cache constants
CACHE_DB = Path('.emanet_cache.db')

# Model constants
VOXTRAL_SMALL = 'mistralai/Voxtral-Small-24B-2507'
VOXTRAL_MINI = 'mistralai/Voxtral-Mini-3B-2507'

# Unit conversion constants
BYTES_TO_GB = 1024**3