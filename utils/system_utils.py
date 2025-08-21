
"""
utils/system_utils.py - Utilities for system interactions like disk and token validation.
"""
import shutil
import logging

logger = logging.getLogger(__name__)

def check_disk_space(path: str, required_gb: int) -> bool:
    """Checks for sufficient disk space at the given path."""
    try:
        total, used, free = shutil.disk_usage(path)
        free_gb = free / (1024**3)
        logger.info(f"Disk space check at '{path}': {free_gb:.2f} GB free.")
        if free_gb < required_gb:
            logger.critical(f"Insufficient disk space. Required: {required_gb} GB, Available: {free_gb:.2f} GB")
            return False
        return True
    except FileNotFoundError:
        logger.error(f"Path for disk space check not found: {path}")
        return False
