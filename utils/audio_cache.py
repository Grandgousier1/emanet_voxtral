#!/usr/bin/env python3
"""
utils/audio_cache.py - Intelligent audio caching system
Prevents redundant audio loading with memory management
"""

import hashlib
import time
import gc
import threading
from pathlib import Path
from typing import Dict, Optional, Tuple, Any, OrderedDict
import weakref
from collections import OrderedDict

import numpy as np

from cli_feedback import CLIFeedback
from constants import BYTES_TO_GB


class AudioCacheEntry:
    """Single audio cache entry with metadata."""
    
    def __init__(self, audio_data: np.ndarray, sample_rate: int, file_size: int):
        self.audio_data = audio_data
        self.sample_rate = sample_rate
        self.file_size = file_size
        self.access_time = time.time()
        self.access_count = 1
        self.memory_size = audio_data.nbytes
    
    def touch(self):
        """Update access time and count."""
        self.access_time = time.time()
        self.access_count += 1


class AudioCache:
    """Intelligent audio cache with LRU eviction and memory management."""
    
    def __init__(self, max_memory_gb: float = 10.0, max_entries: int = 50):
        self.max_memory_bytes = int(max_memory_gb * BYTES_TO_GB)
        self.max_entries = max_entries
        
        # LRU CACHE FIX: Use OrderedDict for LRU behavior
        self.cache: OrderedDict[str, AudioCacheEntry] = OrderedDict()
        self.current_memory = 0
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.feedback: Optional[CLIFeedback] = None
        
        # Thread-safe access to cache and stats
        self._cache_lock = threading.RLock()
        
        # Memory pressure thresholds
        self.memory_warning_threshold = 0.8  # 80% of max memory
        self.memory_critical_threshold = 0.95  # 95% of max memory
    
    def set_feedback(self, feedback: CLIFeedback):
        """Set feedback instance for logging."""
        self.feedback = feedback
    
    def _get_file_hash(self, file_path: Path) -> str:
        """Generate hash for file caching."""
        # Use file path, size and mtime for cache key
        stat = file_path.stat()
        content = f"{file_path}:{stat.st_size}:{stat.st_mtime}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def _cleanup_memory(self, needed_bytes: int = 0):
        """LRU cleanup cache to free memory."""
        if not self.cache:
            return
        
        # Calculate target memory after cleanup
        target_memory = self.max_memory_bytes * 0.7  # Keep 70% max
        
        if needed_bytes > 0:
            target_memory = min(target_memory, self.max_memory_bytes - needed_bytes)
        
        if self.current_memory <= target_memory:
            return
        
        # LRU CACHE FIX: Remove from oldest to newest (FIFO in OrderedDict)
        freed_memory = 0
        removed_count = 0
        keys_to_remove = []
        
        # Iterate through cache in insertion order (oldest first)
        for cache_key, entry in list(self.cache.items()):
            if self.current_memory - freed_memory <= target_memory:
                break
            
            keys_to_remove.append(cache_key)
            freed_memory += entry.memory_size
            removed_count += 1
        
        # Remove the selected entries
        for cache_key in keys_to_remove:
            del self.cache[cache_key]
            self.evictions += 1
        
        self.current_memory -= freed_memory
        
        if self.feedback and removed_count > 0:
            self.feedback.debug(f"Audio cache LRU cleanup: removed {removed_count} entries, "
                              f"freed {freed_memory / (1024*1024):.1f}MB")
        
        # Force garbage collection
        gc.collect()
    
    def _force_cleanup_entries(self):
        """Force cleanup based on entry count (protection anti-fuite) - assumes _cache_lock held."""
        if len(self.cache) < self.max_entries:
            return
            
        # Remove 30% of oldest entries to prevent gradual leaks
        target_entries = int(self.max_entries * 0.7)
        entries_to_remove = len(self.cache) - target_entries
        
        if entries_to_remove <= 0:
            return
        
        # LRU CACHE FIX: Use OrderedDict FIFO order (oldest first)
        freed_memory = 0
        keys_to_remove = []
        
        # Remove oldest entries (first entries in OrderedDict)
        for i, (cache_key, entry) in enumerate(self.cache.items()):
            if i >= entries_to_remove:
                break
            keys_to_remove.append(cache_key)
            freed_memory += entry.memory_size
        
        # Remove the selected entries
        for cache_key in keys_to_remove:
            del self.cache[cache_key]
        
        self.current_memory -= freed_memory
        
        if self.feedback:
            self.feedback.debug(f"Force cleanup entries: removed {len(keys_to_remove)} entries, "
                              f"freed {freed_memory / (BYTES_TO_GB / 1024):.1f}MB")
        
        gc.collect()
    
    def _force_cleanup_aggressive(self):
        """Aggressive cleanup to free 50% of cache memory - assumes _cache_lock held."""
        if not self.cache:
            return
            
        target_memory = self.max_memory_bytes * 0.5  # Keep only 50%
        
        # LRU CACHE FIX: Use OrderedDict FIFO order (oldest first)
        freed_memory = 0
        removed_count = 0
        keys_to_remove = []
        
        # Remove oldest entries until we reach target memory
        for cache_key, entry in list(self.cache.items()):
            if self.current_memory - freed_memory <= target_memory:
                break
                
            keys_to_remove.append(cache_key)
            freed_memory += entry.memory_size
            removed_count += 1
        
        # Remove the selected entries
        for cache_key in keys_to_remove:
            del self.cache[cache_key]
        
        self.current_memory -= freed_memory
        
        if self.feedback:
            self.feedback.warning(f"Aggressive cleanup: removed {removed_count} entries, "
                                f"freed {freed_memory / (BYTES_TO_GB / 1024):.1f}MB")
        
        gc.collect()
    
    def get(self, file_path: Path) -> Optional[Tuple[np.ndarray, int]]:
        """Get audio data from cache with LRU update."""
        cache_key = self._get_file_hash(file_path)
        
        with self._cache_lock:
            if cache_key in self.cache:
                entry = self.cache[cache_key]
                
                # Verify file hasn't changed
                if file_path.exists() and file_path.stat().st_size == entry.file_size:
                    entry.touch()
                    self.hits += 1
                    
                    # LRU CACHE FIX: Move to end (most recently used)
                    self.cache.move_to_end(cache_key)
                    
                    if self.feedback:
                        self.feedback.debug(f"Audio cache hit: {file_path.name} "
                                          f"({entry.memory_size / (BYTES_TO_GB / 1024):.1f}MB)")
                    
                    return entry.audio_data.copy(), entry.sample_rate
                else:
                    # File changed, remove from cache
                    self.current_memory -= entry.memory_size
                    del self.cache[cache_key]
            
            self.misses += 1
            return None
    
    def put(self, file_path: Path, audio_data: np.ndarray, sample_rate: int):
        """Store audio data in cache with LRU insertion."""
        if not file_path.exists():
            return
        
        file_size = file_path.stat().st_size
        memory_needed = audio_data.nbytes
        
        # Skip caching if file is too large
        if memory_needed > self.max_memory_bytes * 0.5:
            if self.feedback:
                self.feedback.debug(f"Audio file too large for cache: {file_path.name} "
                                  f"({memory_needed / (BYTES_TO_GB / 1024):.1f}MB)")
            return
        
        cache_key = self._get_file_hash(file_path)
        
        with self._cache_lock:
            # Check if already exists (update case)
            if cache_key in self.cache:
                old_entry = self.cache[cache_key]
                self.current_memory -= old_entry.memory_size
                del self.cache[cache_key]
            
            # Force cleanup if hitting entry limit (protection contre fuites)
            if len(self.cache) >= self.max_entries:
                self._force_cleanup_entries()
            
            # Cleanup if memory limit reached
            if self.current_memory + memory_needed > self.max_memory_bytes:
                self._cleanup_memory(memory_needed)
            
            # Still not enough space, force aggressive cleanup
            if self.current_memory + memory_needed > self.max_memory_bytes:
                self._force_cleanup_aggressive()
                
            # Final check - skip caching if still not enough space
            if self.current_memory + memory_needed > self.max_memory_bytes:
                if self.feedback:
                    self.feedback.warning(f"Cannot cache audio: insufficient memory after cleanup")
                return
            
            # Create entry and add to cache (automatically goes to end in OrderedDict)
            entry = AudioCacheEntry(audio_data.copy(), sample_rate, file_size)
            self.cache[cache_key] = entry
            self.current_memory += memory_needed
            
            if self.feedback:
                self.feedback.debug(f"Audio cached: {file_path.name} "
                                  f"({memory_needed / (BYTES_TO_GB / 1024):.1f}MB), "
                                  f"total cache: {self.current_memory / (BYTES_TO_GB / 1024):.1f}MB")
    
    def clear(self):
        """Clear all cache - thread-safe."""
        with self._cache_lock:
            self.cache.clear()
            self.current_memory = 0
            
        gc.collect()
        
        if self.feedback:
            self.feedback.debug("Audio cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics - thread-safe."""
        with self._cache_lock:
            total_requests = self.hits + self.misses
            hit_rate = (self.hits / total_requests * 100) if total_requests > 0 else 0
            
            return {
                'entries': len(self.cache),
                'memory_mb': self.current_memory / (BYTES_TO_GB / 1024),
                'max_memory_mb': self.max_memory_bytes / (BYTES_TO_GB / 1024),
                'hits': self.hits,
                'misses': self.misses,
                'hit_rate_percent': hit_rate
            }


# Dependency injection support - backward compatibility functions
def create_audio_cache() -> AudioCache:
    """Factory function to create audio cache with optimal configuration."""
    try:
        from config import get_optimal_config
        config = get_optimal_config()
        memory_limit = config['memory']['audio_buffer_gb']
        max_entries = config['memory']['audio_cache_max_entries']
    except:
        memory_limit = 8.0  # Default 8GB
        max_entries = 50 # Default 50 entries
    
    return AudioCache(max_memory_gb=memory_limit, max_entries=max_entries)


# Legacy global cache for backward compatibility
import threading

_global_audio_cache = None
_audio_cache_lock = threading.Lock()


def get_audio_cache() -> AudioCache:
    """Get global audio cache instance - thread-safe. DEPRECATED: Use service container instead."""
    global _global_audio_cache
    with _audio_cache_lock:
        if _global_audio_cache is None:
            _global_audio_cache = create_audio_cache()
        return _global_audio_cache


def clear_audio_cache():
    """Clear global audio cache - thread-safe. DEPRECATED: Use service container instead."""
    global _global_audio_cache
    with _audio_cache_lock:
        if _global_audio_cache:
            _global_audio_cache.clear()