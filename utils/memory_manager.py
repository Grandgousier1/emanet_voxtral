#!/usr/bin/env python3
"""
utils/memory_manager.py - Unified memory management system
Centralizes GPU and system memory cleanup with smart scheduling
"""

import time
import gc
import threading
from typing import Optional, Dict, Any
from dataclasses import dataclass

from utils.gpu_utils import free_cuda_mem, gpu_mem_info
from cli_feedback import CLIFeedback
from config import get_optimal_config


@dataclass
class CleanupStats:
    """Statistics for cleanup operations."""
    total_cleanups: int = 0
    forced_cleanups: int = 0
    last_cleanup: float = 0
    gpu_memory_freed_mb: float = 0
    system_memory_freed_mb: float = 0


class MemoryManager:
    """Unified memory management with intelligent cleanup scheduling."""
    
    def __init__(self, feedback: Optional[CLIFeedback] = None):
        self.feedback = feedback
        self.config = get_optimal_config()
        self.stats = CleanupStats()
        
        # RACE CONDITION FIX: Single lock for all operations to prevent deadlocks
        self._lock = threading.RLock()
        
        # Cleanup intervals from config
        self.segment_interval = self.config['memory']['gpu_memory_cleanup_interval_segments']
        self.batch_interval = self.config['memory']['gpu_memory_cleanup_interval_batches']
        self.forced_interval = self.config['memory']['gpu_memory_cleanup_forced_interval']
        
        # Counters (protégés par _lock unique)
        self.segments_processed = 0
        self.batches_processed = 0
        self.last_forced_cleanup = time.time()
        
        if self.feedback:
            self.feedback.debug(f"Memory manager initialized: "
                              f"segment_interval={self.segment_interval}, "
                              f"batch_interval={self.batch_interval}, "
                              f"forced_interval={self.forced_interval}s")
    
    def _should_cleanup_segments_unsafe(self) -> bool:
        """Check if cleanup is needed based on segments processed (assumes _stats_lock held)."""
        # Protection contre modulo par zéro
        if self.segment_interval <= 0:
            return False
        return self.segments_processed > 0 and self.segments_processed % self.segment_interval == 0
    
    def should_cleanup_segments(self) -> bool:
        """Check if cleanup is needed based on segments processed."""
        with self._lock:
            return self._should_cleanup_segments_unsafe()
    
    def _should_cleanup_batches_unsafe(self) -> bool:
        """Check if cleanup is needed based on batches processed (assumes _stats_lock held)."""
        # Protection contre modulo par zéro
        if self.batch_interval <= 0:
            return False
        return self.batches_processed > 0 and self.batches_processed % self.batch_interval == 0
    
    def should_cleanup_batches(self) -> bool:
        """Check if cleanup is needed based on batches processed."""
        with self._lock:
            return self._should_cleanup_batches_unsafe()
    
    def _should_cleanup_forced_unsafe(self) -> bool:
        """Check if forced cleanup is needed based on time (assumes _stats_lock held)."""
        return time.time() - self.last_forced_cleanup > self.forced_interval
    
    def should_cleanup_forced(self) -> bool:
        """Check if forced cleanup is needed based on time."""
        with self._lock:
            return self._should_cleanup_forced_unsafe()
    
    def get_gpu_memory_pressure(self) -> float:
        """Get current GPU memory pressure (0.0 to 1.0)."""
        try:
            mem_info = gpu_mem_info()
            if mem_info and 'total' in mem_info and 'used' in mem_info and mem_info['total'] > 0:
                return mem_info['used'] / mem_info['total']
        except:
            pass
        return 0.0
    
    def cleanup_gpu_memory(self, force: bool = False, context: str = ""):
        """Perform GPU memory cleanup with logging."""
        with self._lock:
            try:
                # Get memory before cleanup
                mem_before = self.get_gpu_memory_pressure()
                
                free_cuda_mem()
                gc.collect()
                
                # Get memory after cleanup
                mem_after = self.get_gpu_memory_pressure()
                freed_percent = (mem_before - mem_after) * 100
                
                # Thread-safe update des stats
                with self._lock:
                    self.stats.total_cleanups += 1
                    if force:
                        self.stats.forced_cleanups += 1
                    self.stats.last_cleanup = time.time()
                
                if self.feedback:
                    cleanup_type = "forced" if force else "scheduled"
                    self.feedback.debug(f"GPU memory cleanup ({cleanup_type})"
                                      f"{f' - {context}' if context else ''}: "
                                      f"{mem_before:.1%} → {mem_after:.1%} "
                                      f"(freed {freed_percent:.1f}%)")
                
            except Exception as e:
                if self.feedback:
                    self.feedback.warning(f"GPU cleanup failed: {e}")
    
    def on_segment_processed(self, force_check: bool = False):
        """Called when a segment is processed."""
        with self._lock:
            self.segments_processed += 1
            segment_num = self.segments_processed
            should_cleanup = force_check or self._should_cleanup_segments_unsafe() or self._should_cleanup_forced_unsafe()
            
            if should_cleanup:
                # RACE CONDITION FIX: Call cleanup without nested locks
                self._cleanup_gpu_memory_unsafe(context=f"segment {segment_num}")
    
    def on_batch_processed(self, force_check: bool = False):
        """Called when a batch is processed."""
        with self._lock:
            self.batches_processed += 1
            batch_num = self.batches_processed
            should_cleanup = force_check or self._should_cleanup_batches_unsafe() or self._should_cleanup_forced_unsafe()
            
            if should_cleanup:
                # RACE CONDITION FIX: Call cleanup without nested locks
                self._cleanup_gpu_memory_unsafe(context=f"batch {batch_num}")
    
    def _cleanup_gpu_memory_unsafe(self, context: str = ""):
        """Internal cleanup method called with lock already acquired."""
        try:
            # Get memory info before cleanup
            initial_info = gpu_mem_info()
            initial_used = initial_info.get('used_mb', 0) if initial_info else 0
            
            # Perform GPU memory cleanup
            free_cuda_mem()
            gc.collect()
            
            # Get memory info after cleanup
            final_info = gpu_mem_info()
            final_used = final_info.get('used_mb', 0) if final_info else 0
            freed_mb = max(0, initial_used - final_used)
            
            # Update stats
            self.stats.total_cleanups += 1
            self.stats.last_cleanup = time.time()
            self.stats.gpu_memory_freed_mb += freed_mb
            
            if self.feedback and freed_mb > 10:  # Only log if significant memory freed
                self.feedback.debug(f"GPU cleanup {context}: freed {freed_mb:.1f}MB")
                
        except Exception as e:
            if self.feedback:
                self.feedback.warning(f"GPU cleanup failed: {e}")
    
    def force_cleanup(self, context: str = ""):
        """Force immediate memory cleanup."""
        # cleanup_gpu_memory acquires _lock internally
        self.cleanup_gpu_memory(force=True, context=context)
    
    def check_memory_pressure(self, threshold: float = 0.85) -> bool:
        """Check if memory pressure is above threshold."""
        pressure = self.get_gpu_memory_pressure()
        
        if pressure > threshold:
            if self.feedback:
                self.feedback.warning(f"High GPU memory pressure: {pressure:.1%}")
            return True
        
        return False
    
    def auto_cleanup_if_needed(self, threshold: float = 0.9):
        """Automatically cleanup if memory pressure is too high."""
        if self.check_memory_pressure(threshold):
            self.force_cleanup("high memory pressure")
            return True
        return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory management statistics - thread-safe."""
        with self._lock:
            current_time = time.time()
            return {
                'segments_processed': self.segments_processed,
                'batches_processed': self.batches_processed,
                'total_cleanups': self.stats.total_cleanups,
                'forced_cleanups': self.stats.forced_cleanups,
                'last_cleanup_ago_s': current_time - self.stats.last_cleanup,
                'gpu_memory_pressure': self.get_gpu_memory_pressure(),
                'next_forced_cleanup_s': max(0, self.forced_interval - (current_time - self.last_forced_cleanup))
            }
    
    def reset_counters(self):
        """Reset processing counters - thread-safe."""
        with self._lock:
            self.segments_processed = 0
            self.batches_processed = 0
        
        if self.feedback:
            self.feedback.debug("Memory manager counters reset")


# Dependency injection support - factory function
def create_memory_manager(feedback: Optional[CLIFeedback] = None) -> MemoryManager:
    """Factory function to create memory manager."""
    return MemoryManager(feedback)


# Legacy global memory manager for backward compatibility
_global_memory_manager = None
_memory_manager_lock = threading.Lock()


def get_memory_manager(feedback: Optional[CLIFeedback] = None) -> MemoryManager:
    """Get global memory manager instance - thread-safe. DEPRECATED: Use service container instead."""
    global _global_memory_manager
    with _memory_manager_lock:
        if _global_memory_manager is None:
            _global_memory_manager = create_memory_manager(feedback)
        elif feedback and not _global_memory_manager.feedback:
            _global_memory_manager.feedback = feedback
        
        return _global_memory_manager


def reset_memory_manager():
    """Reset global memory manager - thread-safe. DEPRECATED: Use service container instead."""
    global _global_memory_manager
    with _memory_manager_lock:
        _global_memory_manager = None