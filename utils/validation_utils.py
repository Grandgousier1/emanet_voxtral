#!/usr/bin/env python3
"""
utils/validation_utils.py - File size and timeout validation utilities
Provides comprehensive validation for security and resource protection
"""

import time
import signal
import threading
from pathlib import Path
from typing import Optional, Callable, Any, Dict
from contextlib import contextmanager
from functools import wraps

from cli_feedback import CLIFeedback
from validator import CodeValidator
from config import get_optimal_config, setup_runpod_environment, detect_hardware
from constants import BYTES_TO_GB


class ValidationError(Exception):
    """Custom exception for validation failures."""
    pass


class TimeoutError(Exception):
    """Custom exception for timeout errors."""
    pass


class FileValidator:
    """Validates file sizes and types for security and resource protection."""
    
    def __init__(self, feedback: Optional[CLIFeedback] = None):
        self.feedback = feedback
        self.config = get_optimal_config()
        self.validation_config = self.config.get('validation', {})
    
    def validate_audio_file_size(self, file_path: Path) -> bool:
        """Validate audio file size against configured limits."""
        try:
            if not file_path.exists():
                if self.feedback:
                    self.feedback.error(f"File not found: {file_path}")
                return False
            
            file_size_gb = file_path.stat().st_size / BYTES_TO_GB
            max_size_gb = self.validation_config.get('max_audio_file_size_gb', 5.0)
            
            if file_size_gb > max_size_gb:
                if self.feedback:
                    self.feedback.error(f"Audio file too large: {file_size_gb:.2f}GB > {max_size_gb}GB limit")
                raise ValidationError(f"Audio file size {file_size_gb:.2f}GB exceeds limit of {max_size_gb}GB")
            
            if self.feedback:
                self.feedback.debug(f"Audio file size OK: {file_size_gb:.2f}GB (limit: {max_size_gb}GB)")
            
            return True
            
        except Exception as e:
            if self.feedback:
                self.feedback.error(f"File size validation failed: {e}")
            return False
    
    def validate_batch_file_size(self, file_path: Path) -> bool:
        """Validate batch file size against configured limits."""
        try:
            if not file_path.exists():
                if self.feedback:
                    self.feedback.error(f"Batch file not found: {file_path}")
                return False
            
            file_size_mb = file_path.stat().st_size / (1024 * 1024)
            max_size_mb = self.validation_config.get('max_batch_file_size_mb', 10.0)
            
            if file_size_mb > max_size_mb:
                if self.feedback:
                    self.feedback.error(f"Batch file too large: {file_size_mb:.2f}MB > {max_size_mb}MB limit")
                raise ValidationError(f"Batch file size {file_size_mb:.2f}MB exceeds limit of {max_size_mb}MB")
            
            return True
            
        except Exception as e:
            if self.feedback:
                self.feedback.error(f"Batch file validation failed: {e}")
            return False
    
    def validate_audio_duration(self, duration_seconds: float) -> bool:
        """Validate audio duration against configured limits."""
        try:
            min_duration = self.validation_config.get('min_audio_duration_seconds', 1.0)
            max_duration_hours = self.validation_config.get('max_audio_duration_hours', 12.0)
            max_duration_seconds = max_duration_hours * 3600
            
            if duration_seconds < min_duration:
                if self.feedback:
                    self.feedback.error(f"Audio too short: {duration_seconds:.1f}s < {min_duration}s minimum")
                raise ValidationError(f"Audio duration {duration_seconds:.1f}s below minimum of {min_duration}s")
            
            if duration_seconds > max_duration_seconds:
                if self.feedback:
                    self.feedback.error(f"Audio too long: {duration_seconds/3600:.1f}h > {max_duration_hours}h maximum")
                raise ValidationError(f"Audio duration {duration_seconds/3600:.1f}h exceeds maximum of {max_duration_hours}h")
            
            return True
            
        except ValidationError:
            raise
        except Exception as e:
            if self.feedback:
                self.feedback.error(f"Audio duration validation failed: {e}")
            return False
    
    def validate_segment_count(self, segment_count: int) -> bool:
        """Validate number of segments to prevent excessive processing."""
        try:
            max_segments = self.validation_config.get('max_segments_per_file', 10000)
            
            if segment_count > max_segments:
                if self.feedback:
                    self.feedback.error(f"Too many segments: {segment_count} > {max_segments} maximum")
                raise ValidationError(f"Segment count {segment_count} exceeds maximum of {max_segments}")
            
            return True
            
        except ValidationError:
            raise
        except Exception as e:
            if self.feedback:
                self.feedback.error(f"Segment count validation failed: {e}")
            return False


class TimeoutManager:
    """Manages timeouts for various operations to prevent resource exhaustion."""
    
    def __init__(self, feedback: Optional[CLIFeedback] = None):
        self.feedback = feedback
        self.config = get_optimal_config()
        self.validation_config = self.config.get('validation', {})
    
    @contextmanager
    def timeout_context(self, timeout_seconds: float, operation_name: str = "operation"):
        """Context manager for timeout handling with signal-based interruption."""
        def timeout_handler(signum, frame):
            raise TimeoutError(f"Operation '{operation_name}' timed out after {timeout_seconds}s")
        
        # Set the signal handler and a alarm
        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(int(timeout_seconds))
        
        start_time = time.time()
        
        try:
            if self.feedback:
                self.feedback.debug(f"Starting {operation_name} with {timeout_seconds}s timeout")
            yield
            
        except TimeoutError:
            if self.feedback:
                self.feedback.error(f"Timeout: {operation_name} exceeded {timeout_seconds}s limit")
            raise
            
        finally:
            # Restore the old signal handler and cancel the alarm
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
            
            elapsed = time.time() - start_time
            if self.feedback:
                self.feedback.debug(f"Completed {operation_name} in {elapsed:.1f}s")
    
    def get_download_timeout(self) -> float:
        """Get configured download timeout."""
        return self.validation_config.get('download_timeout_seconds', 1800)
    
    def get_processing_timeout(self) -> float:
        """Get configured processing timeout."""
        return self.validation_config.get('processing_timeout_seconds', 3600)
    
    def get_model_load_timeout(self) -> float:
        """Get configured model loading timeout."""
        return self.validation_config.get('model_load_timeout_seconds', 600)
    
    def get_vad_timeout(self) -> float:
        """Get configured VAD timeout."""
        return self.validation_config.get('vad_timeout_seconds', 900)


class RateLimiter:
    """Rate limiter to prevent resource exhaustion."""
    
    def __init__(self, feedback: Optional[CLIFeedback] = None):
        self.feedback = feedback
        self.config = get_optimal_config()
        self.validation_config = self.config.get('validation', {})
        self.last_start_time = 0.0
        self._lock = threading.Lock()
    
    def can_start_processing(self) -> bool:
        """Check if processing can start based on rate limits."""
        with self._lock:
            current_time = time.time()
            min_interval = self.validation_config.get('min_processing_interval_seconds', 5)
            
            if current_time - self.last_start_time < min_interval:
                remaining = min_interval - (current_time - self.last_start_time)
                if self.feedback:
                    self.feedback.warning(f"Rate limit: must wait {remaining:.1f}s before starting new process")
                return False
            
            self.last_start_time = current_time
            return True


def enhanced_preflight_checks(feedback: CLIFeedback) -> bool:
    """Enhanced preflight checks with comprehensive validation."""
    
    feedback.major_step("Pre-execution Validation", 1, 6)
    
    # Run full validation suite
    feedback.substep("Running comprehensive validation suite")
    validator = CodeValidator()
    validation_success = validator.run_validation()
    
    if not validation_success:
        feedback.critical("Validation failed - execution cannot proceed safely",
                         solution="Fix validation errors shown above")
        return False
    
    feedback.success("All validation checks passed")
    
    # Setup RunPod environment
    feedback.substep("Configuring RunPod B200 environment")
    try:
        setup_runpod_environment()
        feedback.success("RunPod environment optimized")
    except Exception as e:
        feedback.warning(f"Environment setup issues: {e}")
    
    # Hardware detection with detailed feedback (optimisé - un seul appel)
    feedback.substep("Hardware detection and optimization")
    try:
        # Optimisation N+1: detect_hardware() une seule fois puis config
        hw = detect_hardware()
        config = get_optimal_config()  # Utilise le cache hardware
        
        if hw['is_b200'] and hw['gpu_memory_gb']:
            # Protection contre accès index vide
            vram_gb = hw['gpu_memory_gb'][0] if hw['gpu_memory_gb'] else 0
            feedback.success(f"🚀 B200 GPU detected: {vram_gb:.0f}GB VRAM")
            feedback.info(f"Optimization: {config['audio']['batch_size']} batch size, {config['audio']['parallel_workers']} workers")
        else:
            feedback.info(f"Hardware: {hw['gpu_count']} GPU(s), {hw['total_ram_gb']:.1f}GB RAM")
        
        feedback.success("Hardware detection complete")
        
    except Exception as e:
        feedback.error(f"Hardware detection failed: {e}",
                      solution="Check GPU drivers and system configuration")
        return False
    
    return True


# Dependency injection support - factory functions
def create_file_validator(feedback: Optional[CLIFeedback] = None) -> FileValidator:
    """Factory function to create file validator."""
    return FileValidator(feedback)


def create_timeout_manager(feedback: Optional[CLIFeedback] = None) -> TimeoutManager:
    """Factory function to create timeout manager."""
    return TimeoutManager(feedback)


def create_rate_limiter(feedback: Optional[CLIFeedback] = None) -> RateLimiter:
    """Factory function to create rate limiter."""
    return RateLimiter(feedback)


# Legacy global instances for backward compatibility
_global_file_validator = None
_global_timeout_manager = None
_global_rate_limiter = None
_validator_lock = threading.Lock()
_timeout_lock = threading.Lock()
_rate_lock = threading.Lock()


def get_file_validator(feedback: Optional[CLIFeedback] = None) -> FileValidator:
    """Get global file validator instance - thread-safe. DEPRECATED: Use service container instead."""
    global _global_file_validator
    with _validator_lock:
        if _global_file_validator is None:
            _global_file_validator = create_file_validator(feedback)
        elif feedback and not _global_file_validator.feedback:
            _global_file_validator.feedback = feedback
        return _global_file_validator


def get_timeout_manager(feedback: Optional[CLIFeedback] = None) -> TimeoutManager:
    """Get global timeout manager instance - thread-safe. DEPRECATED: Use service container instead."""
    global _global_timeout_manager
    with _timeout_lock:
        if _global_timeout_manager is None:
            _global_timeout_manager = create_timeout_manager(feedback)
        elif feedback and not _global_timeout_manager.feedback:
            _global_timeout_manager.feedback = feedback
        return _global_timeout_manager


def get_rate_limiter(feedback: Optional[CLIFeedback] = None) -> RateLimiter:
    """Get global rate limiter instance - thread-safe. DEPRECATED: Use service container instead."""
    global _global_rate_limiter
    with _rate_lock:
        if _global_rate_limiter is None:
            _global_rate_limiter = create_rate_limiter(feedback)
        elif feedback and not _global_rate_limiter.feedback:
            _global_rate_limiter.feedback = feedback
        return _global_rate_limiter


# Convenience functions
def validate_audio_file(file_path: Path, feedback: Optional[CLIFeedback] = None) -> bool:
    """Convenience function to validate audio file."""
    validator = get_file_validator(feedback)
    return validator.validate_audio_file_size(file_path)


def validate_batch_file(file_path: Path, feedback: Optional[CLIFeedback] = None) -> bool:
    """Convenience function to validate batch file."""
    validator = get_file_validator(feedback)
    return validator.validate_batch_file_size(file_path)


def validate_audio_duration(duration_seconds: float, feedback: Optional[CLIFeedback] = None) -> bool:
    """Convenience function to validate audio duration."""
    validator = get_file_validator(feedback)
    return validator.validate_audio_duration(duration_seconds)


def validate_segment_count(segment_count: int, feedback: Optional[CLIFeedback] = None) -> bool:
    """Convenience function to validate segment count."""
    validator = get_file_validator(feedback)
    return validator.validate_segment_count(segment_count)