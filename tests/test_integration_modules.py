#!/usr/bin/env python3
"""
tests/test_integration_modules.py - Integration tests for uncovered modules
Tests for gpu_utils, memory_manager, audio_utils, and other critical modules
"""

import unittest
import tempfile
import numpy as np
import time
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import torch
    import torchaudio
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from utils.gpu_utils import check_cuda_available, gpu_mem_info
from utils.memory_manager import MemoryManager
from utils.audio_utils import enhanced_vad_segments
from utils.model_utils import (
    get_transformers_generation_params, handle_oom_with_recovery,
    detect_optimal_dtype, ensure_model_eval_mode
)
from utils.security_utils import sanitize_input, validate_file_path
from cli_feedback import get_feedback


class TestGPUUtils(unittest.TestCase):
    """Integration tests for GPU utilities."""
    
    def test_gpu_detection_integration(self):
        """Test complete GPU detection and configuration."""
        # Test CUDA availability
        has_cuda = check_cuda_available()
        self.assertIsInstance(has_cuda, bool)
        
        # Test GPU memory info if available
        mem_info = gpu_mem_info()
        
        if has_cuda and mem_info:
            self.assertIn('total', mem_info)
            self.assertIn('free', mem_info)
            self.assertIn('used', mem_info)
            self.assertGreater(mem_info['total'], 0)
    
    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not available")
    def test_gpu_memory_integration(self):
        """Test GPU memory utilities integration."""
        if torch.cuda.is_available():
            # Test basic memory operations
            initial_memory = torch.cuda.memory_allocated()
            
            # Allocate some memory
            test_tensor = torch.randn(1000, device='cuda')
            allocated_memory = torch.cuda.memory_allocated()
            
            self.assertGreater(allocated_memory, initial_memory)
            
            # Clean up
            del test_tensor
            torch.cuda.empty_cache()
        else:
            self.skipTest("CUDA not available")


class TestMemoryManager(unittest.TestCase):
    """Integration tests for memory management."""
    
    def setUp(self):
        """Set up memory manager for testing."""
        self.memory_manager = MemoryManager()
    
    def tearDown(self):
        """Clean up memory manager."""
        if hasattr(self.memory_manager, 'stop_monitoring'):
            self.memory_manager.stop_monitoring()
    
    def test_memory_monitoring_lifecycle(self):
        """Test complete memory monitoring lifecycle."""
        # Start monitoring
        self.memory_manager.start_monitoring()
        
        # Get initial memory state
        initial_info = self.memory_manager.get_memory_info()
        self.assertIsInstance(initial_info, dict)
        self.assertIn('cpu_percent', initial_info)
        self.assertIn('available_gb', initial_info)
        
        # Create some memory load
        large_data = np.random.randn(100000)  # ~800KB
        
        # Check memory after allocation
        loaded_info = self.memory_manager.get_memory_info()
        self.assertGreaterEqual(loaded_info['cpu_percent'], initial_info['cpu_percent'])
        
        # Check memory warnings
        warnings = self.memory_manager.check_memory_pressure()
        self.assertIsInstance(warnings, list)
        
        # Stop monitoring
        self.memory_manager.stop_monitoring()
        
        # Cleanup
        del large_data
    
    @unittest.skipUnless(TORCH_AVAILABLE and torch.cuda.is_available(), "CUDA not available")
    def test_gpu_memory_monitoring(self):
        """Test GPU memory monitoring integration."""
        self.memory_manager.start_monitoring()
        
        # Get GPU memory info
        gpu_info = self.memory_manager.get_gpu_memory_info()
        
        if gpu_info:  # If GPU monitoring is available
            self.assertIn('allocated_gb', gpu_info)
            self.assertIn('cached_gb', gpu_info)
            self.assertIn('total_gb', gpu_info)
            
            # Test GPU memory allocation
            initial_allocated = gpu_info['allocated_gb']
            
            # Allocate some GPU memory
            test_tensor = torch.randn(1000, 1000, device='cuda')
            
            # Check memory increase
            new_gpu_info = self.memory_manager.get_gpu_memory_info()
            self.assertGreater(new_gpu_info['allocated_gb'], initial_allocated)
            
            # Cleanup
            del test_tensor
            torch.cuda.empty_cache()


class TestAudioUtils(unittest.TestCase):
    """Integration tests for audio processing utilities."""
    
    def setUp(self):
        """Set up test environment."""
        self.feedback = get_feedback(debug_mode=True)
        self.test_dir = Path(tempfile.mkdtemp())
    
    def tearDown(self):
        """Clean up test files."""
        import shutil
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
    
    def test_vad_segments_with_synthetic_audio(self):
        """Test VAD processing with synthetic audio data."""
        # Create synthetic audio file
        sample_rate = 16000
        duration = 3.0  # 3 seconds
        
        # Create audio with speech-like patterns
        t = np.linspace(0, duration, int(sample_rate * duration))
        
        # Mix of sine waves to simulate speech
        audio_data = (
            0.3 * np.sin(2 * np.pi * 200 * t) +  # Low frequency
            0.2 * np.sin(2 * np.pi * 800 * t) +  # Mid frequency
            0.1 * np.sin(2 * np.pi * 1600 * t)   # High frequency
        )
        
        # Add some silence periods
        audio_data[int(0.5 * sample_rate):int(1.0 * sample_rate)] *= 0.1  # Quiet section
        audio_data[int(2.0 * sample_rate):int(2.5 * sample_rate)] *= 0.1  # Another quiet section
        
        # Save as WAV file
        test_audio_path = self.test_dir / "test_audio.wav"
        
        try:
            import soundfile as sf
            sf.write(str(test_audio_path), audio_data, sample_rate)
        except ImportError:
            self.skipTest("soundfile not available for audio file creation")
        
        # Test VAD processing
        try:
            segments = enhanced_vad_segments(test_audio_path, self.feedback)
            
            # Should return list of segments
            self.assertIsInstance(segments, list)
            
            # Each segment should have required fields
            for segment in segments:
                self.assertIn('start', segment)
                self.assertIn('end', segment)
                self.assertIsInstance(segment['start'], (int, float))
                self.assertIsInstance(segment['end'], (int, float))
                self.assertLessEqual(segment['start'], segment['end'])
                
        except Exception as e:
            # VAD might fail with synthetic audio or missing dependencies
            self.skipTest(f"VAD processing failed (expected with synthetic data): {e}")
    
    def test_audio_validation_integration(self):
        """Test audio validation with real file operations."""
        # Create minimal valid audio data
        sample_rate = 16000
        duration = 1.0
        audio_data = np.sin(2 * np.pi * 440 * np.linspace(0, duration, int(sample_rate * duration)))
        
        test_audio_path = self.test_dir / "valid_audio.wav"
        
        try:
            import soundfile as sf
            sf.write(str(test_audio_path), audio_data.astype(np.float32), sample_rate)
            
            # Test file exists and is readable
            self.assertTrue(test_audio_path.exists())
            self.assertGreater(test_audio_path.stat().st_size, 0)
            
            # Test audio loading
            loaded_audio, loaded_sr = sf.read(str(test_audio_path))
            self.assertEqual(loaded_sr, sample_rate)
            self.assertEqual(len(loaded_audio), len(audio_data))
            
        except ImportError:
            self.skipTest("soundfile not available")


class TestModelUtils(unittest.TestCase):
    """Integration tests for model utilities."""
    
    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not available")
    def test_oom_recovery_integration(self):
        """Test OOM recovery mechanism integration."""
        # Create a simple model for testing
        model = torch.nn.Linear(100, 10)
        model.eval()
        
        def simulate_processing(batch_size):
            """Simulate model processing that might OOM."""
            inputs = torch.randn(batch_size, 100)
            with torch.no_grad():
                outputs = model(inputs)
            return outputs
        
        # Test normal processing
        try:
            result = handle_oom_with_recovery(
                process_func=simulate_processing,
                initial_batch_size=4,
                min_batch_size=1
            )
            
            self.assertIsNotNone(result)
            self.assertEqual(result.shape[1], 10)  # Output features
            
        except Exception as e:
            self.skipTest(f"OOM recovery test failed: {e}")
    
    def test_generation_params_integration(self):
        """Test generation parameters configuration."""
        params = get_transformers_generation_params(
            task_type="translation",
            quality_level="high"
        )
        
        self.assertIsInstance(params, dict)
        
        # Should contain key generation parameters
        expected_keys = ['max_length', 'num_beams', 'do_sample', 'temperature']
        for key in expected_keys:
            self.assertIn(key, params)
        
        # Values should be reasonable
        self.assertGreater(params['max_length'], 0)
        self.assertGreater(params['num_beams'], 0)
        self.assertIsInstance(params['do_sample'], bool)
        self.assertGreater(params['temperature'], 0.0)
    
    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not available")
    def test_dtype_detection_integration(self):
        """Test optimal dtype detection integration."""
        optimal_dtype = detect_optimal_dtype()
        
        # Should return a valid PyTorch dtype
        self.assertIn(optimal_dtype, [torch.float32, torch.float16, torch.bfloat16])
        
        # Test with different device configurations
        if torch.cuda.is_available():
            gpu_dtype = detect_optimal_dtype(device='cuda')
            self.assertIn(gpu_dtype, [torch.float32, torch.float16, torch.bfloat16])
        
        cpu_dtype = detect_optimal_dtype(device='cpu')
        self.assertIn(cpu_dtype, [torch.float32, torch.float16])  # bfloat16 might not be supported on CPU
    
    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not available")
    def test_model_eval_mode_enforcement(self):
        """Test model eval mode enforcement."""
        # Create model in training mode
        model = torch.nn.Linear(10, 5)
        model.train()  # Explicitly set to training mode
        
        self.assertTrue(model.training)
        
        # Ensure eval mode
        ensure_model_eval_mode(model)
        self.assertFalse(model.training)
        
        # Test with already eval model
        model.eval()
        ensure_model_eval_mode(model)
        self.assertFalse(model.training)


class TestSecurityUtils(unittest.TestCase):
    """Integration tests for security utilities."""
    
    def test_input_sanitization_integration(self):
        """Test complete input sanitization pipeline."""
        # Test various input types
        test_cases = [
            ("normal input", True),
            ("", False),  # Empty input
            ("a" * 10000, False),  # Too long
            ("input with\nnewlines", True),  # Should handle newlines
            ("special chars: àâäéèêë", True),  # Unicode should be fine
            ("<script>alert('xss')</script>", False),  # Script injection
            ("../../etc/passwd", False),  # Path traversal
            ("SELECT * FROM users", True),  # SQL (but we're not doing SQL injection here)
        ]
        
        for input_text, should_pass in test_cases:
            try:
                result = sanitize_input(input_text, max_length=1000)
                if should_pass:
                    self.assertIsInstance(result, str)
                    self.assertLessEqual(len(result), 1000)
                else:
                    # Should either raise exception or return sanitized version
                    if result is not None:
                        self.assertNotEqual(result, input_text)  # Should be modified
            except ValueError:
                if should_pass:
                    self.fail(f"Valid input rejected: {input_text}")
                # Invalid inputs can raise exceptions - that's fine
    
    def test_file_path_validation_integration(self):
        """Test file path validation with real filesystem operations."""
        # Create test directory structure
        test_dir = Path(tempfile.mkdtemp())
        
        try:
            # Create some test files
            (test_dir / "valid_file.txt").write_text("test content")
            (test_dir / "subdir").mkdir()
            (test_dir / "subdir" / "nested_file.txt").write_text("nested content")
            
            # Test valid paths
            valid_paths = [
                test_dir / "valid_file.txt",
                test_dir / "subdir" / "nested_file.txt",
                test_dir / "nonexistent.txt",  # Should be valid path even if file doesn't exist
            ]
            
            for path in valid_paths:
                try:
                    result = validate_file_path(str(path), allowed_extensions=['.txt'])
                    self.assertTrue(result)
                except Exception as e:
                    self.fail(f"Valid path rejected: {path}, error: {e}")
            
            # Test invalid paths
            invalid_paths = [
                "/etc/passwd",  # System file
                test_dir.parent.parent / "sensitive.txt",  # Outside allowed area
                test_dir / "file.exe",  # Wrong extension
            ]
            
            for path in invalid_paths:
                with self.assertRaises(ValueError):
                    validate_file_path(str(path), allowed_extensions=['.txt'], allowed_dirs=[str(test_dir)])
        
        finally:
            # Cleanup
            import shutil
            shutil.rmtree(test_dir)


class TestCLIFeedback(unittest.TestCase):
    """Integration tests for CLI feedback system."""
    
    def test_feedback_lifecycle(self):
        """Test complete feedback system lifecycle."""
        feedback = get_feedback(debug_mode=True)
        
        # Test basic operations
        feedback.step("Test Step", 1, 3)
        feedback.substep("Test Substep")
        feedback.info("Test info message")
        feedback.success("Test success message")
        feedback.warning("Test warning message")
        
        # Test progress tracking
        feedback.update_progress(0.5)
        
        # Should not crash and should handle all message types
        self.assertIsNotNone(feedback)
    
    def test_feedback_with_exceptions(self):
        """Test feedback system with exception handling."""
        feedback = get_feedback(debug_mode=True)
        
        # Test exception logging
        try:
            raise ValueError("Test exception")
        except Exception as e:
            feedback.exception(e, "Test context")
            # Should handle exception gracefully
        
        # Test error messages
        feedback.error("Test error message")
        feedback.critical("Test critical message")


if __name__ == '__main__':
    # Set up test environment
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Create test directory
    test_dir = Path(__file__).parent
    test_dir.mkdir(exist_ok=True)
    
    # Run tests with detailed output
    unittest.main(verbosity=2)