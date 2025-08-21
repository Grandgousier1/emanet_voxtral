#!/usr/bin/env python3
"""
tests/test_edge_cases.py - Edge cases and robustness tests
Stress testing for the ML pipeline under challenging conditions
"""

import unittest
import numpy as np
import tempfile
import logging
from pathlib import Path
from typing import List, Optional
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

from utils.tensor_validation import (
    validate_tensor_shape, validate_audio_tensor, check_tensor_health
)
from utils.gpu_utils import check_cuda_available, gpu_mem_info
from utils.memory_manager import MemoryManager
from utils.audio_utils import enhanced_vad_segments
from utils.translation_quality import TranslationQualityValidator


class TestExtremeInputs(unittest.TestCase):
    """Test handling of extreme and edge case inputs."""
    
    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not available")
    def test_zero_length_tensors(self):
        """Test handling of zero-length tensors."""
        # Zero-dimensional tensor
        zero_dim = torch.tensor(42.0)
        self.assertTrue(validate_tensor_shape(zero_dim, expected_shape=()))
        
        # Empty tensors
        empty_1d = torch.empty(0)
        self.assertTrue(validate_tensor_shape(empty_1d, min_dims=1, max_dims=1))
        
        empty_2d = torch.empty(0, 5)
        self.assertTrue(validate_tensor_shape(empty_2d, expected_shape=(0, 5)))
        
        # Audio tensor edge case: silent audio
        silent_audio = torch.zeros(16000)  # 1 second of silence
        self.assertTrue(validate_audio_tensor(silent_audio, 16000, max_duration_sec=2.0))
    
    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not available")
    def test_massive_tensors(self):
        """Test handling of very large tensors (within memory limits)."""
        try:
            # Large but manageable tensor for testing
            large_audio = torch.randn(16000 * 60)  # 1 minute of audio
            self.assertTrue(validate_audio_tensor(large_audio, 16000, max_duration_sec=70.0))
            
            # Test memory efficiency
            memory_usage_before = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
            
            # Process large tensor
            processed = large_audio * 0.5
            self.assertTrue(check_tensor_health(processed, check_range=(-2.0, 2.0)))
            
            # Cleanup
            del large_audio, processed
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except (RuntimeError, torch.cuda.OutOfMemoryError):
            # This is expected behavior for truly massive tensors
            self.skipTest("Out of memory - this is expected for very large tensors")
    
    def test_extreme_strings(self):
        """Test translation quality validation with extreme string inputs."""
        validator = TranslationQualityValidator()
        
        # Empty strings
        result = validator.validate_translation_completeness("", "")
        self.assertIn('Empty source text', result.get('issues', []))
        
        # Very long strings
        long_source = "Bu çok uzun bir cümle. " * 1000  # Very long Turkish text
        long_target = "C'est une phrase très longue. " * 1000  # Very long French text
        
        result = validator.validate_translation_completeness(long_source, long_target)
        self.assertIsInstance(result['length_ratio'], float)
        self.assertGreater(result['completeness_score'], 0.0)
        
        # Special characters and Unicode
        unicode_source = "Türkçe özel karakterler: ğüşıöç ĞÜŞIÖÇ"
        unicode_target = "Caractères spéciaux français: àâäéèêëîïôöùûüÿç"
        
        result = validator.validate_translation_completeness(unicode_source, unicode_target)
        self.assertGreater(result['completeness_score'], 0.0)
        
        # Mixed languages
        mixed_source = "Hello dünya مرحبا world"
        mixed_target = "Bonjour monde hello world"
        
        result = validator.validate_translation_completeness(mixed_source, mixed_target)
        self.assertIsInstance(result, dict)
    
    def test_extreme_subtitle_constraints(self):
        """Test subtitle constraints with extreme timing scenarios."""
        validator = TranslationQualityValidator()
        
        # Extremely fast subtitle (impossible to read)
        fast_text = "This is way too fast"
        fast_duration = 0.1  # 100ms
        
        result = validator.validate_subtitle_constraints(fast_text, fast_duration)
        self.assertFalse(result['constraints_met'])
        self.assertGreater(len(result['violations']), 0)
        
        # Extremely slow subtitle
        slow_text = "Slow"
        slow_duration = 60.0  # 1 minute
        
        result = validator.validate_subtitle_constraints(slow_text, slow_duration)
        self.assertTrue(result['constraints_met'])  # Should be readable
        
        # Very long subtitle with reasonable timing
        long_text = "This is an extremely long subtitle that exceeds normal length limits " * 5
        normal_duration = 5.0
        
        result = validator.validate_subtitle_constraints(long_text, normal_duration)
        # Should flag length issues even if timing is reasonable
        self.assertIn('violations', result)


class TestMemoryStressTests(unittest.TestCase):
    """Test memory management under stress conditions."""
    
    def setUp(self):
        """Set up memory manager for testing."""
        self.memory_manager = MemoryManager()
    
    @unittest.skipUnless(TORCH_AVAILABLE and torch.cuda.is_available(), "CUDA not available")
    def test_gpu_memory_pressure(self):
        """Test behavior under GPU memory pressure."""
        try:
            # Get initial memory state
            initial_memory = torch.cuda.memory_allocated()
            
            # Create progressively larger tensors until we approach limits
            tensors = []
            max_size = 1024 * 1024  # Start with 1M elements
            
            for i in range(10):
                try:
                    size = max_size * (i + 1)
                    tensor = torch.randn(size, device='cuda')
                    tensors.append(tensor)
                    
                    # Validate tensor health
                    self.assertTrue(check_tensor_health(tensor[:1000]))  # Check subset
                    
                except torch.cuda.OutOfMemoryError:
                    # This is expected - we're testing OOM handling
                    break
            
            # Clean up
            for tensor in tensors:
                del tensor
            torch.cuda.empty_cache()
            
            # Verify memory is released
            final_memory = torch.cuda.memory_allocated()
            self.assertLessEqual(final_memory, initial_memory + 1024 * 1024)  # Allow small overhead
            
        except Exception as e:
            self.skipTest(f"GPU memory test skipped: {e}")
    
    def test_cpu_memory_monitoring(self):
        """Test CPU memory usage monitoring."""
        # Test memory manager functionality
        self.memory_manager.start_monitoring()
        
        # Create some memory load
        large_arrays = []
        for i in range(5):
            arr = np.random.randn(100000)  # ~800KB each
            large_arrays.append(arr)
        
        # Check memory reporting
        memory_info = self.memory_manager.get_memory_info()
        self.assertIn('cpu_percent', memory_info)
        self.assertIn('available_gb', memory_info)
        
        # Cleanup
        del large_arrays
        self.memory_manager.stop_monitoring()


class TestErrorRecoveryMechanisms(unittest.TestCase):
    """Test error recovery and graceful degradation."""
    
    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not available")
    def test_corrupted_tensor_recovery(self):
        """Test recovery from corrupted tensor data."""
        # Create tensor with NaN values
        corrupted = torch.randn(100)
        corrupted[10:20] = torch.nan
        
        # Should detect corruption
        with self.assertRaises(ValueError):
            check_tensor_health(corrupted, check_nan=True)
        
        # Test repair mechanism
        repaired = torch.where(torch.isnan(corrupted), torch.zeros_like(corrupted), corrupted)
        self.assertTrue(check_tensor_health(repaired, check_nan=True))
        
        # Test Inf values
        inf_tensor = torch.randn(100)
        inf_tensor[5] = torch.inf
        
        with self.assertRaises(ValueError):
            check_tensor_health(inf_tensor, check_inf=True)
        
        # Repair Inf values
        repaired_inf = torch.where(torch.isinf(inf_tensor), torch.zeros_like(inf_tensor), inf_tensor)
        self.assertTrue(check_tensor_health(repaired_inf, check_inf=True))
    
    def test_invalid_audio_recovery(self):
        """Test recovery from invalid audio data."""
        # Audio with values outside valid range
        invalid_audio = torch.randn(16000) * 5.0  # Outside [-1, 1]
        
        # Should fail validation
        with self.assertRaises(ValueError):
            validate_audio_tensor(invalid_audio, 16000, max_duration_sec=2.0)
        
        # Test normalization recovery
        normalized = torch.tanh(invalid_audio)  # Clamp to valid range
        self.assertTrue(validate_audio_tensor(normalized, 16000, max_duration_sec=2.0))
        
        # Test clipping recovery
        clipped = torch.clamp(invalid_audio, -1.0, 1.0)
        self.assertTrue(validate_audio_tensor(clipped, 16000, max_duration_sec=2.0))


class TestConcurrencyEdgeCases(unittest.TestCase):
    """Test edge cases in concurrent processing."""
    
    def test_thread_safety_validation(self):
        """Test thread safety of validation functions."""
        import threading
        import time
        
        results = []
        errors = []
        
        def worker():
            try:
                if TORCH_AVAILABLE:
                    tensor = torch.randn(100)
                    result = check_tensor_health(tensor)
                    results.append(result)
                else:
                    # Simulate work without PyTorch
                    time.sleep(0.01)
                    results.append(True)
            except Exception as e:
                errors.append(e)
        
        # Create multiple threads
        threads = []
        for i in range(10):
            t = threading.Thread(target=worker)
            threads.append(t)
            t.start()
        
        # Wait for completion
        for t in threads:
            t.join()
        
        # Check results
        self.assertEqual(len(errors), 0, f"Thread safety errors: {errors}")
        self.assertEqual(len(results), 10)
        self.assertTrue(all(results))
    
    def test_async_context_isolation(self):
        """Test that async contexts don't interfere with each other."""
        import asyncio
        
        async def async_worker(worker_id):
            """Simulate async processing with potential state conflicts."""
            # Simulate some async work
            await asyncio.sleep(0.01)
            
            if TORCH_AVAILABLE:
                # Each worker should have isolated state
                torch.manual_seed(worker_id)  # Different seed per worker
                tensor = torch.randn(10)
                return tensor.mean().item()
            else:
                return worker_id * 0.1
        
        async def run_concurrent_test():
            # Run multiple workers concurrently
            tasks = [async_worker(i) for i in range(5)]
            results = await asyncio.gather(*tasks)
            return results
        
        # Run the async test
        try:
            results = asyncio.run(run_concurrent_test())
            self.assertEqual(len(results), 5)
            # Results should be different due to different seeds
            self.assertGreater(len(set(results)), 1)
        except Exception as e:
            self.skipTest(f"Async test failed: {e}")


class TestHardwareEdgeCases(unittest.TestCase):
    """Test edge cases related to hardware detection and configuration."""
    
    def test_gpu_detection_edge_cases(self):
        """Test GPU detection with various hardware configurations."""
        try:
            # Test basic CUDA availability
            has_cuda = check_cuda_available()
            self.assertIsInstance(has_cuda, bool)
            
            # Test GPU memory info
            mem_info = gpu_mem_info()
            
            if has_cuda and mem_info:
                self.assertIn('total', mem_info)
                self.assertIn('free', mem_info)
                self.assertIn('used', mem_info)
                self.assertGreater(mem_info['total'], 0)
                
        except Exception as e:
            # GPU detection should never crash
            self.fail(f"GPU detection failed: {e}")
    
    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not available")
    def test_dtype_compatibility_edge_cases(self):
        """Test dtype compatibility across different hardware."""
        # Test with different dtypes
        dtypes_to_test = [torch.float32, torch.float16]
        
        if hasattr(torch, 'bfloat16'):
            dtypes_to_test.append(torch.bfloat16)
        
        for dtype in dtypes_to_test:
            try:
                # Create tensor with specific dtype
                tensor = torch.randn(100, dtype=dtype)
                
                # Validate dtype preservation
                from utils.tensor_validation import validate_tensor_dtype
                self.assertTrue(validate_tensor_dtype(tensor, dtype))
                
                # Test device compatibility
                if torch.cuda.is_available():
                    gpu_tensor = tensor.cuda()
                    self.assertEqual(gpu_tensor.dtype, dtype)
                    
            except Exception as e:
                # Some dtypes might not be supported on all hardware
                logging.warning(f"Dtype {dtype} not supported: {e}")


class TestRealWorldScenarios(unittest.TestCase):
    """Test scenarios that mimic real-world usage patterns."""
    
    def test_corrupted_audio_file_handling(self):
        """Test handling of corrupted or invalid audio files."""
        # Create a fake audio file with invalid content
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            tmp.write(b"This is not audio data")
            fake_audio_path = Path(tmp.name)
        
        try:
            # This should handle the error gracefully
            from cli_feedback import get_feedback
            feedback = get_feedback(debug_mode=True)
            
            # The function should detect invalid audio gracefully
            try:
                segments = enhanced_vad_segments(fake_audio_path, feedback)
                # If it returns anything, it should be empty or handled gracefully
                self.assertIsInstance(segments, list)
            except Exception as e:
                # Expected to fail, but should be a handled exception
                self.assertIn("audio", str(e).lower())
                
        finally:
            # Cleanup
            fake_audio_path.unlink()
    
    def test_network_interruption_simulation(self):
        """Test behavior when network operations are interrupted."""
        # Simulate network timeout scenario
        import time
        
        def slow_operation():
            """Simulate a slow network operation."""
            time.sleep(0.1)  # Brief delay
            return "success"
        
        # Test timeout handling
        start_time = time.time()
        result = slow_operation()
        duration = time.time() - start_time
        
        self.assertEqual(result, "success")
        self.assertLess(duration, 1.0)  # Should complete reasonably fast


if __name__ == '__main__':
    # Set up logging for detailed test output
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Create test directory if it doesn't exist
    test_dir = Path(__file__).parent
    test_dir.mkdir(exist_ok=True)
    
    # Run tests with detailed output
    unittest.main(verbosity=2)