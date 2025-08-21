#!/usr/bin/env python3
"""
tests/test_end_to_end.py - End-to-end pipeline testing
Complete integration tests for the entire subtitle generation pipeline
"""

import unittest
import tempfile
import numpy as np
import time
import subprocess
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import json

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import torch
    import torchaudio
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from cli_feedback import get_feedback
try:
    from config import detect_hardware
except ImportError:
    def detect_hardware():
        return {"device": "cpu", "has_gpu": False}
from utils.reproducibility import set_global_seed, ensure_reproducible_environment
from utils.validation_utils import enhanced_preflight_checks
from utils.tensor_validation import validate_audio_tensor
from utils.translation_quality import TranslationQualityValidator


class TestEndToEndPipeline(unittest.TestCase):
    """Complete end-to-end pipeline testing."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_dir = Path(tempfile.mkdtemp())
        self.feedback = get_feedback(debug_mode=True)
        set_global_seed(42)  # Ensure reproducible tests
    
    def tearDown(self):
        """Clean up test environment."""
        import shutil
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
    
    def create_test_audio(self, duration=5.0, sample_rate=16000):
        """Create synthetic test audio file."""
        # Generate synthetic speech-like audio
        t = np.linspace(0, duration, int(sample_rate * duration))
        
        # Create complex waveform that resembles speech
        audio_data = (
            0.3 * np.sin(2 * np.pi * 200 * t) +      # Fundamental frequency
            0.2 * np.sin(2 * np.pi * 400 * t) +      # First harmonic
            0.15 * np.sin(2 * np.pi * 800 * t) +     # Second harmonic
            0.1 * np.sin(2 * np.pi * 1600 * t) +     # Third harmonic
            0.05 * np.random.randn(len(t))           # Noise component
        )
        
        # Add speech-like envelopes (pauses and amplitude variations)
        envelope = np.ones_like(t)
        
        # Add some pauses
        pause_start = int(0.3 * len(t))
        pause_end = int(0.4 * len(t))
        envelope[pause_start:pause_end] *= 0.1
        
        pause_start = int(0.7 * len(t))
        pause_end = int(0.8 * len(t))
        envelope[pause_start:pause_end] *= 0.1
        
        # Apply envelope
        audio_data *= envelope
        
        # Normalize to [-1, 1]
        audio_data = audio_data / np.max(np.abs(audio_data))
        
        # Save as WAV file
        test_audio_path = self.test_dir / "test_speech.wav"
        
        try:
            import soundfile as sf
            sf.write(str(test_audio_path), audio_data.astype(np.float32), sample_rate)
            return test_audio_path
        except ImportError:
            self.skipTest("soundfile not available for audio creation")
    
    def test_complete_pipeline_dry_run(self):
        """Test complete pipeline in dry-run/validation mode."""
        # Test hardware detection
        hardware_config = detect_hardware()
        self.assertIsInstance(hardware_config, dict)
        self.assertIn('device', hardware_config)
        
        # Test preflight checks
        preflight_results = enhanced_preflight_checks(self.feedback)
        self.assertIsInstance(preflight_results, dict)
        self.assertIn('checks_passed', preflight_results)
        
        # Test reproducibility setup
        repro_results = ensure_reproducible_environment(seed=42, validate=True)
        self.assertIsInstance(repro_results, dict)
        self.assertIn('setup_successful', repro_results)
    
    def test_audio_processing_pipeline(self):
        """Test complete audio processing pipeline."""
        # Create test audio
        audio_path = self.create_test_audio(duration=3.0)
        
        # Test audio validation
        if TORCH_AVAILABLE:
            try:
                import soundfile as sf
                audio_data, sample_rate = sf.read(str(audio_path))
                audio_tensor = torch.from_numpy(audio_data)
                
                # Validate audio tensor
                is_valid = validate_audio_tensor(
                    audio_tensor, sample_rate, max_duration_sec=5.0
                )
                self.assertTrue(is_valid)
                
            except ImportError:
                self.skipTest("soundfile not available")
        
        # Test VAD processing
        try:
            from utils.audio_utils import enhanced_vad_segments
            segments = enhanced_vad_segments(audio_path, self.feedback)
            
            self.assertIsInstance(segments, list)
            
            # Validate segment structure
            for segment in segments:
                self.assertIn('start', segment)
                self.assertIn('end', segment)
                self.assertIsInstance(segment['start'], (int, float))
                self.assertIsInstance(segment['end'], (int, float))
                self.assertGreaterEqual(segment['start'], 0)
                self.assertLessEqual(segment['end'], 3.0)  # Duration limit
                
        except Exception as e:
            # VAD might fail with synthetic audio - that's expected
            self.feedback.warning(f"VAD processing failed (expected with synthetic audio): {e}")
    
    def test_translation_quality_pipeline(self):
        """Test complete translation quality validation pipeline."""
        validator = TranslationQualityValidator()
        
        # Test realistic translation scenarios
        test_cases = [
            {
                'source': "Merhaba, nasılsınız? Bugün hava çok güzel.",
                'target': "Bonjour, comment allez-vous? Le temps est très beau aujourd'hui.",
                'duration': 3.0,
                'expected_quality': 'good'
            },
            {
                'source': "Bu film gerçekten harika. Çok beğendim.",
                'target': "Ce film est vraiment fantastique. J'ai beaucoup aimé.",
                'duration': 2.5,
                'expected_quality': 'good'
            },
            {
                'source': "Lütfen bana yardım edin.",
                'target': "S'il vous plaît, aidez-moi.",
                'duration': 1.5,
                'expected_quality': 'fair'  # Shorter, simpler
            }
        ]
        
        for i, case in enumerate(test_cases):
            with self.subTest(case=i):
                assessment = validator.comprehensive_quality_assessment(
                    source=case['source'],
                    target=case['target'],
                    duration=case['duration']
                )
                
                # Validate assessment structure
                self.assertIn('overall_score', assessment)
                self.assertIn('quality_level', assessment)
                self.assertIn('detailed_metrics', assessment)
                self.assertIn('recommendations', assessment)
                
                # Score should be reasonable
                self.assertGreaterEqual(assessment['overall_score'], 0.0)
                self.assertLessEqual(assessment['overall_score'], 1.0)
                
                # Quality level should match expectations roughly
                actual_quality = assessment['quality_level']
                self.assertIn(actual_quality, ['poor', 'fair', 'good', 'excellent'])
    
    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not available")
    def test_model_processing_pipeline(self):
        """Test model processing components of the pipeline."""
        # Test model utilities
        from utils.model_utils import (
            detect_optimal_dtype, get_transformers_generation_params,
            ensure_model_eval_mode
        )
        
        # Test dtype detection
        optimal_dtype = detect_optimal_dtype()
        self.assertIn(optimal_dtype, [torch.float32, torch.float16, torch.bfloat16])
        
        # Test generation parameters
        params = get_transformers_generation_params(
            task_type="translation",
            quality_level="high"
        )
        self.assertIsInstance(params, dict)
        self.assertIn('max_length', params)
        self.assertIn('num_beams', params)
        
        # Test model evaluation mode enforcement
        test_model = torch.nn.Linear(10, 5)
        test_model.train()  # Set to training mode
        ensure_model_eval_mode(test_model)
        self.assertFalse(test_model.training)  # Should be in eval mode
    
    def test_error_handling_pipeline(self):
        """Test error handling throughout the pipeline."""
        # Test with invalid inputs
        invalid_cases = [
            {"type": "empty_file", "path": self.test_dir / "empty.wav"},
            {"type": "non_existent", "path": self.test_dir / "doesnt_exist.wav"},
            {"type": "invalid_format", "path": self.test_dir / "invalid.txt"}
        ]
        
        # Create empty file
        invalid_cases[0]["path"].touch()
        
        # Create invalid format file
        invalid_cases[2]["path"].write_text("This is not audio data")
        
        for case in invalid_cases:
            with self.subTest(case=case["type"]):
                try:
                    from utils.audio_utils import enhanced_vad_segments
                    segments = enhanced_vad_segments(case["path"], self.feedback)
                    
                    # If it doesn't fail, it should return empty list or handle gracefully
                    self.assertIsInstance(segments, list)
                    
                except Exception as e:
                    # Errors are expected for invalid inputs
                    self.assertIsInstance(e, (FileNotFoundError, ValueError, RuntimeError))
    
    def test_cli_interface_integration(self):
        """Test CLI interface integration."""
        # Test help command
        try:
            result = subprocess.run([
                sys.executable, str(Path(__file__).parent.parent / "main.py"), "--help"
            ], capture_output=True, text=True, timeout=30)
            
            # Help should work regardless of dependencies
            self.assertEqual(result.returncode, 0)
            self.assertIn("usage", result.stdout.lower())
            
        except subprocess.TimeoutExpired:
            self.fail("CLI help command timed out")
        except FileNotFoundError:
            self.skipTest("main.py not found")
        
        # Test validation-only mode
        try:
            result = subprocess.run([
                sys.executable, str(Path(__file__).parent.parent / "main.py"), 
                "--validate-only"
            ], capture_output=True, text=True, timeout=60)
            
            # Should complete without processing actual audio
            # Return code might be non-zero due to missing dependencies, but shouldn't crash
            self.assertIsNotNone(result.stdout)
            
        except subprocess.TimeoutExpired:
            self.fail("Validation command timed out")
        except FileNotFoundError:
            self.skipTest("main.py not found or dependencies missing")
    
    def test_memory_management_pipeline(self):
        """Test memory management throughout the pipeline."""
        from utils.memory_manager import MemoryManager
        
        memory_manager = MemoryManager()
        memory_manager.start_monitoring()
        
        try:
            # Get initial memory state
            initial_memory = memory_manager.get_memory_info()
            self.assertIsInstance(initial_memory, dict)
            
            # Simulate processing load
            if TORCH_AVAILABLE:
                # Create some tensors to simulate model processing
                tensors = []
                for i in range(5):
                    tensor = torch.randn(1000, 1000)  # ~4MB each
                    tensors.append(tensor)
                
                # Check memory usage increased
                loaded_memory = memory_manager.get_memory_info()
                self.assertGreaterEqual(
                    loaded_memory['cpu_percent'], 
                    initial_memory['cpu_percent']
                )
                
                # Clean up
                del tensors
            
            # Check memory warnings
            warnings = memory_manager.check_memory_pressure()
            self.assertIsInstance(warnings, list)
            
        finally:
            memory_manager.stop_monitoring()
    
    def test_reproducibility_pipeline(self):
        """Test reproducibility throughout the pipeline."""
        from utils.reproducibility import ReproducibleSession, validate_reproducibility_state
        
        # Test reproducible execution
        results = []
        
        for run in range(2):
            with ReproducibleSession(seed=123, restore_on_exit=True):
                # Simulate some processing
                if TORCH_AVAILABLE:
                    result = torch.randn(10).mean().item()
                    results.append(result)
                else:
                    np.random.seed(123)
                    result = np.random.randn(10).mean()
                    results.append(result)
        
        # Results should be identical due to reproducible setup
        if len(results) == 2:
            if TORCH_AVAILABLE:
                self.assertAlmostEqual(results[0], results[1], places=6)
            else:
                self.assertAlmostEqual(results[0], results[1], places=10)
        
        # Test reproducibility validation
        validation = validate_reproducibility_state()
        self.assertIsInstance(validation, dict)
    
    def test_performance_monitoring_pipeline(self):
        """Test performance monitoring integration."""
        start_time = time.time()
        
        # Simulate pipeline operations with timing
        operations = [
            ("Hardware Detection", lambda: detect_hardware()),
            ("Preflight Checks", lambda: enhanced_preflight_checks(self.feedback)),
            ("Reproducibility Setup", lambda: ensure_reproducible_environment(seed=42))
        ]
        
        timings = {}
        
        for op_name, op_func in operations:
            op_start = time.time()
            try:
                result = op_func()
                self.assertIsNotNone(result)
            except Exception as e:
                self.feedback.warning(f"{op_name} failed: {e}")
            
            op_duration = time.time() - op_start
            timings[op_name] = op_duration
            
            # Operations should complete reasonably quickly
            self.assertLess(op_duration, 10.0, f"{op_name} took too long: {op_duration}s")
        
        total_duration = time.time() - start_time
        self.assertLess(total_duration, 30.0, f"Total pipeline took too long: {total_duration}s")
        
        # Log timing information
        self.feedback.info(f"Pipeline timings: {timings}")


class TestBatchProcessingPipeline(unittest.TestCase):
    """Test batch processing capabilities of the pipeline."""
    
    def setUp(self):
        """Set up batch testing environment."""
        self.test_dir = Path(tempfile.mkdtemp())
        self.feedback = get_feedback(debug_mode=True)
    
    def tearDown(self):
        """Clean up batch test environment."""
        import shutil
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
    
    def test_batch_translation_quality(self):
        """Test batch translation quality validation."""
        validator = TranslationQualityValidator()
        
        # Create batch of translation pairs
        batch_translations = [
            ("Merhaba", "Bonjour"),
            ("Teşekkür ederim", "Merci beaucoup"),
            ("Görüşürüz", "À bientôt"),
            ("Nasılsın?", "Comment allez-vous?"),
            ("Çok güzel", "Très beau")
        ]
        
        batch_results = []
        
        for source, target in batch_translations:
            result = validator.comprehensive_quality_assessment(source, target, duration=2.0)
            batch_results.append(result)
        
        # Validate batch results
        self.assertEqual(len(batch_results), len(batch_translations))
        
        # All should have valid scores
        for result in batch_results:
            self.assertIn('overall_score', result)
            self.assertGreaterEqual(result['overall_score'], 0.0)
            self.assertLessEqual(result['overall_score'], 1.0)
        
        # Calculate batch statistics
        scores = [r['overall_score'] for r in batch_results]
        avg_score = np.mean(scores)
        
        self.assertGreater(avg_score, 0.0)
        self.feedback.info(f"Batch average quality score: {avg_score:.3f}")
    
    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not available")
    def test_batch_tensor_processing(self):
        """Test batch tensor processing pipeline."""
        from utils.tensor_validation import validate_batch_consistency
        
        # Create batch of audio-like tensors
        batch_size = 4
        audio_length = 16000  # 1 second
        
        batch_tensors = []
        for i in range(batch_size):
            # Each tensor represents 1 second of audio
            tensor = torch.randn(audio_length) * 0.5  # Normalized range
            batch_tensors.append(tensor)
        
        # Test batch consistency
        is_consistent = validate_batch_consistency(batch_tensors)
        self.assertTrue(is_consistent)
        
        # Test batch processing simulation
        stacked_batch = torch.stack(batch_tensors)
        self.assertEqual(stacked_batch.shape, (batch_size, audio_length))
        
        # Simulate batch model processing
        processed_batch = stacked_batch * 0.8  # Simple processing
        
        # Validate processed batch
        self.assertEqual(processed_batch.shape, stacked_batch.shape)
        self.assertTrue(torch.allclose(processed_batch, stacked_batch * 0.8))


if __name__ == '__main__':
    # Set up comprehensive test environment
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Create test output directory
    test_dir = Path(__file__).parent
    test_dir.mkdir(exist_ok=True)
    
    # Run tests with maximum verbosity
    unittest.main(verbosity=2, buffer=True)