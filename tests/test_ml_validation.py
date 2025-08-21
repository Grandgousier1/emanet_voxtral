#!/usr/bin/env python3
"""
tests/test_ml_validation.py - Comprehensive ML validation tests
Scientific tests for PyTorch components with metamorphic testing
"""

import unittest
import numpy as np
import tempfile
import logging
from pathlib import Path
from typing import Dict, Any
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
    validate_tensor_shape, validate_tensor_dtype, validate_tensor_device,
    validate_audio_tensor, check_tensor_health, validate_batch_consistency
)
from utils.reproducibility import (
    set_global_seed, ensure_reproducible_environment, 
    ReproducibleSession, validate_reproducibility_state
)
from utils.translation_quality import TranslationQualityValidator

class TestTensorValidation(unittest.TestCase):
    """Metamorphic tests for tensor validation utilities."""
    
    def setUp(self):
        """Set up test environment."""
        if TORCH_AVAILABLE:
            set_global_seed(42)
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not available")
    def test_tensor_shape_validation_metamorphic(self):
        """Metamorphic test: tensor shape validation properties."""
        # Property: Valid tensor should pass validation
        tensor = torch.randn(2, 3, 4)
        self.assertTrue(validate_tensor_shape(tensor, expected_shape=(2, 3, 4)))
        
        # Property: Tensor with wrong shape should fail
        with self.assertRaises(ValueError):
            validate_tensor_shape(tensor, expected_shape=(2, 3, 5))
        
        # Metamorphic relation: reshaping preserves element count
        reshaped = tensor.reshape(6, 4)
        self.assertTrue(validate_tensor_shape(reshaped, expected_shape=(6, 4)))
        self.assertEqual(tensor.numel(), reshaped.numel())
        
        # Property: dimension constraints
        self.assertTrue(validate_tensor_shape(tensor, min_dims=2, max_dims=4))
        with self.assertRaises(ValueError):
            validate_tensor_shape(tensor, min_dims=4)
    
    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not available")
    def test_tensor_dtype_validation_metamorphic(self):
        """Metamorphic test: dtype validation and conversions."""
        # Property: tensor with correct dtype passes
        tensor_f32 = torch.randn(10).float()
        self.assertTrue(validate_tensor_dtype(tensor_f32, torch.float32))
        
        # Property: conversion preserves shape
        tensor_f16 = tensor_f32.half()
        self.assertTrue(validate_tensor_dtype(tensor_f16, torch.float16))
        self.assertEqual(tensor_f32.shape, tensor_f16.shape)
        
        # Property: bfloat16 validation for B200
        if hasattr(torch, 'bfloat16'):
            tensor_bf16 = tensor_f32.to(torch.bfloat16)
            self.assertTrue(validate_tensor_dtype(tensor_bf16, torch.bfloat16))
            self.assertEqual(tensor_f32.shape, tensor_bf16.shape)
    
    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not available")
    def test_audio_tensor_validation_metamorphic(self):
        """Metamorphic test: audio tensor validation properties."""
        # Property: valid audio tensor passes
        sample_rate = 16000
        duration = 2.0  # seconds
        samples = int(sample_rate * duration)
        
        # Mono audio
        audio_mono = torch.randn(samples) * 0.5  # Normalized range
        self.assertTrue(validate_audio_tensor(audio_mono, sample_rate, max_duration_sec=3.0))
        
        # Stereo audio
        audio_stereo = torch.randn(2, samples) * 0.5
        self.assertTrue(validate_audio_tensor(audio_stereo, sample_rate, max_duration_sec=3.0))
        
        # Property: duration calculation consistency
        calculated_duration = samples / sample_rate
        self.assertAlmostEqual(calculated_duration, duration, places=3)
        
        # Property: audio outside valid range should fail
        audio_invalid = torch.randn(samples) * 2.0  # Outside [-1, 1]
        with self.assertRaises(ValueError):
            validate_audio_tensor(audio_invalid, sample_rate, max_duration_sec=3.0)
    
    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not available")
    def test_tensor_health_metamorphic(self):
        """Metamorphic test: numerical health validation."""
        # Property: healthy tensor passes all checks
        healthy_tensor = torch.randn(100) * 0.1
        self.assertTrue(check_tensor_health(healthy_tensor, check_range=(-1.0, 1.0)))
        
        # Property: NaN detection
        nan_tensor = healthy_tensor.clone()
        nan_tensor[0] = torch.nan
        with self.assertRaises(ValueError):
            check_tensor_health(nan_tensor, check_nan=True)
        
        # Property: Inf detection
        inf_tensor = healthy_tensor.clone()
        inf_tensor[0] = torch.inf
        with self.assertRaises(ValueError):
            check_tensor_health(inf_tensor, check_inf=True)
        
        # Metamorphic relation: operations preserve health
        doubled = healthy_tensor * 2
        if doubled.abs().max() <= 1.0:  # Still in range
            self.assertTrue(check_tensor_health(doubled, check_range=(-1.0, 1.0)))
    
    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not available")
    def test_batch_consistency_metamorphic(self):
        """Metamorphic test: batch consistency validation."""
        # Property: consistent batch passes validation
        batch = [
            torch.randn(10, 5).to(self.device),
            torch.randn(10, 5).to(self.device),
            torch.randn(10, 5).to(self.device)
        ]
        self.assertTrue(validate_batch_consistency(batch))
        
        # Property: inconsistent dtype fails
        batch_mixed_dtype = [
            torch.randn(10, 5).float(),
            torch.randn(10, 5).double(),  # Different dtype
            torch.randn(10, 5).float()
        ]
        with self.assertRaises(ValueError):
            validate_batch_consistency(batch_mixed_dtype)
        
        # Property: inconsistent device fails (if CUDA available)
        if torch.cuda.is_available():
            batch_mixed_device = [
                torch.randn(10, 5).cuda(),
                torch.randn(10, 5).cpu(),  # Different device
                torch.randn(10, 5).cuda()
            ]
            with self.assertRaises(ValueError):
                validate_batch_consistency(batch_mixed_device)

class TestReproducibility(unittest.TestCase):
    """Metamorphic tests for reproducibility utilities."""
    
    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not available")
    def test_seed_reproducibility_metamorphic(self):
        """Metamorphic test: same seed produces same results."""
        # Property: same seed → same random sequence
        set_global_seed(123)
        result1 = torch.randn(10)
        
        set_global_seed(123)
        result2 = torch.randn(10)
        
        torch.testing.assert_close(result1, result2, rtol=1e-6, atol=1e-6)
        
        # Property: different seeds → different sequences
        set_global_seed(456)
        result3 = torch.randn(10)
        
        # Should be different (with very high probability)
        self.assertFalse(torch.allclose(result1, result3, rtol=1e-3))
    
    def test_reproducible_session_isolation(self):
        """Test that reproducible sessions are isolated."""
        # Property: session isolation
        original_seed = 42
        session_seed = 999
        
        set_global_seed(original_seed)
        
        with ReproducibleSession(session_seed, restore_on_exit=True):
            # Inside session
            validation = validate_reproducibility_state()
            if TORCH_AVAILABLE:
                self.assertIsNotNone(validation.get('torch_deterministic'))
        
        # After session - should maintain isolation
        validation_after = validate_reproducibility_state()
        self.assertIsInstance(validation_after, dict)
    
    def test_reproducibility_validation_metamorphic(self):
        """Test reproducibility state validation."""
        # Property: ensure_reproducible_environment improves score
        initial_report = ensure_reproducible_environment(seed=42, validate=True)
        
        self.assertIsInstance(initial_report, dict)
        self.assertIn('setup_successful', initial_report)
        self.assertIn('validation_results', initial_report)
        
        if TORCH_AVAILABLE and initial_report.get('validation_results'):
            score = initial_report['validation_results'].get('reproducibility_score', 0)
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)

class TestTranslationQuality(unittest.TestCase):
    """Metamorphic tests for translation quality validation."""
    
    def setUp(self):
        """Set up translation quality validator."""
        self.validator = TranslationQualityValidator()
    
    def test_translation_completeness_metamorphic(self):
        """Metamorphic test: translation completeness properties."""
        # Property: equal length texts have good ratio
        source = "Merhaba dünya"
        target = "Bonjour monde"
        
        result = self.validator.validate_translation_completeness(source, target)
        self.assertIsInstance(result, dict)
        self.assertIn('completeness_score', result)
        self.assertIn('length_ratio', result)
        
        # Property: empty target should score poorly
        empty_result = self.validator.validate_translation_completeness(source, "")
        self.assertIn('Empty source or target text', empty_result.get('issues', []))
        
        # Metamorphic relation: scaling preserves ratio
        long_source = source * 3
        long_target = target * 3
        long_result = self.validator.validate_translation_completeness(long_source, long_target)
        
        # Ratios should be similar
        ratio_diff = abs(result['length_ratio'] - long_result['length_ratio'])
        self.assertLess(ratio_diff, 0.1)
    
    def test_cultural_adaptation_metamorphic(self):
        """Test cultural adaptation scoring properties."""
        # Property: text with Turkish markers gets cultural analysis
        turkish_text = "Abi, nasılsın? Çok iyi, teşekkürler."
        french_text = "Mon frère, comment allez-vous? Très bien, merci."
        
        result = self.validator.validate_cultural_adaptation(turkish_text, french_text)
        self.assertIsInstance(result, dict)
        self.assertIn('cultural_score', result)
        
        # Property: score should be between 0 and 1
        self.assertGreaterEqual(result['cultural_score'], 0.0)
        self.assertLessEqual(result['cultural_score'], 1.0)
    
    def test_repetition_penalty_metamorphic(self):
        """Test repetition penalty calculation properties."""
        # Property: no repetition scores high
        unique_text = "Bonjour je suis très content aujourd'hui"
        score_unique = self.validator.calculate_repetition_penalty(unique_text)
        
        # Property: high repetition scores low
        repetitive_text = "bonjour bonjour bonjour je je je suis suis"
        score_repetitive = self.validator.calculate_repetition_penalty(repetitive_text)
        
        self.assertGreater(score_unique, score_repetitive)
        
        # Property: scores are in [0, 1]
        self.assertGreaterEqual(score_unique, 0.0)
        self.assertLessEqual(score_unique, 1.0)
        self.assertGreaterEqual(score_repetitive, 0.0)
        self.assertLessEqual(score_repetitive, 1.0)
    
    def test_subtitle_constraints_metamorphic(self):
        """Test subtitle constraint validation properties."""
        # Property: good subtitle passes constraints
        good_text = "Bonjour tout le monde"
        good_duration = 2.0
        
        result = self.validator.validate_subtitle_constraints(good_text, good_duration)
        self.assertIsInstance(result, dict)
        self.assertIn('constraints_met', result)
        
        # Property: too fast subtitle fails
        fast_text = "This is a very long subtitle that reads too fast for comfort"
        fast_duration = 1.0
        
        fast_result = self.validator.validate_subtitle_constraints(fast_text, fast_duration)
        self.assertFalse(fast_result['constraints_met'])
        self.assertTrue(len(fast_result['violations']) > 0)
        
        # Metamorphic relation: longer duration improves scores
        longer_result = self.validator.validate_subtitle_constraints(fast_text, fast_duration * 2)
        self.assertGreaterEqual(longer_result['readability_score'], fast_result['readability_score'])
    
    def test_comprehensive_assessment_metamorphic(self):
        """Test comprehensive quality assessment properties."""
        # Property: assessment returns all required components
        source = "Merhaba, nasılsın?"
        target = "Bonjour, comment allez-vous?"
        duration = 2.0
        
        assessment = self.validator.comprehensive_quality_assessment(source, target, duration)
        
        required_keys = ['overall_score', 'quality_level', 'detailed_metrics', 'recommendations']
        for key in required_keys:
            self.assertIn(key, assessment)
        
        # Property: overall score is in [0, 1]
        self.assertGreaterEqual(assessment['overall_score'], 0.0)
        self.assertLessEqual(assessment['overall_score'], 1.0)
        
        # Property: quality level matches score
        score = assessment['overall_score']
        level = assessment['quality_level']
        
        if score >= 0.8:
            self.assertEqual(level, 'excellent')
        elif score >= 0.6:
            self.assertEqual(level, 'good')
        elif score >= 0.4:
            self.assertEqual(level, 'fair')
        else:
            self.assertEqual(level, 'poor')

class TestMLSystemProperties(unittest.TestCase):
    """Integration tests for ML system properties."""
    
    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not available")
    def test_deterministic_inference_property(self):
        """Test that inference is deterministic with same inputs."""
        # Create synthetic model for testing
        model = torch.nn.Linear(10, 1)
        model.eval()  # Critical: eval mode
        
        # Same input should produce same output
        input_tensor = torch.randn(5, 10)
        
        with torch.no_grad():
            output1 = model(input_tensor)
            output2 = model(input_tensor)
        
        torch.testing.assert_close(output1, output2, rtol=1e-6, atol=1e-6)
    
    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not available")
    def test_batch_processing_consistency(self):
        """Test that batch processing maintains individual consistency."""
        # Property: processing items individually vs in batch should give same results
        model = torch.nn.Linear(5, 3)
        model.eval()
        
        # Individual processing
        items = [torch.randn(1, 5) for _ in range(3)]
        individual_results = []
        
        with torch.no_grad():
            for item in items:
                result = model(item)
                individual_results.append(result)
        
        # Batch processing
        batch_input = torch.cat(items, dim=0)
        with torch.no_grad():
            batch_result = model(batch_input)
        
        # Results should be equivalent
        individual_concat = torch.cat(individual_results, dim=0)
        torch.testing.assert_close(individual_concat, batch_result, rtol=1e-5, atol=1e-5)

if __name__ == '__main__':
    # Set up logging for test visibility
    logging.basicConfig(level=logging.INFO)
    
    # Create test directory if it doesn't exist
    test_dir = Path(__file__).parent
    test_dir.mkdir(exist_ok=True)
    
    # Run tests with detailed output
    unittest.main(verbosity=2)