#!/usr/bin/env python3
"""
Property-based tests for domain models using Hypothesis.

These tests verify invariants and edge cases that traditional unit tests might miss
by generating thousands of random test cases automatically.
"""

import pytest
from hypothesis import given, strategies as st, assume, settings, Verbosity
from hypothesis.stateful import RuleBasedStateMachine, Bundle, rule, initialize
import math
import time
from typing import Dict, Any

from domain_models import (
    AudioSegment, ProcessingResult, ModelConfig, ProcessingConfig,
    ErrorSeverity, ErrorContext, BatchMetrics, BatchStatus, ProcessingBatch,
    validate_audio_segment, validate_processing_result, create_audio_segment,
    create_processing_result, create_batch_with_id
)


# ====================== STRATEGIES DEFINITION ======================

@st.composite
def audio_segment_strategy(draw):
    """Generate valid AudioSegment data."""
    # Generate start time between 0 and 1 hour
    start = draw(st.floats(min_value=0.0, max_value=3600.0, exclude_nan=True, exclude_infinity=True))
    # Duration between 0.1s and 60s
    duration = draw(st.floats(min_value=0.1, max_value=60.0, exclude_nan=True, exclude_infinity=True))
    end = start + duration
    
    sample_rate = 16000
    start_sample = int(start * sample_rate)
    end_sample = int(end * sample_rate)
    
    return {
        'start': start,
        'end': end,
        'duration': duration,
        'start_sample': start_sample,
        'end_sample': end_sample,
        'confidence': draw(st.one_of(st.none(), st.floats(min_value=0.0, max_value=1.0))),
        'speaker_id': draw(st.one_of(st.none(), st.text(min_size=1, max_size=20))),
        'language': draw(st.one_of(st.none(), st.sampled_from(['en', 'fr', 'es', 'de', 'it'])))
    }


@st.composite
def processing_result_strategy(draw):
    """Generate valid ProcessingResult data."""
    start = draw(st.floats(min_value=0.0, max_value=3600.0, exclude_nan=True, exclude_infinity=True))
    duration = draw(st.floats(min_value=0.1, max_value=60.0, exclude_nan=True, exclude_infinity=True))
    end = start + duration
    
    quality_score = draw(st.floats(min_value=0.0, max_value=1.0, exclude_nan=True, exclude_infinity=True))
    
    # Quality level should match score
    if quality_score >= 0.9:
        quality_level = 'excellent'
    elif quality_score >= 0.7:
        quality_level = 'good'
    elif quality_score >= 0.5:
        quality_level = 'fair'
    elif quality_score > 0:
        quality_level = 'poor'
    else:
        quality_level = 'failed'
    
    return {
        'text': draw(st.text(min_size=0, max_size=1000)),
        'start': start,
        'end': end,
        'quality_score': quality_score,
        'quality_level': quality_level,
        'model_used': draw(st.one_of(st.none(), st.text(min_size=1, max_size=50))),
        'processing_time': draw(st.one_of(st.none(), st.floats(min_value=0.001, max_value=300.0))),
        'retry_count': draw(st.one_of(st.none(), st.integers(min_value=0, max_value=5))),
        'error_message': draw(st.one_of(st.none(), st.text(min_size=0, max_size=200)))
    }


# ====================== PROPERTY-BASED TESTS ======================

class TestAudioSegmentProperties:
    """Property-based tests for AudioSegment invariants."""
    
    @given(segment_data=audio_segment_strategy())
    @settings(max_examples=200, verbosity=Verbosity.normal)
    def test_audio_segment_time_invariants(self, segment_data):
        """Test that AudioSegment time relationships are always consistent."""
        segment = validate_audio_segment(segment_data)
        
        # Core invariants
        assert segment['end'] > segment['start'], f"End ({segment['end']}) must be > start ({segment['start']})"
        assert segment['duration'] > 0, f"Duration ({segment['duration']}) must be positive"
        assert segment['end_sample'] > segment['start_sample'], "End sample must be > start sample"
        
        # Consistency invariants
        calculated_duration = segment['end'] - segment['start']
        assert abs(calculated_duration - segment['duration']) < 0.001, \
            f"Duration inconsistency: {calculated_duration} vs {segment['duration']}"
        
        # Sample rate consistency (16kHz)
        expected_start_sample = int(segment['start'] * 16000)
        expected_end_sample = int(segment['end'] * 16000)
        assert abs(segment['start_sample'] - expected_start_sample) <= 1, \
            "Start sample must match start time * 16000"
        assert abs(segment['end_sample'] - expected_end_sample) <= 1, \
            "End sample must match end time * 16000"
    
    @given(
        start=st.floats(min_value=0.0, max_value=1800.0, exclude_nan=True, exclude_infinity=True),
        duration=st.floats(min_value=0.1, max_value=30.0, exclude_nan=True, exclude_infinity=True)
    )
    def test_create_audio_segment_factory(self, start, duration):
        """Test the factory function maintains invariants."""
        end = start + duration
        segment = create_audio_segment(start, end)
        
        assert segment['start'] == start
        assert segment['end'] == end
        assert abs(segment['duration'] - duration) < 0.001
        assert segment['start_sample'] == int(start * 16000)
        assert segment['end_sample'] == int(end * 16000)
    
    @given(
        segments=st.lists(audio_segment_strategy(), min_size=1, max_size=50)
    )
    def test_audio_batch_invariants(self, segments):
        """Test that audio batches maintain consistency."""
        validated_segments = [validate_audio_segment(seg) for seg in segments]
        batch = create_batch_with_id(validated_segments)
        
        # Batch invariants
        assert len(batch['segments']) == len(validated_segments)
        assert batch['total_duration'] > 0
        assert batch['max_segment_length'] >= 0
        assert batch['created_at'] <= time.time() + 1  # Allow for small time differences
        assert len(batch['batch_id']) > 0
        
        # Duration consistency
        expected_duration = sum(seg['duration'] for seg in validated_segments)
        assert abs(batch['total_duration'] - expected_duration) < 0.001
        
        # Max length consistency  
        if validated_segments:
            expected_max_length = max(seg['end_sample'] - seg['start_sample'] for seg in validated_segments)
            assert batch['max_segment_length'] == expected_max_length


class TestProcessingResultProperties:
    """Property-based tests for ProcessingResult invariants."""
    
    @given(result_data=processing_result_strategy())
    @settings(max_examples=150)
    def test_processing_result_quality_invariants(self, result_data):
        """Test that ProcessingResult quality metrics are consistent."""
        result = validate_processing_result(result_data)
        
        # Quality score bounds
        assert 0.0 <= result['quality_score'] <= 1.0, \
            f"Quality score {result['quality_score']} must be between 0.0 and 1.0"
        
        # Quality level consistency with score
        score = result['quality_score']
        level = result['quality_level']
        
        if score >= 0.9:
            assert level == 'excellent', f"Score {score} should be 'excellent', got '{level}'"
        elif score >= 0.7:
            assert level == 'good', f"Score {score} should be 'good', got '{level}'"
        elif score >= 0.5:
            assert level == 'fair', f"Score {score} should be 'fair', got '{level}'"
        elif score > 0:
            assert level == 'poor', f"Score {score} should be 'poor', got '{level}'"
        else:
            assert level == 'failed', f"Score {score} should be 'failed', got '{level}'"
        
        # Time invariants
        assert result['end'] > result['start'], "End time must be after start time"
        
        # Optional field invariants
        if result['processing_time'] is not None:
            assert result['processing_time'] > 0, "Processing time must be positive"
        
        if result['retry_count'] is not None:
            assert result['retry_count'] >= 0, "Retry count must be non-negative"
    
    @given(
        text=st.text(min_size=0, max_size=500),
        start=st.floats(min_value=0.0, max_value=1800.0, exclude_nan=True, exclude_infinity=True),
        end=st.floats(min_value=0.1, max_value=1801.0, exclude_nan=True, exclude_infinity=True),
        quality_score=st.floats(min_value=0.0, max_value=1.0, exclude_nan=True, exclude_infinity=True)
    )
    def test_create_processing_result_factory(self, text, start, end, quality_score):
        """Test the factory function for ProcessingResult."""
        assume(end > start)  # Hypothesis assumption to filter valid inputs
        
        result = create_processing_result(text, start, end, quality_score)
        
        assert result['text'] == text
        assert result['start'] == start
        assert result['end'] == end
        assert result['quality_score'] == quality_score
        
        # Quality level should be correctly determined
        if quality_score >= 0.9:
            assert result['quality_level'] == 'excellent'
        elif quality_score >= 0.7:
            assert result['quality_level'] == 'good'
        elif quality_score >= 0.5:
            assert result['quality_level'] == 'fair'
        elif quality_score > 0:
            assert result['quality_level'] == 'poor'
        else:
            assert result['quality_level'] == 'failed'


class TestConfigurationProperties:
    """Property-based tests for configuration models."""
    
    @given(
        name=st.text(min_size=1, max_size=100),
        backend=st.sampled_from(['vllm', 'transformers']),
        dtype=st.sampled_from(['float32', 'float16', 'bfloat16']),
        device_map=st.text(min_size=1, max_size=20),
        max_batch_size=st.integers(min_value=1, max_value=1024)
    )
    def test_model_config_invariants(self, name, backend, dtype, device_map, max_batch_size):
        """Test ModelConfig validation invariants."""
        config = ModelConfig(
            name=name,
            backend=backend,
            dtype=dtype,
            device_map=device_map,
            max_batch_size=max_batch_size
        )
        
        # Immutability invariant
        assert config.name == name
        assert config.backend == backend
        assert config.dtype == dtype
        assert config.device_map == device_map
        assert config.max_batch_size == max_batch_size
        
        # Validation invariants  
        assert config.max_batch_size > 0
        assert config.name.strip() != ""
    
    @given(
        quality_level=st.sampled_from(['fast', 'balanced', 'best']),
        target_language=st.text(min_size=1, max_size=50),
        max_workers=st.integers(min_value=1, max_value=128),
        gpu_memory_limit=st.floats(min_value=0.1, max_value=0.95),
        timeout_seconds=st.integers(min_value=60, max_value=7200)
    )
    def test_processing_config_invariants(self, quality_level, target_language, max_workers, gpu_memory_limit, timeout_seconds):
        """Test ProcessingConfig validation invariants."""
        config = ProcessingConfig(
            quality_level=quality_level,
            target_language=target_language,
            max_workers=max_workers,
            gpu_memory_limit=gpu_memory_limit,
            timeout_seconds=timeout_seconds
        )
        
        # Validation invariants
        assert 0.1 <= config.gpu_memory_limit <= 0.95
        assert config.max_workers > 0
        assert config.timeout_seconds >= 60
        
        # Immutability
        assert config.quality_level == quality_level
        assert config.target_language == target_language


class TestErrorHandlingProperties:
    """Property-based tests for error handling models."""
    
    @given(
        operation=st.text(min_size=1, max_size=100),
        component=st.text(min_size=1, max_size=50),
        severity=st.sampled_from(ErrorSeverity),
        recovery_strategy=st.one_of(st.none(), st.text(min_size=1, max_size=200))
    )
    def test_error_context_invariants(self, operation, component, severity, recovery_strategy):
        """Test ErrorContext invariants."""
        start_time = time.time()
        context = ErrorContext(
            operation=operation,
            component=component,
            severity=severity,
            recovery_strategy=recovery_strategy
        )
        end_time = time.time()
        
        # Basic invariants
        assert context.operation == operation
        assert context.component == component  
        assert context.severity == severity
        assert context.recovery_strategy == recovery_strategy
        
        # Generated fields invariants
        assert len(context.error_id) == 8  # UUID[:8]
        assert start_time <= context.timestamp <= end_time + 0.1  # Allow small time diff
        
        # Severity comparison invariant
        assert context.severity < ErrorSeverity.CRITICAL or context.severity == ErrorSeverity.CRITICAL


# ====================== STATEFUL TESTING ======================

class BatchProcessingStateMachine(RuleBasedStateMachine):
    """Stateful testing for batch processing workflows."""
    
    def __init__(self):
        super().__init__()
        self.batches = []
        self.completed_batches = []
    
    segments = Bundle('segments')
    batches = Bundle('batches')
    
    @rule(target=segments, segment_data=audio_segment_strategy())
    def create_segment(self, segment_data):
        """Create a valid audio segment."""
        return validate_audio_segment(segment_data)
    
    @rule(target=batches, segment_list=st.lists(segments, min_size=1, max_size=10))
    def create_batch(self, segment_list):
        """Create a batch from segments."""
        batch = create_batch_with_id(segment_list)
        self.batches.append(batch)
        return batch
    
    @rule(batch=batches)
    def process_batch(self, batch):
        """Simulate batch processing."""
        # Simulate processing logic invariants
        assert len(batch['segments']) > 0
        assert batch['total_duration'] > 0
        assert batch['created_at'] <= time.time()
        
        # Move to completed
        if batch in self.batches:
            self.batches.remove(batch)
            self.completed_batches.append(batch)


# Test the stateful machine
TestBatchProcessing = BatchProcessingStateMachine.TestCase


# ====================== EDGE CASE DETECTION ======================

@given(
    segments=st.lists(
        audio_segment_strategy(),
        min_size=0,
        max_size=1000  # Test with large batches
    )
)
@settings(max_examples=50, deadline=5000)  # 5 second timeout
def test_large_batch_performance_invariants(segments):
    """Test that large batches don't break invariants."""
    if not segments:
        return
    
    validated_segments = [validate_audio_segment(seg) for seg in segments]
    batch = create_batch_with_id(validated_segments)
    
    # Performance invariants
    assert len(batch['segments']) == len(validated_segments)
    assert batch['total_duration'] >= 0
    assert batch['max_segment_length'] >= 0
    
    # Memory efficiency - batch shouldn't duplicate all data
    import sys
    batch_size = sys.getsizeof(batch)
    segments_size = sum(sys.getsizeof(seg) for seg in validated_segments)
    # Batch metadata should be small compared to segments
    assert batch_size < segments_size * 2  # Reasonable overhead


@given(
    start=st.floats(min_value=0.0, max_value=1e6),  # Very large times
    duration=st.floats(min_value=1e-6, max_value=1e6)  # Very small to very large durations
)
@settings(max_examples=100)
def test_extreme_time_values_handling(start, duration):
    """Test handling of extreme time values."""
    assume(not math.isnan(start) and not math.isnan(duration))
    assume(not math.isinf(start) and not math.isinf(duration))
    assume(duration > 0)
    
    end = start + duration
    
    try:
        segment = create_audio_segment(start, end)
        
        # Even with extreme values, invariants should hold
        assert segment['end'] > segment['start']
        assert segment['duration'] > 0
        assert segment['end_sample'] > segment['start_sample']
        
    except (OverflowError, ValueError):
        # It's acceptable to reject truly extreme values
        pass


if __name__ == '__main__':
    # Run property-based tests with increased examples for CI
    pytest.main([
        __file__, 
        '--hypothesis-show-statistics',
        '--hypothesis-verbosity=normal',
        '-v'
    ])