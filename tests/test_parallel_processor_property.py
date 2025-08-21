#!/usr/bin/env python3
"""
Property-based tests for parallel_processor module.

Tests concurrent execution properties, batch processing invariants,
and GPU memory management under various conditions.
"""

import pytest
from hypothesis import given, strategies as st, assume, settings, Verbosity, HealthCheck
from hypothesis.stateful import RuleBasedStateMachine, Bundle, rule, initialize, invariant
from unittest.mock import Mock, patch, MagicMock
import asyncio
import numpy as np
import time
from typing import List, Dict, Any
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from parallel_processor import (
    HardwareConfigurator, AudioLoader, AudioBatcher, B200OptimizedProcessor,
    DiskSpaceManager
)
from domain_models import AudioSegment, create_audio_segment


# ====================== STRATEGIES ======================

@st.composite 
def hardware_config_strategy(draw):
    """Generate realistic hardware configurations."""
    return {
        'cpu_count': draw(st.integers(min_value=1, max_value=128)),
        'gpu_memory_gb': draw(st.floats(min_value=1.0, max_value=200.0)),
        'audio': {
            'parallel_workers': draw(st.integers(min_value=1, max_value=32)),
            'batch_size': draw(st.integers(min_value=1, max_value=256))
        },
        'vllm': {
            'semaphore_limit': draw(st.integers(min_value=1, max_value=8))
        }
    }


@st.composite
def audio_batch_strategy(draw, min_segments=1, max_segments=50):
    """Generate realistic audio batch data."""
    num_segments = draw(st.integers(min_value=min_segments, max_value=max_segments))
    segments = []
    
    current_time = 0.0
    for _ in range(num_segments):
        duration = draw(st.floats(min_value=0.1, max_value=10.0))
        start_time = current_time
        end_time = start_time + duration
        
        segment_data = {
            'start': start_time,
            'end': end_time,
            'duration': duration,
            'start_sample': int(start_time * 16000),
            'end_sample': int(end_time * 16000)
        }
        segments.append(segment_data)
        current_time = end_time + draw(st.floats(min_value=0.0, max_value=1.0))  # Gap between segments
    
    # Add audio data reference
    total_samples = int(current_time * 16000) + 16000  # Add buffer
    audio_data = np.random.random(total_samples).astype(np.float32)
    
    return segments, audio_data


# ====================== HARDWARE CONFIGURATION PROPERTIES ======================

class TestHardwareConfiguratorProperties:
    """Property-based tests for HardwareConfigurator."""
    
    @given(config_data=hardware_config_strategy())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_hardware_configuration_invariants(self, config_data):
        """Test hardware configuration invariants."""
        with patch('parallel_processor.get_optimal_config', return_value=config_data['audio']), \
             patch('parallel_processor.detect_hardware', return_value=config_data):
            
            configurator = HardwareConfigurator()
            
            # Worker allocation invariants
            assert configurator.audio_workers > 0
            assert configurator.io_workers >= 0
            assert configurator.audio_workers + configurator.io_workers <= config_data['cpu_count'] * 2  # Reasonable upper bound
            
            # GPU batch size invariants
            assert configurator.gpu_batch_size > 0
            assert configurator.gpu_batch_size >= config_data['audio']['batch_size'] / 4  # Shouldn't be too small
            assert configurator.gpu_batch_size <= config_data['audio']['batch_size'] * 8  # Shouldn't be too large
            
            # B200 optimization invariants
            if config_data.get('gpu_memory_gb', 0) >= 100:
                # Should optimize for B200
                assert configurator.gpu_batch_size >= config_data['audio']['batch_size']
            
            # Semaphore limit invariant
            assert configurator.semaphore_limit > 0
            assert configurator.semaphore_limit <= 16  # Reasonable upper bound


# ====================== AUDIO BATCH PROCESSING PROPERTIES ======================

class TestAudioBatcherProperties:
    """Property-based tests for AudioBatcher."""
    
    @given(
        segments_and_audio=audio_batch_strategy(min_segments=1, max_segments=100),
        gpu_batch_size=st.integers(min_value=1, max_value=64)
    )
    @settings(max_examples=80)
    def test_batch_creation_invariants(self, segments_and_audio, gpu_batch_size):
        """Test that batch creation maintains invariants regardless of input."""
        segments, audio_data = segments_and_audio
        batcher = AudioBatcher(gpu_batch_size=gpu_batch_size)
        
        with patch('parallel_processor.console'):
            batches = batcher.create_optimal_batches(segments, audio_data)
        
        # Core invariants
        assert isinstance(batches, list)
        
        if segments:  # Non-empty input should produce non-empty output
            assert len(batches) > 0
            
            # Verify all segments are included
            total_segments_in_batches = 0
            for batch in batches:
                batch_segments = [item for item in batch if '_audio_data_ref' not in item]
                total_segments_in_batches += len(batch_segments)
            
            assert total_segments_in_batches == len(segments), \
                f"Lost segments: input {len(segments)}, output {total_segments_in_batches}"
            
            # Each batch should have audio reference
            for batch in batches:
                audio_refs = [item for item in batch if '_audio_data_ref' in item]
                assert len(audio_refs) == 1, "Each batch should have exactly one audio reference"
                assert audio_refs[0]['_audio_data_ref'] is audio_data
            
            # Batch size constraints
            for batch in batches:
                batch_segments = [item for item in batch if '_audio_data_ref' not in item]
                assert len(batch_segments) <= gpu_batch_size, \
                    f"Batch too large: {len(batch_segments)} > {gpu_batch_size}"
                assert len(batch_segments) > 0, "No empty batches"
        else:  # Empty input should produce empty output
            assert len(batches) == 0
    
    @given(
        segments_and_audio=audio_batch_strategy(min_segments=5, max_segments=20),
        gpu_batch_size=st.integers(min_value=2, max_value=10)
    )
    def test_duration_based_grouping_properties(self, segments_and_audio, gpu_batch_size):
        """Test that duration-based grouping improves batch efficiency."""
        segments, audio_data = segments_and_audio
        batcher = AudioBatcher(gpu_batch_size=gpu_batch_size)
        
        with patch('parallel_processor.console'):
            batches = batcher.create_optimal_batches(segments, audio_data)
        
        if len(batches) > 1:
            # Verify that segments within batches have similar durations
            for batch in batches:
                batch_segments = [item for item in batch if '_audio_data_ref' not in item]
                if len(batch_segments) > 1:
                    durations = [seg['duration'] for seg in batch_segments]
                    min_duration = min(durations)
                    max_duration = max(durations)
                    
                    # Segments in same batch should be reasonably similar in duration
                    # (allowing for some variance due to bucketing algorithm)
                    if min_duration > 0:
                        duration_ratio = max_duration / min_duration
                        assert duration_ratio <= 10.0, \
                            f"Duration variance too high in batch: {duration_ratio}"
    
    @given(
        segments_and_audio=audio_batch_strategy(min_segments=1, max_segments=5),
        gpu_batch_size=st.integers(min_value=1, max_value=3)
    )
    def test_sample_index_consistency(self, segments_and_audio, gpu_batch_size):
        """Test that sample indices remain consistent through batching."""
        segments, audio_data = segments_and_audio
        batcher = AudioBatcher(gpu_batch_size=gpu_batch_size)
        
        with patch('parallel_processor.console'):
            batches = batcher.create_optimal_batches(segments, audio_data)
        
        for batch in batches:
            for item in batch:
                if '_audio_data_ref' not in item:
                    # Sample indices should be valid
                    assert item['start_sample'] >= 0
                    assert item['end_sample'] > item['start_sample']
                    assert item['end_sample'] <= len(audio_data)
                    
                    # Duration consistency
                    expected_duration = (item['end_sample'] - item['start_sample']) / 16000
                    assert abs(item['duration'] - expected_duration) < 0.1


# ====================== DISK SPACE MANAGEMENT PROPERTIES ======================

class TestDiskSpaceManagerProperties:
    """Property-based tests for DiskSpaceManager."""
    
    @given(
        max_work_size_gb=st.floats(min_value=0.1, max_value=100.0),
        num_dirs=st.integers(min_value=0, max_value=20)
    )
    @settings(max_examples=50)
    def test_disk_manager_invariants(self, max_work_size_gb, num_dirs):
        """Test disk space manager maintains invariants."""
        manager = DiskSpaceManager(max_work_size_gb=max_work_size_gb)
        
        # Initial state invariants
        assert manager.max_work_size_gb == max_work_size_gb
        assert len(manager.work_dirs) == 0
        
        # Simulate directory creation
        with patch('pathlib.Path.mkdir'), \
             patch('pathlib.Path.exists', return_value=True), \
             patch('pathlib.Path.rglob', return_value=[]), \
             patch('shutil.rmtree'):
            
            created_dirs = []
            for _ in range(num_dirs):
                work_dir = manager.create_work_dir()
                created_dirs.append(work_dir)
                
                # Directory tracking invariants
                assert work_dir in manager.work_dirs
                assert len(manager.work_dirs) <= num_dirs + 1  # +1 for current dir
            
            # Cleanup invariants
            manager.cleanup_all()
            assert len(manager.work_dirs) == 0


# ====================== STATEFUL TESTING FOR CONCURRENT OPERATIONS ======================

class ParallelProcessorStateMachine(RuleBasedStateMachine):
    """Stateful testing for parallel processor concurrent operations."""
    
    def __init__(self):
        super().__init__()
        self.active_batches = []
        self.completed_batches = []
        self.processor = None
    
    batches = Bundle('batches')
    
    @initialize()
    def setup_processor(self):
        """Initialize the processor with mocked dependencies."""
        with patch('parallel_processor.HardwareConfigurator'), \
             patch('parallel_processor.AudioLoader'), \
             patch('parallel_processor.AudioBatcher'):
            self.processor = B200OptimizedProcessor()
    
    @rule(target=batches)
    def create_batch(self):
        """Create a new batch for processing."""
        segments, audio_data = audio_batch_strategy(min_segments=1, max_segments=5).example()
        
        batch = []
        batch.append({'_audio_data_ref': audio_data})
        for segment_data in segments:
            batch.append(segment_data)
        
        self.active_batches.append(batch)
        return batch
    
    @rule(batch=batches)
    def validate_batch(self, batch):
        """Validate batch structure."""
        audio_refs = [item for item in batch if '_audio_data_ref' in item]
        segments = [item for item in batch if '_audio_data_ref' not in item]
        
        # Batch structure invariants
        assert len(audio_refs) == 1, "Exactly one audio reference per batch"
        assert len(segments) > 0, "At least one segment per batch"
        
        # Audio reference validity
        audio_data = audio_refs[0]['_audio_data_ref']
        assert hasattr(audio_data, 'shape') or hasattr(audio_data, '__len__')
        
        # Segment validity
        for segment in segments:
            assert 'start' in segment
            assert 'end' in segment
            assert 'start_sample' in segment
            assert 'end_sample' in segment
            assert segment['end'] > segment['start']
            assert segment['end_sample'] > segment['start_sample']
    
    @rule(batch=batches)
    def process_batch_validation(self, batch):
        """Test batch validation logic."""
        if self.processor:
            try:
                audio_data_ref, segments = self.processor._validate_batch_data(batch)
                
                # Validation invariants
                assert audio_data_ref is not None
                assert len(segments) > 0
                assert all('_audio_data_ref' not in seg for seg in segments)
                
            except ValueError:
                # Validation rejection is acceptable for invalid batches
                pass
    
    @invariant()
    def batches_consistency(self):
        """Invariant: batch management remains consistent."""
        total_batches = len(self.active_batches) + len(self.completed_batches)
        assert total_batches >= 0
        
        # No batch should be in both lists
        for batch in self.active_batches:
            assert batch not in self.completed_batches


# Convert stateful machine to test case
TestParallelProcessorStateful = ParallelProcessorStateMachine.TestCase


# ====================== MEMORY AND PERFORMANCE PROPERTIES ======================

@given(
    batch_sizes=st.lists(
        st.integers(min_value=1, max_value=100), 
        min_size=1, 
        max_size=20
    ),
    max_segments=st.integers(min_value=10, max_value=1000)
)
@settings(max_examples=30, deadline=10000)  # 10 second timeout
def test_memory_usage_properties(batch_sizes, max_segments):
    """Test memory usage properties under various batch configurations."""
    import gc
    import psutil
    import os
    
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss
    
    # Create multiple batches with different sizes
    batches = []
    for batch_size in batch_sizes[:5]:  # Limit to first 5 to avoid excessive memory usage
        segments, audio_data = audio_batch_strategy(
            min_segments=min(batch_size, max_segments // 10),
            max_segments=min(batch_size * 2, max_segments)
        ).example()
        
        batcher = AudioBatcher(gpu_batch_size=batch_size)
        with patch('parallel_processor.console'):
            batch_list = batcher.create_optimal_batches(segments, audio_data)
        
        batches.extend(batch_list)
    
    # Memory growth should be reasonable
    current_memory = process.memory_info().rss
    memory_growth = current_memory - initial_memory
    
    # Memory growth should be proportional to data size (with reasonable overhead)
    total_segments = sum(len(batch) for batch in batches)
    if total_segments > 0:
        memory_per_segment = memory_growth / total_segments
        # Should not exceed 1MB per segment (very generous bound)
        assert memory_per_segment < 1024 * 1024, \
            f"Memory usage too high: {memory_per_segment} bytes per segment"
    
    # Cleanup
    del batches
    gc.collect()


@given(
    num_workers=st.integers(min_value=1, max_value=8),
    segments_per_batch=st.integers(min_value=1, max_value=20)
)
@settings(max_examples=20)
def test_concurrency_properties(num_workers, segments_per_batch):
    """Test properties of concurrent batch processing."""
    # Create test data
    segments, audio_data = audio_batch_strategy(
        min_segments=segments_per_batch,
        max_segments=segments_per_batch + 5
    ).example()
    
    # Test semaphore behavior
    semaphore_limit = min(num_workers, 4)  # Reasonable limit
    
    with patch('parallel_processor.HardwareConfigurator') as mock_hw:
        mock_hw.return_value.semaphore_limit = semaphore_limit
        mock_hw.return_value.gpu_batch_size = 32
        mock_hw.return_value.audio_workers = num_workers
        mock_hw.return_value.io_workers = max(1, num_workers // 2)
        
        processor = B200OptimizedProcessor()
        
        # Semaphore properties
        assert hasattr(processor.hardware_config, 'semaphore_limit')
        assert processor.hardware_config.semaphore_limit <= num_workers
        assert processor.hardware_config.semaphore_limit > 0


if __name__ == '__main__':
    # Run with hypothesis statistics
    pytest.main([
        __file__, 
        '--hypothesis-show-statistics',
        '--hypothesis-verbosity=verbose',
        '--tb=short',
        '-v'
    ])