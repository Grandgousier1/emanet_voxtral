#!/usr/bin/env python3
"""
_test_parallel_processor.py - Unit tests for the parallel processor
"""

import unittest
import numpy as np

from parallel_processor import B200OptimizedProcessor

class TestParallelProcessor(unittest.TestCase):
    def test_create_optimal_batches(self):
        """Test the _create_optimal_batches method."""
        
        processor = B200OptimizedProcessor()
        
        # Create mock segments
        segments = [
            {'start': 0, 'end': 1},
            {'start': 1, 'end': 2},
            {'start': 2, 'end': 3},
            {'start': 3, 'end': 8},
            {'start': 8, 'end': 9},
            {'start': 9, 'end': 10},
        ]
        
        # Create mock audio data
        audio_data = np.random.randn(10 * 16000).astype(np.float32)
        
        # Create batches
        batches = processor._create_optimal_batches(segments, audio_data)
        
        # Check that the batches are created correctly
        self.assertEqual(len(batches), 2)
        self.assertEqual(len(batches[0]), 3)
        self.assertEqual(len(batches[1]), 3)

if __name__ == '__main__':
    unittest.main()