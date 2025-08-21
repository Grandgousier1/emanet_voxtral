#!/usr/bin/env python3
"""
_test_main.py - Unit tests for the main script
"""

import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path

from utils.processing_utils import get_audio_path, process_audio

class TestMain(unittest.TestCase):
    @patch('utils.audio_utils.enhanced_download_audio')
    def test_get_audio_path_url(self, mock_download_audio):
        """Test the get_audio_path function with a URL."""
        
        mock_feedback = MagicMock()
        mock_download_audio.return_value = Path("/tmp/audio.wav")
        
        audio_path = get_audio_path("http://example.com/video.mp4", Path("/tmp"), mock_feedback)
        
        self.assertEqual(audio_path, Path("/tmp/audio.wav"))
        mock_download_audio.assert_called_once()

    def test_get_audio_path_local_file(self):
        """Test the get_audio_path function with a local file."""
        
        mock_feedback = MagicMock()
        
        # Create a dummy file
        with open("dummy_audio.wav", "w") as f:
            f.write("dummy audio")
            
        audio_path = get_audio_path("dummy_audio.wav", Path("/tmp"), mock_feedback)
        
        self.assertEqual(audio_path, Path("dummy_audio.wav"))
        
        # Clean up the dummy file
        import os
        os.remove("dummy_audio.wav")

    @patch('utils.audio_utils.enhanced_vad_segments')
    @patch('utils.processing_utils.enhanced_voxtral_process')
    def test_process_audio(self, mock_voxtral_process, mock_vad_segments):
        """Test the process_audio function."""
        
        mock_feedback = MagicMock()
        mock_vad_segments.return_value = [{'start': 0, 'end': 1}]
        mock_voxtral_process.return_value = [{'text': 'hello', 'start': 0, 'end': 1}]
        
        translated_segments = process_audio(Path("dummy_audio.wav"), mock_feedback, True)
        
        self.assertEqual(len(translated_segments), 1)
        self.assertEqual(translated_segments[0]['text'], 'hello')
        mock_vad_segments.assert_called_once()
        mock_voxtral_process.assert_called_once()

if __name__ == '__main__':
    unittest.main()