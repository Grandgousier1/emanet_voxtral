#!/usr/bin/env python3
"""
Tests unitaires pour parallel_processor optimisé
"""

import pytest
from unittest.mock import Mock, patch, MagicMock, AsyncMock
import torch
import numpy as np
import asyncio
from pathlib import Path
import sys
import os

# Ajout du répertoire parent au path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from parallel_processor import (
    B200OptimizedProcessor, HardwareConfigurator, AudioLoader, AudioBatcher,
    DiskSpaceManager
)
from domain_models import ErrorSeverity, AudioSegment


class TestHardwareConfigurator:
    """Tests pour HardwareConfigurator."""
    
    def test_hardware_configurator_init(self):
        """Test initialisation du configurateur hardware."""
        with patch('parallel_processor.get_optimal_config') as mock_config, \
             patch('parallel_processor.detect_hardware') as mock_hw:
            
            mock_config.return_value = {
                'audio': {'parallel_workers': 4, 'batch_size': 32},
                'vllm': {'semaphore_limit': 2}
            }
            mock_hw.return_value = {
                'cpu_count': 8,
                'gpu_memory_gb': 180
            }
            
            config = HardwareConfigurator()
            
            assert config.audio_workers == 4
            assert config.io_workers == 4  # cpu_count - audio_workers
            assert config.semaphore_limit == 2

    def test_optimize_batch_size_b200(self):
        """Test optimisation batch size pour B200."""
        with patch('parallel_processor.get_optimal_config') as mock_config, \
             patch('parallel_processor.detect_hardware') as mock_hw, \
             patch('parallel_processor.console') as mock_console:
            
            mock_config.return_value = {
                'audio': {'parallel_workers': 4, 'batch_size': 32},
                'vllm': {'semaphore_limit': 2}
            }
            mock_hw.return_value = {
                'cpu_count': 8,
                'gpu_memory_gb': 180  # B200 detected
            }
            
            config = HardwareConfigurator()
            
            # Should be optimized to 128 (min(32*4, 128))
            assert config.gpu_batch_size == 128
            mock_console.log.assert_called_with('[cyan]B200 detected: increasing batch size to 128[/cyan]')

    def test_optimize_batch_size_regular_gpu(self):
        """Test batch size normal pour GPU standard."""
        with patch('parallel_processor.get_optimal_config') as mock_config, \
             patch('parallel_processor.detect_hardware') as mock_hw:
            
            mock_config.return_value = {
                'audio': {'parallel_workers': 4, 'batch_size': 32},
                'vllm': {'semaphore_limit': 2}
            }
            mock_hw.return_value = {
                'cpu_count': 8,
                'gpu_memory_gb': 24  # Regular GPU
            }
            
            config = HardwareConfigurator()
            
            # Should keep base batch size
            assert config.gpu_batch_size == 32


class TestAudioLoader:
    """Tests pour AudioLoader."""
    
    def test_load_and_resample_success(self):
        """Test chargement audio réussi."""
        with patch('parallel_processor.sf.read') as mock_sf_read, \
             patch('parallel_processor.LIBROSA_AVAILABLE', True), \
             patch('parallel_processor.librosa.resample') as mock_resample, \
             patch('parallel_processor.console') as mock_console:
            
            # Audio à 44100 Hz
            mock_audio_data = np.random.random(44100)
            mock_sf_read.return_value = (mock_audio_data, 44100)
            mock_resample.return_value = np.random.random(16000)
            
            result = AudioLoader.load_and_resample(Path("/test/audio.wav"))
            
            assert result is not None
            audio_data, sr = result
            mock_sf_read.assert_called_once_with("/test/audio.wav")
            mock_resample.assert_called_once_with(mock_audio_data, orig_sr=44100, target_sr=16000)

    def test_load_and_resample_no_librosa(self):
        """Test chargement sans librosa disponible."""
        with patch('parallel_processor.sf.read') as mock_sf_read, \
             patch('parallel_processor.LIBROSA_AVAILABLE', False), \
             patch('parallel_processor.console') as mock_console:
            
            mock_audio_data = np.random.random(44100)
            mock_sf_read.return_value = (mock_audio_data, 44100)
            
            result = AudioLoader.load_and_resample(Path("/test/audio.wav"))
            
            assert result is not None
            audio_data, sr = result
            assert sr == 44100  # Original sample rate kept
            mock_console.log.assert_called_with('[yellow]Librosa not available, keeping original sample rate[/yellow]')

    def test_load_and_resample_failure(self):
        """Test échec chargement audio."""
        with patch('parallel_processor.sf.read', side_effect=Exception("Read error")), \
             patch('parallel_processor.console') as mock_console:
            
            result = AudioLoader.load_and_resample(Path("/test/nonexistent.wav"))
            
            assert result is None
            mock_console.log.assert_called_with('[red]Failed to load audio: Read error[/red]')


class TestAudioBatcher:
    """Tests pour AudioBatcher."""
    
    def test_create_optimal_batches_empty_segments(self):
        """Test avec segments vides."""
        batcher = AudioBatcher(gpu_batch_size=32)
        audio_data = np.random.random(16000)
        
        result = batcher.create_optimal_batches([], audio_data)
        
        assert result == []

    def test_create_optimal_batches_valid_segments(self):
        """Test création de batches avec segments valides."""
        batcher = AudioBatcher(gpu_batch_size=2)
        audio_data = np.random.random(32000)  # 2 secondes à 16kHz
        
        segments = [
            {'start': 0.0, 'end': 1.0},
            {'start': 1.0, 'end': 2.0},
            {'start': 2.0, 'end': 2.5}
        ]
        
        with patch('parallel_processor.console') as mock_console:
            batches = batcher.create_optimal_batches(segments, audio_data)
            
            assert len(batches) >= 1
            
            # Vérifier que chaque segment a les champs requis
            for batch in batches:
                audio_ref_found = False
                for item in batch:
                    if '_audio_data_ref' in item:
                        audio_ref_found = True
                        assert item['_audio_data_ref'] is audio_data
                    else:
                        assert 'duration' in item
                        assert 'start_sample' in item
                        assert 'end_sample' in item
                assert audio_ref_found

    def test_create_optimal_batches_invalid_segments(self):
        """Test gestion de segments invalides."""
        batcher = AudioBatcher(gpu_batch_size=32)
        audio_data = np.random.random(16000)
        
        invalid_segments = [
            {'start': -1.0, 'end': 0.5},  # start négatif
            {'start': 2.0, 'end': 1.0},   # end < start
            {'invalid': 'data'}            # champs manquants
        ]
        
        with patch('parallel_processor.console') as mock_console:
            batches = batcher.create_optimal_batches(invalid_segments, audio_data)
            
            # Les segments invalides doivent être corrigés avec des valeurs par défaut
            for batch in batches:
                for item in batch:
                    if '_audio_data_ref' not in item:
                        assert item['duration'] >= 0
                        assert item['start_sample'] >= 0
                        assert item['end_sample'] > item['start_sample']


class TestB200OptimizedProcessor:
    """Tests pour B200OptimizedProcessor."""
    
    @pytest.fixture
    def processor(self):
        """Fixture pour créer un processeur optimisé."""
        with patch('parallel_processor.HardwareConfigurator'), \
             patch('parallel_processor.AudioLoader'), \
             patch('parallel_processor.AudioBatcher'):
            return B200OptimizedProcessor()

    def test_validate_batch_data_success(self, processor):
        """Test validation batch réussie."""
        mock_audio_data = np.random.random(16000)
        mock_audio_data.shape = (16000,)  # Mock shape attribute
        
        batch = [
            {'_audio_data_ref': mock_audio_data},
            {'start': 0.0, 'end': 1.0, 'start_sample': 0, 'end_sample': 16000},
            {'start': 1.0, 'end': 2.0, 'start_sample': 16000, 'end_sample': 32000}
        ]
        
        audio_ref, segments = processor._validate_batch_data(batch)
        
        assert audio_ref is mock_audio_data
        assert len(segments) == 2
        assert segments[0]['start'] == 0.0
        assert segments[1]['start'] == 1.0

    def test_validate_batch_data_no_audio_ref(self, processor):
        """Test validation batch sans référence audio."""
        batch = [
            {'start': 0.0, 'end': 1.0, 'start_sample': 0, 'end_sample': 16000}
        ]
        
        with pytest.raises(ValueError, match="No audio data reference found in batch"):
            processor._validate_batch_data(batch)

    def test_validate_batch_data_corrupted_audio(self, processor):
        """Test validation batch avec audio corrompu."""
        mock_audio_data = Mock()
        # Mock corrupted audio data without shape attribute
        del mock_audio_data.shape
        
        batch = [{'_audio_data_ref': mock_audio_data}]
        
        with pytest.raises(AttributeError):
            processor._validate_batch_data(batch)

    def test_process_vllm_batch_success(self, processor):
        """Test traitement batch vLLM réussi."""
        mock_audio_data = np.random.random(32000)  # 2 secondes
        segments = [
            {'start': 0.0, 'end': 1.0, 'start_sample': 0, 'end_sample': 16000},
            {'start': 1.0, 'end': 2.0, 'start_sample': 16000, 'end_sample': 32000}
        ]
        
        mock_model = Mock()
        mock_output1 = Mock()
        mock_output1.outputs = [Mock(text="Texte traduit 1")]
        mock_output2 = Mock()
        mock_output2.outputs = [Mock(text="Texte traduit 2")]
        mock_model.generate.return_value = [mock_output1, mock_output2]
        
        mock_vllm_params_getter = Mock(return_value={})
        mock_prompt_getter = Mock(return_value="Translate this audio")
        
        results = processor._process_vllm_batch(
            segments, mock_audio_data, mock_model, "French",
            mock_vllm_params_getter, mock_prompt_getter
        )
        
        assert len(results) == 2
        assert results[0]['text'] == "Texte traduit 1"
        assert results[1]['text'] == "Texte traduit 2"
        assert results[0]['start'] == 0.0
        assert results[1]['start'] == 1.0

    def test_process_transformers_batch_success(self, processor):
        """Test traitement batch Transformers réussi."""
        mock_audio_data = np.random.random(32000)
        segments = [
            {'start': 0.0, 'end': 1.0, 'start_sample': 0, 'end_sample': 16000}
        ]
        
        mock_model = Mock()
        mock_model.device = 'cuda:0'
        mock_processor = Mock()
        mock_processor.return_value = {'input_features': torch.randn(1, 80, 3000)}
        mock_processor.batch_decode.return_value = ["Texte traduit"]
        mock_processor.tokenizer.eos_token_id = 50256
        
        # Mock des imports
        with patch('parallel_processor.get_transformers_generation_params', return_value={}), \
             patch('parallel_processor.validate_translation_quality') as mock_quality:
            
            mock_quality.return_value = {
                'overall_score': 0.85,
                'quality_level': 'good'
            }
            
            results = processor._process_transformers_batch(
                segments, mock_audio_data, mock_model, mock_processor
            )
            
            assert len(results) == 1
            assert results[0]['text'] == "Texte traduit"
            assert results[0]['quality_score'] == 0.85
            assert results[0]['quality_level'] == 'good'

    def test_handle_oom_recovery_single_segment(self, processor):
        """Test récupération OOM avec segment unique."""
        mock_audio_data = np.random.random(16000)
        segments = [{'start': 0.0, 'end': 1.0, 'start_sample': 0, 'end_sample': 16000}]
        
        with patch('parallel_processor.free_cuda_mem'), \
             patch('parallel_processor.logger'):
            
            results = processor._handle_oom_recovery(
                segments, mock_audio_data, Mock(), Mock(), "French", Mock(), Mock()
            )
            
            assert len(results) == 1
            assert "critical oom" in results[0]['text']

    def test_handle_oom_recovery_multiple_segments(self, processor):
        """Test récupération OOM avec plusieurs segments."""
        mock_audio_data = np.random.random(32000)
        segments = [
            {'start': 0.0, 'end': 1.0, 'start_sample': 0, 'end_sample': 16000},
            {'start': 1.0, 'end': 2.0, 'start_sample': 16000, 'end_sample': 32000},
            {'start': 2.0, 'end': 3.0, 'start_sample': 32000, 'end_sample': 48000},
            {'start': 3.0, 'end': 4.0, 'start_sample': 48000, 'end_sample': 64000}
        ]
        
        with patch('parallel_processor.free_cuda_mem'), \
             patch('parallel_processor.logger'), \
             patch.object(processor, '_process_batch_gpu') as mock_process:
            
            # Mock successful recovery
            mock_process.side_effect = [
                [{'text': 'first_half', 'start': 0.0, 'end': 1.0}],  # First half
                [{'text': 'second_half', 'start': 2.0, 'end': 3.0}]  # Second half  
            ]
            
            results = processor._handle_oom_recovery(
                segments, mock_audio_data, Mock(), Mock(), "French", Mock(), Mock()
            )
            
            assert len(results) == 2
            assert mock_process.call_count == 2

    @pytest.mark.asyncio
    async def test_process_batches_async_success(self, processor):
        """Test traitement asynchrone de batches."""
        mock_batches = [
            [{'_audio_data_ref': np.random.random(16000)}, 
             {'start': 0.0, 'end': 1.0, 'start_sample': 0, 'end_sample': 16000}]
        ]
        
        mock_model = Mock()
        mock_processor = Mock() 
        mock_progress = Mock()
        mock_task = Mock()
        
        # Mock du processeur GPU pour retourner des résultats
        with patch.object(processor, '_process_batch_gpu', return_value=[
            {'text': 'processed', 'start': 0.0, 'end': 1.0}
        ]):
            
            results = await processor._process_batches_async(
                mock_batches, mock_model, mock_processor, "French", mock_progress, mock_task
            )
            
            assert len(results) == 1
            assert results[0]['text'] == 'processed'

    @pytest.mark.asyncio  
    async def test_process_batches_async_timeout(self, processor):
        """Test timeout lors du traitement asynchrone."""
        mock_batches = [
            [{'_audio_data_ref': np.random.random(16000)}]
        ]
        
        mock_model = Mock()
        mock_processor = Mock()
        mock_progress = Mock()
        mock_task = Mock()
        
        # Mock timeout
        with patch.object(processor, '_process_batch_gpu', side_effect=asyncio.TimeoutError), \
             patch('parallel_processor.logger'):
            
            results = await processor._process_batches_async(
                mock_batches, mock_model, mock_processor, "French", mock_progress, mock_task
            )
            
            assert len(results) >= 1
            assert 'timeout' in results[0]['text']


class TestDiskSpaceManager:
    """Tests pour DiskSpaceManager."""
    
    def test_disk_space_manager_init(self):
        """Test initialisation du gestionnaire d'espace disque."""
        manager = DiskSpaceManager(max_work_size_gb=10)
        
        assert manager.max_work_size_gb == 10
        assert manager.work_dirs == []

    def test_create_work_dir(self):
        """Test création d'un répertoire de travail."""
        manager = DiskSpaceManager()
        
        with patch('pathlib.Path.mkdir') as mock_mkdir:
            work_dir = manager.create_work_dir()
            
            assert isinstance(work_dir, Path)
            assert 'work_' in str(work_dir)
            mock_mkdir.assert_called_once_with(exist_ok=True)
            assert work_dir in manager.work_dirs

    def test_cleanup_old_dirs(self):
        """Test nettoyage des anciens répertoires."""
        manager = DiskSpaceManager(max_work_size_gb=0.1)  # Very small limit
        
        # Mock plusieurs répertoires
        mock_dir1 = Mock(spec=Path)
        mock_dir1.exists.return_value = True
        mock_dir1.rglob.return_value = []  # Empty directory
        
        mock_dir2 = Mock(spec=Path)
        mock_dir2.exists.return_value = True
        mock_dir2.rglob.return_value = []
        
        mock_dir3 = Mock(spec=Path) 
        mock_dir3.exists.return_value = True
        mock_dir3.rglob.return_value = []
        
        manager.work_dirs = [mock_dir1, mock_dir2, mock_dir3]
        
        with patch('shutil.rmtree') as mock_rmtree, \
             patch('parallel_processor.console'):
            
            manager._cleanup_old_dirs()
            
            # Should keep last 2 directories
            assert len(manager.work_dirs) == 2
            assert mock_dir1 not in manager.work_dirs

    def test_cleanup_all(self):
        """Test nettoyage complet."""
        manager = DiskSpaceManager()
        
        mock_dir = Mock(spec=Path)
        mock_dir.exists.return_value = True
        manager.work_dirs = [mock_dir]
        
        with patch('shutil.rmtree') as mock_rmtree:
            manager.cleanup_all()
            
            mock_rmtree.assert_called_once_with(mock_dir)
            assert manager.work_dirs == []


if __name__ == '__main__':
    pytest.main([__file__, '-v'])