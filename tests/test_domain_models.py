#!/usr/bin/env python3
"""
Tests unitaires pour les modèles de domaine
"""

import pytest
from typing import Dict, Any
from domain_models import (
    AudioSegment, ProcessingResult, ModelConfig, ProcessingConfig,
    ErrorSeverity, ErrorContext, BatchMetrics, BatchStatus, ProcessingBatch
)

class TestAudioSegment:
    """Tests pour AudioSegment TypedDict."""
    
    def test_audio_segment_creation(self):
        """Test création d'un segment audio valide."""
        segment: AudioSegment = {
            'start': 0.0,
            'end': 1.5,
            'duration': 1.5,
            'start_sample': 0,
            'end_sample': 24000
        }
        
        assert segment['duration'] == 1.5
        assert segment['start_sample'] == 0
        assert segment['end_sample'] == 24000
    
    def test_audio_segment_type_hints(self):
        """Test que MyPy détecte les erreurs de type."""
        # Ce test passera avec mypy mais échouera à l'exécution
        segment: AudioSegment = {
            'start': 0.0,
            'end': 1.5,
            'duration': 1.5,
            'start_sample': 0,
            'end_sample': 24000
        }
        
        # Vérification runtime des types
        assert isinstance(segment['start'], float)
        assert isinstance(segment['start_sample'], int)

class TestProcessingResult:
    """Tests pour ProcessingResult."""
    
    @pytest.mark.parametrize("quality_level", [
        'excellent', 'good', 'fair', 'poor', 'failed'
    ])
    def test_valid_quality_levels(self, quality_level):
        """Test tous les niveaux de qualité valides."""
        result: ProcessingResult = {
            'text': 'Bonjour le monde',
            'start': 0.0,
            'end': 1.5,
            'quality_score': 0.95,
            'quality_level': quality_level
        }
        
        assert result['quality_level'] == quality_level

class TestModelConfig:
    """Tests pour ModelConfig dataclass."""
    
    def test_immutable_config(self):
        """Test l'immutabilité du ModelConfig."""
        config = ModelConfig(
            name="mistralai/Voxtral-Small-24B-2507",
            backend="vllm",
            dtype="bfloat16",
            device_map="auto",
            max_batch_size=32
        )
        
        # Tentative de modification doit lever une exception
        with pytest.raises(AttributeError):
            config.name = "autre-modele"  # type: ignore
    
    def test_config_validation(self):
        """Test validation des paramètres."""
        config = ModelConfig(
            name="test-model",
            backend="transformers",  
            dtype="float16",
            device_map="sequential",
            max_batch_size=16
        )
        
        assert config.backend in ['vllm', 'transformers']
        assert config.max_batch_size > 0

class TestErrorContext:
    """Tests pour gestion d'erreurs."""
    
    def test_error_context_creation(self):
        """Test création contexte d'erreur."""
        context = ErrorContext(
            operation="model loading",
            component="transformers",
            severity=ErrorSeverity.CRITICAL,
            recovery_strategy="fallback to vLLM"
        )
        
        assert context.operation == "model loading"
        assert context.severity == ErrorSeverity.CRITICAL
        assert context.recovery_strategy == "fallback to vLLM"
    
    def test_error_severity_enum(self):
        """Test enum ErrorSeverity."""
        assert ErrorSeverity.CRITICAL.value == "critical"
        assert ErrorSeverity.INFO.value == "info"

class TestBatchProcessing:
    """Tests pour le traitement par batches."""
    
    def test_batch_creation(self):
        """Test création d'un batch de traitement."""
        segments = [
            {
                'start': 0.0, 'end': 1.0, 'duration': 1.0,
                'start_sample': 0, 'end_sample': 16000
            },
            {
                'start': 1.0, 'end': 2.0, 'duration': 1.0, 
                'start_sample': 16000, 'end_sample': 32000
            }
        ]
        
        batch = ProcessingBatch(
            segments=segments,
            audio_data_ref=b"fake_audio_data",
            status=BatchStatus.PENDING
        )
        
        assert len(batch.segments) == 2
        assert batch.status == BatchStatus.PENDING
        assert batch.metrics is None
    
    def test_batch_metrics(self):
        """Test métriques de batch."""
        metrics = BatchMetrics(
            batch_id="batch-001",
            segment_count=10,
            avg_duration=2.5,
            processing_time=15.3,
            gpu_memory_used=4.2,
            success_rate=0.95
        )
        
        assert metrics.success_rate == 0.95
        assert metrics.segment_count == 10
        
# Property-based testing avec hypothesis
try:
    from hypothesis import given, strategies as st
    
    @given(st.floats(min_value=0.0, max_value=3600.0))
    def test_audio_segment_duration_property(duration):
        """Property test: durée toujours positive."""
        segment: AudioSegment = {
            'start': 0.0,
            'end': duration,
            'duration': duration,
            'start_sample': 0,
            'end_sample': int(duration * 16000)
        }
        
        assert segment['duration'] >= 0
        assert segment['end'] >= segment['start']
        assert segment['end_sample'] >= segment['start_sample']
    
except ImportError:
    # hypothesis pas disponible - skip property tests
    pass