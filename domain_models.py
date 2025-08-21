#!/usr/bin/env python3
"""
Domain models for Voxtral - Type-safe data structures and protocols
"""

from typing import (
    TypedDict, Optional, List, Literal, Protocol, Union, Dict, Any,
    runtime_checkable, NewType
)
import time
from dataclasses import dataclass, field
from enum import Enum
import uuid
from pathlib import Path

# Type aliases for better semantics
AudioData = NewType('AudioData', bytes)
TimestampSeconds = NewType('TimestampSeconds', float)  
SampleIndex = NewType('SampleIndex', int)
QualityScore = NewType('QualityScore', float)  # 0.0 to 1.0


# Audio Domain Models
class AudioSegment(TypedDict):
    """Structure typée d'un segment audio avec validation."""
    start: TimestampSeconds
    end: TimestampSeconds
    duration: TimestampSeconds
    start_sample: SampleIndex
    end_sample: SampleIndex
    # Métadonnées optionnelles
    confidence: Optional[float]
    speaker_id: Optional[str]
    language: Optional[str]


class ProcessingResult(TypedDict):
    """Résultat de traitement d'un segment avec métriques de qualité."""
    text: str
    start: TimestampSeconds
    end: TimestampSeconds
    quality_score: QualityScore
    quality_level: Literal['excellent', 'good', 'fair', 'poor', 'failed']
    # Métadonnées de traitement
    model_used: Optional[str]
    processing_time: Optional[float]
    retry_count: Optional[int]
    error_message: Optional[str]


# Configuration Models avec validation
@dataclass(frozen=True, slots=True)
class ModelConfig:
    """Configuration immutable d'un modèle avec validation."""
    name: str
    backend: Literal['vllm', 'transformers']
    dtype: Literal['float32', 'float16', 'bfloat16']
    device_map: str
    max_batch_size: int
    
    def __post_init__(self):
        """Validation des paramètres."""
        if self.max_batch_size <= 0:
            raise ValueError(f"max_batch_size doit être > 0, reçu: {self.max_batch_size}")
        
        if self.name.strip() == "":
            raise ValueError("name ne peut pas être vide")


@dataclass(frozen=True, slots=True) 
class ProcessingConfig:
    """Configuration de traitement avec validation."""
    quality_level: Literal['fast', 'balanced', 'best']
    target_language: str
    max_workers: int
    gpu_memory_limit: float
    timeout_seconds: int
    
    def __post_init__(self):
        """Validation des paramètres."""
        if not (0.1 <= self.gpu_memory_limit <= 0.95):
            raise ValueError(f"gpu_memory_limit doit être entre 0.1 et 0.95, reçu: {self.gpu_memory_limit}")
        
        if self.max_workers <= 0:
            raise ValueError(f"max_workers doit être > 0, reçu: {self.max_workers}")
        
        if self.timeout_seconds < 60:
            raise ValueError(f"timeout_seconds doit être >= 60, reçu: {self.timeout_seconds}")


# Hardware Detection Models
@dataclass(frozen=True, slots=True)
class HardwareInfo:
    """Information hardware détectée."""
    cpu_count: int
    gpu_count: int
    gpu_memory_gb: float
    gpu_name: str
    cuda_version: Optional[str]
    is_b200: bool = field(init=False)
    
    def __post_init__(self):
        """Calcul automatique des propriétés dérivées."""
        # B200 detection based on memory and name
        is_b200 = (
            self.gpu_memory_gb >= 180 or 
            "B200" in self.gpu_name or
            "H200" in self.gpu_name
        )
        object.__setattr__(self, 'is_b200', is_b200)


# Error Domain avec hiérarchie
class ErrorSeverity(Enum):
    """Niveaux de sévérité des erreurs avec priorités."""
    INFO = ("info", 0)
    WARNING = ("warning", 1)
    ERROR = ("error", 2) 
    CRITICAL = ("critical", 3)
    
    def __init__(self, level_name: str, priority: int):
        self.level_name = level_name
        self.priority = priority
    
    def __lt__(self, other):
        """Comparaison pour tri par priorité."""
        if isinstance(other, ErrorSeverity):
            return self.priority < other.priority
        return NotImplemented


@dataclass(frozen=True, slots=True)
class ErrorContext:
    """Contexte d'une erreur avec métadonnées."""
    operation: str
    component: str
    severity: ErrorSeverity
    recovery_strategy: Optional[str] = None
    error_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp: float = field(default_factory=time.time)


# Batch Processing Models
@dataclass(frozen=True, slots=True)
class BatchMetrics:
    """Métriques d'un batch de traitement."""
    batch_id: str
    segment_count: int
    avg_duration: TimestampSeconds
    processing_time: float
    gpu_memory_used: float
    success_rate: QualityScore
    throughput_segments_per_second: float = field(init=False)
    
    def __post_init__(self):
        """Calcul automatique du throughput."""
        if self.processing_time > 0:
            throughput = self.segment_count / self.processing_time
        else:
            throughput = 0.0
        object.__setattr__(self, 'throughput_segments_per_second', throughput)


class BatchStatus(Enum):
    """État d'un batch avec transitions valides."""
    PENDING = "pending"
    PROCESSING = "processing" 
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    
    def can_transition_to(self, new_status: 'BatchStatus') -> bool:
        """Vérifie si une transition d'état est valide."""
        valid_transitions = {
            BatchStatus.PENDING: {BatchStatus.PROCESSING, BatchStatus.CANCELLED},
            BatchStatus.PROCESSING: {BatchStatus.COMPLETED, BatchStatus.FAILED, BatchStatus.CANCELLED},
            BatchStatus.COMPLETED: set(),  # État final
            BatchStatus.FAILED: {BatchStatus.PENDING},  # Peut être relancé
            BatchStatus.CANCELLED: set()  # État final
        }
        return new_status in valid_transitions.get(self, set())


@dataclass
class ProcessingBatch:
    """Batch mutable pour le traitement avec validation d'état."""
    segments: List[AudioSegment]
    audio_data_ref: AudioData
    status: BatchStatus = BatchStatus.PENDING
    metrics: Optional[BatchMetrics] = None
    error_context: Optional[ErrorContext] = None
    batch_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    created_at: float = field(default_factory=time.time)
    
    def transition_to(self, new_status: BatchStatus, error_context: Optional[ErrorContext] = None) -> bool:
        """Transition d'état sécurisée."""
        if not self.status.can_transition_to(new_status):
            return False
        
        self.status = new_status
        if error_context and new_status == BatchStatus.FAILED:
            self.error_context = error_context
        return True


# Protocols for type safety
@runtime_checkable
class AudioProcessor(Protocol):
    """Interface pour les processeurs audio."""
    
    def process_segments(
        self, 
        audio_path: Path,
        segments: List[AudioSegment],
        config: ProcessingConfig
    ) -> List[ProcessingResult]:
        """Traite des segments audio et retourne les résultats."""
        ...


@runtime_checkable  
class ModelManager(Protocol):
    """Interface pour la gestion de modèles."""
    
    def load_model(self, config: ModelConfig) -> Any:
        """Charge un modèle selon la configuration."""
        ...
    
    def __enter__(self) -> 'ModelManager':
        """Support du context manager."""
        ...
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Cleanup garanti des ressources."""
        ...


@runtime_checkable
class FeedbackProvider(Protocol):
    """Interface pour le feedback utilisateur."""
    
    def info(self, message: str) -> None:
        """Affiche un message informatif."""
        ...
    
    def warning(self, message: str) -> None:
        """Affiche un avertissement."""
        ...
    
    def error(self, message: str) -> None:
        """Affiche une erreur."""
        ...
    
    def success(self, message: str) -> None:
        """Affiche un message de succès."""
        ...


# Validation utilities
def validate_audio_segment(segment: Dict[str, Any]) -> AudioSegment:
    """
    Valide et convertit un dictionnaire en AudioSegment typé.
    
    Args:
        segment: Dictionnaire à valider
        
    Returns:
        AudioSegment validé
        
    Raises:
        ValueError: Si la validation échoue
    """
    required_fields = {'start', 'end', 'duration', 'start_sample', 'end_sample'}
    missing_fields = required_fields - set(segment.keys())
    if missing_fields:
        raise ValueError(f"Champs manquants: {missing_fields}")
    
    # Validation des contraintes métier
    if segment['start'] < 0:
        raise ValueError(f"start doit être >= 0, reçu: {segment['start']}")
    
    if segment['end'] <= segment['start']:
        raise ValueError(f"end ({segment['end']}) doit être > start ({segment['start']})")
    
    if segment['start_sample'] < 0:
        raise ValueError(f"start_sample doit être >= 0, reçu: {segment['start_sample']}")
    
    if segment['end_sample'] <= segment['start_sample']:
        raise ValueError(f"end_sample ({segment['end_sample']}) doit être > start_sample ({segment['start_sample']})")
    
    # Validation de cohérence durée vs échantillons (16kHz)
    expected_duration = (segment['end_sample'] - segment['start_sample']) / 16000
    actual_duration = segment['end'] - segment['start']
    if abs(expected_duration - actual_duration) > 0.1:  # Tolérance 100ms
        raise ValueError(f"Incohérence durée: calculée {expected_duration:.3f}s, fournie {actual_duration:.3f}s")
    
    return AudioSegment(
        start=TimestampSeconds(segment['start']),
        end=TimestampSeconds(segment['end']),
        duration=TimestampSeconds(segment['duration']),
        start_sample=SampleIndex(segment['start_sample']),
        end_sample=SampleIndex(segment['end_sample']),
        confidence=segment.get('confidence'),
        speaker_id=segment.get('speaker_id'),
        language=segment.get('language')
    )


def validate_processing_result(result: Dict[str, Any]) -> ProcessingResult:
    """
    Valide et convertit un dictionnaire en ProcessingResult typé.
    
    Args:
        result: Dictionnaire à valider
        
    Returns:
        ProcessingResult validé
        
    Raises:
        ValueError: Si la validation échoue
    """
    required_fields = {'text', 'start', 'end', 'quality_score', 'quality_level'}
    missing_fields = required_fields - set(result.keys())
    if missing_fields:
        raise ValueError(f"Champs manquants: {missing_fields}")
    
    # Validation quality_score
    quality_score = result['quality_score']
    if not (0.0 <= quality_score <= 1.0):
        raise ValueError(f"quality_score doit être entre 0.0 et 1.0, reçu: {quality_score}")
    
    # Validation quality_level
    valid_levels = {'excellent', 'good', 'fair', 'poor', 'failed'}
    if result['quality_level'] not in valid_levels:
        raise ValueError(f"quality_level invalide: {result['quality_level']}, valides: {valid_levels}")
    
    return ProcessingResult(
        text=result['text'],
        start=TimestampSeconds(result['start']),
        end=TimestampSeconds(result['end']),
        quality_score=QualityScore(quality_score),
        quality_level=result['quality_level'],
        model_used=result.get('model_used'),
        processing_time=result.get('processing_time'),
        retry_count=result.get('retry_count'),
        error_message=result.get('error_message')
    )