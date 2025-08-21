# Suggestions de Refactoring - Voxtral

## 1. main.py - Décomposition de la fonction principale

### Observation
La fonction `main()` a une complexité cyclomatique de 18 avec des responsabilités mélangées (parsing, validation, orchestration).

### Code "Avant"
```python
def main():
    start_time = time.time()
    args = parse_args()
    
    # 1. Setup services (logging and feedback)
    setup_logging(log_level=args.log_level.upper())
    feedback = get_feedback(debug_mode=args.debug)
    error_resolver = ErrorResolver(feedback)
    # ... 80 lignes de logique mélangée
```

### Code "Après" 
```python
@dataclass
class AppConfig:
    """Configuration immutable pour l'application."""
    args: argparse.Namespace
    feedback: CLIFeedback
    start_time: float

def create_app_config() -> AppConfig:
    """Crée la configuration de l'application."""
    args = parse_args()
    setup_logging(log_level=args.log_level.upper())
    feedback = get_feedback(debug_mode=args.debug)
    return AppConfig(args, feedback, time.time())

def validate_environment(config: AppConfig) -> bool:
    """Valide l'environnement d'exécution."""
    if config.args.force:
        return True
        
    validations = [
        ("Token Hugging Face", validate_hf_token()),
        ("Espace Disque", check_disk_space('.', required_gb=25)),
        ("Dépendances GPU", enhanced_preflight_checks(config.feedback))
    ]
    
    for check_name, is_valid in validations:
        if not is_valid:
            config.feedback.error(f"Validation échouée: {check_name}")
            return False
    
    return True

def main() -> int:
    """Point d'entrée principal - orchestration pure."""
    config = create_app_config()
    
    if not validate_environment(config):
        return 1
    
    if config.args.validate_only or config.args.dry_run:
        return handle_validation_mode(config)
    
    return process_media(config)
```

### Justification
- **Complexité réduite** : De 18 à 4 par fonction
- **Testabilité** : Chaque fonction pure est testable indépendamment  
- **Lisibilité** : Intent clair à chaque niveau d'abstraction

## 2. parallel_processor.py - Algorithme de batching

### Observation
L'algorithme actuel de création de batches est O(n²) avec une logique imbriquée difficile à suivre.

### Code "Avant"
```python
def create_optimal_batches(self, segments: List[Dict], audio_data) -> List[List[Dict]]:
    # Sort segments by duration
    segments_with_duration = sorted(segments, key=lambda s: s['duration'])
    
    batches = []
    current_batch = []
    
    for segment in segments_with_duration:
        if not current_batch:
            current_batch.append(segment)
        else:
            # Check if the new segment is close in length...
            last_segment_duration = current_batch[-1]['duration']
            duration_diff = abs(segment['duration'] - last_segment_duration)
            
            if duration_diff > 1.0 or len(current_batch) >= self.gpu_batch_size:
                batches.append(current_batch)
                current_batch = [segment]
            else:
                current_batch.append(segment)
```

### Code "Après"
```python
from dataclasses import dataclass
from typing import Iterator

@dataclass(frozen=True)
class BatchingStrategy:
    """Stratégie de création de batches immutable."""
    max_batch_size: int
    duration_tolerance: float = 1.0
    
    def group_by_duration(self, segments: List[Dict]) -> Iterator[List[Dict]]:
        """Groupe les segments par durée similaire."""
        if not segments:
            return
            
        sorted_segments = sorted(segments, key=lambda s: s['duration'])
        current_batch = [sorted_segments[0]]
        
        for segment in sorted_segments[1:]:
            if (self._should_start_new_batch(current_batch, segment)):
                yield current_batch
                current_batch = [segment]
            else:
                current_batch.append(segment)
        
        if current_batch:
            yield current_batch
    
    def _should_start_new_batch(self, current_batch: List[Dict], segment: Dict) -> bool:
        """Détermine si un nouveau batch doit être créé."""
        return (
            len(current_batch) >= self.max_batch_size or
            abs(segment['duration'] - current_batch[-1]['duration']) > self.duration_tolerance
        )

def create_optimal_batches(self, segments: List[Dict], audio_data) -> List[List[Dict]]:
    """Crée des batches optimaux avec une stratégie configurable."""
    strategy = BatchingStrategy(max_batch_size=self.gpu_batch_size)
    
    batches = [
        list(batch) + [{'_audio_data_ref': audio_data}]
        for batch in strategy.group_by_duration(segments)
    ]
    
    self._log_batch_statistics(batches, len(segments))
    return batches
```

### Justification
- **Séparation des préoccupations** : Stratégie de batching extraite
- **Immutabilité** : Configuration figée, moins de bugs  
- **Complexité O(n log n)** : Plus efficace pour gros volumes
- **Testabilité** : `BatchingStrategy` testable indépendamment

## 3. utils/auth_manager.py - Gestion sécurisée des tokens

### Observation
La classe `TokenManager` mélange trop de responsabilités et la logique de fallback est complexe.

### Code "Avant"
```python
def get_hf_token(self) -> Optional[str]:
    """Get HuggingFace token from various sources in priority order."""
    
    # 1. Environment variable (highest priority)
    token = os.getenv('HF_TOKEN') or os.getenv('HUGGINGFACE_HUB_TOKEN')
    if token:
        self.feedback.debug("Using HF token from environment variable")
        return token
    
    # 2. .env file
    if self.env_file.exists():
        token = self._get_token_from_env_file()
        if token:
            self.feedback.debug("Using HF token from .env file")
            return token
    # ... plus de logique imbriquée
```

### Code "Après"  
```python
from abc import ABC, abstractmethod
from typing import Optional

class TokenSource(ABC):
    """Interface pour les sources de tokens."""
    
    @abstractmethod
    def get_token(self) -> Optional[str]:
        """Récupère le token depuis cette source."""
        pass
    
    @property
    @abstractmethod 
    def source_name(self) -> str:
        """Nom de la source pour le logging."""
        pass

class EnvironmentTokenSource(TokenSource):
    """Source de tokens depuis variables d'environnement."""
    
    def get_token(self) -> Optional[str]:
        return os.getenv('HF_TOKEN') or os.getenv('HUGGINGFACE_HUB_TOKEN')
    
    @property
    def source_name(self) -> str:
        return "environment variable"

class EnvFileTokenSource(TokenSource):
    """Source de tokens depuis fichier .env."""
    
    def __init__(self, env_file: Path, decryptor: 'TokenDecryptor'):
        self.env_file = env_file
        self.decryptor = decryptor
    
    def get_token(self) -> Optional[str]:
        if not self.env_file.exists():
            return None
            
        # Logique d'extraction simplifiée
        raw_token = self._extract_raw_token()
        return self.decryptor.decrypt(raw_token) if raw_token else None
    
    @property
    def source_name(self) -> str:
        return ".env file"

class TokenManager:
    """Gestionnaire de tokens avec sources multiples."""
    
    def __init__(self, feedback: CLIFeedback, sources: List[TokenSource]):
        self.feedback = feedback
        self.sources = sources
    
    def get_hf_token(self) -> Optional[str]:
        """Récupère le premier token disponible."""
        for source in self.sources:
            if token := source.get_token():
                self.feedback.debug(f"Using HF token from {source.source_name}")
                return token
        
        return self._prompt_for_token()
```

### Justification
- **Single Responsibility** : Chaque source a une seule responsabilité
- **Strategy Pattern** : Sources interchangeables et testables
- **Walrus Operator** : Code plus concis et moderne
- **Interface claire** : Facilite l'ajout de nouvelles sources

## 4. Configuration et Types

### Observation
Types hints incomplets et configuration dispersée dans le code.

### Code "Avant"
```python
def process_audio_segments_parallel(self, audio_path: Path, segments: List[Dict], 
                                  model: ModelType, processor: ProcessorType, target_lang: str) -> List[Dict]:
```

### Code "Après"
```python
from typing import Protocol, TypedDict

class AudioSegment(TypedDict):
    """Structure typée d'un segment audio."""
    start: float
    end: float
    duration: float
    start_sample: int 
    end_sample: int

class ProcessingResult(TypedDict):
    """Résultat de traitement d'un segment."""
    text: str
    start: float
    end: float
    quality_score: float
    quality_level: str

class AudioProcessor(Protocol):
    """Interface pour les processeurs audio."""
    def process_segments(
        self, 
        audio_path: Path, 
        segments: List[AudioSegment],
        model: ModelType,
        processor: ProcessorType,
        target_lang: str
    ) -> List[ProcessingResult]:
        """Traite les segments audio en parallèle."""
        ...
```

### Justification
- **Type Safety** : TypedDict pour structures de données
- **Protocol** : Duck typing avec vérification statique
- **Documentation** : Types auto-documentent l'API
- **IDE Support** : Meilleure completion et détection d'erreurs

## 5. Gestion d'erreurs unifiée

### Observation
Gestion d'erreurs inconsistante avec code dupliqué dans plusieurs modules.

### Code "Avant"
```python
try:
    model = AutoModelForSpeechSeq2Seq.from_pretrained(model_name, **auth_kwargs)
except torch.cuda.OutOfMemoryError as e:
    self.feedback.critical(f"GPU Out of Memory: {e}")
    # Code de recovery dupliqué...
except ImportError as e:
    error_handler.handle_import_error('transformers', e, optional=False)
except Exception as e:
    error_handler.handle_gpu_error(e, "Model loading")
```

### Code "Après"
```python
from contextlib import contextmanager
from enum import Enum

class ErrorSeverity(Enum):
    """Niveaux de sévérité des erreurs."""
    INFO = "info"
    WARNING = "warning" 
    ERROR = "error"
    CRITICAL = "critical"

@dataclass(frozen=True)
class ErrorContext:
    """Contexte d'une erreur."""
    operation: str
    component: str
    recovery_strategy: Optional[str] = None

@contextmanager
def error_boundary(
    context: ErrorContext, 
    feedback: CLIFeedback,
    oom_handler: Optional[Callable] = None
):
    """Gestionnaire d'erreurs contextualisé."""
    try:
        yield
    except torch.cuda.OutOfMemoryError as e:
        feedback.error(f"GPU OOM during {context.operation}")
        if oom_handler:
            return oom_handler(e, context)
        raise
    except ImportError as e:
        feedback.warning(f"Import failed for {context.component}: {e}")
        if not context.recovery_strategy:
            raise
        feedback.info(f"Recovery: {context.recovery_strategy}")
    except Exception as e:
        feedback.critical(f"Unexpected error in {context.operation}: {e}")
        raise

# Usage
def load_model(self, model_name: str) -> Tuple[Any, Any]:
    context = ErrorContext(
        operation="model loading",
        component="transformers",
        recovery_strategy="fallback to vLLM backend"
    )
    
    with error_boundary(context, self.feedback, self._handle_model_oom):
        return self._do_load_model(model_name)
```

### Justification
- **DRY** : Logique d'erreur centralisée
- **Context Manager** : Gestion automatique des ressources
- **Strategy** : Recovery handlers configurables  
- **Traçabilité** : Contexte riche pour debugging