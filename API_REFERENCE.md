# 📚 API REFERENCE - VOXTRAL B200

## Table des matières
- [Core Components](#core-components)
- [B200 Optimizations](#b200-optimizations)
- [Audio Processing](#audio-processing)
- [Model Management](#model-management)
- [Quality Validation](#quality-validation)
- [Performance Monitoring](#performance-monitoring)

---

## Core Components

### `main.py`

#### `main()`
Point d'entrée principal du pipeline de traduction.

**Signature:**
```python
def main() -> None
```

**Exemple:**
```bash
python main.py input.mp4 --target-lang French --quality high
```

---

## B200 Optimizations

### `utils/b200_optimizer.py`

#### `B200Optimizer`
Optimiseur principal pour hardware B200.

**Classe:**
```python
class B200Optimizer:
    def __init__(self, config: Optional[B200Config] = None)
```

**Méthodes principales:**

##### `optimize_model()`
Applique optimisations B200 à un modèle PyTorch.

```python
def optimize_model(self, model: torch.nn.Module, 
                  compile_mode: str = "max-autotune") -> torch.nn.Module
```

**Paramètres:**
- `model`: Modèle PyTorch à optimiser
- `compile_mode`: Mode de compilation ("default", "reduce-overhead", "max-autotune")

**Retour:** Modèle optimisé avec torch.compile et optimisations B200

**Exemple:**
```python
from utils.b200_optimizer import get_b200_optimizer

optimizer = get_b200_optimizer()
optimized_model = optimizer.optimize_model(model, "max-autotune")
```

##### `optimize_tensor()`
Optimise un tensor pour traitement B200.

```python
def optimize_tensor(self, tensor: torch.Tensor, 
                   target_dtype: Optional[torch.dtype] = None) -> torch.Tensor
```

**Paramètres:**
- `tensor`: Tensor d'entrée
- `target_dtype`: Type cible (défaut: bfloat16 pour B200)

**Retour:** Tensor optimisé (dtype, device, memory format)

**Exemple:**
```python
audio_tensor = torch.randn(16000)
optimized = optimizer.optimize_tensor(audio_tensor)  # → bfloat16, GPU, optimized layout
```

#### `B200BatchProcessor`
Processeur de batches optimisé B200.

```python
class B200BatchProcessor:
    def __init__(self, optimizer: B200Optimizer)
```

##### `find_optimal_batch_size()`
Trouve la taille de batch optimale pour B200.

```python
def find_optimal_batch_size(self, model: torch.nn.Module, 
                           input_shape: tuple, 
                           max_memory_gb: float = 180.0) -> int
```

**Exemple:**
```python
processor = B200BatchProcessor(optimizer)
optimal_batch = processor.find_optimal_batch_size(model, (16000,), 180.0)
# → 128 pour B200, 32 pour GPU standard
```

##### `process_batch()`
Traite un batch avec optimisations B200.

```python
@b200_performance_monitor
def process_batch(self, model: torch.nn.Module, batch: torch.Tensor) -> torch.Tensor
```

---

## Audio Processing

### `utils/audio_utils.py`

#### `enhanced_vad_segments()`
VAD (Voice Activity Detection) optimisé avec cache.

```python
def enhanced_vad_segments(audio_path: Path, 
                         feedback: CLIFeedback,
                         min_segment_duration: float = 0.5,
                         max_segment_duration: float = 30.0) -> List[Dict[str, float]]
```

**Paramètres:**
- `audio_path`: Chemin vers fichier audio
- `feedback`: Instance de feedback CLI
- `min_segment_duration`: Durée minimale segment (sec)
- `max_segment_duration`: Durée maximale segment (sec)

**Retour:** Liste segments avec `start`, `end`, `confidence`

**Exemple:**
```python
from utils.audio_utils import enhanced_vad_segments
from cli_feedback import get_feedback

segments = enhanced_vad_segments(
    Path("audio.wav"), 
    get_feedback(),
    min_segment_duration=1.0
)
# → [{'start': 0.0, 'end': 2.5, 'confidence': 0.95}, ...]
```

#### `AudioCache`
Cache intelligent pour données audio.

```python
class AudioCache:
    def __init__(self, max_size_gb: float = 10.0)
```

**Méthodes:**

##### `get_cached_audio()`
```python
def get_cached_audio(self, audio_path: Path) -> Optional[Tuple[np.ndarray, int]]
```

##### `cache_audio()`
```python
def cache_audio(self, audio_path: Path, audio_data: np.ndarray, sample_rate: int) -> None
```

---

## Model Management

### `utils/model_utils.py`

#### `ModelManager`
Gestionnaire de modèles avec optimisations B200.

```python
class ModelManager:
    def __init__(self, feedback: CLIFeedback)
```

##### `load_voxtral_model()`
Charge un modèle Voxtral avec optimisations B200.

```python
def load_voxtral_model(self, model_name: str, 
                      use_vllm: bool = True) -> Optional[Tuple[Any, Any]]
```

**Paramètres:**
- `model_name`: Nom du modèle (ex: "mistralai/Voxtral-Mini-3B-2507")
- `use_vllm`: Utiliser vLLM si disponible

**Retour:** Tuple (processor, model) ou (None, model) pour vLLM

**Exemple:**
```python
manager = ModelManager(feedback)
processor, model = manager.load_voxtral_model(
    "mistralai/Voxtral-Mini-3B-2507",
    use_vllm=True
)
```

#### Fonctions utilitaires

##### `detect_optimal_dtype()`
Détecte le dtype optimal selon hardware.

```python
def detect_optimal_dtype(device: str = "auto") -> torch.dtype
```

**Retour:**
- `torch.bfloat16` pour B200/Ampere+ (capability ≥8.0)
- `torch.float16` pour GPU plus anciens
- `torch.float32` pour CPU

##### `handle_oom_with_recovery()`
Gestion OOM avec réduction progressive batch.

```python
def handle_oom_with_recovery(process_func, 
                           initial_batch_size: int = 16,
                           min_batch_size: int = 1, 
                           **kwargs) -> Optional[torch.Tensor]
```

**Exemple:**
```python
def my_processing(batch_size, data):
    return model(data[:batch_size])

result = handle_oom_with_recovery(
    my_processing,
    initial_batch_size=32,
    data=large_batch
)  # → Réduit automatiquement si OOM: 32→16→8→4→1
```

##### `get_transformers_generation_params()`
Paramètres génération optimisés B200.

```python
def get_transformers_generation_params(task_type: str = "translation",
                                     quality_level: str = "high") -> Dict[str, Any]
```

**Paramètres:**
- `task_type`: "translation" ou "transcription"
- `quality_level`: "low", "medium", "high", "max"

**Retour:** Dict paramètres génération optimisés

---

## Quality Validation

### `utils/translation_quality.py`

#### `TranslationQualityValidator`
Validateur qualité traductions scientifique.

```python
class TranslationQualityValidator:
    def __init__(self, turkish_markers: Optional[List[str]] = None,
                 french_markers: Optional[List[str]] = None)
```

##### `comprehensive_quality_assessment()`
Évaluation qualité complète.

```python
def comprehensive_quality_assessment(self, source: str, target: str, 
                                   duration: Optional[float] = None) -> Dict[str, Any]
```

**Paramètres:**
- `source`: Texte source (turc)
- `target`: Texte cible (français)
- `duration`: Durée audio (optionnel)

**Retour:** Dict avec scores et recommandations
```python
{
    'overall_score': 0.85,           # Score global [0-1]
    'quality_level': 'good',         # 'poor'|'fair'|'good'|'excellent'
    'detailed_metrics': {
        'completeness_score': 0.90,
        'cultural_score': 0.80,
        'repetition_penalty': 0.95,
        'subtitle_score': 0.75
    },
    'recommendations': [...]
}
```

##### `validate_subtitle_constraints()`
Validation contraintes sous-titres.

```python
def validate_subtitle_constraints(self, text: str, 
                                duration: float) -> Dict[str, Any]
```

**Contraintes validées:**
- Vitesse lecture (caractères/seconde)
- Longueur maximale (42 caractères/ligne)
- Nombre lignes (max 2)
- Temps affichage minimal

---

## Performance Monitoring

### `utils/performance_profiler.py`

#### `PerformanceProfiler`
Profileur performance pour B200.

```python
class PerformanceProfiler:
    def __init__(self, enable_detailed_profiling: bool = True)
```

##### `profile_operation()`
Context manager pour profiler opérations.

```python
@contextmanager
def profile_operation(self, operation_name: str, batch_size: int = 1)
```

**Exemple:**
```python
profiler = PerformanceProfiler()

with profiler.profile_operation("model_inference", batch_size=32):
    output = model(input_batch)

# Métriques automatiquement enregistrées
```

#### `B200Benchmarker`
Suite benchmark complète B200.

```python
class B200Benchmarker:
    def run_comprehensive_benchmark(self) -> B200BenchmarkResult
```

**Exemple:**
```python
benchmarker = B200Benchmarker()
results = benchmarker.run_comprehensive_benchmark()

print(f"Average throughput: {results.summary_stats['max_throughput_ops_per_sec']:.1f} ops/sec")
print(f"Recommendations: {results.recommendations}")
```

#### Décorateurs de monitoring

##### `@b200_performance_monitor`
Décorateur monitoring automatique.

```python
@b200_performance_monitor
def process_audio_batch(batch):
    # Fonction automatiquement monitorée
    return model(batch)
```

##### `@profile_function`
Décorateur profiling fonction.

```python
@profile_function("custom_operation")
def my_function():
    # Profiling automatique avec nom personnalisé
    pass
```

---

## Validation et Tests

### `utils/tensor_validation.py`

#### Fonctions de validation

##### `validate_audio_tensor()`
```python
def validate_audio_tensor(audio_tensor: torch.Tensor,
                         sample_rate: int = 16000,
                         max_duration_sec: float = 30.0,
                         name: str = "audio") -> bool
```

##### `validate_batch_consistency()`
```python
def validate_batch_consistency(batch: List[torch.Tensor]) -> bool
```

### `utils/reproducibility.py`

#### `ensure_reproducible_environment()`
Configuration reproductibilité globale.

```python
def ensure_reproducible_environment(seed: int = 42, 
                                  validate: bool = False) -> Dict[str, Any]
```

#### `ReproducibleSession`
Context manager session reproductible.

```python
with ReproducibleSession(seed=123):
    # Toutes opérations déterministes
    result = model(input_data)
```

---

## Configuration et Hardware

### `config.py`

#### `detect_hardware()`
```python
def detect_hardware() -> Dict[str, Any]
```

**Retour:**
```python
{
    'device': 'cuda:0',
    'has_gpu': True,
    'gpu_name': 'NVIDIA B200',
    'gpu_memory_gb': 180.0,
    'cpu_count': 28,
    'total_ram_gb': 188.0,
    'is_b200': True
}
```

#### `get_optimal_config()`
Configuration optimale selon hardware.

```python
def get_optimal_config() -> Dict[str, Any]
```

---

## Exemples d'utilisation

### Pipeline complet optimisé B200
```python
from utils.b200_optimizer import get_b200_optimizer
from utils.model_utils import ModelManager
from cli_feedback import get_feedback

# Setup
feedback = get_feedback()
optimizer = get_b200_optimizer()
manager = ModelManager(feedback)

# Chargement modèle optimisé
processor, model = manager.load_voxtral_model("mistralai/Voxtral-Mini-3B-2507")
optimized_model = optimizer.optimize_model(model, "max-autotune")

# Traitement audio
from utils.audio_utils import enhanced_vad_segments
segments = enhanced_vad_segments(Path("input.wav"), feedback)

# Processing parallèle
from parallel_processor import B200OptimizedProcessor
processor = B200OptimizedProcessor()
results = await processor.process_audio_segments_parallel(
    Path("input.wav"), segments, optimized_model, processor, "French"
)
```

### Benchmark performance
```python
from utils.performance_profiler import B200Benchmarker

benchmarker = B200Benchmarker()
results = benchmarker.run_comprehensive_benchmark()

# Sauvegarde résultats
benchmarker.save_benchmark_results(results, Path("benchmark_results.json"))
```

Cette API reference complète permet une utilisation optimale de tous les composants B200.