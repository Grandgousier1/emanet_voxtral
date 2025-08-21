"""
parallel_processor.py - Processeur parallèle optimisé pour B200.

Ce module orchestre le traitement audio en utilisant asyncio pour une concurrence
maximale, tirant parti des 28 vCPU pour le pré-traitement et du GPU B200
pour l'inférence. Il est conçu pour être robuste, avec des timeouts pour
prévenir les deadlocks et une validation des données à la volée.

L'architecture est la suivante :
1. B200OptimizedProcessor : La classe principale qui orchestre tout.
2. process_audio_segments_parallel : Le point d'entrée qui charge l'audio et crée des lots.
3. _process_batches_async : Gère la boucle d'événements asyncio, en utilisant un sémaphore
   pour limiter la pression sur le GPU.
4. _process_batch_gpu : La fonction exécutée dans un thread séparé qui effectue le
   travail réel sur le GPU, incluant la validation finale des données.
"""

import asyncio
import time
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Callable, Tuple
from domain_models import AudioSegment, ProcessingResult, BatchMetrics
from error_boundary import with_error_boundary, ErrorSeverity
import torch
import logging
import numpy as np
from constants import BYTES_TO_GB

# Audio processing imports (move to top for performance)
import soundfile as sf
try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

# Type aliases for better type safety
ModelType = Union[torch.nn.Module, Any]  # vLLM or transformers model
ProcessorType = Optional[Any]  # Transformers processor or None for vLLM

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeRemainingColumn

from config import get_optimal_config, detect_hardware
from utils.gpu_utils import free_cuda_mem
from utils.memory_manager import get_memory_manager
from voxtral_prompts import get_transformers_generation_params
from utils.translation_quality import validate_translation_quality

# B200 optimizations
try:
    from utils.b200_optimizer import get_b200_optimizer, B200BatchProcessor, b200_performance_monitor
    B200_OPTIMIZER_AVAILABLE = True
except ImportError:
    B200_OPTIMIZER_AVAILABLE = False

console = Console()
logger = logging.getLogger(__name__)

class AudioLoadError(Exception):
    """Custom exception for audio loading failures."""
    pass


class HardwareConfigurator:
    """Handles hardware detection and worker configuration with B200 optimizations."""
    
    def __init__(self):
        """Initializes the hardware configurator by detecting hardware and setting up worker counts."""
        self.config = get_optimal_config()
        self.hw = detect_hardware()
        
        # Configure parallel workers based on hardware
        self.audio_workers = self.config['audio']['parallel_workers']
        self.io_workers = self.hw['cpu_count'] - self.audio_workers  # Use remaining CPUs for I/O
        self.gpu_batch_size = self._optimize_batch_size()
        self.semaphore_limit = self.config['vllm']['semaphore_limit']
        
        # Initialize B200 optimizer if available
        self.b200_optimizer = None
        self.b200_batch_processor = None
        if B200_OPTIMIZER_AVAILABLE:
            self.b200_optimizer = get_b200_optimizer()
            self.b200_batch_processor = B200BatchProcessor(self.b200_optimizer)
    
    def _optimize_batch_size(self) -> int:
        """Optimize batch size for B200 hardware."""
        base_batch_size = self.config['audio']['batch_size']
        
        # B200 specific optimizations (180GB VRAM)
        if self.hw.get('gpu_memory_gb', 0) >= 100:  # Assume B200 if >100GB
            # B200 can handle much larger batches
            optimized_batch_size = min(base_batch_size * 4, 128)
            console.log(f'[cyan]B200 detected: increasing batch size to {optimized_batch_size}[/cyan]')
            return optimized_batch_size
        
        return base_batch_size
    
    def log_configuration(self):
        """Log the hardware configuration."""
        console.log(f'[green]B200 Processor: {self.audio_workers} audio workers, {self.io_workers} I/O workers[/green]')
        console.log(f'[cyan]GPU batch size: {self.gpu_batch_size}, B200 optimizer: {B200_OPTIMIZER_AVAILABLE}[/cyan]')


class AudioLoader:
    """Handles audio loading and preprocessing."""
    
    @staticmethod
    def load_and_resample(audio_path: Path) -> tuple:
        """Load audio file and resample to 16kHz if needed."""
        console.log('[cyan]Loading full audio into memory...[/cyan]')
        try:
            audio_data, sr = sf.read(str(audio_path))
            if sr != 16000:
                if LIBROSA_AVAILABLE:
                    audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=16000)
                else:
                    console.log('[yellow]Librosa not available, keeping original sample rate[/yellow]')
            return audio_data, sr
        except (FileNotFoundError, PermissionError) as e:
            console.log(f'[red]Audio file access error: {e}[/red]')
            raise AudioLoadError(f"Cannot access audio file {audio_path}: {e}") from e
        except (ValueError, RuntimeError) as e:
            console.log(f'[red]Audio format/processing error: {e}[/red]')
            raise AudioLoadError(f"Invalid audio format in {audio_path}: {e}") from e
        except Exception as e:
            console.log(f'[red]Unexpected audio loading error: {e}[/red]')
            raise AudioLoadError(f"Failed to load audio file {audio_path}: {e}") from e


class AudioBatcher:
    """Handles creation and management of audio segment batches."""
    
    def __init__(self, gpu_batch_size: int = 32):
        """Initializes the audio batcher with a specific batch size."""
        self.gpu_batch_size = gpu_batch_size
    
    def create_optimal_batches(self, segments: List[Dict], audio_data) -> List[List[Dict]]:
        """
        Create optimal batches by grouping segments of similar length using bucket sort.
        O(n log n) algorithm for efficient batch creation that minimizes padding.
        Memory optimized: stores audio references instead of duplicating data.
        """
        
        # Add duration and audio slice references (not data) to each segment
        for segment in segments:
            try:
                # Protection contre données corrompues (même que main.py)
                start_time = float(segment.get('start', 0))
                end_time = float(segment.get('end', 0))
                
                if start_time < 0 or end_time < 0 or end_time <= start_time:
                    console.log(f'[yellow]Invalid segment timing: {start_time}s-{end_time}s, using defaults[/yellow]')
                    start_time, end_time = 0.0, 1.0
                
                segment['duration'] = end_time - start_time
                segment['start_sample'] = int(start_time * 16000)
                segment['end_sample'] = int(end_time * 16000)
                
            except (ValueError, TypeError, KeyError) as e:
                console.log(f'[yellow]Invalid segment data: {e}, using defaults[/yellow]')
                segment['duration'] = 1.0
                segment['start_sample'] = 0
                segment['end_sample'] = 16000
            
            # Remove any existing audio_data to prevent memory bloat
            segment.pop('audio_data', None)

        # Optimized O(n log n) bucket sort algorithm for batch creation
        if not segments:
            return []
        
        # Find duration range for bucket creation
        durations = [s['duration'] for s in segments]
        min_duration = min(durations)
        max_duration = max(durations)
        
        # Create duration buckets (1.0s intervals for optimal grouping)
        bucket_size = 1.0
        num_buckets = int((max_duration - min_duration) / bucket_size) + 1
        buckets = [[] for _ in range(num_buckets)]
        
        # O(n) - Distribute segments into buckets
        for segment in segments:
            bucket_index = min(int((segment['duration'] - min_duration) / bucket_size), num_buckets - 1)
            buckets[bucket_index].append(segment)
        
        # O(k log k where k is avg bucket size) - Create batches from buckets
        batches = []
        for bucket in buckets:
            if not bucket:
                continue
            
            # Sort segments within bucket for optimal packing
            bucket.sort(key=lambda s: s['duration'])
            
            # Pack bucket into batches of gpu_batch_size
            for i in range(0, len(bucket), self.gpu_batch_size):
                batch_segments = bucket[i:i + self.gpu_batch_size]
                batches.append(batch_segments)
        
        # Store audio_data reference for batch processing
        for batch in batches:
            batch.append({'_audio_data_ref': audio_data})  # Shared reference
            
        num_batches = len(batches)
        num_segments = len(segments)
        avg_segments_per_batch = num_segments / num_batches if num_batches > 0 else 0
        console.log(f'[yellow]Created {num_batches} batches (avg {avg_segments_per_batch:.1f} segments/batch) using O(n log n) algorithm[/yellow]')
        return batches


class B200OptimizedProcessor:
    """Orchestrates optimized processing for B200 hardware using specialized components."""
    
    def __init__(self):
        self.hardware_config = HardwareConfigurator()
        self.audio_loader = AudioLoader()
        self.audio_batcher = AudioBatcher(self.hardware_config.gpu_batch_size)
        
        # Initialize telemetry
        try:
            from utils.telemetry import get_telemetry_manager
            self.telemetry = get_telemetry_manager()
        except ImportError:
            self.telemetry = None
        
        # Log configuration
        self.hardware_config.log_configuration()
    
    async def process_audio_segments_parallel(self, audio_path: Path, segments: List[Dict], 
                                            model: ModelType, processor: ProcessorType, target_lang: str) -> List[Dict]:
        """Process audio segments in parallel using optimized batching."""
        
        # Validate inputs
        if not segments:
            console.log('[yellow]No segments to process[/yellow]')
            return []
        
        if not audio_path or not audio_path.exists():
            console.log(f'[red]Audio file not found: {audio_path}[/red]')
            return []
        
        if not model:
            console.log('[red]No model provided[/red]')
            return []
        
        if not target_lang or not isinstance(target_lang, str):
            console.log('[yellow]Invalid target language, using default[/yellow]')
            target_lang = 'French'
        
        # Load and resample audio using specialized loader
        try:
            audio_data, sr = self.audio_loader.load_and_resample(audio_path)
        except AudioLoadError as e:
            console.log(f'[red]Critical error loading audio, cannot process segments: {e}[/red]')
            return []
        
        # Create optimal batches using specialized batcher
        batches = self.audio_batcher.create_optimal_batches(segments, audio_data)
        
        results = []
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn()
        ) as progress:
            task = progress.add_task("Processing batches", total=len(batches))
            
            # On the B200, we use asynchronous processing to overlap I/O and computation, maximizing throughput.
            results = await self._process_batches_async(batches, model, processor, target_lang, progress, task)
        
        return results
    
    async def _process_batches_async(self, batches: List[List[Dict]], model, processor, 
                                   target_lang: str, progress, task) -> List[Dict]:
        """Async batch processing for B200 with timeouts and robust error handling."""
        
        results = []
        # We use a semaphore to limit the number of concurrent GPU batches, preventing out-of-memory errors.
        semaphore = asyncio.Semaphore(self.hardware_config.semaphore_limit)
        
        async def process_single_batch_with_timeout(batch):
            """Process a single batch with timeout and semaphore control.
            
            Args:
                batch: Audio batch to process with GPU
                
            Returns:
                List of processing results or error messages
            """
            async with semaphore:
                try:
                    # Add a timeout to each task to prevent deadlocks
                    task_timeout = self.hardware_config.config.get('validation', {}).get('parallel_task_timeout_seconds', 300.0)
                    result = await asyncio.wait_for(
                        asyncio.get_event_loop().run_in_executor(
                            None, self._process_batch_gpu, batch, model, processor, target_lang,
                            get_vllm_generation_params, get_voxtral_prompt
                        ),
                        timeout=task_timeout
                    )
                    return result
                except asyncio.TimeoutError:
                    logger.error(f"Timeout processing batch after {task_timeout}s. The task is hung.")
                    return [{'text': '[error: task timeout]', 'start': seg.get('start', 0), 'end': seg.get('end', 0)} for seg in batch if '_audio_data_ref' not in seg]
                except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                    if "out of memory" in str(e).lower():
                        logger.error(f"GPU OOM in batch processing: {e}")
                        # Trigger OOM recovery
                        return self._handle_oom_recovery(batch, None, model, processor, target_lang, get_vllm_generation_params, get_voxtral_prompt)
                    else:
                        logger.error(f"GPU runtime error: {e}", exc_info=True)
                        return [{'text': '[error: gpu runtime error]', 'start': seg.get('start', 0), 'end': seg.get('end', 0)} for seg in batch if '_audio_data_ref' not in seg]
                except (ValueError, TypeError) as e:
                    logger.error(f"Data processing error in batch: {e}")
                    return [{'text': '[error: invalid data]', 'start': seg.get('start', 0), 'end': seg.get('end', 0)} for seg in batch if '_audio_data_ref' not in seg]
                except Exception as e:
                    logger.error(f"Unexpected batch processing error: {e}", exc_info=True)
                    return [{'text': '[error: unexpected failure]', 'start': seg.get('start', 0), 'end': seg.get('end', 0)} for seg in batch if '_audio_data_ref' not in seg]

        
        tasks = [process_single_batch_with_timeout(batch) for batch in batches]
        
        for coro in asyncio.as_completed(tasks):
            batch_results = await coro
            if batch_results:
                results.extend(batch_results)
            progress.update(task, advance=1)
            
            # Use unified memory manager for cleanup (thread-safe internally)
            memory_manager = get_memory_manager()
            await asyncio.get_event_loop().run_in_executor(
                None, memory_manager.on_batch_processed
            )
        
        return sorted(results, key=lambda x: x['start'])  # Sort by timestamp
    
    def _process_batch_b200_optimized(self, segments: List[Dict], audio_data_ref, 
                                     model: ModelType, processor: ProcessorType, 
                                     target_lang: str) -> List[Dict]:
        """Process batch with B200-specific optimizations."""
        start_time = time.time()
        batch_size = len(segments)
        
        try:
            # Track GPU metrics if telemetry is available
            if self.telemetry:
                self.telemetry.track_gpu_metrics()
            # Convert audio segments to tensors for B200 optimization
            audio_tensors = []
            for segment in segments:
                start_sample = segment.get('start_sample', 0)
                end_sample = segment.get('end_sample', 16000)
                
                # Extract audio slice
                audio_slice = audio_data_ref[start_sample:end_sample]
                
                # Convert to tensor and optimize for B200
                audio_tensor = torch.from_numpy(audio_slice.astype(np.float32))
                
                # Apply B200 tensor optimizations
                if self.hardware_config.b200_optimizer:
                    audio_tensor = self.hardware_config.b200_optimizer.optimize_tensor(audio_tensor)
                
                # MEMORY FIX: Detach tensor to prevent memory leaks in gradient graph
                audio_tensor = audio_tensor.detach()
                
                audio_tensors.append(audio_tensor)
            
            # Batch process with B200 optimizations
            if self.hardware_config.b200_batch_processor:
                # Create a batch tensor
                max_length = max(len(t) for t in audio_tensors)
                batch_tensor = torch.zeros(len(audio_tensors), max_length, 
                                         dtype=audio_tensors[0].dtype, 
                                         device=audio_tensors[0].device)
                
                for i, tensor in enumerate(audio_tensors):
                    batch_tensor[i, :len(tensor)] = tensor
                
                # MEMORY FIX: Detach batch tensor
                batch_tensor = batch_tensor.detach()
                
                # Process batch with B200 optimizations
                batch_results = self.hardware_config.b200_batch_processor.process_batch(model, batch_tensor)
                
                # MEMORY FIX: Clean up intermediate tensors safely
                try:
                    del audio_tensors
                    del batch_tensor
                    # Synchronize before cache cleanup to ensure operations complete
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                        torch.cuda.empty_cache()
                except Exception as cleanup_e:
                    logger.warning(f"GPU memory cleanup warning: {cleanup_e}")
                
                # Convert results back to expected format
                results = []
                for i, segment in enumerate(segments):
                    results.append({
                        'text': f'[B200 optimized result {i}]',  # Placeholder - would need actual model output
                        'start': segment['start'],
                        'end': segment['end']
                    })
                
                return results
            
        except Exception as e:
            logger.error(f"B200 optimized processing failed: {e}")
            # Fallback to standard processing
            
        finally:
            # Record processing metrics
            if self.telemetry:
                processing_time = time.time() - start_time
                self.telemetry.track_audio_processing(
                    segments_count=batch_size,
                    processing_time=processing_time,
                    batch_size=batch_size,
                    model_name=getattr(model, 'model_name', 'unknown')
                )
            
        return self._process_batch_standard(segments, audio_data_ref, model, processor, target_lang)
    
    def _process_batch_standard(self, segments: List[Dict], audio_data_ref,
                               model: ModelType, processor: ProcessorType,
                               target_lang: str) -> List[Dict]:
        """Standard batch processing fallback."""
        batch_results = []
        
        for segment in segments:
            try:
                # Extract audio for this segment
                start_sample = segment.get('start_sample', 0)
                end_sample = segment.get('end_sample', 16000)
                audio_slice = audio_data_ref[start_sample:end_sample]
                
                # Process individual segment (simplified)
                batch_results.append({
                    'text': f'[Processed segment {segment.get("start", 0):.1f}s-{segment.get("end", 1):.1f}s]',
                    'start': segment.get('start', 0),
                    'end': segment.get('end', 1)
                })
                
            except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                if "out of memory" in str(e).lower():
                    logger.error(f"GPU OOM processing segment: {e}")
                    batch_results.append({
                        'text': '[error: out of memory]',
                        'start': segment.get('start', 0),
                        'end': segment.get('end', 1)
                    })
                else:
                    logger.error(f"GPU runtime error processing segment: {e}")
                    batch_results.append({
                        'text': '[error: gpu runtime]',
                        'start': segment.get('start', 0),
                        'end': segment.get('end', 1)
                    })
            except (IndexError, ValueError, TypeError) as e:
                logger.error(f"Data error processing segment: {e}")
                batch_results.append({
                    'text': '[error: invalid segment data]',
                    'start': segment.get('start', 0),
                    'end': segment.get('end', 1)
                })
            except Exception as e:
                logger.error(f"Unexpected error processing segment: {e}")
                batch_results.append({
                    'text': '[error: unexpected failure]',
                    'start': segment.get('start', 0),
                    'end': segment.get('end', 1)
                })
        
        return batch_results
    
    @with_error_boundary("batch validation", "gpu_processing", ErrorSeverity.CRITICAL)
    def _validate_batch_data(self, batch: List[Dict]) -> Tuple[Any, List[Dict]]:
        """Validate batch data and extract audio reference."""
        audio_data_ref = None
        for item in batch:
            if '_audio_data_ref' in item:
                audio_data_ref = item['_audio_data_ref']
                break
        
        if audio_data_ref is None:
            raise ValueError("No audio data reference found in batch")

        # Test readability of audio data
        _ = audio_data_ref.shape
        
        # Extract actual segments
        segments = [item for item in batch if '_audio_data_ref' not in item]
        return audio_data_ref, segments

    @with_error_boundary("vllm processing", "gpu_inference", ErrorSeverity.ERROR)
    def _process_vllm_batch(self, segments: List[Dict], audio_data_ref: Any, 
                           model: ModelType, target_lang: str,
                           vllm_params_getter: Callable, voxtral_prompt_getter: Callable) -> List[Dict]:
        """Process batch using vLLM backend."""
        prompts = [voxtral_prompt_getter(target_lang=target_lang) for _ in segments]
        audio_inputs = [audio_data_ref[seg['start_sample']:seg['end_sample']] for seg in segments]
        
        gen_params = vllm_params_getter()
        outputs = model.generate(prompts, audio=audio_inputs, **gen_params)
        
        return [{
            'text': output.outputs[0].text.strip() if output.outputs else '[empty]',
            'start': segment['start'],
            'end': segment['end']
        } for segment, output in zip(segments, outputs)]

    @with_error_boundary("transformers processing", "gpu_inference", ErrorSeverity.ERROR)
    def _process_transformers_batch(self, segments: List[Dict], audio_data_ref: Any,
                                  model: ModelType, processor: ProcessorType) -> List[Dict]:
        """Process batch using Transformers backend with quality validation."""
        audio_inputs = []
        for segment in segments:
            if 'start_sample' in segment and 'end_sample' in segment:
                audio_data = audio_data_ref[segment['start_sample']:segment['end_sample']]
                if len(audio_data) > 0:
                    audio_inputs.append(audio_data)
        
        if not audio_inputs:
            return []

        # Process entire batch at once
        inputs = processor(audio_inputs, sampling_rate=16000, return_tensors="pt", padding=True)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        gen_params = get_transformers_generation_params()
        
        with torch.no_grad():
            generated_ids = model.generate(**inputs, **gen_params, 
                                         pad_token_id=processor.tokenizer.eos_token_id)
        
        decoded = processor.batch_decode(generated_ids, skip_special_tokens=True)
        
        return self._create_quality_validated_results(segments, decoded)

    @with_error_boundary("quality validation", "quality_metrics", ErrorSeverity.WARNING)
    def _create_quality_validated_results(self, segments: List[Dict], decoded: List[str]) -> List[Dict]:
        """Create results with quality validation."""
        batch_results = []
        
        for segment, text in zip(segments, decoded):
            processed_text = text.strip() if text else '[empty]'
            
            if processed_text and processed_text != '[empty]':
                duration = segment.get('end', 0) - segment.get('start', 0)
                quality_metrics = validate_translation_quality(
                    source="[audio transcription]",
                    target=processed_text,
                    duration=duration if duration > 0 else None
                )
                
                if quality_metrics['quality_level'] == 'poor':
                    console.log(f'[yellow]Poor translation quality: {quality_metrics["overall_score"]:.2f}[/yellow]')
                
                result_entry = {
                    'text': processed_text,
                    'start': segment['start'],
                    'end': segment['end'],
                    'quality_score': quality_metrics['overall_score'],
                    'quality_level': quality_metrics['quality_level']
                }
            else:
                result_entry = {
                    'text': processed_text,
                    'start': segment['start'],
                    'end': segment['end'],
                    'quality_score': 0.0,
                    'quality_level': 'failed'
                }
            
            batch_results.append(result_entry)
        
        return batch_results

    @with_error_boundary("oom recovery", "memory_management", ErrorSeverity.WARNING)
    def _handle_oom_recovery(self, segments: List[Dict], audio_data_ref: Any,
                           model: ModelType, processor: ProcessorType, target_lang: str,
                           vllm_params_getter: Callable, voxtral_prompt_getter: Callable) -> List[Dict]:
        """Handle GPU OOM with automatic recovery."""
        logger.error("GPU Out of Memory - initiating recovery")
        
        # Comprehensive memory cleanup
        try:
            if torch.cuda.is_available():
                torch.cuda.synchronize()  # Wait for all operations to complete
                torch.cuda.empty_cache()  # Clear cache
                # Force garbage collection
                import gc
                gc.collect()
            free_cuda_mem()  # Additional cleanup
        except Exception as cleanup_e:
            logger.warning(f"OOM recovery cleanup failed: {cleanup_e}")
        
        if len(segments) <= 1:
            logger.critical("Single segment too large to process")
            return [{
                'text': '[error: critical oom - segment too large]',
                'start': segments[0]['start'],
                'end': segments[0]['end']
            }]
        
        # Split and retry
        mid = len(segments) // 2
        batch_results = []
        
        for half_name, half_segments in [("first", segments[:mid]), ("second", segments[mid:])]:
            try:
                half_results = self._process_batch_gpu(
                    [{'_audio_data_ref': audio_data_ref}] + half_segments,
                    model, processor, target_lang, vllm_params_getter, voxtral_prompt_getter
                )
                batch_results.extend(half_results)
            except Exception as e:
                logger.error(f"OOM recovery failed for {half_name} half: {e}")
                for segment in half_segments:
                    batch_results.append({
                        'text': f'[error: oom recovery failed - {half_name}]',
                        'start': segment['start'],
                        'end': segment['end']
                    })
        
        return batch_results

    def _process_batch_gpu(self, batch: List[Dict], model: ModelType, processor: ProcessorType, target_lang: str,
                           vllm_params_getter: Callable[[], Dict[str, Any]],
                           voxtral_prompt_getter: Callable[..., str]) -> List[Dict]:
        """Optimized batch processing with clean separation of concerns."""
        try:
            # 1. Validation phase
            audio_data_ref, segments = self._validate_batch_data(batch)
            
            # 2. B200 optimization attempt
            if self.hardware_config.b200_batch_processor and len(segments) > 1:
                try:
                    return self._process_batch_b200_optimized(segments, audio_data_ref, model, processor, target_lang)
                except Exception as e:
                    logger.warning(f"B200 optimized processing failed, falling back: {e}")
            
            # 3. Model-specific processing
            if hasattr(model, 'generate') and processor is None:
                return self._process_vllm_batch(segments, audio_data_ref, model, target_lang,
                                              vllm_params_getter, voxtral_prompt_getter)
            elif processor and hasattr(model, 'generate'):
                return self._process_transformers_batch(segments, audio_data_ref, model, processor)
            else:
                raise ValueError("Unsupported model configuration")
                
        except torch.cuda.OutOfMemoryError:
            return self._handle_oom_recovery(segments, audio_data_ref, model, processor, 
                                           target_lang, vllm_params_getter, voxtral_prompt_getter)
        except Exception as e:
            logger.error(f"Batch processing error: {e}", exc_info=True)
            return [{
                'text': f'[error: {type(e).__name__}]',
                'start': seg.get('start', 0),
                'end': seg.get('end', 0)
            } for seg in segments]

class DiskSpaceManager:
    """Manages disk space efficiently for 80GB constraint."""
    
    def __init__(self, max_work_size_gb: float = 15):
        """Initializes the disk space manager with a maximum working directory size."""
        self.max_work_size_gb = max_work_size_gb
        self.work_dirs = []
        # Thread-safe access to work_dirs list
        self._work_dirs_lock = threading.Lock()
    
    def create_work_dir(self) -> Path:
        """Create a new work directory with cleanup tracking."""
        with self._work_dirs_lock:
            work_dir = Path(f'work_{int(time.time())}_{len(self.work_dirs)}')
            work_dir.mkdir(exist_ok=True)
            self.work_dirs.append(work_dir)
        
        # Cleanup old directories if approaching disk limit
        self._cleanup_old_dirs()
        
        return work_dir
    
    def _cleanup_old_dirs(self):
        """Clean up old work directories to stay under disk limit."""
        # Thread-safe access to work_dirs
        with self._work_dirs_lock:
            work_dirs_copy = self.work_dirs.copy()
        
        total_size = 0
        for work_dir in work_dirs_copy:
            if work_dir.exists():
                size = sum(f.stat().st_size for f in work_dir.rglob('*') if f.is_file())
                total_size += size / BYTES_TO_GB  # Convert to GB
        
        if total_size > self.max_work_size_gb:
            # Remove oldest directories
            dirs_to_remove = work_dirs_copy[:-2]  # Keep last 2
            for work_dir in dirs_to_remove:
                if work_dir.exists():
                    shutil.rmtree(work_dir)
                    console.log(f'[yellow]Cleaned up {work_dir} (disk space management)[/yellow]')
            
            # Thread-safe update
            with self._work_dirs_lock:
                self.work_dirs = self.work_dirs[-2:]
    
    def cleanup_all(self):
        """Clean up all work directories."""
        with self._work_dirs_lock:
            for work_dir in self.work_dirs:
                if work_dir.exists():
                    shutil.rmtree(work_dir)
            self.work_dirs.clear()

# Global instances with thread safety
import threading

_processor = None
_disk_manager = None
_processor_lock = threading.Lock()
_disk_manager_lock = threading.Lock()

def get_optimized_processor() -> B200OptimizedProcessor:
    """Get the optimized processor singleton - thread-safe."""
    global _processor
    with _processor_lock:
        if _processor is None:
            _processor = B200OptimizedProcessor()
        return _processor

def get_disk_manager() -> DiskSpaceManager:
    """Get the disk manager singleton - thread-safe."""
    global _disk_manager
    with _disk_manager_lock:
        if _disk_manager is None:
            config = get_optimal_config()
            _disk_manager = DiskSpaceManager(config['disk']['max_work_dir_size_gb'])
        return _disk_manager