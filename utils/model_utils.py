#!/usr/bin/env python3
"""
utils/model_utils.py - Model management utilities
Enhanced with B200 optimizations and torch.compile
"""

import torch
import gc
import threading
from typing import Optional, Tuple, Any, Dict, Union
from dataclasses import dataclass
import logging

from cli_feedback import ErrorHandler, CLIFeedback
from utils.gpu_utils import free_cuda_mem
from config import get_vllm_args

logger = logging.getLogger(__name__)


@dataclass
class ModelState:
    """Immutable model state container for atomic updates."""
    model: Optional[Any] = None
    processor: Optional[Any] = None
    model_name: Optional[str] = None
    backend: Optional[str] = None  # 'vllm' or 'transformers'

class ModelManager:
    """A class to manage the loading and caching of models with atomic state transitions."""

    def __init__(self, feedback: CLIFeedback):
        self.feedback = feedback
        self._state_lock = threading.RLock()
        self._current_state = ModelState()
    
    def __enter__(self) -> 'ModelManager':
        """Context manager entry - returns self for model operations."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        """Context manager exit - guarantees model cleanup."""
        self._cleanup_all_models()
        return False  # Don't suppress exceptions
    
    def _update_state_atomic(self, new_state: ModelState) -> None:
        """Atomically update the model state with proper cleanup."""
        with self._state_lock:
            # Clean up previous state if exists
            if self._current_state.model is not None:
                self.feedback.substep("Cleaning up previous model")
                try:
                    if self._current_state.processor is not None:
                        del self._current_state.processor
                    del self._current_state.model
                    free_cuda_mem()
                    gc.collect()
                except Exception as e:
                    self.feedback.warning(f"Cleanup warning: {e}")
            
            # Atomically set new state
            self._current_state = new_state
    
    def _get_current_state(self) -> ModelState:
        """Thread-safe access to current state."""
        with self._state_lock:
            return self._current_state
    
    def _cleanup_all_models(self) -> None:
        """Guaranteed cleanup of all loaded models and processors."""
        with self._state_lock:
            if self._current_state.model is not None:
                self.feedback.debug("Context manager: cleaning up model resources")
                try:
                    # Clean up processor first
                    if self._current_state.processor is not None:
                        del self._current_state.processor
                        self.feedback.debug("Processor cleaned up")
                    
                    # Clean up model
                    if hasattr(self._current_state.model, 'cpu'):
                        # Move model to CPU before deletion to free GPU memory
                        self._current_state.model.cpu()
                    del self._current_state.model
                    self.feedback.debug("Model cleaned up")
                    
                    # Aggressive cleanup for B200
                    free_cuda_mem()
                    gc.collect()
                    
                    # Reset state
                    self._current_state = ModelState()
                    self.feedback.debug("Model state reset - cleanup guaranteed")
                    
                except Exception as e:
                    # Even if cleanup fails, reset state to prevent memory leaks
                    self._current_state = ModelState()
                    self.feedback.error(f"Model cleanup warning: {e}, state reset anyway")

    def load_voxtral_model(self, model_name: str, use_vllm: bool = True) -> Optional[Tuple[Any, Any]]:
        """Enhanced model loading with detailed feedback and error handling."""
        error_handler = ErrorHandler(self.feedback)
        
        # Check if model is already loaded (thread-safe)
        current_state = self._get_current_state()
        if current_state.model_name == model_name and current_state.model is not None:
            self.feedback.info(f"Using cached model: {model_name.split('/')[-1]} ({current_state.backend})")
            return current_state.processor, current_state.model

        self.feedback.substep(f"Loading {model_name.split('/')[-1]}")

        try:
            if use_vllm:
                try:
                    from vllm import LLM

                    vllm_args = get_vllm_args(model_name)
                    self.feedback.info(f"vLLM config: {vllm_args['gpu_memory_utilization']:.1%} GPU, {vllm_args['max_num_seqs']} max sequences")

                    with self.feedback.status(f"Loading {model_name} with vLLM..."):
                        llm = LLM(
                            model=model_name,
                            tokenizer_mode="mistral",
                            config_format="mistral",
                            load_format="mistral",
                            **vllm_args
                        )

                    # Atomically update state
                    new_state = ModelState(
                        model=llm,
                        processor=None,
                        model_name=model_name,
                        backend='vllm'
                    )
                    self._update_state_atomic(new_state)

                    self.feedback.success(f"vLLM model loaded: {model_name.split('/')[-1]}")
                    return None, llm

                except ImportError as e:
                    self.feedback.warning(f"vLLM not available: {e}")
                    self.feedback.info("Will use transformers backend instead")
                    error_handler.handle_import_error('vllm', e, optional=True)
                except Exception as e:
                    self.feedback.warning(f"vLLM failed to load model: {e}")
                    self.feedback.info("Falling back to transformers backend")
                    error_handler.handle_gpu_error(e, "vLLM model loading")

            self.feedback.substep("Falling back to transformers backend")
            try:
                from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
                import os

                # Check for HuggingFace token if accessing private models
                hf_token = os.getenv('HF_TOKEN') or os.getenv('HUGGINGFACE_HUB_TOKEN')
                auth_kwargs = {}
                if hf_token:
                    auth_kwargs['token'] = hf_token
                    self.feedback.debug("Using HuggingFace authentication token")
                elif 'mistralai' in model_name.lower():
                    self.feedback.warning("Mistral models may require HF_TOKEN environment variable")

                with self.feedback.status(f"Loading {model_name} with transformers..."):
                    processor = AutoProcessor.from_pretrained(model_name, **auth_kwargs)
                    # B200 optimization: use bfloat16 for optimal Tensor Core performance
                    if torch.cuda.is_available():
                        # Verify CUDA device capability for bfloat16
                        device_capability = torch.cuda.get_device_capability()
                        if device_capability[0] >= 8:  # Ampere+ architecture supports bfloat16
                            dtype = torch.bfloat16
                            self.feedback.debug("Using bfloat16 for B200/Ampere+ optimization")
                        else:
                            dtype = torch.float16
                            self.feedback.debug("Using float16 for pre-Ampere GPU")
                    else:
                        dtype = torch.float32
                        self.feedback.warning("CUDA not available, using float32")
                    
                    model = AutoModelForSpeechSeq2Seq.from_pretrained(
                        model_name,
                        torch_dtype=dtype,
                        device_map="auto",
                        trust_remote_code=True,
                        **auth_kwargs
                    )

                # Apply B200 optimizations
                model = self._apply_b200_optimizations(model, dtype)
                
                # Set model to evaluation mode for inference (critical for science)
                if hasattr(model, 'eval'):
                    model.eval()
                    self.feedback.debug("Model set to evaluation mode for deterministic inference")
                    
                    # Ensure all submodules are in eval mode
                    for module in model.modules():
                        if hasattr(module, 'training'):
                            module.training = False
                
                # Atomically update state
                new_state = ModelState(
                    model=model,
                    processor=processor,
                    model_name=model_name,
                    backend='transformers'
                )
                self._update_state_atomic(new_state)

                self.feedback.success(f"Transformers model loaded: {model_name.split('/')[-1]}")
                return processor, model
                
            except torch.cuda.OutOfMemoryError as e:
                # Critical: B200 OOM handling with recovery
                self.feedback.critical(f"GPU Out of Memory during model loading: {e}")
                self.feedback.info("B200 Recovery: Clearing GPU cache and retrying with optimizations...")
                
                # Emergency cleanup with proper synchronization
                if torch.cuda.is_available():
                    torch.cuda.synchronize()  # Ensure all GPU operations complete
                free_cuda_mem()
                gc.collect()
                # Brief pause to allow system to stabilize
                import time
                time.sleep(0.5)
                
                # Retry with memory optimizations for B200
                try:
                    self.feedback.substep("Retrying with B200 memory optimizations...")
                    model = AutoModelForSpeechSeq2Seq.from_pretrained(
                        model_name,
                        torch_dtype=torch.bfloat16,  # Force bfloat16 for memory efficiency
                        device_map="sequential",     # Sequential loading to minimize memory peaks
                        low_cpu_mem_usage=True,     # Minimize CPU memory during loading
                        trust_remote_code=True,
                        **auth_kwargs
                    )
                    
                    # Critical: Set to eval mode even in recovery path
                    if hasattr(model, 'eval'):
                        model.eval()
                        for module in model.modules():
                            if hasattr(module, 'training'):
                                module.training = False
                    
                    new_state = ModelState(
                        model=model,
                        processor=processor,
                        model_name=model_name,
                        backend='transformers'
                    )
                    self._update_state_atomic(new_state)
                    
                    self.feedback.success(f"Model loaded with B200 memory optimizations")
                    return processor, model
                    
                except torch.cuda.OutOfMemoryError:
                    self.feedback.critical("B200 OOM persists even with optimizations",
                                         solution="Reduce model size or enable model sharding")
                    return None, None

            except ImportError as e:
                error_handler.handle_import_error('transformers', e, optional=False)
            except Exception as e:
                error_handler.handle_gpu_error(e, "Transformers model loading")

            return None, None

        except Exception as e:
            self.feedback.critical(f"Model loading completely failed: {e}",
                             solution="Check GPU memory and model availability")
            return None, None
    
    def _apply_b200_optimizations(self, model: torch.nn.Module, dtype: torch.dtype) -> torch.nn.Module:
        """Apply B200-specific optimizations to model."""
        try:
            from utils.b200_optimizer import optimize_for_b200
            
            # Apply B200 optimizations with torch.compile
            optimized_model = optimize_for_b200(model, compile_mode="max-autotune")
            self.feedback.info("Applied B200 optimizations with torch.compile")
            return optimized_model
            
        except ImportError:
            self.feedback.debug("B200 optimizer not available, using standard optimizations")
            
            # Fallback B200 optimizations without external optimizer
            if torch.cuda.is_available() and dtype == torch.bfloat16:
                # Enable Tensor Core optimizations
                if hasattr(torch.backends.cuda, 'matmul'):
                    torch.backends.cuda.matmul.allow_tf32 = True
                    torch.backends.cudnn.allow_tf32 = True
                
                # Apply torch.compile if available
                if hasattr(torch, 'compile'):
                    try:
                        model = torch.compile(model, mode="reduce-overhead")
                        self.feedback.info("Applied basic torch.compile optimization")
                    except Exception as e:
                        self.feedback.debug(f"torch.compile failed: {e}")
            
            return model
        
        except Exception as e:
            self.feedback.warning(f"B200 optimization failed: {e}, using unoptimized model")
            return model


def get_transformers_generation_params(task_type: str = "translation", 
                                     quality_level: str = "high") -> Dict[str, Any]:
    """
    Get optimized generation parameters for different tasks and quality levels.
    Enhanced with B200-specific optimizations.
    
    Args:
        task_type: Type of task ("translation", "transcription", etc.)
        quality_level: Quality level ("low", "medium", "high", "max")
    
    Returns:
        Dictionary of generation parameters
    """
    base_params = {
        "do_sample": True,
        "temperature": 0.1,  # Low temperature for deterministic results
        "pad_token_id": 50256,
        "eos_token_id": 50256,
        "use_cache": True,
        "return_dict_in_generate": True,
        "output_scores": False,
        "length_penalty": 1.0,
    }
    
    # Quality-specific parameters
    quality_configs = {
        "low": {
            "max_length": 256,
            "num_beams": 1,
            "early_stopping": True,
            "no_repeat_ngram_size": 3,
        },
        "medium": {
            "max_length": 512,
            "num_beams": 3,
            "early_stopping": True,
            "no_repeat_ngram_size": 3,
        },
        "high": {
            "max_length": 1024,
            "num_beams": 5,
            "early_stopping": False,
            "no_repeat_ngram_size": 4,
            "repetition_penalty": 1.1,
        },
        "max": {
            "max_length": 2048,
            "num_beams": 8,
            "early_stopping": False,
            "no_repeat_ngram_size": 4,
            "repetition_penalty": 1.1,
            "diversity_penalty": 0.1,
            "num_beam_groups": 2,
        }
    }
    
    # Task-specific adjustments
    task_adjustments = {
        "translation": {
            "forced_decoder_ids": None,
            "suppress_tokens": None,
        },
        "transcription": {
            "language": "turkish",
            "task": "transcribe",
        }
    }
    
    # Combine parameters
    params = base_params.copy()
    params.update(quality_configs.get(quality_level, quality_configs["high"]))
    params.update(task_adjustments.get(task_type, {}))
    
    # B200-specific optimizations
    if torch.cuda.is_available():
        device_cap = torch.cuda.get_device_capability()
        if device_cap[0] >= 8:  # B200/Ampere+
            # Increase batch size for better B200 utilization
            if quality_level in ["low", "medium"]:
                params["num_return_sequences"] = min(params.get("num_beams", 1), 3)
    
    return params


def handle_oom_with_recovery(process_func, initial_batch_size: int = 16, 
                           min_batch_size: int = 1, **kwargs) -> Optional[torch.Tensor]:
    """
    Handle OOM errors with progressive batch size reduction and B200 optimizations.
    
    Args:
        process_func: Function to execute with batch processing
        initial_batch_size: Starting batch size
        min_batch_size: Minimum batch size before giving up
        **kwargs: Additional arguments for process_func
    
    Returns:
        Result tensor or None if all attempts failed
    """
    current_batch_size = initial_batch_size
    
    while current_batch_size >= min_batch_size:
        try:
            # Clear GPU cache before attempt
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            logger.info(f"Attempting processing with batch size: {current_batch_size}")
            result = process_func(batch_size=current_batch_size, **kwargs)
            
            logger.info(f"Successfully processed with batch size: {current_batch_size}")
            return result
            
        except torch.cuda.OutOfMemoryError as e:
            logger.warning(f"OOM with batch size {current_batch_size}: {e}")
            
            # Reduce batch size
            current_batch_size = max(min_batch_size, current_batch_size // 2)
            
            # Additional cleanup for B200
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
            
            if current_batch_size < min_batch_size:
                logger.error("Minimum batch size reached, processing failed")
                break
                
        except Exception as e:
            logger.error(f"Non-OOM error during processing: {e}")
            break
    
    return None


def detect_optimal_dtype(device: str = "auto") -> torch.dtype:
    """
    Detect optimal dtype for the current hardware.
    Enhanced for B200 with bfloat16 preference.
    
    Args:
        device: Target device ("auto", "cuda", "cpu")
    
    Returns:
        Optimal dtype
    """
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if device == "cpu":
        # CPU typically works best with float32
        return torch.float32
    
    if device == "cuda" and torch.cuda.is_available():
        device_cap = torch.cuda.get_device_capability()
        
        # B200 and Ampere+ support bfloat16 with excellent performance
        if device_cap[0] >= 8:
            if hasattr(torch, 'bfloat16'):
                logger.info("Using bfloat16 for B200/Ampere+ optimization")
                return torch.bfloat16
            else:
                logger.warning("bfloat16 not available, falling back to float16")
                return torch.float16
        
        # Older GPUs: use float16
        elif device_cap[0] >= 7:  # Volta+
            return torch.float16
        
        # Very old GPUs: use float32
        else:
            logger.warning("Old GPU detected, using float32")
            return torch.float32
    
    # Default fallback
    return torch.float32


def ensure_model_eval_mode(model: torch.nn.Module) -> None:
    """
    Ensure model is in evaluation mode for deterministic inference.
    Critical for scientific reproducibility.
    
    Args:
        model: Model to set to eval mode
    """
    if hasattr(model, 'eval'):
        model.eval()
        
        # Ensure all submodules are in eval mode
        for module in model.modules():
            if hasattr(module, 'training'):
                module.training = False
        
        # Disable gradients for inference
        for param in model.parameters():
            param.requires_grad = False
        
        logger.debug("Model set to evaluation mode with gradients disabled")


def create_model_manager(feedback: CLIFeedback):
    """
    Create a ModelManager with guaranteed cleanup using context manager pattern.
    
    Usage:
        with create_model_manager(feedback) as manager:
            processor, model = manager.load_voxtral_model("mistralai/Voxtral-Small-24B-2507")
            # Model automatically cleaned up on exit
    
    Args:
        feedback: CLIFeedback instance for user communication
    
    Returns:
        ModelManager with context manager support
    """
    return ModelManager(feedback)
