#!/usr/bin/env python3
"""
utils/b200_optimizer.py - Advanced B200 GPU optimizations
NVIDIA B200 specific optimizations for maximum performance
"""

import torch
import torch.nn.functional as F
from typing import Optional, Dict, Any, Callable, Union, List
import logging
from dataclasses import dataclass
from functools import wraps
import time

logger = logging.getLogger(__name__)


@dataclass
class B200Config:
    """B200-specific optimization configuration."""
    use_bfloat16: bool = True
    use_tensor_cores: bool = True
    enable_torch_compile: bool = True
    memory_format: torch.memory_format = torch.channels_last
    use_fused_kernels: bool = True
    optimize_for_inference: bool = True
    enable_cudnn_benchmark: bool = True
    tensor_core_precision: str = "highest"  # "highest", "high", "medium"


class B200Optimizer:
    """Advanced B200 GPU optimizer for maximum performance."""
    
    def __init__(self, config: Optional[B200Config] = None):
        self.config = config or B200Config()
        self.compiled_models = {}
        self.performance_cache = {}
        self._setup_environment()
    
    def _setup_environment(self):
        """Setup optimal environment for B200."""
        if not torch.cuda.is_available():
            logger.warning("CUDA not available, B200 optimizations disabled")
            return
        
        # Check for B200 compatibility
        device_cap = torch.cuda.get_device_capability()
        if device_cap[0] < 8:
            logger.warning(f"GPU compute capability {device_cap} < 8.0, some B200 optimizations may not be available")
        
        # Enable optimizations
        if self.config.enable_cudnn_benchmark:
            torch.backends.cudnn.benchmark = True
            logger.info("Enabled cuDNN benchmark for B200")
        
        # Setup tensor core precision
        if hasattr(torch.backends.cuda, 'matmul'):
            if self.config.tensor_core_precision == "highest":
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
                torch.backends.cudnn.allow_tf32 = True
            logger.info(f"Tensor Core precision: {self.config.tensor_core_precision}")
    
    def optimize_model(self, model: torch.nn.Module, 
                      compile_mode: str = "max-autotune") -> torch.nn.Module:
        """
        Apply comprehensive B200 optimizations to a model.
        
        Args:
            model: PyTorch model to optimize
            compile_mode: torch.compile mode ("default", "reduce-overhead", "max-autotune")
        
        Returns:
            Optimized model
        """
        model_id = id(model)
        
        # Check cache
        if model_id in self.compiled_models:
            logger.info("Using cached compiled model")
            return self.compiled_models[model_id]
        
        logger.info(f"Applying B200 optimizations with compile mode: {compile_mode}")
        
        # 1. Set optimal dtype for B200
        if self.config.use_bfloat16 and torch.cuda.is_available():
            device_cap = torch.cuda.get_device_capability()
            if device_cap[0] >= 8:  # Ampere+ supports bfloat16
                model = model.to(dtype=torch.bfloat16)
                logger.info("Converted model to bfloat16 for B200 Tensor Cores")
        
        # 2. Optimize memory format for tensor cores
        if self.config.use_tensor_cores:
            try:
                model = model.to(memory_format=self.config.memory_format)
                logger.info(f"Applied memory format: {self.config.memory_format}")
            except Exception as e:
                logger.warning(f"Could not apply memory format: {e}")
        
        # 3. Set model to eval mode for inference optimizations
        if self.config.optimize_for_inference:
            model.eval()
            # Disable gradients for all parameters
            for param in model.parameters():
                param.requires_grad = False
            logger.info("Model optimized for inference (eval mode, no gradients)")
        
        # 4. Apply torch.compile for maximum performance
        if self.config.enable_torch_compile and hasattr(torch, 'compile'):
            try:
                # Configure compilation options for B200
                compile_options = {
                    "mode": compile_mode,
                    "fullgraph": False,  # Allow graph breaks for flexibility
                    "dynamic": True,     # Handle dynamic shapes
                }
                
                # Additional B200-specific optimizations
                if compile_mode == "max-autotune":
                    compile_options.update({
                        "options": {
                            "triton.cudagraphs": True,
                            "epilogue_fusion": True,
                            "max_autotune": True,
                            "shape_padding": True,
                        }
                    })
                
                optimized_model = torch.compile(model, **compile_options)
                self.compiled_models[model_id] = optimized_model
                
                logger.info(f"Model compiled with torch.compile ({compile_mode})")
                return optimized_model
                
            except Exception as e:
                logger.warning(f"torch.compile failed: {e}, falling back to non-compiled model")
        
        # Cache even non-compiled model to avoid reprocessing
        self.compiled_models[model_id] = model
        return model
    
    def optimize_tensor(self, tensor: torch.Tensor, 
                       target_dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        """
        Optimize tensor for B200 processing.
        
        Args:
            tensor: Input tensor
            target_dtype: Target dtype (default: bfloat16 for B200)
        
        Returns:
            Optimized tensor
        """
        if not torch.cuda.is_available():
            return tensor
        
        # Default to bfloat16 for B200
        if target_dtype is None and self.config.use_bfloat16:
            device_cap = torch.cuda.get_device_capability()
            if device_cap[0] >= 8:
                target_dtype = torch.bfloat16
            else:
                target_dtype = torch.float16
        
        # Convert dtype if needed
        if target_dtype and tensor.dtype != target_dtype:
            tensor = tensor.to(dtype=target_dtype)
        
        # Ensure tensor is on GPU
        if not tensor.is_cuda:
            tensor = tensor.cuda()
        
        # Apply optimal memory format for tensor cores
        if self.config.use_tensor_cores and tensor.dim() >= 4:
            try:
                tensor = tensor.to(memory_format=self.config.memory_format)
            except Exception as e:
                logger.debug(f"Could not apply memory format to tensor: {e}")
        
        return tensor
    
    def create_fused_operation(self, operations: List[Callable]) -> Callable:
        """
        Create a fused operation for multiple tensor operations.
        
        Args:
            operations: List of operations to fuse
        
        Returns:
            Fused operation function
        """
        if not self.config.use_fused_kernels:
            # Return sequential execution if fusion disabled
            def sequential_ops(x):
                for op in operations:
                    x = op(x)
                return x
            return sequential_ops
        
        def fused_ops(x):
            """Fused operations with B200 optimizations."""
            # Apply operations in sequence (torch.compile will fuse them)
            for op in operations:
                x = op(x)
            return x
        
        # Compile the fused operation if torch.compile available
        if self.config.enable_torch_compile and hasattr(torch, 'compile'):
            try:
                fused_ops = torch.compile(fused_ops, mode="max-autotune")
                logger.info("Created fused operation with torch.compile")
            except Exception as e:
                logger.warning(f"Failed to compile fused operation: {e}")
        
        return fused_ops
    
    def benchmark_operation(self, operation: Callable, input_tensor: torch.Tensor,
                          num_runs: int = 100, warmup_runs: int = 10) -> Dict[str, float]:
        """
        Benchmark an operation for performance analysis.
        
        Args:
            operation: Operation to benchmark
            input_tensor: Input tensor for benchmarking
            num_runs: Number of timing runs
            warmup_runs: Number of warmup runs
        
        Returns:
            Performance metrics
        """
        if not torch.cuda.is_available():
            logger.warning("CUDA not available for benchmarking")
            return {}
        
        device = input_tensor.device
        
        # Warmup runs
        for _ in range(warmup_runs):
            with torch.no_grad():
                _ = operation(input_tensor)
        
        torch.cuda.synchronize(device)
        
        # Timing runs
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        times = []
        for _ in range(num_runs):
            start_event.record()
            with torch.no_grad():
                result = operation(input_tensor)
            end_event.record()
            torch.cuda.synchronize(device)
            times.append(start_event.elapsed_time(end_event))
        
        # Calculate statistics
        times = torch.tensor(times).detach()  # MEMORY FIX: Detach to prevent gradient tracking
        metrics = {
            "mean_ms": float(times.mean()),
            "std_ms": float(times.std()),
            "min_ms": float(times.min()),
            "max_ms": float(times.max()),
            "throughput_ops_per_sec": 1000.0 / float(times.mean()),
            "memory_allocated_mb": torch.cuda.memory_allocated(device) / 1024**2,
            "memory_cached_mb": torch.cuda.memory_reserved(device) / 1024**2
        }
        
        return metrics


def b200_performance_monitor(func):
    """Decorator to monitor B200 performance of functions."""
    
    @wraps(func)
    def wrapper(*args, **kwargs):
        if not torch.cuda.is_available():
            return func(*args, **kwargs)
        
        # Record initial state
        start_time = time.time()
        initial_memory = torch.cuda.memory_allocated()
        
        # Execute function
        try:
            result = func(*args, **kwargs)
            
            # Record final state
            end_time = time.time()
            final_memory = torch.cuda.memory_allocated()
            
            # Log performance metrics
            duration = end_time - start_time
            memory_delta = final_memory - initial_memory
            
            logger.info(f"B200 Performance - {func.__name__}: "
                       f"{duration:.3f}s, memory Δ: {memory_delta/1024**2:.1f}MB")
            
            return result
            
        except Exception as e:
            logger.error(f"B200 Performance error in {func.__name__}: {e}")
            raise
    
    return wrapper


class B200BatchProcessor:
    """Optimized batch processor for B200."""
    
    def __init__(self, optimizer: B200Optimizer):
        self.optimizer = optimizer
        self.optimal_batch_sizes = {}
    
    def find_optimal_batch_size(self, model: torch.nn.Module, 
                               input_shape: tuple, 
                               max_memory_gb: float = 180.0) -> int:
        """
        Find optimal batch size for B200 (180GB VRAM).
        
        Args:
            model: Model to test
            input_shape: Shape of single input (without batch dimension)
            max_memory_gb: Maximum memory to use
        
        Returns:
            Optimal batch size
        """
        cache_key = (id(model), input_shape, max_memory_gb)
        if cache_key in self.optimal_batch_sizes:
            return self.optimal_batch_sizes[cache_key]
        
        if not torch.cuda.is_available():
            return 1
        
        logger.info(f"Finding optimal batch size for B200 with {max_memory_gb}GB limit")
        
        # Start with a reasonable batch size for B200
        batch_size = 32
        max_batch_size = 1
        
        device = next(model.parameters()).device
        
        while batch_size <= 512:  # Reasonable upper limit
            try:
                # Clear cache
                torch.cuda.empty_cache()
                
                # Test batch
                test_input = torch.randn(batch_size, *input_shape, 
                                       device=device, dtype=torch.bfloat16)
                test_input = self.optimizer.optimize_tensor(test_input)
                
                with torch.no_grad():
                    _ = model(test_input)
                
                # Check memory usage
                memory_used_gb = torch.cuda.memory_allocated(device) / 1024**3
                
                if memory_used_gb <= max_memory_gb * 0.8:  # Leave 20% headroom
                    max_batch_size = batch_size
                    batch_size *= 2
                else:
                    break
                    
            except torch.cuda.OutOfMemoryError:
                break
            except Exception as e:
                logger.warning(f"Error testing batch size {batch_size}: {e}")
                break
        
        # Use 80% of max found batch size for safety
        optimal_batch_size = max(1, int(max_batch_size * 0.8))
        
        self.optimal_batch_sizes[cache_key] = optimal_batch_size
        logger.info(f"Optimal batch size for B200: {optimal_batch_size}")
        
        return optimal_batch_size
    
    @b200_performance_monitor
    def process_batch(self, model: torch.nn.Module, batch: torch.Tensor) -> torch.Tensor:
        """
        Process batch with B200 optimizations.
        
        Args:
            model: Optimized model
            batch: Input batch
        
        Returns:
            Processed batch
        """
        # Optimize input tensor
        batch = self.optimizer.optimize_tensor(batch)
        
        # Process with optimized model
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=self.optimizer.config.use_bfloat16):
            result = model(batch)
        
        return result


# Global B200 optimizer instance
_b200_optimizer = None

def get_b200_optimizer(config: Optional[B200Config] = None) -> B200Optimizer:
    """Get global B200 optimizer instance."""
    global _b200_optimizer
    if _b200_optimizer is None:
        _b200_optimizer = B200Optimizer(config)
    return _b200_optimizer


def optimize_for_b200(model: torch.nn.Module, 
                     compile_mode: str = "max-autotune") -> torch.nn.Module:
    """
    Convenience function to optimize model for B200.
    
    Args:
        model: Model to optimize
        compile_mode: Compilation mode
    
    Returns:
        Optimized model
    """
    optimizer = get_b200_optimizer()
    return optimizer.optimize_model(model, compile_mode)


def b200_tensor_optimize(tensor: torch.Tensor) -> torch.Tensor:
    """Convenience function to optimize tensor for B200."""
    optimizer = get_b200_optimizer()
    return optimizer.optimize_tensor(tensor)