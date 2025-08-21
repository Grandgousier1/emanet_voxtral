#!/usr/bin/env python3
"""
utils/performance_profiler.py - Advanced performance profiling for B200
Comprehensive profiling and benchmarking utilities for B200 optimization
"""

import torch
import time
import psutil
import threading
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from contextlib import contextmanager
import numpy as np
import logging
from pathlib import Path
import json

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""
    operation: str
    duration_ms: float
    gpu_memory_used_mb: float = 0.0
    cpu_percent: float = 0.0
    memory_mb: float = 0.0
    throughput_ops_per_sec: float = 0.0
    batch_size: int = 1
    tensor_ops: int = 0
    cuda_memory_allocated_mb: float = 0.0
    cuda_memory_cached_mb: float = 0.0
    additional_metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class B200BenchmarkResult:
    """Comprehensive benchmark results for B200."""
    test_name: str
    hardware_info: Dict[str, Any]
    performance_metrics: List[PerformanceMetrics]
    summary_stats: Dict[str, float]
    recommendations: List[str]
    timestamp: str


class PerformanceProfiler:
    """Advanced performance profiler for B200 systems."""
    
    def __init__(self, enable_detailed_profiling: bool = True):
        self.enable_detailed_profiling = enable_detailed_profiling
        self.metrics_history = []
        self.profiling_active = False
        self._start_time = None
        self._initial_gpu_memory = 0
        self._initial_cpu_memory = 0
    
    @contextmanager
    def profile_operation(self, operation_name: str, batch_size: int = 1):
        """Context manager for profiling operations."""
        # Record initial state
        start_time = time.perf_counter()
        initial_cpu_percent = psutil.cpu_percent()
        initial_memory = psutil.virtual_memory().used / 1024**2
        
        initial_gpu_memory = 0
        if torch.cuda.is_available():
            initial_gpu_memory = torch.cuda.memory_allocated() / 1024**2
        
        try:
            yield self
        finally:
            # Record final state
            end_time = time.perf_counter()
            duration_ms = (end_time - start_time) * 1000
            
            final_cpu_percent = psutil.cpu_percent()
            final_memory = psutil.virtual_memory().used / 1024**2
            
            final_gpu_memory = 0
            gpu_memory_cached = 0
            if torch.cuda.is_available():
                final_gpu_memory = torch.cuda.memory_allocated() / 1024**2
                gpu_memory_cached = torch.cuda.memory_reserved() / 1024**2
            
            # Calculate metrics
            metrics = PerformanceMetrics(
                operation=operation_name,
                duration_ms=duration_ms,
                gpu_memory_used_mb=final_gpu_memory - initial_gpu_memory,
                cpu_percent=(initial_cpu_percent + final_cpu_percent) / 2,
                memory_mb=final_memory - initial_memory,
                throughput_ops_per_sec=1000.0 / duration_ms if duration_ms > 0 else 0,
                batch_size=batch_size,
                cuda_memory_allocated_mb=final_gpu_memory,
                cuda_memory_cached_mb=gpu_memory_cached
            )
            
            self.metrics_history.append(metrics)
            
            if self.enable_detailed_profiling:
                logger.info(f"Performance - {operation_name}: {duration_ms:.2f}ms, "
                           f"GPU Δ: {metrics.gpu_memory_used_mb:.1f}MB, "
                           f"Throughput: {metrics.throughput_ops_per_sec:.1f} ops/s")
    
    def benchmark_tensor_operations(self, tensor_shapes: List[tuple], 
                                  dtypes: List[torch.dtype] = None) -> Dict[str, PerformanceMetrics]:
        """Benchmark basic tensor operations for B200."""
        if dtypes is None:
            dtypes = [torch.float32, torch.float16]
            if hasattr(torch, 'bfloat16'):
                dtypes.append(torch.bfloat16)
        
        results = {}
        
        for shape in tensor_shapes:
            for dtype in dtypes:
                if not torch.cuda.is_available() and dtype == torch.bfloat16:
                    continue
                
                test_name = f"tensor_ops_{shape}_{dtype}"
                
                with self.profile_operation(test_name):
                    try:
                        # Create tensors
                        device = 'cuda' if torch.cuda.is_available() else 'cpu'
                        a = torch.randn(shape, dtype=dtype, device=device)
                        b = torch.randn(shape, dtype=dtype, device=device)
                        
                        # Benchmark operations
                        _ = torch.matmul(a, b.T) if len(shape) >= 2 else a * b
                        _ = torch.nn.functional.relu(a)
                        _ = torch.sum(a, dim=-1)
                        
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                        
                    except Exception as e:
                        logger.warning(f"Tensor benchmark failed for {test_name}: {e}")
                        continue
                
                if self.metrics_history:
                    results[test_name] = self.metrics_history[-1]
        
        return results
    
    def benchmark_model_inference(self, model: torch.nn.Module, 
                                 input_shapes: List[tuple],
                                 batch_sizes: List[int] = None) -> Dict[str, PerformanceMetrics]:
        """Benchmark model inference performance."""
        if batch_sizes is None:
            batch_sizes = [1, 4, 8, 16, 32]
        
        results = {}
        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype
        
        model.eval()
        
        for input_shape in input_shapes:
            for batch_size in batch_sizes:
                test_name = f"model_inference_bs{batch_size}_{input_shape}"
                
                try:
                    # Create input batch
                    full_shape = (batch_size,) + input_shape
                    input_tensor = torch.randn(full_shape, dtype=dtype, device=device)
                    
                    with self.profile_operation(test_name, batch_size):
                        with torch.no_grad():
                            _ = model(input_tensor)
                        
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                    
                    if self.metrics_history:
                        result = self.metrics_history[-1]
                        # Calculate per-sample throughput
                        result.throughput_ops_per_sec = batch_size * 1000.0 / result.duration_ms
                        results[test_name] = result
                
                except torch.cuda.OutOfMemoryError:
                    logger.warning(f"OOM for {test_name}")
                    break
                except Exception as e:
                    logger.warning(f"Model benchmark failed for {test_name}: {e}")
        
        return results
    
    def profile_memory_usage(self, duration_seconds: float = 10.0) -> Dict[str, List[float]]:
        """Profile memory usage over time."""
        memory_samples = {
            'timestamps': [],
            'cpu_memory_mb': [],
            'gpu_memory_mb': [],
            'gpu_cached_mb': []
        }
        
        start_time = time.time()
        
        while time.time() - start_time < duration_seconds:
            current_time = time.time() - start_time
            memory_samples['timestamps'].append(current_time)
            
            # CPU memory
            cpu_memory = psutil.virtual_memory().used / 1024**2
            memory_samples['cpu_memory_mb'].append(cpu_memory)
            
            # GPU memory
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.memory_allocated() / 1024**2
                gpu_cached = torch.cuda.memory_reserved() / 1024**2
                memory_samples['gpu_memory_mb'].append(gpu_memory)
                memory_samples['gpu_cached_mb'].append(gpu_cached)
            else:
                memory_samples['gpu_memory_mb'].append(0)
                memory_samples['gpu_cached_mb'].append(0)
            
            time.sleep(0.1)  # Sample every 100ms
        
        return memory_samples


class B200Benchmarker:
    """Comprehensive benchmarking suite for B200 systems."""
    
    def __init__(self):
        self.profiler = PerformanceProfiler(enable_detailed_profiling=True)
        self.hardware_info = self._get_hardware_info()
    
    def _get_hardware_info(self) -> Dict[str, Any]:
        """Get comprehensive hardware information."""
        info = {
            'cpu_count': psutil.cpu_count(),
            'cpu_freq_mhz': psutil.cpu_freq().current if psutil.cpu_freq() else 0,
            'total_memory_gb': psutil.virtual_memory().total / 1024**3,
            'cuda_available': torch.cuda.is_available(),
        }
        
        if torch.cuda.is_available():
            info.update({
                'gpu_name': torch.cuda.get_device_name(),
                'gpu_count': torch.cuda.device_count(),
                'gpu_memory_gb': torch.cuda.get_device_properties(0).total_memory / 1024**3,
                'cuda_version': torch.version.cuda,
                'compute_capability': torch.cuda.get_device_capability(),
            })
        
        return info
    
    def run_comprehensive_benchmark(self) -> B200BenchmarkResult:
        """Run comprehensive B200 benchmark suite."""
        logger.info("Starting comprehensive B200 benchmark")
        
        all_metrics = []
        
        # 1. Tensor operations benchmark
        logger.info("Benchmarking tensor operations")
        tensor_shapes = [(1000, 1000), (2048, 2048), (4096, 1024)]
        tensor_results = self.profiler.benchmark_tensor_operations(tensor_shapes)
        all_metrics.extend(tensor_results.values())
        
        # 2. Memory bandwidth test
        logger.info("Testing memory bandwidth")
        with self.profiler.profile_operation("memory_bandwidth_test"):
            self._test_memory_bandwidth()
        all_metrics.extend(self.profiler.metrics_history[-1:])
        
        # 3. B200-specific optimizations test
        if torch.cuda.is_available():
            logger.info("Testing B200-specific optimizations")
            self._test_b200_optimizations()
            all_metrics.extend(self.profiler.metrics_history[-3:])  # Last 3 tests
        
        # Calculate summary statistics
        summary_stats = self._calculate_summary_stats(all_metrics)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(all_metrics, self.hardware_info)
        
        result = B200BenchmarkResult(
            test_name="comprehensive_b200_benchmark",
            hardware_info=self.hardware_info,
            performance_metrics=all_metrics,
            summary_stats=summary_stats,
            recommendations=recommendations,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
        )
        
        logger.info("Comprehensive benchmark completed")
        return result
    
    def _test_memory_bandwidth(self):
        """Test memory bandwidth performance."""
        if not torch.cuda.is_available():
            return
        
        # Large tensor for bandwidth testing
        size = (10000, 10000)
        dtype = torch.bfloat16 if hasattr(torch, 'bfloat16') else torch.float16
        
        # GPU to GPU copy
        a = torch.randn(size, dtype=dtype, device='cuda')
        b = torch.empty_like(a)
        
        for _ in range(10):  # Multiple iterations for stability
            b.copy_(a)
            torch.cuda.synchronize()
    
    def _test_b200_optimizations(self):
        """Test B200-specific optimizations."""
        if not torch.cuda.is_available():
            return
        
        # Test bfloat16 performance
        with self.profiler.profile_operation("bfloat16_matmul"):
            if hasattr(torch, 'bfloat16'):
                a = torch.randn(2048, 2048, dtype=torch.bfloat16, device='cuda')
                b = torch.randn(2048, 2048, dtype=torch.bfloat16, device='cuda')
                _ = torch.matmul(a, b)
                torch.cuda.synchronize()
        
        # Test Tensor Core utilization
        with self.profiler.profile_operation("tensor_core_test"):
            # Shapes optimized for Tensor Cores (multiples of 8 for bfloat16)
            a = torch.randn(1024, 1024, dtype=torch.bfloat16, device='cuda')
            b = torch.randn(1024, 1024, dtype=torch.bfloat16, device='cuda')
            _ = torch.matmul(a, b)
            torch.cuda.synchronize()
        
        # Test torch.compile if available
        if hasattr(torch, 'compile'):
            with self.profiler.profile_operation("torch_compile_test"):
                def simple_matmul(x, y):
                    return torch.matmul(x, y)
                
                compiled_fn = torch.compile(simple_matmul)
                a = torch.randn(1024, 1024, dtype=torch.bfloat16, device='cuda')
                b = torch.randn(1024, 1024, dtype=torch.bfloat16, device='cuda')
                _ = compiled_fn(a, b)
                torch.cuda.synchronize()
    
    def _calculate_summary_stats(self, metrics: List[PerformanceMetrics]) -> Dict[str, float]:
        """Calculate summary statistics from metrics."""
        if not metrics:
            return {}
        
        durations = [m.duration_ms for m in metrics]
        throughputs = [m.throughput_ops_per_sec for m in metrics if m.throughput_ops_per_sec > 0]
        gpu_memory_usage = [m.gpu_memory_used_mb for m in metrics]
        
        return {
            'avg_duration_ms': np.mean(durations),
            'min_duration_ms': np.min(durations),
            'max_duration_ms': np.max(durations),
            'avg_throughput_ops_per_sec': np.mean(throughputs) if throughputs else 0,
            'max_throughput_ops_per_sec': np.max(throughputs) if throughputs else 0,
            'total_gpu_memory_used_mb': np.sum(gpu_memory_usage),
            'avg_gpu_memory_per_op_mb': np.mean(gpu_memory_usage) if gpu_memory_usage else 0,
        }
    
    def _generate_recommendations(self, metrics: List[PerformanceMetrics], 
                                 hardware_info: Dict[str, Any]) -> List[str]:
        """Generate optimization recommendations based on benchmark results."""
        recommendations = []
        
        # Check if B200 optimizations are being used
        if hardware_info.get('gpu_memory_gb', 0) >= 100:  # Likely B200
            bfloat16_tests = [m for m in metrics if 'bfloat16' in m.operation]
            if not bfloat16_tests:
                recommendations.append("Consider using bfloat16 for B200 optimization")
            
            compile_tests = [m for m in metrics if 'compile' in m.operation]
            if not compile_tests:
                recommendations.append("Consider using torch.compile for additional performance")
        
        # Check memory usage patterns
        high_memory_ops = [m for m in metrics if m.gpu_memory_used_mb > 1000]
        if high_memory_ops:
            recommendations.append("Consider batch size optimization for memory-intensive operations")
        
        # Check throughput
        low_throughput_ops = [m for m in metrics if m.throughput_ops_per_sec < 10]
        if low_throughput_ops:
            recommendations.append("Some operations have low throughput - consider optimization")
        
        # Check for OOM indicators
        if hardware_info.get('cuda_available') and not any('cuda' in m.operation for m in metrics):
            recommendations.append("GPU not being utilized - ensure CUDA operations are used")
        
        return recommendations
    
    def save_benchmark_results(self, result: B200BenchmarkResult, 
                             output_path: Path) -> None:
        """Save benchmark results to file."""
        # Convert to serializable format
        result_dict = {
            'test_name': result.test_name,
            'hardware_info': result.hardware_info,
            'performance_metrics': [
                {
                    'operation': m.operation,
                    'duration_ms': m.duration_ms,
                    'gpu_memory_used_mb': m.gpu_memory_used_mb,
                    'cpu_percent': m.cpu_percent,
                    'memory_mb': m.memory_mb,
                    'throughput_ops_per_sec': m.throughput_ops_per_sec,
                    'batch_size': m.batch_size,
                    'cuda_memory_allocated_mb': m.cuda_memory_allocated_mb,
                    'cuda_memory_cached_mb': m.cuda_memory_cached_mb,
                }
                for m in result.performance_metrics
            ],
            'summary_stats': result.summary_stats,
            'recommendations': result.recommendations,
            'timestamp': result.timestamp
        }
        
        with open(output_path, 'w') as f:
            json.dump(result_dict, f, indent=2)
        
        logger.info(f"Benchmark results saved to {output_path}")


# Global profiler instance
_global_profiler = None

def get_performance_profiler() -> PerformanceProfiler:
    """Get global performance profiler instance."""
    global _global_profiler
    if _global_profiler is None:
        _global_profiler = PerformanceProfiler()
    return _global_profiler


def profile_function(operation_name: str = None):
    """Decorator to profile function performance."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            profiler = get_performance_profiler()
            op_name = operation_name or f"{func.__module__}.{func.__name__}"
            
            with profiler.profile_operation(op_name):
                result = func(*args, **kwargs)
            
            return result
        
        return wrapper
    return decorator