#!/usr/bin/env python3
"""
benchmark.py - Comprehensive B200 benchmarking script
Performance testing and optimization validation for B200 hardware
"""

import argparse
import time
import json
from pathlib import Path
from typing import Dict, Any, List
import logging

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

console = Console()

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    console.print("[yellow]PyTorch not available, some benchmarks will be skipped[/yellow]")

try:
    from utils.performance_profiler import B200Benchmarker, get_performance_profiler
    from utils.b200_optimizer import get_b200_optimizer, B200Config
    from config import detect_hardware
    from cli_feedback import get_feedback
    BENCHMARK_AVAILABLE = True
except ImportError as e:
    BENCHMARK_AVAILABLE = False
    console.print(f"[yellow]Benchmark modules not available: {e}[/yellow]")


def benchmark_basic_operations() -> Dict[str, Any]:
    """Benchmark basic system operations."""
    console.print("[cyan]Running basic operations benchmark...[/cyan]")
    
    results = {
        "cpu_operations": {},
        "memory_operations": {},
        "disk_operations": {}
    }
    
    # CPU benchmark
    start_time = time.perf_counter()
    for i in range(1000000):
        _ = i ** 2
    cpu_time = time.perf_counter() - start_time
    results["cpu_operations"]["cpu_intensive_loop"] = {
        "duration_ms": cpu_time * 1000,
        "operations_per_sec": 1000000 / cpu_time
    }
    
    # Memory benchmark
    if TORCH_AVAILABLE:
        start_time = time.perf_counter()
        large_tensor = torch.randn(10000, 1000)
        _ = torch.sum(large_tensor)
        memory_time = time.perf_counter() - start_time
        results["memory_operations"]["large_tensor_sum"] = {
            "duration_ms": memory_time * 1000,
            "tensor_size_mb": large_tensor.numel() * 4 / 1024**2  # float32 = 4 bytes
        }
    
    return results


def benchmark_b200_optimizations() -> Dict[str, Any]:
    """Benchmark B200-specific optimizations."""
    if not BENCHMARK_AVAILABLE or not TORCH_AVAILABLE:
        return {"error": "B200 benchmark modules not available"}
    
    console.print("[cyan]Running B200 optimization benchmarks...[/cyan]")
    
    benchmarker = B200Benchmarker()
    result = benchmarker.run_comprehensive_benchmark()
    
    # Convert to dict format for JSON serialization
    return {
        "test_name": result.test_name,
        "hardware_info": result.hardware_info,
        "summary_stats": result.summary_stats,
        "recommendations": result.recommendations,
        "timestamp": result.timestamp,
        "num_metrics": len(result.performance_metrics)
    }

def benchmark_model_optimizations() -> Dict[str, Any]:
    """Benchmark model optimization features."""
    if not TORCH_AVAILABLE:
        return {"error": "PyTorch not available"}
    
    console.print("[cyan]Running model optimization benchmarks...[/cyan]")
    
    results = {}
    
    # Test torch.compile availability
    if hasattr(torch, 'compile'):
        def simple_model(x):
            return torch.nn.functional.relu(x)
        
        # Benchmark without compilation
        x = torch.randn(1000, 1000)
        if torch.cuda.is_available():
            x = x.cuda()
        
        start_time = time.perf_counter()
        for _ in range(100):
            _ = simple_model(x)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        uncompiled_time = time.perf_counter() - start_time
        
        # Benchmark with compilation
        try:
            compiled_model = torch.compile(simple_model)
            
            start_time = time.perf_counter()
            for _ in range(100):
                _ = compiled_model(x)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            compiled_time = time.perf_counter() - start_time
            
            results["torch_compile"] = {
                "uncompiled_time_ms": uncompiled_time * 1000,
                "compiled_time_ms": compiled_time * 1000,
                "speedup_ratio": uncompiled_time / compiled_time if compiled_time > 0 else 0
            }
        except Exception as e:
            results["torch_compile"] = {"error": str(e)}
    
    # Test dtype optimizations
    if torch.cuda.is_available():
        dtypes_to_test = [torch.float32, torch.float16]
        if hasattr(torch, 'bfloat16'):
            dtypes_to_test.append(torch.bfloat16)
        
        dtype_results = {}
        for dtype in dtypes_to_test:
            try:
                a = torch.randn(2048, 2048, dtype=dtype, device='cuda')
                b = torch.randn(2048, 2048, dtype=dtype, device='cuda')
                
                start_time = time.perf_counter()
                for _ in range(10):
                    _ = torch.matmul(a, b)
                torch.cuda.synchronize()
                duration = time.perf_counter() - start_time
                
                dtype_results[str(dtype)] = {
                    "duration_ms": duration * 1000,
                    "operations_per_sec": 10 / duration
                }
            except Exception as e:
                dtype_results[str(dtype)] = {"error": str(e)}
        
        results["dtype_optimization"] = dtype_results
    
    return results


def display_benchmark_results(results: Dict[str, Any]) -> None:
    """Display benchmark results in a formatted table."""
    console.print("\n")
    console.print(Panel.fit(
        "🚀 B200 BENCHMARK RESULTS",
        style="bold green"
    ))
    
    # Hardware Information
    if "hardware_info" in results.get("b200_benchmarks", {}):
        hw_info = results["b200_benchmarks"]["hardware_info"]
        
        hw_table = Table(title="🔧 Hardware Configuration", show_header=True, header_style="bold magenta")
        hw_table.add_column("Component", style="cyan")
        hw_table.add_column("Value", style="green")
        
        hw_table.add_row("CPU Count", str(hw_info.get("cpu_count", "Unknown")))
        hw_table.add_row("Total Memory", f"{hw_info.get('total_memory_gb', 0):.1f} GB")
        hw_table.add_row("CUDA Available", str(hw_info.get("cuda_available", False)))
        
        if hw_info.get("cuda_available"):
            hw_table.add_row("GPU Name", hw_info.get("gpu_name", "Unknown"))
            hw_table.add_row("GPU Memory", f"{hw_info.get('gpu_memory_gb', 0):.1f} GB")
            hw_table.add_row("Compute Capability", str(hw_info.get("compute_capability", "Unknown")))
        
        console.print(hw_table)
        console.print()
    
    # Performance Summary
    if "b200_benchmarks" in results and "summary_stats" in results["b200_benchmarks"]:
        stats = results["b200_benchmarks"]["summary_stats"]
        
        perf_table = Table(title="📊 Performance Summary", show_header=True, header_style="bold magenta")
        perf_table.add_column("Metric", style="cyan")
        perf_table.add_column("Value", style="green")
        
        if "avg_duration_ms" in stats:
            perf_table.add_row("Average Duration", f"{stats['avg_duration_ms']:.2f} ms")
        if "max_throughput_ops_per_sec" in stats:
            perf_table.add_row("Max Throughput", f"{stats['max_throughput_ops_per_sec']:.1f} ops/sec")
        if "total_gpu_memory_used_mb" in stats:
            perf_table.add_row("Total GPU Memory Used", f"{stats['total_gpu_memory_used_mb']:.1f} MB")
        
        console.print(perf_table)
        console.print()
    
    # Model Optimizations
    if "model_optimizations" in results:
        model_results = results["model_optimizations"]
        
        model_table = Table(title="⚡ Model Optimization Results", show_header=True, header_style="bold magenta")
        model_table.add_column("Optimization", style="cyan")
        model_table.add_column("Result", style="green")
        model_table.add_column("Speedup", style="yellow")
        
        if "torch_compile" in model_results and "speedup_ratio" in model_results["torch_compile"]:
            speedup = model_results["torch_compile"]["speedup_ratio"]
            model_table.add_row(
                "torch.compile",
                "✅ Available" if speedup > 0 else "❌ Failed",
                f"{speedup:.2f}x" if speedup > 0 else "N/A"
            )
        
        if "dtype_optimization" in model_results:
            for dtype, data in model_results["dtype_optimization"].items():
                if "error" not in data:
                    model_table.add_row(
                        f"dtype: {dtype}",
                        "✅ Working",
                        f"{data['operations_per_sec']:.1f} ops/sec"
                    )
        
        console.print(model_table)
        console.print()
    
    # Recommendations
    if "b200_benchmarks" in results and "recommendations" in results["b200_benchmarks"]:
        recommendations = results["b200_benchmarks"]["recommendations"]
        
        if recommendations:
            console.print(Panel(
                "\n".join([f"• {rec}" for rec in recommendations]),
                title="💡 Optimization Recommendations",
                style="yellow"
            ))


def save_results(results: Dict[str, Any], output_path: Path) -> None:
    """Save benchmark results to file."""
    try:
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        console.print(f"[green]Results saved to {output_path}[/green]")
    except Exception as e:
        console.print(f"[red]Failed to save results: {e}[/red]")


def main():
    """Main benchmark execution function."""
    parser = argparse.ArgumentParser(description="B200 Comprehensive Benchmark Suite")
    parser.add_argument("--output", "-o", type=Path, default="benchmark_results.json",
                       help="Output file for results")
    parser.add_argument("--basic-only", action="store_true",
                       help="Run only basic benchmarks")
    parser.add_argument("--b200-only", action="store_true",
                       help="Run only B200-specific benchmarks")
    parser.add_argument("--model-only", action="store_true",
                       help="Run only model optimization benchmarks")
    parser.add_argument("--quick", action="store_true",
                       help="Run quick benchmark (reduced iterations)")
    
    args = parser.parse_args()
    
    console.print(Panel.fit(
        "🔥 B200 PERFORMANCE BENCHMARK SUITE\n"
        "Comprehensive performance testing and optimization validation",
        style="bold blue"
    ))
    
    results = {
        "benchmark_info": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "torch_available": TORCH_AVAILABLE,
            "benchmark_modules_available": BENCHMARK_AVAILABLE,
            "cuda_available": torch.cuda.is_available() if TORCH_AVAILABLE else False
        }
    }
    
    # Determine which benchmarks to run
    run_basic = not args.b200_only and not args.model_only
    run_b200 = not args.basic_only and not args.model_only
    run_model = not args.basic_only and not args.b200_only
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
    ) as progress:
        
        total_tasks = sum([run_basic, run_b200, run_model])
        overall_task = progress.add_task("Overall Progress", total=total_tasks)
        
        if run_basic:
            progress.update(overall_task, description="Running basic benchmarks...")
            results["basic_operations"] = benchmark_basic_operations()
            progress.advance(overall_task)
        
        if run_b200:
            progress.update(overall_task, description="Running B200 benchmarks...")
            results["b200_benchmarks"] = benchmark_b200_optimizations()
            progress.advance(overall_task)
        
        if run_model:
            progress.update(overall_task, description="Running model optimization benchmarks...")
            results["model_optimizations"] = benchmark_model_optimizations()
            progress.advance(overall_task)
    
    # Display results
    display_benchmark_results(results)
    
    # Save results
    save_results(results, args.output)
    
    console.print("\n")
    console.print(Panel.fit(
        "✅ BENCHMARK COMPLETED\n"
        f"Results saved to: {args.output}",
        style="bold green"
    ))


if __name__ == "__main__":
    main()