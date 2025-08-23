#!/usr/bin/env python3
"""
monitor.py - B200 Resource Monitor
Real-time monitoring of GPU, RAM, and disk usage optimized for RunPod B200
"""

import time
import psutil
import threading
from pathlib import Path
from typing import Dict, Any

try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from rich.console import Console
    from rich.table import Table
    from rich.columns import Columns
    from rich.live import Live
    console = Console()
    RICH_AVAILABLE = True
except ImportError:
    # Fallback console without rich formatting
    class Console:
        def print(self, *args, **kwargs):
            print(*args)
    console = Console()
    RICH_AVAILABLE = False

from constants import BYTES_TO_GB

class B200Monitor:
    """Real-time resource monitor for B200 with circuit breaker protection."""
    
    def __init__(self, max_failures: int = 5, failure_timeout: int = 30):
        self.running = False
        # Thread-safe stats avec lock pour éviter race conditions
        self._stats_lock = threading.RLock()
        self.stats = {
            'gpu': {},
            'memory': {},
            'disk': {},
            'process': {}
        }
        
        # Initialize telemetry
        try:
            from utils.telemetry import get_telemetry_manager
            self.telemetry = get_telemetry_manager()
        except ImportError:
            self.telemetry = None
        
        # Circuit breaker pattern for robust monitoring
        self.max_failures = max_failures
        self.failure_timeout = failure_timeout
        self.failure_count = 0
        self.last_failure_time = 0
        self.circuit_open = False
        
        # Monitoring state
        self.last_update_time = 0
        self.update_errors = 0
        self.total_updates = 0
        
        if PYNVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                self.gpu_count = pynvml.nvmlDeviceGetCount()
            except:
                self.gpu_count = 0
        else:
            self.gpu_count = 0
    
    def get_gpu_stats(self) -> Dict[str, Any]:
        """Get GPU statistics."""
        stats = {}
        
        if TORCH_AVAILABLE and torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                device_stats = {}
                
                # PyTorch memory info
                try:
                    device_stats['allocated_gb'] = torch.cuda.memory_allocated(i) / BYTES_TO_GB
                    device_stats['reserved_gb'] = torch.cuda.memory_reserved(i) / BYTES_TO_GB
                    device_stats['max_allocated_gb'] = torch.cuda.max_memory_allocated(i) / BYTES_TO_GB
                except:
                    device_stats['allocated_gb'] = 0
                    device_stats['reserved_gb'] = 0
                    device_stats['max_allocated_gb'] = 0
                
                # NVML info
                if PYNVML_AVAILABLE and i < self.gpu_count:
                    try:
                        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                        device_stats['total_gb'] = mem_info.total / BYTES_TO_GB
                        device_stats['free_gb'] = mem_info.free / BYTES_TO_GB
                        device_stats['used_gb'] = mem_info.used / BYTES_TO_GB
                        device_stats['utilization'] = mem_info.used / mem_info.total * 100
                        
                        # GPU utilization
                        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                        device_stats['gpu_util'] = util.gpu
                        device_stats['memory_util'] = util.memory
                        
                        # Temperature
                        device_stats['temperature'] = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                        
                        # Power
                        try:
                            device_stats['power_watts'] = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000
                            device_stats['power_limit'] = pynvml.nvmlDeviceGetPowerManagementLimitConstraints(handle)[1] / 1000
                        except:
                            device_stats['power_watts'] = 0
                            device_stats['power_limit'] = 0
                            
                    except Exception as e:
                        console.log(f"[yellow]NVML error for GPU {i}: {e}[/yellow]")
                
                device_stats['name'] = torch.cuda.get_device_name(i) if TORCH_AVAILABLE else f"GPU {i}"
                stats[f'gpu_{i}'] = device_stats
        
        return stats
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get system memory statistics."""
        mem = psutil.virtual_memory()
        swap = psutil.swap_memory()
        
        return {
            'total_gb': mem.total / BYTES_TO_GB,
            'available_gb': mem.available / BYTES_TO_GB,
            'used_gb': mem.used / BYTES_TO_GB,
            'free_gb': mem.free / BYTES_TO_GB,
            'utilization': mem.percent,
            'swap_total_gb': swap.total / BYTES_TO_GB,
            'swap_used_gb': swap.used / BYTES_TO_GB,
            'swap_utilization': swap.percent
        }
    
    def get_disk_stats(self) -> Dict[str, Any]:
        """Get disk usage statistics."""
        disk = psutil.disk_usage('/')
        
        # Check work directories
        work_dirs_size = 0
        try:
            for work_dir in Path('.').glob('work_*'):
                if work_dir.is_dir():
                    size = sum(f.stat().st_size for f in work_dir.rglob('*') if f.is_file())
                    work_dirs_size += size
        except:
            pass
        
        return {
            'total_gb': disk.total / BYTES_TO_GB,
            'used_gb': disk.used / BYTES_TO_GB,
            'free_gb': disk.free / BYTES_TO_GB,
            'utilization': (disk.used / disk.total) * 100,
            'work_dirs_gb': work_dirs_size / BYTES_TO_GB
        }
    
    def get_process_stats(self) -> Dict[str, Any]:
        """Get current process statistics."""
        process = psutil.Process()
        
        try:
            cpu_percent = process.cpu_percent()
            memory_info = process.memory_info()
            
            return {
                'cpu_percent': cpu_percent,
                'memory_gb': memory_info.rss / BYTES_TO_GB,
                'num_threads': process.num_threads(),
                'num_fds': process.num_fds() if hasattr(process, 'num_fds') else 0
            }
        except:
            return {'cpu_percent': 0, 'memory_gb': 0, 'num_threads': 0, 'num_fds': 0}
    
    def _should_skip_update(self) -> bool:
        """Check if update should be skipped due to circuit breaker."""
        current_time = time.time()
        
        # Reset circuit breaker if timeout has passed
        if self.circuit_open and (current_time - self.last_failure_time) > self.failure_timeout:
            self.circuit_open = False
            self.failure_count = 0
            console.log("[green]Monitoring circuit breaker reset[/green]")
        
        return self.circuit_open
    
    def _record_failure(self):
        """Record a monitoring failure."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        self.update_errors += 1
        
        if self.failure_count >= self.max_failures:
            self.circuit_open = True
            console.log(f"[red]Monitoring circuit breaker opened after {self.failure_count} failures[/red]")
    
    def _record_success(self):
        """Record a successful monitoring update."""
        if self.failure_count > 0:
            # Gradually reduce failure count on success
            self.failure_count = max(0, self.failure_count - 1)
        
        self.last_update_time = time.time()
        self.total_updates += 1
    
    def update_stats(self):
        """Update all statistics with circuit breaker protection and thread safety."""
        if self._should_skip_update():
            return
        
        try:
            # Collecter les stats en local d'abord (pas de lock nécessaire)
            gpu_stats = self.get_gpu_stats()
            memory_stats = self.get_memory_stats()
            disk_stats = self.get_disk_stats()
            process_stats = self.get_process_stats()
            
            # Mise à jour atomique avec lock
            with self._stats_lock:
                self.stats['gpu'] = gpu_stats
                self.stats['memory'] = memory_stats
                self.stats['disk'] = disk_stats
                self.stats['process'] = process_stats
            
            # Export to telemetry
            if self.telemetry:
                self._export_to_telemetry(memory_stats, disk_stats, process_stats, gpu_stats)
            
            self._record_success()
            
        except Exception as e:
            console.log(f"[yellow]Monitoring update failed: {e}[/yellow]")
            self._record_failure()
            
            # Use cached stats if available - thread-safe access
            with self._stats_lock:
                if not self.stats['memory']:
                    # Provide minimal fallback stats
                    self.stats = {
                        'gpu': {},
                        'memory': {'utilization': 0, 'total_gb': 0, 'used_gb': 0},
                        'disk': {'utilization': 0, 'total_gb': 0, 'used_gb': 0},
                        'process': {'cpu_percent': 0, 'memory_gb': 0}
                    }
    
    def create_display_table(self) -> Table:
        """Create a rich table for displaying stats."""
        
        # GPU Table
        gpu_table = Table(title="🚀 B200 GPU Status", show_header=True, header_style="bold magenta")
        gpu_table.add_column("GPU", style="cyan")
        gpu_table.add_column("Memory", justify="right")
        gpu_table.add_column("Utilization", justify="right")
        gpu_table.add_column("Temperature", justify="right")
        gpu_table.add_column("Power", justify="right")
        
        # Thread-safe access aux GPU stats
        with self._stats_lock:
            gpu_stats_copy = self.stats['gpu'].copy()
        
        for gpu_id, stats in gpu_stats_copy.items():
            if 'total_gb' in stats:
                memory_str = f"{stats['used_gb']:.1f}/{stats['total_gb']:.1f}GB ({stats['utilization']:.1f}%)"
                util_str = f"GPU: {stats.get('gpu_util', 0)}%, Mem: {stats.get('memory_util', 0)}%"
                temp_str = f"{stats.get('temperature', 0)}°C"
                power_str = f"{stats.get('power_watts', 0):.0f}W"
                
                gpu_table.add_row(
                    stats['name'][:20],
                    memory_str,
                    util_str,
                    temp_str,
                    power_str
                )
        
        # System Table
        sys_table = Table(title="💾 System Resources", show_header=True, header_style="bold green")
        sys_table.add_column("Resource", style="cyan")
        sys_table.add_column("Usage", justify="right")
        sys_table.add_column("Available", justify="right")
        sys_table.add_column("Utilization", justify="right")
        
        # Thread-safe access aux stats
        with self._stats_lock:
            mem = self.stats['memory'].copy()
            disk = self.stats['disk'].copy()
            proc = self.stats['process'].copy()
        
        sys_table.add_row(
            "RAM",
            f"{mem['used_gb']:.1f}GB",
            f"{mem['available_gb']:.1f}GB",
            f"{mem['utilization']:.1f}%"
        )
        
        sys_table.add_row(
            "Disk",
            f"{disk['used_gb']:.1f}GB",
            f"{disk['free_gb']:.1f}GB",
            f"{disk['utilization']:.1f}%"
        )
        
        sys_table.add_row(
            "Work Dirs",
            f"{disk['work_dirs_gb']:.2f}GB",
            "",
            ""
        )
        
        sys_table.add_row(
            "Process",
            f"{proc['memory_gb']:.2f}GB",
            f"CPU: {proc['cpu_percent']:.1f}%",
            f"Threads: {proc['num_threads']}"
        )
        
        return Columns([gpu_table, sys_table])
    
    def monitor_loop(self, update_interval: float = 2.0):
        """Main monitoring loop with robust error handling."""
        try:
            with Live(self.create_display_table(), refresh_per_second=0.5) as live:
                while self.running:
                    try:
                        self.update_stats()
                        live.update(self.create_display_table())
                        time.sleep(update_interval)
                    except KeyboardInterrupt:
                        console.log("[yellow]Monitoring interrupted by user[/yellow]")
                        break
                    except Exception as e:
                        console.log(f"[red]Monitoring loop error: {e}[/red]")
                        # Add exponential backoff on errors
                        error_sleep = min(update_interval * 2, 10.0)
                        time.sleep(error_sleep)
        except Exception as e:
            console.log(f"[red]Critical monitoring error: {e}[/red]")
        finally:
            self.running = False
    
    def start_monitoring(self, update_interval: float = 2.0):
        """Start monitoring in a separate thread."""
        self.running = True
        self.monitor_thread = threading.Thread(target=self.monitor_loop, args=(update_interval,))
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        console.log("[green]🔍 B200 monitoring started[/green]")
    
    def stop_monitoring(self):
        """Stop monitoring."""
        self.running = False
        if hasattr(self, 'monitor_thread'):
            self.monitor_thread.join(timeout=1.0)
        console.log("[yellow]📊 Monitoring stopped[/yellow]")
    
    def get_summary(self) -> str:
        """Get a summary string of current resource usage."""
        self.update_stats()
        
        # Add circuit breaker status to summary
        status_indicator = ""
        if self.circuit_open:
            status_indicator = "[CIRCUIT OPEN] "
        elif self.failure_count > 0:
            status_indicator = f"[{self.failure_count} failures] "
        
        gpu_summary = ""
        # Thread-safe access aux GPU stats
        with self._stats_lock:
            gpu_stats_copy = self.stats['gpu'].copy()
        
        for gpu_id, stats in gpu_stats_copy.items():
            if 'utilization' in stats:
                gpu_summary += f"GPU: {stats['utilization']:.1f}% "
        
        # Thread-safe access aux autres stats
        with self._stats_lock:
            mem = self.stats['memory'].copy()
            disk = self.stats['disk'].copy()
        
        return (f"{status_indicator}{gpu_summary}| "
                f"RAM: {mem.get('utilization', 0):.1f}% ({mem.get('used_gb', 0):.1f}GB) | "
                f"Disk: {disk.get('utilization', 0):.1f}% ({disk.get('free_gb', 0):.1f}GB free) | "
                f"Work: {disk.get('work_dirs_gb', 0):.2f}GB | "
                f"Updates: {self.total_updates}/{self.total_updates + self.update_errors}")
    
    def _export_to_telemetry(self, memory_stats: Dict, disk_stats: Dict, process_stats: Dict, gpu_stats: Dict) -> None:
        """Export monitoring data to telemetry system."""
        try:
            # System metrics
            self.telemetry.record_gauge("system_memory_utilization", memory_stats.get('utilization', 0))
            self.telemetry.record_gauge("system_disk_utilization", disk_stats.get('utilization', 0))
            self.telemetry.record_gauge("process_cpu_percent", process_stats.get('cpu_percent', 0))
            self.telemetry.record_gauge("process_memory_gb", process_stats.get('memory_gb', 0))
            
            # GPU metrics
            for gpu_id, stats in gpu_stats.items():
                gpu_tags = {"gpu_id": gpu_id, "gpu_name": stats.get('name', 'unknown')}
                
                if 'utilization' in stats:
                    self.telemetry.record_gauge("gpu_memory_utilization", stats['utilization'], gpu_tags)
                if 'gpu_util' in stats:
                    self.telemetry.record_gauge("gpu_compute_utilization", stats['gpu_util'], gpu_tags)
                if 'temperature' in stats:
                    self.telemetry.record_gauge("gpu_temperature_celsius", stats['temperature'], gpu_tags)
                if 'power_watts' in stats:
                    self.telemetry.record_gauge("gpu_power_watts", stats['power_watts'], gpu_tags)
                    
        except Exception as e:
            console.log(f"[yellow]Telemetry export failed: {e}[/yellow]")

