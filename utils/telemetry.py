#!/usr/bin/env python3
"""
telemetry.py - OpenTelemetry Metrics and Observability
Advanced telemetry system for B200 GPU monitoring and performance analysis
"""

import time
import functools
from typing import Dict, Any, Optional, Callable, List
from dataclasses import dataclass, field
from contextlib import contextmanager
import threading
from pathlib import Path
import json

try:
    from opentelemetry import metrics
    from opentelemetry.metrics import MeterProvider, Counter, Histogram, Gauge
    from opentelemetry.sdk.metrics import MeterProvider as SDKMeterProvider
    from opentelemetry.sdk.metrics.export import MetricExporter, MetricExportResult
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.exporter.prometheus import PrometheusMetricReader
    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from rich.console import Console
    console = Console()
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    # Fallback console for environments without rich
    class FallbackConsole:
        def log(self, msg):
            print(msg.replace('[green]', '').replace('[/green]', '')
                     .replace('[yellow]', '').replace('[/yellow]', '')
                     .replace('[red]', '').replace('[/red]', ''))
    console = FallbackConsole()


@dataclass
class MetricData:
    """Structured metric data container."""
    name: str
    value: float
    timestamp: float
    tags: Dict[str, str] = field(default_factory=dict)
    unit: str = "count"


class LocalMetricExporter:
    """Local metric exporter that writes to JSON files for analysis."""
    
    def __init__(self, output_dir: str = "metrics"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.metrics_buffer: List[MetricData] = []
        self.buffer_lock = threading.RLock()
        self.max_buffer_size = 1000
        
    def export_metric(self, metric: MetricData):
        """Export a single metric."""
        with self.buffer_lock:
            self.metrics_buffer.append(metric)
            
            if len(self.metrics_buffer) >= self.max_buffer_size:
                self._flush_buffer()
    
    def _flush_buffer(self):
        """Flush metrics buffer to file."""
        if not self.metrics_buffer:
            return
            
        timestamp = int(time.time())
        filename = self.output_dir / f"metrics_{timestamp}.json"
        
        try:
            with open(filename, 'w') as f:
                json.dump([
                    {
                        'name': m.name,
                        'value': m.value,
                        'timestamp': m.timestamp,
                        'tags': m.tags,
                        'unit': m.unit
                    }
                    for m in self.metrics_buffer
                ], f, indent=2)
            
            console.log(f"[green]Exported {len(self.metrics_buffer)} metrics to {filename}[/green]")
            self.metrics_buffer.clear()
            
        except Exception as e:
            console.log(f"[red]Failed to export metrics: {e}[/red]")
    
    def flush(self):
        """Force flush all buffered metrics."""
        with self.buffer_lock:
            self._flush_buffer()


class TelemetryManager:
    """Advanced telemetry manager with OpenTelemetry integration."""
    
    def __init__(self, service_name: str = "emanet_voxtral"):
        self.service_name = service_name
        self.local_exporter = LocalMetricExporter()
        self.metrics_cache: Dict[str, Any] = {}
        self.metric_definitions: Dict[str, Dict[str, Any]] = {}
        
        # Performance tracking
        self.operation_timers: Dict[str, float] = {}
        self.operation_counters: Dict[str, int] = {}
        
        # Initialize OpenTelemetry if available
        if OTEL_AVAILABLE:
            self._setup_opentelemetry()
        else:
            console.log("[yellow]OpenTelemetry not available, using local metrics only[/yellow]")
            self.meter = None
    
    def _setup_opentelemetry(self):
        """Setup OpenTelemetry metrics."""
        try:
            # Create resource
            resource = Resource.create({
                "service.name": self.service_name,
                "service.version": "1.0.0",
                "deployment.environment": "production"
            })
            
            # Create meter provider
            self.meter_provider = SDKMeterProvider(resource=resource)
            metrics.set_meter_provider(self.meter_provider)
            
            # Create meter
            self.meter = metrics.get_meter(self.service_name)
            
            # Define core metrics
            self._define_core_metrics()
            
            console.log("[green]OpenTelemetry metrics initialized[/green]")
            
        except Exception as e:
            console.log(f"[yellow]OpenTelemetry setup failed: {e}[/yellow]")
            self.meter = None
    
    def _define_core_metrics(self):
        """Define core application metrics."""
        if not self.meter:
            return
        
        # Performance metrics
        self.request_duration = self.meter.create_histogram(
            name="request_duration_seconds",
            description="Request duration in seconds",
            unit="s"
        )
        
        self.request_counter = self.meter.create_counter(
            name="requests_total",
            description="Total number of requests"
        )
        
        # GPU metrics
        self.gpu_memory_usage = self.meter.create_gauge(
            name="gpu_memory_usage_bytes",
            description="GPU memory usage in bytes"
        )
        
        self.gpu_utilization = self.meter.create_gauge(
            name="gpu_utilization_percent",
            description="GPU utilization percentage"
        )
        
        # Audio processing metrics
        self.audio_processing_duration = self.meter.create_histogram(
            name="audio_processing_duration_seconds",
            description="Audio processing duration in seconds"
        )
        
        self.audio_segments_processed = self.meter.create_counter(
            name="audio_segments_processed_total",
            description="Total number of audio segments processed"
        )
        
        # Batch processing metrics
        self.batch_size = self.meter.create_histogram(
            name="batch_size",
            description="Batch size distribution"
        )
        
        self.batch_processing_time = self.meter.create_histogram(
            name="batch_processing_time_seconds",
            description="Batch processing time in seconds"
        )
        
        # Error metrics
        self.error_counter = self.meter.create_counter(
            name="errors_total",
            description="Total number of errors"
        )
        
        # Store metric definitions for local export
        self.metric_definitions.update({
            "request_duration": {"type": "histogram", "unit": "seconds"},
            "requests_total": {"type": "counter", "unit": "count"},
            "gpu_memory_usage": {"type": "gauge", "unit": "bytes"},
            "gpu_utilization": {"type": "gauge", "unit": "percent"},
            "audio_processing_duration": {"type": "histogram", "unit": "seconds"},
            "audio_segments_processed": {"type": "counter", "unit": "count"},
            "batch_size": {"type": "histogram", "unit": "count"},
            "batch_processing_time": {"type": "histogram", "unit": "seconds"},
            "errors_total": {"type": "counter", "unit": "count"}
        })
    
    def record_counter(self, name: str, value: int = 1, tags: Optional[Dict[str, str]] = None):
        """Record a counter metric."""
        tags = tags if tags is not None else {}
        
        # Record to OpenTelemetry
        if self.meter and hasattr(self, name.replace("_total", "_counter")):
            metric = getattr(self, name.replace("_total", "_counter"))
            metric.add(value, tags)
        
        # Record to local exporter
        self.local_exporter.export_metric(MetricData(
            name=name,
            value=value,
            timestamp=time.time(),
            tags=tags,
            unit="count"
        ))
        
        # Update local cache
        cache_key = f"{name}_{hash(frozenset(tags.items()))}"
        self.operation_counters[cache_key] = self.operation_counters.get(cache_key, 0) + value
    
    def record_gauge(self, name: str, value: float, tags: Optional[Dict[str, str]] = None):
        """Record a gauge metric."""
        tags = tags if tags is not None else {}
        
        # Record to OpenTelemetry
        if self.meter and hasattr(self, name):
            metric = getattr(self, name)
            metric.set(value, tags)
        
        # Record to local exporter
        unit = "bytes" if "memory" in name else "percent" if "utilization" in name else "count"
        self.local_exporter.export_metric(MetricData(
            name=name,
            value=value,
            timestamp=time.time(),
            tags=tags,
            unit=unit
        ))
    
    def record_histogram(self, name: str, value: float, tags: Optional[Dict[str, str]] = None):
        """Record a histogram metric."""
        tags = tags if tags is not None else {}
        
        # Record to OpenTelemetry
        if self.meter and hasattr(self, name):
            metric = getattr(self, name)
            metric.record(value, tags)
        
        # Record to local exporter
        unit = "seconds" if "duration" in name or "time" in name else "count"
        self.local_exporter.export_metric(MetricData(
            name=name,
            value=value,
            timestamp=time.time(),
            tags=tags,
            unit=unit
        ))
    
    @contextmanager
    def time_operation(self, operation_name: str, tags: Optional[Dict[str, str]] = None):
        """Context manager for timing operations."""
        start_time = time.time()
        tags = tags if tags is not None else {}
        
        try:
            yield
            
        except Exception as e:
            # Record error
            error_tags = {**tags, "error_type": type(e).__name__}
            self.record_counter("errors_total", 1, error_tags)
            raise
            
        finally:
            # Record duration
            duration = time.time() - start_time
            duration_tags = {**tags, "operation": operation_name}
            self.record_histogram("request_duration", duration, duration_tags)
    
    def track_gpu_metrics(self):
        """Track GPU metrics if available."""
        if not TORCH_AVAILABLE or not torch.cuda.is_available():
            return
        
        try:
            for i in range(torch.cuda.device_count()):
                device_tags = {"gpu_id": str(i), "device_name": torch.cuda.get_device_name(i)}
                
                # Memory metrics
                allocated = torch.cuda.memory_allocated(i)
                reserved = torch.cuda.memory_reserved(i)
                
                self.record_gauge("gpu_memory_usage", allocated, {**device_tags, "type": "allocated"})
                self.record_gauge("gpu_memory_usage", reserved, {**device_tags, "type": "reserved"})
                
                # Try to get utilization if pynvml available
                try:
                    import pynvml
                    pynvml.nvmlInit()
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    
                    self.record_gauge("gpu_utilization", util.gpu, {**device_tags, "type": "compute"})
                    self.record_gauge("gpu_utilization", util.memory, {**device_tags, "type": "memory"})
                    
                except ImportError:
                    pass
                    
        except Exception as e:
            console.log(f"[yellow]GPU metrics collection failed: {e}[/yellow]")
    
    def track_audio_processing(self, segments_count: int, processing_time: float, 
                             batch_size: int, model_name: str = "unknown"):
        """Track audio processing metrics."""
        tags = {"model": model_name, "batch_size_range": self._get_batch_size_range(batch_size)}
        
        self.record_counter("audio_segments_processed", segments_count, tags)
        self.record_histogram("audio_processing_duration", processing_time, tags)
        self.record_histogram("batch_size", batch_size, tags)
        self.record_histogram("batch_processing_time", processing_time, tags)
    
    def _get_batch_size_range(self, batch_size: int) -> str:
        """Get batch size range for better aggregation."""
        if batch_size <= 10:
            return "small"
        elif batch_size <= 50:
            return "medium"
        elif batch_size <= 100:
            return "large"
        else:
            return "xlarge"
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of collected metrics."""
        summary = {
            "counters": self.operation_counters.copy(),
            "cache_size": len(self.metrics_cache),
            "local_buffer_size": len(self.local_exporter.metrics_buffer),
            "opentelemetry_enabled": self.meter is not None
        }
        
        # Add GPU summary if available
        if TORCH_AVAILABLE and torch.cuda.is_available():
            summary["gpu_devices"] = torch.cuda.device_count()
            summary["gpu_memory_allocated"] = sum(
                torch.cuda.memory_allocated(i) for i in range(torch.cuda.device_count())
            )
        
        return summary
    
    def flush_all(self):
        """Flush all metrics to exporters."""
        self.local_exporter.flush()
        
        if hasattr(self, 'meter_provider') and self.meter_provider and hasattr(self.meter_provider, 'force_flush'):
            try:
                self.meter_provider.force_flush(timeout_millis=5000)
            except Exception as e:
                console.log(f"[yellow]OpenTelemetry flush failed: {e}[/yellow]")


def telemetry_decorator(operation_name: str, track_gpu: bool = False):
    """Decorator for automatic telemetry tracking."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            telemetry = get_telemetry_manager()
            
            with telemetry.time_operation(operation_name, {"function": func.__name__}):
                if track_gpu:
                    telemetry.track_gpu_metrics()
                
                result = func(*args, **kwargs)
                
                # Track successful completion
                telemetry.record_counter("requests_total", 1, {
                    "operation": operation_name,
                    "function": func.__name__,
                    "status": "success"
                })
                
                return result
        
        return wrapper
    return decorator


# Global telemetry manager instance
_telemetry_manager: Optional[TelemetryManager] = None
_telemetry_lock = threading.Lock()


def get_telemetry_manager() -> TelemetryManager:
    """Get or create global telemetry manager instance."""
    global _telemetry_manager
    
    if _telemetry_manager is None:
        with _telemetry_lock:
            if _telemetry_manager is None:
                _telemetry_manager = TelemetryManager()
    
    return _telemetry_manager


def init_telemetry(service_name: str = "emanet_voxtral") -> TelemetryManager:
    """Initialize telemetry system."""
    global _telemetry_manager
    
    with _telemetry_lock:
        _telemetry_manager = TelemetryManager(service_name)
    
    console.log(f"[green]Telemetry initialized for service: {service_name}[/green]")
    return _telemetry_manager


def shutdown_telemetry():
    """Shutdown telemetry system and flush all metrics."""
    global _telemetry_manager
    
    if _telemetry_manager:
        _telemetry_manager.flush_all()
        console.log("[green]Telemetry shutdown complete[/green]")