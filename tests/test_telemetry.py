#!/usr/bin/env python3
"""
test_telemetry.py - Tests for telemetry system
"""

import pytest
import time
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.telemetry import (
    TelemetryManager, MetricData, LocalMetricExporter,
    telemetry_decorator, get_telemetry_manager, init_telemetry
)


class TestMetricData:
    """Test MetricData container."""
    
    def test_metric_data_creation(self):
        """Test metric data creation with defaults."""
        metric = MetricData(name="test_metric", value=42.0, timestamp=time.time())
        
        assert metric.name == "test_metric"
        assert metric.value == 42.0
        assert metric.timestamp > 0
        assert metric.tags == {}
        assert metric.unit == "count"
    
    def test_metric_data_with_tags(self):
        """Test metric data with custom tags and unit."""
        tags = {"environment": "test", "service": "api"}
        metric = MetricData(
            name="response_time", 
            value=0.123, 
            timestamp=time.time(),
            tags=tags,
            unit="seconds"
        )
        
        assert metric.tags == tags
        assert metric.unit == "seconds"


class TestLocalMetricExporter:
    """Test local metric exporter."""
    
    def test_export_metric(self):
        """Test exporting a single metric."""
        with tempfile.TemporaryDirectory() as tmpdir:
            exporter = LocalMetricExporter(tmpdir)
            
            metric = MetricData("test_counter", 1.0, time.time())
            exporter.export_metric(metric)
            
            assert len(exporter.metrics_buffer) == 1
            assert exporter.metrics_buffer[0].name == "test_counter"
    
    def test_buffer_flush_on_size(self):
        """Test automatic buffer flush when size limit reached."""
        with tempfile.TemporaryDirectory() as tmpdir:
            exporter = LocalMetricExporter(tmpdir)
            exporter.max_buffer_size = 5
            
            # Add metrics to trigger flush
            for i in range(6):
                metric = MetricData(f"metric_{i}", float(i), time.time())
                exporter.export_metric(metric)
            
            # Buffer should be flushed and contain only the last metric
            assert len(exporter.metrics_buffer) == 1
            
            # Check that file was created
            metric_files = list(Path(tmpdir).glob("metrics_*.json"))
            assert len(metric_files) == 1
    
    def test_manual_flush(self):
        """Test manual buffer flush."""
        with tempfile.TemporaryDirectory() as tmpdir:
            exporter = LocalMetricExporter(tmpdir)
            
            # Add some metrics
            for i in range(3):
                metric = MetricData(f"metric_{i}", float(i), time.time())
                exporter.export_metric(metric)
            
            # Manual flush
            exporter.flush()
            
            # Buffer should be empty
            assert len(exporter.metrics_buffer) == 0
            
            # File should exist
            metric_files = list(Path(tmpdir).glob("metrics_*.json"))
            assert len(metric_files) == 1
            
            # Verify content
            with open(metric_files[0]) as f:
                data = json.load(f)
            assert len(data) == 3
            assert data[0]['name'] == 'metric_0'


class TestTelemetryManager:
    """Test telemetry manager."""
    
    def test_telemetry_manager_creation(self):
        """Test telemetry manager creation."""
        manager = TelemetryManager("test_service")
        
        assert manager.service_name == "test_service"
        assert isinstance(manager.local_exporter, LocalMetricExporter)
        assert isinstance(manager.metrics_cache, dict)
        assert isinstance(manager.operation_counters, dict)
    
    def test_record_counter(self):
        """Test recording counter metrics."""
        manager = TelemetryManager("test_service")
        
        # Record some counters
        manager.record_counter("requests_total", 1, {"endpoint": "/api/test"})
        manager.record_counter("requests_total", 2, {"endpoint": "/api/test"})
        
        # Check local exporter received metrics
        assert len(manager.local_exporter.metrics_buffer) == 2
        
        # Check operation counters
        assert len(manager.operation_counters) == 1
        counter_key = list(manager.operation_counters.keys())[0]
        assert manager.operation_counters[counter_key] == 3
    
    def test_record_gauge(self):
        """Test recording gauge metrics."""
        manager = TelemetryManager("test_service")
        
        manager.record_gauge("cpu_usage", 75.5, {"host": "test-host"})
        
        # Check metric was exported
        assert len(manager.local_exporter.metrics_buffer) == 1
        metric = manager.local_exporter.metrics_buffer[0]
        assert metric.name == "cpu_usage"
        assert metric.value == 75.5
        assert metric.tags == {"host": "test-host"}
        assert metric.unit == "percent"
    
    def test_record_histogram(self):
        """Test recording histogram metrics."""
        manager = TelemetryManager("test_service")
        
        manager.record_histogram("request_duration", 0.123, {"method": "GET"})
        
        # Check metric was exported
        assert len(manager.local_exporter.metrics_buffer) == 1
        metric = manager.local_exporter.metrics_buffer[0]
        assert metric.name == "request_duration"
        assert metric.value == 0.123
        assert metric.unit == "seconds"
    
    def test_time_operation_success(self):
        """Test timing operation context manager - success case."""
        manager = TelemetryManager("test_service")
        
        with manager.time_operation("test_operation", {"service": "api"}):
            time.sleep(0.01)  # Small delay
        
        # Should have recorded duration
        assert len(manager.local_exporter.metrics_buffer) == 1
        metric = manager.local_exporter.metrics_buffer[0]
        assert metric.name == "request_duration"
        assert metric.value > 0.005  # Should be more than 5ms
        assert metric.tags == {"service": "api", "operation": "test_operation"}
    
    def test_time_operation_with_error(self):
        """Test timing operation context manager - error case."""
        manager = TelemetryManager("test_service")
        
        with pytest.raises(ValueError):
            with manager.time_operation("test_operation"):
                raise ValueError("Test error")
        
        # Should have recorded both duration and error
        assert len(manager.local_exporter.metrics_buffer) == 2
        
        # Find error metric
        error_metrics = [m for m in manager.local_exporter.metrics_buffer if m.name == "errors_total"]
        assert len(error_metrics) == 1
        assert error_metrics[0].tags["error_type"] == "ValueError"
    
    @patch('utils.telemetry.TORCH_AVAILABLE', True)
    @patch('torch.cuda.is_available', return_value=True)
    @patch('torch.cuda.device_count', return_value=1)
    @patch('torch.cuda.get_device_name', return_value="Test GPU")
    @patch('torch.cuda.memory_allocated', return_value=1024*1024*1024)  # 1GB
    @patch('torch.cuda.memory_reserved', return_value=2*1024*1024*1024)  # 2GB
    def test_track_gpu_metrics(self, *mocks):
        """Test GPU metrics tracking."""
        manager = TelemetryManager("test_service")
        
        manager.track_gpu_metrics()
        
        # Should have recorded GPU metrics
        gpu_metrics = [m for m in manager.local_exporter.metrics_buffer if "gpu" in m.name]
        assert len(gpu_metrics) >= 2  # At least allocated and reserved memory
        
        # Check one of the metrics
        allocated_metrics = [m for m in gpu_metrics if m.tags.get("type") == "allocated"]
        assert len(allocated_metrics) == 1
        assert allocated_metrics[0].value == 1024*1024*1024
    
    def test_track_audio_processing(self):
        """Test audio processing metrics tracking."""
        manager = TelemetryManager("test_service")
        
        manager.track_audio_processing(
            segments_count=10,
            processing_time=2.5,
            batch_size=32,
            model_name="test-model"
        )
        
        # Should have recorded multiple metrics
        assert len(manager.local_exporter.metrics_buffer) >= 4
        
        # Check specific metrics
        segment_metrics = [m for m in manager.local_exporter.metrics_buffer if m.name == "audio_segments_processed"]
        assert len(segment_metrics) == 1
        assert segment_metrics[0].value == 10
        
        duration_metrics = [m for m in manager.local_exporter.metrics_buffer if m.name == "audio_processing_duration"]
        assert len(duration_metrics) == 1
        assert duration_metrics[0].value == 2.5
    
    def test_get_metrics_summary(self):
        """Test metrics summary generation."""
        manager = TelemetryManager("test_service")
        
        # Add some metrics
        manager.record_counter("test_counter", 5)
        manager.record_gauge("test_gauge", 42.0)
        
        summary = manager.get_metrics_summary()
        
        assert "counters" in summary
        assert "cache_size" in summary
        assert "local_buffer_size" in summary
        assert "opentelemetry_enabled" in summary
        
        assert summary["local_buffer_size"] == 2


class TestTelemetryDecorator:
    """Test telemetry decorator."""
    
    def test_decorator_success(self):
        """Test decorator on successful function."""
        manager = TelemetryManager("test_service")
        
        @telemetry_decorator("test_operation")
        def test_function(x):
            return x * 2
        
        # Mock global manager
        with patch('utils.telemetry.get_telemetry_manager', return_value=manager):
            result = test_function(5)
        
        assert result == 10
        
        # Should have recorded timing and success metrics
        assert len(manager.local_exporter.metrics_buffer) >= 2
        
        # Check success counter
        success_metrics = [m for m in manager.local_exporter.metrics_buffer if m.name == "requests_total"]
        assert len(success_metrics) == 1
        assert success_metrics[0].tags["status"] == "success"
    
    def test_decorator_with_error(self):
        """Test decorator on function that raises error."""
        manager = TelemetryManager("test_service")
        
        @telemetry_decorator("test_operation")
        def failing_function():
            raise RuntimeError("Test error")
        
        with patch('utils.telemetry.get_telemetry_manager', return_value=manager):
            with pytest.raises(RuntimeError):
                failing_function()
        
        # Should have recorded error metric
        error_metrics = [m for m in manager.local_exporter.metrics_buffer if m.name == "errors_total"]
        assert len(error_metrics) == 1


class TestGlobalTelemetryManager:
    """Test global telemetry manager functions."""
    
    def test_get_telemetry_manager_singleton(self):
        """Test that get_telemetry_manager returns singleton."""
        manager1 = get_telemetry_manager()
        manager2 = get_telemetry_manager()
        
        assert manager1 is manager2
    
    def test_init_telemetry(self):
        """Test telemetry initialization."""
        manager = init_telemetry("custom_service")
        
        assert manager.service_name == "custom_service"
        
        # Should be same as global manager
        global_manager = get_telemetry_manager()
        assert manager is global_manager


if __name__ == '__main__':
    pytest.main([__file__, '-v'])