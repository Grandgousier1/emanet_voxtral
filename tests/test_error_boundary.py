#!/usr/bin/env python3
"""
Tests pour le système error boundary
"""

import pytest
from unittest.mock import Mock, MagicMock
import torch

from error_boundary import (
    error_boundary, with_error_boundary, create_oom_handler,
    ErrorBoundaryMiddleware
)
from domain_models import ErrorContext, ErrorSeverity
from cli_feedback import CLIFeedback

class TestErrorBoundary:
    """Tests pour error_boundary context manager."""
    
    def test_successful_operation(self):
        """Test opération qui réussit."""
        feedback = Mock(spec=CLIFeedback)
        context = ErrorContext(
            operation="test op",
            component="test component", 
            severity=ErrorSeverity.ERROR
        )
        
        with error_boundary(context, feedback):
            result = "success"
        
        # Aucun appel à feedback en cas de succès
        feedback.error.assert_not_called()
        feedback.warning.assert_not_called()
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_oom_handling(self):
        """Test gestion OOM avec handler."""
        feedback = Mock(spec=CLIFeedback)
        context = ErrorContext(
            operation="gpu operation",
            component="model",
            severity=ErrorSeverity.ERROR,
            recovery_strategy="reduce batch size"
        )
        
        oom_handler = Mock(return_value="recovery_result")
        
        with error_boundary(context, feedback, oom_handler=oom_handler):
            raise torch.cuda.OutOfMemoryError("fake OOM")
        
        feedback.error.assert_called_once()
        feedback.info.assert_called_with("🔄 Recovery: reduce batch size") 
        oom_handler.assert_called_once()
    
    def test_import_error_non_critical(self):
        """Test ImportError non critique avec fallback."""
        feedback = Mock(spec=CLIFeedback)
        context = ErrorContext(
            operation="import optional",
            component="optional_lib",
            severity=ErrorSeverity.WARNING,
            recovery_strategy="use fallback implementation"
        )
        
        with error_boundary(context, feedback, fallback_value="fallback"):
            raise ImportError("optional lib not found")
        
        feedback.warning.assert_called_once()
        feedback.info.assert_called_with("🔄 use fallback implementation")
    
    def test_import_error_critical(self):
        """Test ImportError critique qui propage."""
        feedback = Mock(spec=CLIFeedback)
        context = ErrorContext(
            operation="import critical",
            component="critical_lib",
            severity=ErrorSeverity.CRITICAL
        )
        
        with pytest.raises(ImportError):
            with error_boundary(context, feedback):
                raise ImportError("critical lib missing")
        
        feedback.critical.assert_called_once()

class TestErrorBoundaryDecorator:
    """Tests pour le décorateur error boundary."""
    
    def test_decorator_with_method(self):
        """Test décorateur sur méthode avec self.feedback."""
        
        class TestClass:
            def __init__(self):
                self.feedback = Mock(spec=CLIFeedback)
            
            @with_error_boundary("test operation", "test component")
            def risky_method(self):
                return "success"
        
        instance = TestClass()
        result = instance.risky_method()
        
        assert result == "success"
        instance.feedback.error.assert_not_called()
    
    def test_decorator_with_exception(self):
        """Test décorateur avec exception."""
        
        class TestClass:
            def __init__(self):
                self.feedback = Mock(spec=CLIFeedback)
            
            @with_error_boundary(
                "failing operation", 
                "test component",
                severity=ErrorSeverity.WARNING,
                fallback_value="fallback"
            )
            def failing_method(self):
                raise ValueError("test error")
        
        instance = TestClass()
        result = instance.failing_method()
        
        assert result == "fallback"
        instance.feedback.error.assert_called_once()

class TestOOMHandler:
    """Tests pour les handlers OOM."""
    
    def test_create_oom_handler(self):
        """Test création handler OOM."""
        handler = create_oom_handler(fallback_batch_size=4)
        
        context = ErrorContext(
            operation="batch processing",
            component="model",
            severity=ErrorSeverity.ERROR
        )
        
        # Mock torch.cuda.empty_cache
        with pytest.mock.patch('torch.cuda.empty_cache') as mock_empty_cache:
            result = handler(torch.cuda.OutOfMemoryError("OOM"), context)
        
        mock_empty_cache.assert_called_once()
        assert result["reduce_batch_size"] == 4
        assert result["retry"] is True

class TestErrorBoundaryMiddleware:
    """Tests pour le middleware error boundary."""
    
    def test_middleware_creation(self):
        """Test création du middleware."""
        feedback = Mock(spec=CLIFeedback)
        middleware = ErrorBoundaryMiddleware(feedback)
        
        assert middleware.feedback == feedback
        assert middleware.error_counts == {}
    
    def test_error_tracking(self):
        """Test tracking des erreurs."""
        feedback = Mock(spec=CLIFeedback)
        middleware = ErrorBoundaryMiddleware(feedback)
        
        context = ErrorContext(
            operation="test_op",
            component="test_component", 
            severity=ErrorSeverity.ERROR
        )
        
        # Track plusieurs erreurs
        for _ in range(6):
            middleware.track_error(context)
        
        # Doit déclencher warning après 5 erreurs
        feedback.warning.assert_called_once()
        assert "circuit breaker" in feedback.warning.call_args[0][0]
    
    def test_wrap_operation(self):
        """Test factory pour contextes."""
        feedback = Mock(spec=CLIFeedback)
        middleware = ErrorBoundaryMiddleware(feedback)
        
        context = middleware.wrap_operation("model_loading", "transformers")
        
        assert context.operation == "model_loading"
        assert context.component == "transformers"
        assert context.severity == ErrorSeverity.ERROR

# Tests d'intégration
class TestErrorBoundaryIntegration:
    """Tests d'intégration réels."""
    
    def test_real_model_loading_scenario(self):
        """Test scénario réaliste de chargement modèle."""
        feedback = Mock(spec=CLIFeedback)
        
        @with_error_boundary(
            "model loading",
            "transformers", 
            recovery_strategy="fallback to CPU",
            fallback_value=None
        )
        def load_model():
            # Simule tentative de chargement qui échoue
            raise ImportError("transformers not available")
        
        result = load_model()
        
        assert result is None
        feedback.warning.assert_called()
        feedback.info.assert_called_with("🔄 fallback to CPU")