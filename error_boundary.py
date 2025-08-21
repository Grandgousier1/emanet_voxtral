#!/usr/bin/env python3
"""
Error boundary pattern for unified error handling
"""

import functools
import traceback
from contextlib import contextmanager
from typing import Callable, Optional, Any, Type, TypeVar
import logging

import torch
from domain_models import ErrorContext, ErrorSeverity
from cli_feedback import CLIFeedback

logger = logging.getLogger(__name__)
F = TypeVar('F', bound=Callable[..., Any])

@contextmanager 
def error_boundary(
    context: ErrorContext,
    feedback: CLIFeedback,
    oom_handler: Optional[Callable] = None,
    fallback_value: Any = None
):
    """
    Gestionnaire d'erreurs contextualisé avec recovery automatique.
    
    Usage:
        with error_boundary(context, feedback) as boundary:
            result = risky_operation()
    """
    try:
        yield
    except torch.cuda.OutOfMemoryError as e:
        feedback.error(f"🔥 GPU OOM during {context.operation}")
        logger.error(f"OOM in {context.component}: {e}")
        
        if oom_handler:
            try:
                return oom_handler(e, context)
            except Exception as recovery_error:
                feedback.critical(f"OOM recovery failed: {recovery_error}")
        
        if context.recovery_strategy:
            feedback.info(f"🔄 Recovery: {context.recovery_strategy}")
            
        raise
        
    except ImportError as e:
        if context.severity == ErrorSeverity.CRITICAL:
            feedback.critical(f"❌ Critical import failed: {context.component}")
            raise
        else:
            feedback.warning(f"⚠️ Optional import failed: {context.component}")
            if context.recovery_strategy:
                feedback.info(f"🔄 {context.recovery_strategy}")
            return fallback_value
            
    except (FileNotFoundError, PermissionError) as e:
        feedback.error(f"📁 File access error in {context.operation}: {e}")
        if context.recovery_strategy:
            feedback.info(f"🔄 {context.recovery_strategy}")
        return fallback_value
        
    except Exception as e:
        feedback.error(f"💥 Unexpected error in {context.operation}")
        logger.error(f"Full traceback: {traceback.format_exc()}")
        
        if context.severity == ErrorSeverity.CRITICAL:
            raise
        else:
            return fallback_value

def with_error_boundary(
    operation: str, 
    component: str,
    severity: ErrorSeverity = ErrorSeverity.ERROR,
    recovery_strategy: Optional[str] = None,
    fallback_value: Any = None
):
    """
    Décorateur pour appliquer error boundary à une fonction.
    
    Usage:
        @with_error_boundary("model loading", "transformers")
        def load_model():
            return transformers.load(...)
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Extraire feedback du premier argument si c'est une méthode
            feedback = None
            if args and hasattr(args[0], 'feedback'):
                feedback = args[0].feedback
            elif 'feedback' in kwargs:
                feedback = kwargs['feedback']
            else:
                # Fallback vers un feedback minimal
                from cli_feedback import get_feedback
                feedback = get_feedback()
            
            context = ErrorContext(
                operation=operation,
                component=component, 
                severity=severity,
                recovery_strategy=recovery_strategy
            )
            
            with error_boundary(context, feedback, fallback_value=fallback_value):
                return func(*args, **kwargs)
                
        return wrapper
    return decorator

# Handlers spécialisés
def create_oom_handler(fallback_batch_size: int = 1):
    """Crée un handler OOM avec réduction de batch size."""
    def handle_oom(error: torch.cuda.OutOfMemoryError, context: ErrorContext):
        logger.warning(f"OOM recovery: reducing batch size to {fallback_batch_size}")
        torch.cuda.empty_cache()
        # Retourner des instructions de recovery
        return {"reduce_batch_size": fallback_batch_size, "retry": True}
    return handle_oom

class ErrorBoundaryMiddleware:
    """Middleware pour intégration dans pipelines de traitement."""
    
    def __init__(self, feedback: CLIFeedback):
        self.feedback = feedback
        self.error_counts = {}
    
    def wrap_operation(self, operation: str, component: str):
        """Factory pour créer des contextes d'erreur."""
        return ErrorContext(
            operation=operation,
            component=component,
            severity=ErrorSeverity.ERROR
        )
    
    def track_error(self, context: ErrorContext):
        """Track error frequency pour circuit breaker pattern."""
        key = f"{context.component}:{context.operation}"
        self.error_counts[key] = self.error_counts.get(key, 0) + 1
        
        if self.error_counts[key] > 5:
            self.feedback.warning(f"🔥 High error rate in {key} - consider circuit breaker")