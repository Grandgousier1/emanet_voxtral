#!/usr/bin/env python3
"""
utils/service_container.py - Dependency injection container
Replaces global singletons with configurable service container for better testability
"""

import threading
from typing import Dict, Any, TypeVar, Type, Optional, Callable
from dataclasses import dataclass

from cli_feedback import CLIFeedback

T = TypeVar('T')


@dataclass
class ServiceConfig:
    """Configuration for service instantiation."""
    singleton: bool = True
    factory: Optional[Callable[..., Any]] = None
    dependencies: Optional[Dict[str, str]] = None  # Map of param_name -> service_name


class ServiceContainer:
    """Dependency injection container for managing services and dependencies."""
    
    def __init__(self):
        self._services: Dict[str, Any] = {}
        self._config: Dict[str, ServiceConfig] = {}
        self._lock = threading.RLock()
        self._feedback: Optional[CLIFeedback] = None
    
    def set_feedback(self, feedback: CLIFeedback) -> None:
        """Set the feedback instance used by services."""
        self._feedback = feedback
    
    def register_service(self, 
                        name: str, 
                        service_class: Type[T], 
                        singleton: bool = True,
                        dependencies: Optional[Dict[str, str]] = None) -> None:
        """Register a service class with the container."""
        with self._lock:
            def factory(**kwargs) -> T:
                # Inject feedback if the service accepts it
                if hasattr(service_class, '__init__'):
                    import inspect
                    sig = inspect.signature(service_class.__init__)
                    if 'feedback' in sig.parameters and self._feedback:
                        kwargs['feedback'] = self._feedback
                
                return service_class(**kwargs)
            
            self._config[name] = ServiceConfig(
                singleton=singleton,
                factory=factory,
                dependencies=dependencies if dependencies is not None else {}
            )
    
    def register_factory(self, 
                        name: str, 
                        factory: Callable[..., T], 
                        singleton: bool = True,
                        dependencies: Optional[Dict[str, str]] = None) -> None:
        """Register a factory function with the container."""
        with self._lock:
            self._config[name] = ServiceConfig(
                singleton=singleton,
                factory=factory,
                dependencies=dependencies if dependencies is not None else {}
            )
    
    def register_instance(self, name: str, instance: T) -> None:
        """Register a pre-created instance with the container."""
        with self._lock:
            self._services[name] = instance
            self._config[name] = ServiceConfig(singleton=True, factory=None)
    
    def get_service(self, name: str) -> Any:
        """Get a service instance by name."""
        with self._lock:
            # Return existing singleton if available
            if name in self._services:
                return self._services[name]
            
            # Check if service is registered
            if name not in self._config:
                raise ValueError(f"Service '{name}' not registered")
            
            config = self._config[name]
            
            if config.factory is None:
                raise ValueError(f"Service '{name}' has no factory method")
            
            # Resolve dependencies
            kwargs = {}
            for param_name, service_name in config.dependencies.items():
                kwargs[param_name] = self.get_service(service_name)
            
            # Create the service instance
            instance = config.factory(**kwargs)
            
            # Store if singleton
            if config.singleton:
                self._services[name] = instance
            
            return instance
    
    def clear_services(self) -> None:
        """Clear all service instances (useful for testing)."""
        with self._lock:
            self._services.clear()
    
    def has_service(self, name: str) -> bool:
        """Check if a service is registered."""
        with self._lock:
            return name in self._config
    
    def reset(self) -> None:
        """Reset the container completely (useful for testing)."""
        with self._lock:
            self._services.clear()
            self._config.clear()
            self._feedback = None


# Global container instance (this is the only singleton we keep)
_global_container: Optional[ServiceContainer] = None
_container_lock = threading.Lock()


def get_container() -> ServiceContainer:
    """Get the global service container."""
    global _global_container
    with _container_lock:
        if _global_container is None:
            _global_container = ServiceContainer()
            _setup_default_services(_global_container)
        return _global_container


def _setup_default_services(container: ServiceContainer) -> None:
    """Setup default service registrations."""
    from utils.audio_cache import create_audio_cache
    from utils.memory_manager import create_memory_manager
    from utils.validation_utils import create_file_validator, create_timeout_manager, create_rate_limiter
    
    # Register core services with factory functions
    container.register_factory('audio_cache', create_audio_cache)
    container.register_factory('memory_manager', create_memory_manager)
    container.register_factory('file_validator', create_file_validator)
    container.register_factory('timeout_manager', create_timeout_manager)
    container.register_factory('rate_limiter', create_rate_limiter)


def set_container_feedback(feedback: CLIFeedback) -> None:
    """Set feedback for the global container."""
    container = get_container()
    container.set_feedback(feedback)


def reset_container() -> None:
    """Reset the global container (useful for testing)."""
    global _global_container
    with _container_lock:
        if _global_container:
            _global_container.reset()


# Convenience functions for backward compatibility
def get_audio_cache():
    """Get audio cache service."""
    return get_container().get_service('audio_cache')


def get_memory_manager(feedback: Optional[CLIFeedback] = None):
    """Get memory manager service."""
    container = get_container()
    if feedback:
        container.set_feedback(feedback)
    return container.get_service('memory_manager')


def get_file_validator(feedback: Optional[CLIFeedback] = None):
    """Get file validator service."""
    container = get_container()
    if feedback:
        container.set_feedback(feedback)
    return container.get_service('file_validator')


def get_timeout_manager(feedback: Optional[CLIFeedback] = None):
    """Get timeout manager service."""
    container = get_container()
    if feedback:
        container.set_feedback(feedback)
    return container.get_service('timeout_manager')


def get_rate_limiter(feedback: Optional[CLIFeedback] = None):
    """Get rate limiter service."""
    container = get_container()
    if feedback:
        container.set_feedback(feedback)
    return container.get_service('rate_limiter')


# Testing utilities
class MockServiceContainer:
    """Mock service container for testing."""
    
    def __init__(self):
        self._mocks: Dict[str, Any] = {}
    
    def register_mock(self, name: str, mock_instance: Any) -> None:
        """Register a mock service."""
        self._mocks[name] = mock_instance
    
    def get_service(self, name: str) -> Any:
        """Get a mock service."""
        if name not in self._mocks:
            raise ValueError(f"Mock service '{name}' not registered")
        return self._mocks[name]
    
    def clear_services(self) -> None:
        """Clear all mock services."""
        self._mocks.clear()


def create_test_container() -> MockServiceContainer:
    """Create a mock container for testing."""
    return MockServiceContainer()


def with_test_container(test_container: MockServiceContainer):
    """Context manager to temporarily replace the global container with a test container."""
    from contextlib import contextmanager
    
    @contextmanager
    def context():
        global _global_container
        original_container = _global_container
        try:
            with _container_lock:
                _global_container = test_container
            yield test_container
        finally:
            with _container_lock:
                _global_container = original_container
    
    return context()