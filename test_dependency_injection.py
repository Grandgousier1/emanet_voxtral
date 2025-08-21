#!/usr/bin/env python3
"""
test_dependency_injection.py - Test the dependency injection system
Simple test to verify that services can be created and injected properly
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Mock CLIFeedback for testing
class MockCLIFeedback:
    def debug(self, msg): pass
    def info(self, msg): pass
    def warning(self, msg): pass
    def error(self, msg): pass

def test_service_container():
    """Test basic service container functionality."""
    print("Testing service container...")
    
    try:
        from utils.service_container import get_container, reset_container
        
        # Reset container to clean state
        reset_container()
        container = get_container()
        
        # Test that services can be retrieved
        print("Creating services...")
        audio_cache = container.get_service('audio_cache')
        memory_manager = container.get_service('memory_manager')
        file_validator = container.get_service('file_validator')
        
        print(f"✅ Audio cache created: {type(audio_cache).__name__}")
        print(f"✅ Memory manager created: {type(memory_manager).__name__}")
        print(f"✅ File validator created: {type(file_validator).__name__}")
        
        # Test singleton behavior
        audio_cache2 = container.get_service('audio_cache')
        assert audio_cache is audio_cache2, "Services should be singleton by default"
        print("✅ Singleton behavior verified")
        
        # Test with feedback
        feedback = MockCLIFeedback()
        container.set_feedback(feedback)
        
        # Create new service with feedback
        container.clear_services()  # Clear to force recreation
        timeout_manager = container.get_service('timeout_manager')
        print(f"✅ Timeout manager with feedback: {type(timeout_manager).__name__}")
        
        print("✅ All service container tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Service container test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_backward_compatibility():
    """Test that legacy functions still work."""
    print("\nTesting backward compatibility...")
    
    try:
        # Import the legacy functions
        from utils.audio_cache import create_audio_cache
        from utils.memory_manager import create_memory_manager  
        from utils.validation_utils import create_file_validator
        
        # Test factory functions
        audio_cache = create_audio_cache()
        memory_manager = create_memory_manager()
        file_validator = create_file_validator()
        
        print(f"✅ Factory audio cache: {type(audio_cache).__name__}")
        print(f"✅ Factory memory manager: {type(memory_manager).__name__}")
        print(f"✅ Factory file validator: {type(file_validator).__name__}")
        
        print("✅ All backward compatibility tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Backward compatibility test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = True
    
    print("🧪 Testing Dependency Injection System")
    print("=" * 50)
    
    success &= test_service_container()
    success &= test_backward_compatibility()
    
    if success:
        print("\n🎉 All dependency injection tests passed!")
        print("✅ Services can now be mocked and injected for better testability")
    else:
        print("\n❌ Some tests failed!")
        exit(1)