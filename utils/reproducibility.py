#!/usr/bin/env python3
"""
utils/reproducibility.py - Scientific reproducibility utilities
Ensures deterministic behavior across all ML operations
"""

import os
import random
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

def set_global_seed(seed: int = 42) -> None:
    """
    Set global seed for reproducible ML experiments.
    
    Args:
        seed: Random seed value (default: 42)
    """
    logger.info(f"Setting global seed: {seed}")
    
    # Python random module
    random.seed(seed)
    
    # NumPy
    try:
        import numpy as np
        np.random.seed(seed)
        logger.debug("NumPy seed set")
    except ImportError:
        pass
    
    # PyTorch
    try:
        import torch
        torch.manual_seed(seed)
        
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)  # For multi-GPU
            logger.debug("PyTorch CUDA seeds set")
            
        # Additional PyTorch deterministic settings for B200
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False  # Can hurt performance but ensures reproducibility
        logger.debug("PyTorch deterministic mode enabled")
        
    except ImportError:
        pass
    
    # Set environment variables for additional reproducibility
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    logger.info("Global reproducibility seed configuration completed")

def set_model_deterministic(model) -> None:
    """
    Configure model for deterministic inference.
    
    Args:
        model: PyTorch model to configure
    """
    if hasattr(model, 'eval'):
        model.eval()
        logger.debug("Model set to evaluation mode")
    
    # Disable dropout and batch norm training mode
    for module in model.modules():
        if hasattr(module, 'training'):
            module.training = False
            
    logger.debug("Model configured for deterministic inference")

def get_reproducibility_config(seed: int = 42) -> Dict[str, Any]:
    """
    Get configuration for reproducible generation parameters.
    
    Args:
        seed: Random seed
        
    Returns:
        Reproducible generation configuration
    """
    config = {
        'seed': seed,
        'deterministic': True,
        'generation_params': {
            'do_sample': False,  # Deterministic decoding
            'num_beams': 1,      # No beam search for reproducibility
            'temperature': 1.0,  # Temperature = 1.0 with do_sample=False for deterministic
            'top_k': None,       # Disable top-k sampling
            'top_p': None,       # Disable nucleus sampling
            'repetition_penalty': 1.0,  # No repetition penalty for deterministic
            'use_cache': True,   # Enable KV cache for consistency
        }
    }
    
    logger.debug(f"Generated reproducible config: {config}")
    return config

def validate_reproducibility_state() -> Dict[str, bool]:
    """
    Validate current reproducibility state.
    
    Returns:
        Dictionary with validation results
    """
    state = {}
    
    # Check PyTorch settings
    try:
        import torch
        state['torch_deterministic'] = torch.backends.cudnn.deterministic
        state['torch_benchmark_disabled'] = not torch.backends.cudnn.benchmark
        state['torch_available'] = True
    except ImportError:
        state['torch_available'] = False
    
    # Check environment variables
    state['python_hash_seed_set'] = 'PYTHONHASHSEED' in os.environ
    
    # Check if seeds were set
    try:
        import numpy as np
        # Test if numpy seed is working
        np.random.seed(42)
        test1 = np.random.random()
        np.random.seed(42) 
        test2 = np.random.random()
        state['numpy_deterministic'] = (test1 == test2)
    except ImportError:
        state['numpy_deterministic'] = None
    
    # Overall reproducibility score
    checks = [v for v in state.values() if isinstance(v, bool)]
    state['reproducibility_score'] = sum(checks) / len(checks) if checks else 0.0
    
    logger.info(f"Reproducibility validation: {state}")
    return state

class ReproducibleSession:
    """Context manager for reproducible ML sessions."""
    
    def __init__(self, seed: int = 42, restore_on_exit: bool = True):
        """
        Initialize reproducible session.
        
        Args:
            seed: Random seed to use
            restore_on_exit: Whether to restore original settings on exit
        """
        self.seed = seed
        self.restore_on_exit = restore_on_exit
        self.original_state = {}
        
    def __enter__(self):
        """Enter reproducible session."""
        if self.restore_on_exit:
            # Save original state
            try:
                import torch
                self.original_state['cudnn_deterministic'] = torch.backends.cudnn.deterministic
                self.original_state['cudnn_benchmark'] = torch.backends.cudnn.benchmark
            except ImportError:
                pass
                
        # Set reproducible state
        set_global_seed(self.seed)
        logger.info(f"Entered reproducible session with seed {self.seed}")
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit reproducible session."""
        if self.restore_on_exit and self.original_state:
            # Restore original settings
            try:
                import torch
                torch.backends.cudnn.deterministic = self.original_state.get('cudnn_deterministic', False)
                torch.backends.cudnn.benchmark = self.original_state.get('cudnn_benchmark', True)
                logger.debug("Restored original PyTorch settings")
            except ImportError:
                pass
                
        logger.info("Exited reproducible session")

def ensure_reproducible_environment(seed: int = 42, validate: bool = True) -> Dict[str, Any]:
    """
    Ensure complete reproducible environment setup.
    
    Args:
        seed: Random seed to set
        validate: Whether to validate setup
        
    Returns:
        Setup report and validation results
    """
    report = {
        'seed': seed,
        'setup_successful': False,
        'validation_results': None
    }
    
    try:
        # Set global seed
        set_global_seed(seed)
        
        # Validate if requested
        if validate:
            validation = validate_reproducibility_state()
            report['validation_results'] = validation
            report['setup_successful'] = validation['reproducibility_score'] > 0.8
        else:
            report['setup_successful'] = True
            
        logger.info(f"Reproducible environment setup: {'✓' if report['setup_successful'] else '✗'}")
        
    except Exception as e:
        logger.error(f"Failed to setup reproducible environment: {e}")
        report['error'] = str(e)
        
    return report