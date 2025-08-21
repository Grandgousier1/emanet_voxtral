#!/usr/bin/env python3
"""
utils/tensor_validation.py - Tensor validation utilities for B200 optimization
"""

import torch
from typing import Optional, Tuple, Union, List
import logging

logger = logging.getLogger(__name__)

def validate_tensor_shape(
    tensor: torch.Tensor, 
    expected_shape: Optional[Tuple[int, ...]] = None,
    min_dims: Optional[int] = None,
    max_dims: Optional[int] = None,
    name: str = "tensor"
) -> bool:
    """
    Validate tensor shape with detailed error reporting.
    
    Args:
        tensor: Input tensor to validate
        expected_shape: Expected exact shape (None means any size for that dim)
        min_dims: Minimum number of dimensions
        max_dims: Maximum number of dimensions  
        name: Tensor name for error reporting
        
    Returns:
        True if valid, raises ValueError if not
    """
    if not isinstance(tensor, torch.Tensor):
        raise ValueError(f"{name} must be a torch.Tensor, got {type(tensor)}")
    
    shape = tensor.shape
    
    # Check dimensions count
    if min_dims is not None and len(shape) < min_dims:
        raise ValueError(f"{name} must have at least {min_dims} dimensions, got {len(shape)}")
    
    if max_dims is not None and len(shape) > max_dims:
        raise ValueError(f"{name} must have at most {max_dims} dimensions, got {len(shape)}")
    
    # Check exact shape if provided
    if expected_shape is not None:
        if len(shape) != len(expected_shape):
            raise ValueError(f"{name} shape {shape} doesn't match expected dims {expected_shape}")
        
        for i, (actual, expected) in enumerate(zip(shape, expected_shape)):
            if expected is not None and actual != expected:
                raise ValueError(f"{name} dimension {i}: expected {expected}, got {actual}")
    
    return True

def validate_tensor_dtype(
    tensor: torch.Tensor,
    expected_dtypes: Union[torch.dtype, List[torch.dtype]],
    name: str = "tensor"
) -> bool:
    """
    Validate tensor dtype for B200 optimization.
    
    Args:
        tensor: Input tensor
        expected_dtypes: Expected dtype(s)
        name: Tensor name for error reporting
    """
    if not isinstance(expected_dtypes, list):
        expected_dtypes = [expected_dtypes]
    
    if tensor.dtype not in expected_dtypes:
        expected_str = ", ".join(str(dt) for dt in expected_dtypes)
        raise ValueError(f"{name} dtype {tensor.dtype} not in expected {expected_str}")
    
    return True

def validate_tensor_device(
    tensor: torch.Tensor,
    expected_device: Optional[Union[str, torch.device]] = None,
    require_cuda: bool = True,
    name: str = "tensor"
) -> bool:
    """
    Validate tensor device for B200 processing.
    
    Args:
        tensor: Input tensor
        expected_device: Expected device (None = any CUDA device)
        require_cuda: Whether CUDA is required
        name: Tensor name for error reporting
    """
    device = tensor.device
    
    if require_cuda and device.type != 'cuda':
        raise ValueError(f"{name} must be on CUDA device for B200 processing, got {device}")
    
    if expected_device is not None:
        expected_device = torch.device(expected_device)
        if device != expected_device:
            raise ValueError(f"{name} on device {device}, expected {expected_device}")
    
    return True

def validate_audio_tensor(
    audio_tensor: torch.Tensor,
    sample_rate: int = 16000,
    max_duration_sec: float = 30.0,
    name: str = "audio"
) -> bool:
    """
    Validate audio tensor for Voxtral processing.
    
    Args:
        audio_tensor: Audio tensor [channels, samples] or [samples]
        sample_rate: Expected sample rate
        max_duration_sec: Maximum duration in seconds
        name: Tensor name for error reporting
    """
    # Shape validation
    if audio_tensor.dim() == 1:
        # Mono audio [samples]
        samples = audio_tensor.shape[0]
        channels = 1
    elif audio_tensor.dim() == 2:
        # Multi-channel [channels, samples] or [samples, channels]
        if audio_tensor.shape[0] <= 2:  # Assume [channels, samples]
            channels, samples = audio_tensor.shape
        else:  # Assume [samples, channels]
            samples, channels = audio_tensor.shape
    else:
        raise ValueError(f"{name} must be 1D or 2D tensor, got {audio_tensor.dim()}D")
    
    # Duration validation
    duration_sec = samples / sample_rate
    if duration_sec > max_duration_sec:
        raise ValueError(f"{name} duration {duration_sec:.2f}s exceeds maximum {max_duration_sec}s")
    
    if samples == 0:
        raise ValueError(f"{name} tensor is empty")
    
    # Dtype validation for B200
    if audio_tensor.dtype not in [torch.float32, torch.float16, torch.bfloat16]:
        raise ValueError(f"{name} dtype {audio_tensor.dtype} not suitable for B200 audio processing")
    
    logger.debug(f"Validated {name}: {channels} channels, {samples} samples, {duration_sec:.2f}s")
    return True

def check_tensor_health(
    tensor: torch.Tensor,
    name: str = "tensor",
    check_nan: bool = True,
    check_inf: bool = True,
    check_range: Optional[Tuple[float, float]] = None
) -> bool:
    """
    Check tensor for numerical health (NaN, Inf, range).
    
    Args:
        tensor: Input tensor
        name: Tensor name for error reporting
        check_nan: Check for NaN values
        check_inf: Check for Inf values  
        check_range: Expected value range (min, max)
    """
    if check_nan and torch.isnan(tensor).any():
        raise ValueError(f"{name} contains NaN values")
    
    if check_inf and torch.isinf(tensor).any():
        raise ValueError(f"{name} contains Inf values")
    
    if check_range is not None:
        min_val, max_val = check_range
        tensor_min = tensor.min().item()
        tensor_max = tensor.max().item()
        
        if tensor_min < min_val or tensor_max > max_val:
            raise ValueError(f"{name} values [{tensor_min:.3f}, {tensor_max:.3f}] outside expected range [{min_val}, {max_val}]")
    
    return True

def validate_batch_consistency(
    tensors: List[torch.Tensor],
    name: str = "batch"
) -> bool:
    """
    Validate that tensors in a batch have consistent properties.
    
    Args:
        tensors: List of tensors in batch
        name: Batch name for error reporting
    """
    if not tensors:
        raise ValueError(f"{name} is empty")
    
    first_tensor = tensors[0]
    reference_device = first_tensor.device
    reference_dtype = first_tensor.dtype
    reference_ndim = first_tensor.dim()
    
    for i, tensor in enumerate(tensors[1:], 1):
        if tensor.device != reference_device:
            raise ValueError(f"{name}[{i}] device {tensor.device} != reference {reference_device}")
        
        if tensor.dtype != reference_dtype:
            raise ValueError(f"{name}[{i}] dtype {tensor.dtype} != reference {reference_dtype}")
        
        if tensor.dim() != reference_ndim:
            raise ValueError(f"{name}[{i}] dimensions {tensor.dim()} != reference {reference_ndim}")
    
    return True