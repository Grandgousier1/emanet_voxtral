#!/usr/bin/env bash
set -e

echo "[setup] RunPod B200 Optimized Setup"
echo "Hardware: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits)"

# Install system packages optimized for RunPod
echo "[setup] Installing system packages"
apt-get update -y
apt-get install -y ffmpeg git build-essential wget htop nvtop

# RunPod environment setup (system-wide installation for RunPod)
echo "[setup] Using system Python environment (RunPod standard)"

# Use faster dependency installation
pip install --upgrade pip
pip install uv  # Faster dependency resolver

# Set compilation flags for vLLM on B200
export VLLM_BUILD_WITH_CUDA=1
export NVCC_PREPEND_FLAGS='-ccbin /usr/bin/gcc'

# Install dependencies (RunPod already has PyTorch 2.8.0)
echo "[setup] Installing Python dependencies (PyTorch 2.8.0 from base image)"
uv pip install -r requirements.txt --system

# Check vLLM installation
python -c "import vllm; print(f'[setup] vLLM {vllm.__version__} installed')" || echo "[setup] Warning: vLLM not available"
python -c "import mistral_common; print(f'[setup] mistral-common {mistral_common.__version__} installed')" || echo "[setup] Warning: mistral-common not available"

# Prefetch Voxtral models to HuggingFace cache
echo "[setup] Pre-downloading Voxtral models..."

# Check for HuggingFace token
if [ -z "$HF_TOKEN" ] && [ -z "$HUGGINGFACE_HUB_TOKEN" ]; then
    echo "[setup] Warning: No HF_TOKEN found. Mistral models may not be accessible."
    echo "[setup] Set HF_TOKEN environment variable if needed:"
    echo "[setup]   export HF_TOKEN=your_huggingface_token"
fi

python - <<'PY'
from transformers import AutoProcessor
import os

models_to_cache = [
    'mistralai/Voxtral-Small-24B-2507',
    'mistralai/Voxtral-Mini-3B-2507'
]

# Check for HuggingFace token
hf_token = os.getenv('HF_TOKEN') or os.getenv('HUGGINGFACE_HUB_TOKEN')
auth_kwargs = {}
if hf_token:
    auth_kwargs['token'] = hf_token
    print('[setup] Using HuggingFace authentication token')

for model_name in models_to_cache:
    print(f'[setup] Caching {model_name}...')
    try:
        # Just download the processor/tokenizer (lighter than full model)
        processor = AutoProcessor.from_pretrained(model_name, **auth_kwargs)
        print(f'[setup] ✓ {model_name} processor cached')
    except Exception as e:
        print(f'[setup] ✗ Error caching {model_name}: {e}')
        if "401" in str(e) or "authentication" in str(e).lower():
            print(f'[setup] → Authentication error. Set HF_TOKEN environment variable.')

print('[setup] Model caching completed')
PY

# Set up optimal environment for B200
echo "[setup] Configuring B200 environment optimizations"
echo 'export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512,expandable_segments:True' >> ~/.bashrc
echo 'export CUDA_LAUNCH_BLOCKING=0' >> ~/.bashrc
echo 'export TORCH_CUDA_ARCH_LIST=9.0' >> ~/.bashrc
echo 'export HF_HOME=/tmp/hf_cache' >> ~/.bashrc
echo 'export VLLM_ATTENTION_BACKEND=FLASHINFER' >> ~/.bashrc

# Create optimized cache directories
mkdir -p /tmp/hf_cache
mkdir -p /tmp/pytorch_kernel_cache

# Test hardware detection
echo "[setup] Testing hardware detection..."
python3 -c "
from config import detect_hardware, get_optimal_config
hw = detect_hardware()
config = get_optimal_config()
print(f'Detected: {hw[\"gpu_count\"]} GPU(s), B200: {hw[\"is_b200\"]}')
print(f'Optimized config: batch_size={config[\"audio\"][\"batch_size\"]}, workers={config[\"audio\"][\"parallel_workers\"]}')
"

# Final validation test
echo "[setup] Final validation test..."
python3 -c "
import torch
import sys

try:
    # Test CUDA availability and B200 compatibility
    assert torch.cuda.is_available(), 'CUDA not available'
    assert torch.cuda.device_count() >= 1, 'No CUDA devices found'
    
    # Test bfloat16 support for B200
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        capability = torch.cuda.get_device_capability(0)
        if capability[0] >= 8:
            print('[setup] ✓ bfloat16 support confirmed (Ampere+ GPU)')
        else:
            print('[setup] ⚠ Pre-Ampere GPU detected, will use float16')
    else:
        print('[setup] ⚠ No GPU available, CPU mode only')
    
    # Test basic imports
    import transformers
    import soundfile
    print('[setup] ✓ Core dependencies validated')
    
    print('[setup] ✅ All validation tests passed')
except Exception as e:
    print(f'[setup] ✗ Validation failed: {e}')
    sys.exit(1)
"

echo "[setup] ✅ RunPod B200 setup complete!"
echo "[setup] 🚀 Ready for high-performance Voxtral processing"
