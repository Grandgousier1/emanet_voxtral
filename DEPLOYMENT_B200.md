# 🚀 GUIDE DÉPLOIEMENT B200 - VOXTRAL

## Vue d'ensemble déploiement

Ce guide couvre le déploiement optimisé du système Voxtral sur hardware NVIDIA B200, incluant la configuration RunPod et les optimisations spécifiques.

## 🏗️ Architecture cible

### Hardware B200 (RunPod)
```
NVIDIA B200 GPU
├── 180 GB VRAM (HBM3)
├── Compute Capability 9.0
├── bfloat16 Tensor Cores
└── PCIe 5.0

Host System
├── 28 vCPU (Intel/AMD)
├── 188 GB RAM DDR5
├── 80 GB NVMe SSD
└── Ubuntu 22.04 LTS
```

### Container environnement
```
Base Image: runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04
├── PyTorch 2.8.0
├── CUDA 12.8.1
├── cuDNN optimisé
└── Python 3.11
```

## 📋 Prérequis déploiement

### 1. Vérification hardware
```bash
# Vérification GPU B200
nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv
# Attendu: B200, 9.0, 196608 MiB

# Vérification CUDA
nvcc --version
# Attendu: 12.8.1

# Vérification PyTorch
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
# Attendu: PyTorch: 2.8.0, CUDA: True
```

### 2. Configuration système
```bash
# Optimisations système pour B200
echo 'performance' | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

# Configuration NVIDIA persistence
sudo nvidia-persistenced --persistence-mode

# Optimisations mémoire
echo 'vm.swappiness=10' | sudo tee -a /etc/sysctl.conf
echo 'vm.vfs_cache_pressure=50' | sudo tee -a /etc/sysctl.conf
sudo sysctl -p
```

## 🔧 Installation et configuration

### 1. Clone et setup initial
```bash
# Clone repository
git clone https://github.com/voxtral/voxtral-b200.git
cd voxtral-b200

# Configuration environnement
make setup-runpod
```

### 2. Installation optimisée B200
```bash
# Installation base avec optimisations B200
make install-dev

# Installation vLLM pour performance maximale
make install-vllm

# Validation configuration
make validate-b200
```

### 3. Configuration spécifique B200
```bash
# Variables environnement optimisées
export CUDA_VISIBLE_DEVICES=0
export TORCH_CUDNN_V8_API_ENABLED=1
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1

# Optimisations mémoire GPU
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512,expandable_segments:True"

# Configuration B200 optimizer
export B200_OPTIMIZER_MODE="max-autotune"
export B200_ENABLE_FUSED_KERNELS=1
```

## ⚡ Optimisations B200

### 1. Configuration modèle optimale
```python
# config_b200.py
B200_CONFIG = {
    # Optimisations calcul
    "dtype": "bfloat16",           # Optimal pour B200 Tensor Cores
    "compile_mode": "max-autotune", # torch.compile agressif
    "use_fused_kernels": True,     # Kernel fusion activée
    
    # Optimisations mémoire
    "batch_size": 128,             # 4x plus grand que GPU standard
    "max_memory_gb": 160,          # 88% des 180GB disponibles
    "memory_format": "channels_last",
    
    # Optimisations parallélisme
    "num_workers": 28,             # Utilise tous les vCPU
    "async_processing": True,      # Overlap CPU/GPU
    "semaphore_limit": 8,          # Contrôle concurrence GPU
}
```

### 2. Profil performance B200
```bash
# Benchmark initial
make benchmark-b200

# Résultats attendus B200
# - Throughput: >15x real-time
# - Latency: <1.5s par segment
# - Memory efficiency: 70-80% VRAM
# - GPU utilization: >90%
```

### 3. Monitoring continu
```bash
# Monitoring B200 temps réel
make monitor-b200

# Métriques clés
watch -n 1 "nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits"
```

## 🐳 Déploiement containerisé

### 1. Dockerfile B200 optimisé
```dockerfile
# Dockerfile.b200
FROM runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04

# Optimisations système
RUN apt-get update && apt-get install -y \
    git \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Variables environnement B200
ENV CUDA_VISIBLE_DEVICES=0
ENV TORCH_CUDNN_V8_API_ENABLED=1
ENV PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512,expandable_segments:True"
ENV B200_OPTIMIZER_MODE="max-autotune"

# Installation application
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Installation vLLM optimisé B200
RUN pip install vllm --no-cache-dir

COPY . .

# Configuration optimale B200
RUN chmod +x setup_runpod.sh && ./setup_runpod.sh

# Validation installation
RUN python -c "from utils.b200_optimizer import get_b200_optimizer; print('B200 OK')"

# Point d'entrée
CMD ["python", "main.py"]
```

### 2. Build et déploiement
```bash
# Build image optimisée
docker build -f Dockerfile.b200 -t voxtral-b200:latest .

# Déploiement avec optimisations GPU
docker run --gpus all \
  --shm-size=16g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -v /data:/app/data \
  -p 8080:8080 \
  voxtral-b200:latest
```

## 🚀 Déploiement RunPod

### 1. Configuration Pod B200
```yaml
# runpod-config.yaml
apiVersion: v1
kind: Pod
metadata:
  name: voxtral-b200
spec:
  image: voxtral-b200:latest
  gpu:
    type: "B200"
    count: 1
  resources:
    cpu: "28"
    memory: "188Gi"
    storage: "80Gi"
  environment:
    - name: CUDA_VISIBLE_DEVICES
      value: "0"
    - name: B200_OPTIMIZER_MODE
      value: "max-autotune"
  ports:
    - containerPort: 8080
      hostPort: 8080
```

### 2. Script déploiement automatisé
```bash
#!/bin/bash
# deploy_runpod.sh

set -e

echo "🚀 Déploiement Voxtral B200 sur RunPod"

# 1. Vérification prérequis
echo "📋 Vérification hardware..."
make validate-setup

# 2. Installation optimisée
echo "📦 Installation dependencies..."
make setup-runpod

# 3. Configuration B200
echo "⚡ Configuration B200..."
source setup_runpod.sh

# 4. Tests validation
echo "🧪 Tests validation..."
make test-b200

# 5. Benchmark performance
echo "📊 Benchmark performance..."
make benchmark-b200

# 6. Démarrage service
echo "🎯 Démarrage service..."
python main.py --daemon --b200-mode

echo "✅ Déploiement terminé!"
echo "📊 Monitoring: make monitor-b200"
echo "🔧 Logs: tail -f /var/log/voxtral.log"
```

## 📊 Monitoring et observabilité

### 1. Métriques B200
```python
# monitoring_b200.py
import psutil
import torch
from prometheus_client import Gauge, start_http_server

# Métriques Prometheus
gpu_utilization = Gauge('b200_gpu_utilization_percent', 'B200 GPU utilization')
gpu_memory_used = Gauge('b200_gpu_memory_used_gb', 'B200 GPU memory used')
gpu_memory_total = Gauge('b200_gpu_memory_total_gb', 'B200 GPU memory total')
gpu_temperature = Gauge('b200_gpu_temperature_celsius', 'B200 GPU temperature')

throughput_segments_per_sec = Gauge('voxtral_throughput_segments_per_sec', 'Processing throughput')
latency_per_segment = Gauge('voxtral_latency_per_segment_ms', 'Processing latency')
```

### 2. Dashboard Grafana
```json
{
  "dashboard": {
    "title": "Voxtral B200 Performance",
    "panels": [
      {
        "title": "GPU Utilization",
        "targets": [{"expr": "b200_gpu_utilization_percent"}],
        "thresholds": [{"value": 80, "color": "green"}]
      },
      {
        "title": "Memory Usage", 
        "targets": [{"expr": "b200_gpu_memory_used_gb / b200_gpu_memory_total_gb * 100"}],
        "thresholds": [{"value": 90, "color": "red"}]
      },
      {
        "title": "Throughput",
        "targets": [{"expr": "voxtral_throughput_segments_per_sec"}],
        "unit": "segments/sec"
      }
    ]
  }
}
```

### 3. Alertes automatiques
```yaml
# alerts.yml
groups:
  - name: voxtral-b200
    rules:
      - alert: B200GPUHighMemory
        expr: b200_gpu_memory_used_gb / b200_gpu_memory_total_gb > 0.95
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "B200 GPU memory usage critical"
          
      - alert: B200LowThroughput
        expr: voxtral_throughput_segments_per_sec < 5
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "Processing throughput below expected"
          
      - alert: B200HighTemperature
        expr: b200_gpu_temperature_celsius > 85
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "B200 GPU temperature critical"
```

## 🔧 Maintenance et troubleshooting

### 1. Problèmes courants

#### **OOM (Out of Memory)**
```bash
# Diagnostic
nvidia-smi
make monitor-b200

# Solutions
# 1. Réduire batch size
export B200_BATCH_SIZE=64  # au lieu de 128

# 2. Nettoyer cache GPU
python -c "import torch; torch.cuda.empty_cache()"

# 3. Redémarrer service
sudo systemctl restart voxtral-b200
```

#### **Performance dégradée**
```bash
# Diagnostic performance
make benchmark-b200
make profile

# Vérifications
# 1. Clock speeds GPU
nvidia-smi -q -d CLOCK

# 2. Thermal throttling
nvidia-smi -q -d TEMPERATURE

# 3. PCIe bandwidth
nvidia-smi topo -m
```

#### **Erreurs torch.compile**
```bash
# Diagnostic compilation
export TORCH_LOGS="+dynamo"
python main.py --debug

# Solutions
# 1. Fallback mode compilation
export B200_OPTIMIZER_MODE="reduce-overhead"

# 2. Désactiver torch.compile temporairement
export B200_ENABLE_COMPILE=0
```

### 2. Maintenance préventive

#### **Nettoyage quotidien**
```bash
#!/bin/bash
# daily_maintenance.sh

# Nettoyage cache GPU
python -c "import torch; torch.cuda.empty_cache()"

# Nettoyage logs
find /var/log -name "voxtral*.log" -mtime +7 -delete

# Nettoyage modèles temporaires
find /tmp -name "torch_*" -mtime +1 -delete

# Monitoring santé
make health-check
```

#### **Mise à jour hebdomadaire**
```bash
# weekly_update.sh

# Sauvegarde configuration
cp -r /app/config /backup/config_$(date +%Y%m%d)

# Mise à jour dependencies
make update-deps

# Tests régression
make test-all

# Benchmark performance
make benchmark-b200 > /logs/weekly_benchmark_$(date +%Y%m%d).json
```

### 3. Backup et recovery

#### **Backup modèles**
```bash
# backup_models.sh
BACKUP_DIR="/backup/models/$(date +%Y%m%d)"
mkdir -p $BACKUP_DIR

# Sauvegarde cache modèles
rsync -av ~/.cache/huggingface/ $BACKUP_DIR/huggingface/

# Sauvegarde configuration
cp -r /app/config $BACKUP_DIR/
```

#### **Recovery procédure**
```bash
# recovery.sh

# 1. Arrêt service
sudo systemctl stop voxtral-b200

# 2. Restoration modèles
rsync -av /backup/models/latest/huggingface/ ~/.cache/huggingface/

# 3. Validation configuration
make validate-setup

# 4. Tests fonctionnels
make test-fast

# 5. Redémarrage service
sudo systemctl start voxtral-b200
```

## 🎯 Optimisations avancées

### 1. Tuning fin B200
```python
# fine_tuning_b200.py
import torch

# Configuration optimale identifiée par benchmarks
OPTIMAL_B200_CONFIG = {
    "batch_size": 96,              # Sweet spot pour B200
    "sequence_length": 1024,       # Optimisation Tensor Cores
    "num_attention_heads": 32,     # Multiple de 8 pour bfloat16
    "intermediate_size": 4096,     # Optimisation matmul
    
    # torch.compile options
    "compile_options": {
        "mode": "max-autotune",
        "fullgraph": True,
        "dynamic": False,
        "options": {
            "triton.cudagraphs": True,
            "epilogue_fusion": True,
            "max_autotune": True,
            "shape_padding": True,
        }
    }
}
```

### 2. Profiling automatisé
```bash
# auto_profiling.sh

# Profile avec différentes configurations
for batch_size in 64 96 128; do
    export B200_BATCH_SIZE=$batch_size
    python -m torch.profiler main.py --profile-output "profile_bs${batch_size}.json"
done

# Analyse résultats
python analyze_profiles.py profile_*.json
```

### 3. A/B testing performance
```python
# ab_test_performance.py

configs = [
    {"compile_mode": "default", "batch_size": 64},
    {"compile_mode": "reduce-overhead", "batch_size": 96}, 
    {"compile_mode": "max-autotune", "batch_size": 128},
]

for config in configs:
    benchmark_result = run_benchmark(config)
    print(f"Config {config}: {benchmark_result['throughput']:.1f} segments/sec")
```

Ce guide assure un déploiement optimal et une maintenance efficace du système Voxtral sur hardware B200.