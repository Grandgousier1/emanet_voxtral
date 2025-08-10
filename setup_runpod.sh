#!/usr/bin/env bash
set -e

echo "[setup] Installing system packages"
sudo apt-get update -y
sudo apt-get install -y ffmpeg git build-essential wget

if [ -d "$HOME/emanet-env" ]; then
    source $HOME/emanet-env/bin/activate
else
    echo "[setup] virtualenv not found at $HOME/emanet-env — run 'make install' first"
    exit 0
fi

pip install --upgrade pip
pip install -r requirements.txt

# prefetch mistral-small and optional voxtral-mini to speed first run
python - <<'PY'
from transformers import AutoTokenizer, AutoModelForCausalLM
print('[setup] Pulling mistralai/mistral-small (HF cache)')
try:
    AutoTokenizer.from_pretrained('mistralai/mistral-small')
    AutoModelForCausalLM.from_pretrained('mistralai/mistral-small', device_map='auto')
    print('[setup] Mistral small pulled')
except Exception as e:
    print('[setup] Warning pulling Mistral small:', e)
PY

echo "[setup] Done."
