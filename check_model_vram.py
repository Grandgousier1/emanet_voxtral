
#!/usr/bin/env python3
"""
Outil de vérification de l'empreinte VRAM d'un modèle.

Ce script charge un modèle Hugging Face spécifié, le déplace sur le GPU
et mesure la mémoire VRAM allouée. C'est une étape de validation cruciale
pour s'assurer qu'un modèle peut être chargé avant de lancer une application complète.

Usage: python check_model_vram.py --model_id <ID_du_modèle_huggingface>
"""
import torch
import argparse
import sys
from transformers import AutoModelForCausalLM, AutoConfig

def check_vram(model_id: str):
    print(f"🚀 Démarrage de la vérification VRAM pour le modèle: {model_id}")

    # --- Vérification 1: Disponibilité de CUDA ---
    if not torch.cuda.is_available():
        print("🔥🔥🔥 ERREUR: CUDA n'est pas disponible. Vérification annulée.", file=sys.stderr)
        return False

    device = torch.device("cuda")
    print(f"✅ CUDA disponible. Périphérique: {torch.cuda.get_device_name(0)}")
    print(f"Mémoire totale: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    # --- Étape 2: Chargement du modèle ---
    try:
        print(f"--- Chargement du modèle '{model_id}' sur le CPU (pour commencer)... ---")
        # Utiliser bfloat16 si supporté pour un chargement plus réaliste
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        print(f"Utilisation du dtype: {dtype}")
        
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype)
        print("✅ Modèle chargé sur le CPU.")

        # --- Étape 3: Mesure de la VRAM ---
        torch.cuda.empty_cache() # Vider le cache avant la mesure
        initial_memory = torch.cuda.memory_allocated(device)
        
        print(f"--- Déplacement du modèle vers {device}... ---")
        model.to(device)
        
        final_memory = torch.cuda.memory_allocated(device)
        torch.cuda.empty_cache()

        vram_usage_gb = (final_memory - initial_memory) / 1e9
        
        print(f"\n{'='*60}\nRésultat de la mesure VRAM\n{'='*60}")
        print(f"✅ Empreinte VRAM du modèle '{model_id}': {vram_usage_gb:.2f} GB")
        print('='*60)

        return True

    except Exception as e:
        print(f"🔥🔥🔥 ERREUR: Une exception est survenue: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vérifie l'utilisation de la VRAM pour un modèle Hugging Face.")
    parser.add_argument("--model_id", type=str, required=True, help="L'identifiant du modèle sur le Hub Hugging Face (ex: 'meta-llama/Llama-2-7b-chat-hf').")
    args = parser.parse_args()

    if check_vram(args.model_id):
        sys.exit(0)
    else:
        sys.exit(1)
