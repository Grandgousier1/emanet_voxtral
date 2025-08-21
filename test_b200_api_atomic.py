
"""
Test atomique pour l'API d'optimisation B200.

Ce script vérifie que les fonctions de `utils.b200_optimizer` peuvent être
appelées et exécutées sur un modèle simple avec un tenseur sur le GPU.
Il a pour but de détecter rapidement les incompatibilités entre PyTorch, le driver
NVIDIA, et le code d'optimisation spécifique à la B200.

Usage: python test_b200_api_atomic.py
"""
import torch
import sys

def run_test():
    print("🚀 Démarrage du test atomique de l'API B200...")

    # --- Vérification 1: Disponibilité de CUDA ---
    if not torch.cuda.is_available():
        print("🔥🔥🔥 ERREUR: CUDA n'est pas disponible. Test annulé.", file=sys.stderr)
        return False
    
    device = torch.device("cuda")
    print(f"✅ CUDA disponible. Utilisation du périphérique: {torch.cuda.get_device_name(0)}")

    # --- Vérification 2: Import de l'optimiseur ---
    try:
        # Note: Assurez-vous que le chemin est correct pour votre structure de projet
        from utils.b200_optimizer import get_b200_optimizer, apply_b200_optimizations
        print("✅ Les fonctions de l'optimiseur B200 ont été importées avec succès.")
    except ImportError as e:
        print(f"🔥🔥🔥 ERREUR: Impossible d'importer depuis utils.b200_optimizer: {e}", file=sys.stderr)
        print("Assurez-vous que le projet est installé en mode éditable (`pip install -e .`) et que le PYTHONPATH est correct.")
        return False

    # --- Vérification 3: Application de l'optimisation ---
    print("--- Création d'un modèle et d'un tenseur factices ---")
    try:
        # Modèle simple
        dummy_model = torch.nn.Sequential(
            torch.nn.Linear(16, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 16)
        ).to(device)
        
        # Tenseur factice
        dummy_tensor = torch.randn(4, 16).to(device)

        print("--- Tentative d'application des optimisations B200 ---")
        # Simule l'obtention de la configuration d'optimisation
        optimizer_config = get_b200_optimizer()
        print(f"Configuration d'optimisation obtenue: {optimizer_config}")

        # Applique les optimisations
        optimized_model = apply_b200_optimizations(dummy_model, optimizer_config)
        print("✅ La fonction apply_b200_optimizations s'exécute sans erreur.")

        # Exécute une passe forward pour s'assurer que le modèle est toujours fonctionnel
        print("--- Exécution d'une passe forward sur le modèle optimisé ---")
        output = optimized_model(dummy_tensor)
        print(f"✅ Passe forward réussie. Shape de sortie: {output.shape}")

    except Exception as e:
        print(f"🔥🔥🔥 ERREUR: Une exception est survenue lors de l'application ou de l'exécution des optimisations: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return False

    print("\n✅✅✅ Le test atomique de l'API B200 est un succès!")
    return True

if __name__ == "__main__":
    if run_test():
        sys.exit(0)
    else:
        sys.exit(1)
