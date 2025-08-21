
"""
Script de validation des dépendances à haut risque.

Ce script tente d'installer et d'importer séquentiellement les bibliothèques
qui sont les plus susceptibles de causer des problèmes de compatibilité
sur des architectures GPU spécifiques comme le B200.

Usage: python validate_dependencies.py
"""
import subprocess
import sys
import importlib

# Dépendances à tester. Le format est: (extra_pip, nom_module_import)
# L'extra_pip correspond à celui défini dans pyproject.toml ([project.optional-dependencies])
DEPS_TO_VALIDATE = [
    ("vllm", "vllm"),
    ("flash-attn", "flash_attn"),
    ("bitsandbytes", "bitsandbytes"),
]

def run_command(command):
    """Exécute une commande shell et retourne True en cas de succès."""
    print(f"\n--- Exécution: '{' '.join(command)}' ---")
    try:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8')
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                print(output.strip())
        
        return_code = process.poll()
        if return_code == 0:
            print(f"--- Commande réussie ---")
            return True
        else:
            print(f"--- ERREUR: La commande a échoué avec le code {return_code} ---", file=sys.stderr)
            return False
    except Exception as e:
        print(f"--- EXCEPTION: Une erreur est survenue lors de l'exécution. {e} ---", file=sys.stderr)
        return False

def validate_import(module_name):
    """Tente d'importer un module et retourne True en cas de succès."""
    print(f"--- Validation de l'import: '{module_name}' ---")
    try:
        importlib.import_module(module_name)
        print(f"--- Succès: Le module '{module_name}' a été importé correctement. ---")
        return True
    except ImportError as e:
        print(f"--- ERREUR: Impossible d'importer le module '{module_name}'. Erreur: {e} ---", file=sys.stderr)
        return False
    except Exception as e:
        print(f"--- EXCEPTION: Une erreur inattendue est survenue lors de l'import. {e} ---", file=sys.stderr)
        return False

def main():
    print("🚀 Démarrage de la validation des dépendances critiques...")
    all_success = True
    
    for extra, module in DEPS_TO_VALIDATE:
        print(f"\n{'='*60}\nValidating: {extra} (module: {module})\n{'='*60}")
        
        # Étape 1: Installation
        install_command = [sys.executable, "-m", "pip", "install", f".[{extra}]"]
        if not run_command(install_command):
            print(f"❌ Échec de l'installation pour '{extra}'. Validation arrêtée pour ce paquet.", file=sys.stderr)
            all_success = False
            continue
            
        # Étape 2: Importation
        if not validate_import(module):
            print(f"❌ Échec de l'importation pour '{module}'.", file=sys.stderr)
            all_success = False

    print(f"\n{'='*60}\nRésultat final de la validation\n{'='*60}")
    if all_success:
        print("✅✅✅ Toutes les dépendances critiques ont été installées et importées avec succès!")
        sys.exit(0)
    else:
        print("🔥🔥🔥 Au moins une dépendance critique a échoué. Veuillez vérifier les logs ci-dessus.")
        sys.exit(1)

if __name__ == "__main__":
    main()

