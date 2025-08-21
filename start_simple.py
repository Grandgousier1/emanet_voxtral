#!/usr/bin/env python3
"""
start_simple.py - Démarrage simple sans dépendances lourdes
Alternative robuste à quick_start.py
"""

import os
import sys
import getpass
from pathlib import Path

def show_banner():
    print("=" * 70)
    print("🚀 EMANET VOXTRAL - Configuration Simple")
    print("   Générateur de Sous-titres avec IA")
    print("=" * 70)

def setup_token():
    """Configuration simple du token HF."""
    print("\n🔑 Configuration Token HuggingFace")
    print("-" * 40)
    print("1. Allez sur https://huggingface.co/settings/tokens")
    print("2. Créez un nouveau token (lecture seule suffit)")
    print("3. Copiez le token")
    print("")
    
    try:
        token = getpass.getpass("Entrez votre token HF (saisie masquée): ").strip()
        
        if not token:
            print("❌ Aucun token saisi")
            return False
        
        # Validation basique
        if len(token) < 20:
            print("❌ Token trop court (minimum 20 caractères)")
            return False
        
        # Sauvegarde dans .env
        env_file = Path(".env")
        
        # Lire contenu existant
        existing_lines = []
        if env_file.exists():
            with open(env_file, "r") as f:
                existing_lines = [line for line in f 
                                if not line.strip().startswith("HF_TOKEN=")]
        
        # Écrire le nouveau token
        with open(env_file, "w") as f:
            f.write(f"HF_TOKEN={token}\n")
            f.writelines(existing_lines)
        
        print("✅ Token sauvegardé dans .env")
        
        # Configurer pour la session actuelle
        os.environ['HF_TOKEN'] = token
        print("✅ Token configuré pour cette session")
        
        return True
        
    except (KeyboardInterrupt, EOFError):
        print("\n❌ Configuration annulée")
        return False
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return False

def check_token():
    """Vérifier si le token existe."""
    # Variables d'environnement
    token = os.getenv('HF_TOKEN') or os.getenv('HUGGINGFACE_HUB_TOKEN')
    if token:
        return True
    
    # Fichier .env
    env_file = Path(".env")
    if env_file.exists():
        try:
            with open(env_file, "r") as f:
                for line in f:
                    if line.strip().startswith('HF_TOKEN='):
                        return True
        except:
            pass
    
    return False

def main():
    show_banner()
    
    if not sys.stdin.isatty():
        print("❌ Ce script nécessite un terminal interactif")
        print("Utilisation alternative: export HF_TOKEN='votre_token'")
        sys.exit(1)
    
    print(f"\nPython: {sys.version}")
    print(f"Répertoire: {Path.cwd()}")
    
    # Vérifier le token
    if check_token():
        print("\n✅ Token HuggingFace déjà configuré !")
        print("Vous pouvez maintenant utiliser:")
        print("  • make start (interface complète)")
        print("  • python main.py --help (ligne de commande)")
    else:
        print("\n⚠️  Token HuggingFace manquant")
        
        try:
            if input("Configurer maintenant ? (Y/n): ").lower() != 'n':
                if setup_token():
                    print("\n🎉 Configuration terminée !")
                    print("Vous pouvez maintenant utiliser:")
                    print("  • make start")
                    print("  • python main.py --url 'https://youtube.com/...'")
                else:
                    print("\n⚠️  Configuration non terminée")
                    print("Vous pouvez:")
                    print("  • Relancer ce script: python start_simple.py")
                    print("  • Configurer manuellement: export HF_TOKEN='votre_token'")
        except (KeyboardInterrupt, EOFError):
            print("\n👋 Configuration interrompue")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 Au revoir !")
    except Exception as e:
        print(f"\n💥 Erreur inattendue: {e}")
        print("Contactez le support si le problème persiste.")