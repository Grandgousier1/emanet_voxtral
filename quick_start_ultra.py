#!/usr/bin/env python3
"""
quick_start_ultra.py - Version ultra-simplifiée du démarrage
Pas de dépendances complexes, juste l'essentiel
"""

import os
import sys
import getpass
from pathlib import Path


def main():
    print("=" * 60)
    print("🚀 EMANET VOXTRAL - Configuration Token")
    print("=" * 60)
    
    # Vérifier si le token existe déjà
    existing_token = os.getenv('HF_TOKEN') or os.getenv('HUGGINGFACE_HUB_TOKEN')
    
    if not existing_token:
        # Vérifier dans .env
        env_file = Path('.env')
        if env_file.exists():
            try:
                with open(env_file, 'r') as f:
                    for line in f:
                        if line.strip().startswith('HF_TOKEN='):
                            existing_token = line.split('=', 1)[1].strip('\'"')
                            break
            except:
                pass
    
    if existing_token:
        print(f"✅ Token HuggingFace déjà configuré")
        print(f"   Token: {existing_token[:10]}...")
        print("\n🎯 Tout est prêt ! Vous pouvez utiliser :")
        print("   • python main.py --url 'https://youtube.com/watch?v=...'")
        print("   • python main.py --help (pour toutes les options)")
        return
    
    print("🔑 Configuration Token HuggingFace nécessaire")
    print("-" * 50)
    print("1. Allez sur: https://huggingface.co/settings/tokens")
    print("2. Créez un nouveau token (lecture seule suffit)")
    print("3. Copiez le token")
    print("")
    
    try:
        token = getpass.getpass("Collez votre token HF ici (saisie cachée): ").strip()
        
        if not token:
            print("❌ Pas de token fourni")
            return
        
        # Validation basique
        if len(token) < 15:
            print("❌ Token trop court")
            return
        
        # Sauvegarder
        env_file = Path('.env')
        
        # Contenu existant sans tokens HF
        existing_content = ""
        if env_file.exists():
            with open(env_file, 'r') as f:
                lines = [line for line in f.readlines() 
                        if not line.strip().startswith(('HF_TOKEN=', 'HUGGINGFACE_HUB_TOKEN='))]
                existing_content = ''.join(lines)
        
        # Écrire token
        with open(env_file, 'w') as f:
            f.write(f'HF_TOKEN={token}\n')
            if existing_content.strip():
                f.write(existing_content)
        
        # Session courante
        os.environ['HF_TOKEN'] = token
        
        print("✅ Token configuré avec succès !")
        print("✅ Sauvegardé dans .env pour les prochaines fois")
        print("✅ Configuré pour cette session")
        print("")
        print("🎉 CONFIGURATION TERMINÉE !")
        print("    Plus jamais besoin de reconfigurer")
        print("")
        print("🎯 Vous pouvez maintenant utiliser :")
        print("   • python main.py --url 'https://youtube.com/watch?v=...'")
        print("   • python main.py --help (pour toutes les options)")
        
    except KeyboardInterrupt:
        print("\n❌ Configuration annulée")
    except Exception as e:
        print(f"❌ Erreur: {e}")


if __name__ == "__main__":
    main()