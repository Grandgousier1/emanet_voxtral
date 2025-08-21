#!/usr/bin/env python3
"""
test_token.py - Test si le token HF est configuré correctement
"""

import os
import sys
from pathlib import Path

def main():
    print("🔍 Test de configuration du token HuggingFace")
    print("=" * 50)
    
    # Test 1: Variable d'environnement
    env_token = os.getenv('HF_TOKEN') or os.getenv('HUGGINGFACE_HUB_TOKEN')
    print(f"1. Variable environnement: {'✅' if env_token else '❌'}")
    if env_token:
        print(f"   Token commence par: {env_token[:10]}...")
    
    # Test 2: Fichier .env
    env_file = Path(".env")
    env_file_token = None
    if env_file.exists():
        print("2. Fichier .env: ✅ trouvé")
        try:
            with open(env_file, "r") as f:
                for line in f:
                    line = line.strip()
                    if line.startswith('HF_TOKEN='):
                        env_file_token = line.split('=', 1)[1].strip('\'"')
                        break
            if env_file_token:
                print(f"   Token dans .env: ✅")
                print(f"   Token commence par: {env_file_token[:10]}...")
            else:
                print("   Token dans .env: ❌")
        except Exception as e:
            print(f"   Erreur lecture .env: {e}")
    else:
        print("2. Fichier .env: ❌ non trouvé")
    
    # Test 3: Token final
    final_token = env_token or env_file_token
    print(f"\n🎯 Token final: {'✅ configuré' if final_token else '❌ manquant'}")
    
    if not final_token:
        print("\n💡 Pour configurer le token:")
        print("   1. Lancez: make start")
        print("   2. Ou: python3 simple_auth.py")
        print("   3. Ou manuellement: export HF_TOKEN='votre_token'")
        return False
    
    print("✅ Configuration token OK !")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)