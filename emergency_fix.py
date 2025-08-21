#!/usr/bin/env python3
"""
emergency_fix.py - Script de récupération d'urgence
Installe les dépendances de base en cas de problème
"""

import sys
import subprocess
import os


def emergency_install():
    """Installation d'urgence des dépendances critiques."""
    print("🚨 INSTALLATION D'URGENCE")
    print("=" * 40)
    
    # Dépendances minimales absolues
    critical_packages = [
        "torch",
        "transformers", 
        "rich",
        "soundfile",
        "librosa",
        "numpy",
        "requests",
        "pyyaml",
        "click"
    ]
    
    print("📦 Installation des packages critiques...")
    
    for package in critical_packages:
        print(f"   Installing {package}...")
        try:
            subprocess.run([
                sys.executable, '-m', 'pip', 'install', package, '--quiet'
            ], check=True, timeout=300)
            print(f"   ✅ {package}")
        except subprocess.CalledProcessError:
            print(f"   ❌ {package} - ÉCHEC")
        except subprocess.TimeoutExpired:
            print(f"   ⏱️  {package} - TIMEOUT")
    
    print()
    print("🔧 Installation du projet...")
    try:
        subprocess.run([
            sys.executable, '-m', 'pip', 'install', '-e', '.', '--quiet'
        ], check=True, timeout=60)
        print("✅ Projet installé")
    except Exception as e:
        print(f"❌ Échec installation projet: {e}")
    
    print()
    print("🧪 Test des imports...")
    failed_imports = []
    
    for package in critical_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package}")
            failed_imports.append(package)
    
    print()
    if not failed_imports:
        print("🎉 RÉCUPÉRATION RÉUSSIE !")
        print("✅ Toutes les dépendances critiques sont installées")
        print("🚀 Vous pouvez maintenant utiliser: make start")
    else:
        print("⚠️  RÉCUPÉRATION PARTIELLE")
        print(f"❌ {len(failed_imports)} packages encore manquants: {failed_imports}")
        print("💡 Essayez manuellement:")
        for pkg in failed_imports:
            print(f"   pip install {pkg}")


if __name__ == "__main__":
    emergency_install()