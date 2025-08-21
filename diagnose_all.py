#!/usr/bin/env python3
"""
diagnose_all.py - Diagnostic ultra-complet du projet
Vérifie TOUT ce qui peut poser problème
"""

import sys
import os
import importlib
import subprocess
from pathlib import Path


def print_section(title):
    print(f"\n{'='*60}")
    print(f"🔍 {title}")
    print('='*60)


def check_python_environment():
    """Vérification environnement Python."""
    print_section("ENVIRONNEMENT PYTHON")
    
    print(f"Python version: {sys.version}")
    print(f"Python executable: {sys.executable}")
    print(f"Platform: {sys.platform}")
    
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ requis")
        return False
    else:
        print("✅ Version Python OK")
    
    # Vérifier pip
    try:
        import pip
        print(f"✅ pip disponible: {pip.__version__}")
    except ImportError:
        print("❌ pip non disponible")
        return False
    
    return True


def check_critical_imports():
    """Vérification des imports critiques."""
    print_section("DÉPENDANCES CRITIQUES")
    
    critical_packages = [
        ('torch', 'PyTorch'),
        ('transformers', 'Transformers'), 
        ('rich', 'Rich UI'),
        ('soundfile', 'SoundFile'),
        ('librosa', 'Librosa'),
        ('numpy', 'NumPy'),
        ('requests', 'Requests'),
        ('yaml', 'PyYAML'),
    ]
    
    missing = []
    for package, name in critical_packages:
        try:
            importlib.import_module(package)
            print(f"✅ {name}")
        except ImportError:
            print(f"❌ {name} - MANQUANT")
            missing.append(package)
    
    return missing


def check_project_files():
    """Vérification fichiers projet."""
    print_section("FICHIERS PROJET")
    
    critical_files = [
        'main.py',
        'quick_start.py', 
        'pyproject.toml',
        'requirements.txt',
        'requirements-minimal.txt',
        '.pre-commit-config.yaml',
        'Makefile'
    ]
    
    missing_files = []
    for file in critical_files:
        if Path(file).exists():
            print(f"✅ {file}")
        else:
            print(f"❌ {file} - MANQUANT")
            missing_files.append(file)
    
    return missing_files


def check_python_syntax():
    """Vérification syntaxe Python."""
    print_section("SYNTAXE PYTHON")
    
    python_files = list(Path('.').glob('*.py'))
    python_files.extend(list(Path('utils').glob('*.py')))
    
    errors = []
    checked = 0
    
    for py_file in python_files[:10]:  # Limite pour éviter spam
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                compile(f.read(), str(py_file), 'exec')
            checked += 1
        except SyntaxError as e:
            print(f"❌ {py_file}: {e}")
            errors.append(str(py_file))
        except Exception:
            pass  # Ignore autres erreurs (encoding, etc.)
    
    if not errors:
        print(f"✅ {checked} fichiers Python vérifiés - syntaxe OK")
    
    return errors


def check_disk_space():
    """Vérification espace disque."""
    print_section("RESSOURCES SYSTÈME")
    
    try:
        import shutil
        free_gb = shutil.disk_usage('.').free / (1024**3)
        total_gb = shutil.disk_usage('.').total / (1024**3)
        
        print(f"Espace disque libre: {free_gb:.1f}GB / {total_gb:.1f}GB")
        
        if free_gb < 10:
            print("❌ Espace disque critique (< 10GB)")
            return False
        elif free_gb < 25:
            print("⚠️  Espace disque limité (< 25GB recommandé)")
        else:
            print("✅ Espace disque OK")
            
    except Exception as e:
        print(f"❌ Erreur vérification espace: {e}")
        return False
    
    return True


def check_pip_installable():
    """Test si pip install fonctionne."""
    print_section("TEST INSTALLATION")
    
    try:
        # Test avec un package simple
        result = subprocess.run([
            sys.executable, '-m', 'pip', 'install', '--dry-run', 'requests'
        ], capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            print("✅ pip install fonctionne")
            return True
        else:
            print(f"❌ pip install échoue: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Test pip échoué: {e}")
        return False


def generate_install_command(missing_packages):
    """Génère la commande d'installation optimale."""
    print_section("COMMANDE D'INSTALLATION RECOMMANDÉE")
    
    if not missing_packages:
        print("✅ Toutes les dépendances sont installées !")
        print("🚀 Vous pouvez utiliser: make start")
        return
    
    print("💡 Commandes recommandées:")
    print()
    print("# Option 1: Installation complète")
    print("make install-dev")
    print()
    print("# Option 2: Installation minimale (si option 1 échoue)")
    print("make install-minimal")
    print()
    print("# Option 3: Installation manuelle")
    print(f"pip install {' '.join(missing_packages)}")
    print()


def main():
    """Diagnostic complet."""
    print("🚀 DIAGNOSTIC COMPLET EMANET VOXTRAL")
    
    all_good = True
    
    # Tests séquentiels
    if not check_python_environment():
        all_good = False
    
    missing_packages = check_critical_imports()
    if missing_packages:
        all_good = False
    
    missing_files = check_project_files()
    if missing_files:
        all_good = False
        
    syntax_errors = check_python_syntax()
    if syntax_errors:
        all_good = False
    
    if not check_disk_space():
        all_good = False
    
    if not check_pip_installable():
        all_good = False
    
    # Résumé final
    print_section("RÉSUMÉ FINAL")
    
    if all_good:
        print("🎉 TOUT EST PARFAIT !")
        print("✅ Aucun problème détecté")
        print("🚀 Vous pouvez utiliser: make start")
    else:
        print("⚠️  PROBLÈMES DÉTECTÉS")
        if missing_packages:
            print(f"   • {len(missing_packages)} dépendances manquantes")
        if missing_files:
            print(f"   • {len(missing_files)} fichiers manquants")
        if syntax_errors:
            print(f"   • {len(syntax_errors)} erreurs de syntaxe")
        
        generate_install_command(missing_packages)
    
    return all_good


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)