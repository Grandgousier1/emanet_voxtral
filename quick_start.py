#!/usr/bin/env python3
"""
quick_start.py - Quick Start Guide for EMANET VOXTRAL
User-friendly entry point with step-by-step guidance
"""

import os
import sys
import subprocess
from pathlib import Path


def show_banner():
    """Show simple banner without dependencies."""
    print("=" * 80)
    print("🚀 EMANET VOXTRAL - Générateur de Sous-titres B200")
    print("   Interface Simplifiée pour Débutants")
    print("=" * 80)


def check_environment():
    """Basic environment check."""
    issues = []
    
    # Check Python version
    if sys.version_info < (3, 8):
        issues.append("Python 3.8+ requis")
    
    # Check HF token
    if not os.getenv('HF_TOKEN') and not os.getenv('HUGGING_FACE_HUB_TOKEN'):
        issues.append("Token Hugging Face manquant")
    
    # Check disk space
    try:
        import shutil
        free_gb = shutil.disk_usage('.').free / (1024**3)
        if free_gb < 25:
            issues.append(f"Espace disque insuffisant: {free_gb:.1f}GB (25GB requis)")
    except:
        pass
    
    return issues


def setup_hf_token():
    """Simple HF token setup."""
    print("\n🔑 Configuration Token Hugging Face")
    print("=" * 50)
    print("1. Allez sur https://huggingface.co")
    print("2. Créez un compte (gratuit)")
    print("3. Settings → Access Tokens → Create new token")
    print("4. Copiez le token")
    print("")
    
    import getpass
    token = getpass.getpass("🔑 Entrez votre token HF (input masqué, ou Entrée pour ignorer): ").strip()
    
    if token:
        # Save to .env file with encryption for security
        env_file = Path(".env")
        try:
            # Try to encrypt the token for better security
            import sys
            sys.path.insert(0, '.')
            from utils.auth_manager import TokenManager
            from cli_feedback import get_feedback
            
            token_manager = TokenManager(get_feedback())
            encrypted_token = token_manager._encrypt_token(token)
            
            with open(env_file, "a") as f:
                f.write(f"\nHF_TOKEN={encrypted_token}\n")
            print("✅ Token chiffré et sauvegardé dans .env")
        except ImportError:
            # Fallback to plain text if encryption dependencies not available
            with open(env_file, "a") as f:
                f.write(f"\nHF_TOKEN={token}\n")
            print("✅ Token sauvegardé dans .env (avertissement: non chiffré)")
            print("⚠️  Installez 'cryptography' pour un stockage sécurisé: pip install cryptography")
        print("⚠️  Redémarrez le terminal pour prendre en compte le token")
        return True
    else:
        print("❌ Token non configuré - certaines fonctions seront limitées")
        return False


def show_menu():
    """Show main menu."""
    print("\n🎯 Que souhaitez-vous faire ?")
    print("=" * 40)
    print("1. 🎬 Traiter une vidéo YouTube")
    print("2. 📁 Traiter un fichier local")
    print("3. 📦 Traiter plusieurs fichiers")
    print("4. ⚙️  Configuration")
    print("5. 🏥 Diagnostic système")
    print("6. 📚 Aide")
    print("0. ❌ Quitter")
    print("")
    
    return input("Votre choix (0-6): ").strip()


def process_youtube():
    """Process YouTube video."""
    print("\n🎬 Traitement YouTube")
    print("=" * 30)
    
    url = input("URL YouTube: ").strip()
    if not url:
        print("❌ URL vide")
        return
    
    output = input("Fichier de sortie (sous_titres.srt): ").strip()
    if not output:
        output = "sous_titres.srt"
    
    # Language selection
    print("\nLangues disponibles:")
    langs = {"1": "fr", "2": "en", "3": "es", "4": "de"}
    for k, v in langs.items():
        lang_names = {"fr": "Français", "en": "English", "es": "Español", "de": "Deutsch"}
        print(f"  {k}. {lang_names[v]}")
    
    lang_choice = input("Langue (1-4, défaut 1): ").strip() or "1"
    target_lang = langs.get(lang_choice, "fr")
    
    # Build command
    cmd = [
        sys.executable, "main.py",
        "--url", url,
        "--output", output,
        "--target-lang", target_lang
    ]
    
    print(f"\n🚀 Commande: {' '.join(cmd)}")
    if input("Lancer ? (y/N): ").lower() == 'y':
        subprocess.run(cmd)


def process_local_file():
    """Process local file."""
    print("\n📁 Traitement Fichier Local")
    print("=" * 35)
    
    # Show available files
    print("Fichiers média dans le répertoire actuel:")
    media_exts = {'.mp4', '.mp3', '.wav', '.m4a', '.flac', '.webm', '.mkv'}
    files = []
    
    for f in Path('.').iterdir():
        if f.is_file() and f.suffix.lower() in media_exts:
            files.append(f)
            print(f"  {len(files)}. {f.name}")
    
    if not files:
        print("❌ Aucun fichier média trouvé")
        file_path = input("Chemin complet du fichier: ").strip()
    else:
        print("  0. Autre fichier...")
        choice = input(f"Choisir fichier (1-{len(files)}, 0 pour autre): ").strip()
        
        try:
            if choice == "0":
                file_path = input("Chemin complet du fichier: ").strip()
            else:
                file_path = str(files[int(choice) - 1])
        except (ValueError, IndexError):
            print("❌ Choix invalide")
            return
    
    if not Path(file_path).exists():
        print(f"❌ Fichier non trouvé: {file_path}")
        return
    
    output = input("Fichier de sortie (auto): ").strip()
    if not output:
        output = Path(file_path).stem + "_sous_titres.srt"
    
    # Build command
    cmd = [
        sys.executable, "main.py",
        "--url", file_path,
        "--output", output
    ]
    
    print(f"\n🚀 Commande: {' '.join(cmd)}")
    if input("Lancer ? (y/N): ").lower() == 'y':
        subprocess.run(cmd)


def process_batch():
    """Process multiple files."""
    print("\n📦 Traitement en Lot")
    print("=" * 25)
    
    batch_file = input("Fichier de lot (batch.txt): ").strip() or "batch.txt"
    
    if not Path(batch_file).exists():
        print(f"Création de {batch_file}...")
        urls = []
        print("Entrez les URLs/fichiers (ligne vide pour terminer):")
        
        while True:
            url = input(f"  {len(urls)+1}. ").strip()
            if not url:
                break
            urls.append(url)
        
        if urls:
            with open(batch_file, 'w') as f:
                f.write('\n'.join(urls))
            print(f"✅ {len(urls)} entrées sauvées dans {batch_file}")
        else:
            print("❌ Aucune entrée")
            return
    
    output_dir = input("Répertoire de sortie (./output): ").strip() or "./output"
    
    # Build command
    cmd = [
        sys.executable, "main.py",
        "--batch-list", batch_file,
        "--output-dir", output_dir
    ]
    
    print(f"\n🚀 Commande: {' '.join(cmd)}")
    if input("Lancer ? (y/N): ").lower() == 'y':
        subprocess.run(cmd)


def show_configuration():
    """Show configuration options."""
    print("\n⚙️ Configuration")
    print("=" * 20)
    
    # Check current config
    env_file = Path(".env")
    if env_file.exists():
        print("✅ Fichier .env trouvé")
        with open(env_file) as f:
            content = f.read()
            if "HF_TOKEN" in content:
                print("✅ Token HF configuré")
            else:
                print("❌ Token HF manquant")
    else:
        print("❌ Fichier .env non trouvé")
    
    print("\nOptions:")
    print("1. Configurer token Hugging Face")
    print("2. Voir configuration actuelle")
    print("3. Retour")
    
    choice = input("Choix (1-3): ").strip()
    
    if choice == "1":
        setup_hf_token()
    elif choice == "2":
        print("\nConfiguration actuelle:")
        print(f"  Python: {sys.version}")
        print(f"  Répertoire: {Path.cwd()}")
        print(f"  HF_TOKEN: {'✅' if os.getenv('HF_TOKEN') else '❌'}")


def run_diagnostic():
    """Run basic diagnostic."""
    print("\n🏥 Diagnostic Système")
    print("=" * 25)
    
    issues = check_environment()
    
    if not issues:
        print("✅ Système OK - Prêt pour le traitement")
    else:
        print("❌ Problèmes détectés:")
        for issue in issues:
            print(f"  • {issue}")
        
        print("\nSolutions:")
        if "Token Hugging Face" in str(issues):
            print("  • Configurez votre token HF (option 4 → 1)")
        if "Espace disque" in str(issues):
            print("  • Libérez de l'espace disque")
        if "Python" in str(issues):
            print("  • Mettez à jour Python vers 3.8+")


def show_help():
    """Show help information."""
    print("\n📚 Aide")
    print("=" * 10)
    print("""
UTILISATION RAPIDE:

1. 🎬 YouTube → Copiez l'URL, choisissez la langue, lancez
2. 📁 Local   → Sélectionnez le fichier, lancez  
3. 📦 Batch   → Listez vos fichiers, lancez

PRÉREQUIS:
• Python 3.8+
• Token Hugging Face (gratuit)
• 25GB espace disque libre
• GPU recommandé (B200 optimal)

FORMATS SUPPORTÉS:
• Entrée: YouTube, MP4, MP3, WAV, M4A, FLAC, WebM
• Sortie: SRT (sous-titres)

AIDE AVANCÉE:
• python main.py --help        (options complètes)
• python validator.py          (diagnostic détaillé)
• Consultez README.md          (documentation)

SUPPORT:
• GitHub: emanet_voxtral repository
• Logs: emanet.log (si erreurs)
    """)


def main():
    """Main interactive loop."""
    show_banner()
    
    # Quick environment check
    issues = check_environment()
    if issues:
        print("\n⚠️  Problèmes détectés:")
        for issue in issues:
            print(f"  • {issue}")
        
        if "Token Hugging Face" in str(issues):
            if input("\nConfigurer le token maintenant ? (Y/n): ").lower() != 'n':
                if setup_hf_token():
                    print("✅ Configuration terminée !")
    else:
        print("\n✅ Environnement OK - Prêt à commencer !")
    
    # Main loop
    while True:
        choice = show_menu()
        
        if choice == "0":
            print("👋 Au revoir !")
            break
        elif choice == "1":
            process_youtube()
        elif choice == "2":
            process_local_file()
        elif choice == "3":
            process_batch()
        elif choice == "4":
            show_configuration()
        elif choice == "5":
            run_diagnostic()
        elif choice == "6":
            show_help()
        else:
            print("❌ Choix invalide")
        
        input("\nAppuyez sur Entrée pour continuer...")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 Interruption utilisateur - Au revoir !")
    except Exception as e:
        print(f"\n💥 Erreur: {e}")
        print("Utilisez 'python main.py --help' pour l'aide complète")