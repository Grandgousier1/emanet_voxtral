#!/usr/bin/env python3
"""
cli_enhanced.py - Enhanced CLI with Interactive Setup and User Guidance
Comprehensive user-friendly CLI with step-by-step guidance and validation
"""

import os
import sys
import time
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import getpass

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Prompt, Confirm, IntPrompt
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn
from rich.syntax import Syntax
from rich.text import Text
from rich.align import Align
from rich.columns import Columns
from rich.tree import Tree
from rich.status import Status
from rich.markdown import Markdown

console = Console()


class InteractiveCLI:
    """Enhanced interactive CLI for user-friendly experience."""
    
    def __init__(self):
        self.config_file = Path("cli_config.json")
        self.user_config = self.load_user_config()
        self.session_state = {
            "first_run": not self.config_file.exists(),
            "advanced_mode": False,
            "auto_mode": False
        }
    
    def load_user_config(self) -> Dict[str, Any]:
        """Load user configuration from file."""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                pass
        return {}
    
    def save_user_config(self):
        """Save user configuration to file."""
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self.user_config, f, indent=2, ensure_ascii=False)
        except Exception as e:
            console.print(f"[yellow]Impossible de sauvegarder la configuration: {e}[/yellow]")
    
    def display_welcome_banner(self):
        """Display enhanced welcome banner."""
        banner = """
[bold cyan]████████╗██████╗  █████╗ ██████╗ ██╗   ██╗ ██████╗████████╗██╗ ██████╗ ███╗   ██╗[/bold cyan]
[bold cyan]╚══██╔══╝██╔══██╗██╔══██╗██╔══██╗██║   ██║██╔════╝╚══██╔══╝██║██╔═══██╗████╗  ██║[/bold cyan]
[bold cyan]   ██║   ██████╔╝███████║██║  ██║██║   ██║██║        ██║   ██║██║   ██║██╔██╗ ██║[/bold cyan]
[bold cyan]   ██║   ██╔══██╗██╔══██║██║  ██║██║   ██║██║        ██║   ██║██║   ██║██║╚██╗██║[/bold cyan]
[bold cyan]   ██║   ██║  ██║██║  ██║██████╔╝╚██████╔╝╚██████╗   ██║   ██║╚██████╔╝██║ ╚████║[/bold cyan]
[bold cyan]   ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝╚═════╝  ╚═════╝  ╚═════╝   ╚═╝   ╚═╝ ╚═════╝ ╚═╝  ╚═══╝[/bold cyan]
                                                                                      
[bold yellow]         🚀 EMANET VOXTRAL - Générateur de Sous-titres B200 🚀[/bold yellow]
        """
        
        info_text = """
[dim]Version 2.0 - Optimisé pour GPU B200 (180GB VRAM)[/dim]
[dim]Traduction haute qualité avec modèles Voxtral de Mistral AI[/dim]

✨ [bold]Fonctionnalités principales :[/bold]
   • Traitement vidéo/audio automatique
   • Génération de sous-titres multilingues
   • Optimisation GPU B200 avancée
   • Interface interactive guidée
   • Monitoring temps réel
        """
        
        console.print(Panel.fit(
            f"{banner}\n{info_text}",
            border_style="cyan",
            padding=(1, 2)
        ))
    
    def show_quick_start_menu(self) -> str:
        """Show quick start menu for new users."""
        if not self.session_state["first_run"] and "preferred_mode" in self.user_config:
            return self.user_config["preferred_mode"]
        
        console.print("\n[bold]🎯 Comment souhaitez-vous commencer ?[/bold]\n")
        
        options = {
            "1": ("🎬 Mode Simple", "Traiter une vidéo/audio unique avec assistance"),
            "2": ("📦 Mode Batch", "Traiter plusieurs fichiers en lot"),
            "3": ("⚙️  Configuration", "Configurer l'environnement et les préférences"),
            "4": ("🔧 Mode Avancé", "Options avancées pour utilisateurs expérimentés"),
            "5": ("📚 Guide d'utilisation", "Tutoriel complet pas à pas"),
            "6": ("🏥 Diagnostic", "Vérifier l'état du système")
        }
        
        for key, (title, desc) in options.items():
            console.print(f"  [bold cyan]{key}[/bold cyan]. {title}")
            console.print(f"     [dim]{desc}[/dim]")
        
        choice = Prompt.ask(
            "\n[bold]Votre choix",
            choices=list(options.keys()),
            default="1"
        )
        
        mode_map = {
            "1": "simple",
            "2": "batch", 
            "3": "config",
            "4": "advanced",
            "5": "tutorial",
            "6": "diagnostic"
        }
        
        selected_mode = mode_map[choice]
        
        # Save preference for returning users
        if self.session_state["first_run"]:
            save_pref = Confirm.ask("💾 Sauvegarder ce choix comme préférence par défaut ?")
            if save_pref:
                self.user_config["preferred_mode"] = selected_mode
                self.save_user_config()
        
        return selected_mode
    
    def run_system_diagnostic(self):
        """Run comprehensive system diagnostic with fixes."""
        console.print(Panel("🏥 [bold]Diagnostic Système Complet[/bold]", style="blue"))
        
        checks = [
            ("🔧 Dépendances Python", self._check_dependencies),
            ("🎮 Configuration GPU", self._check_gpu_setup),
            ("💾 Espace disque", self._check_disk_space),
            ("🔑 Token Hugging Face", self._check_hf_token),
            ("📁 Structure projet", self._check_project_structure),
            ("🚀 Modèles disponibles", self._check_models),
            ("⚡ Performance système", self._check_performance)
        ]
        
        results = []
        with Progress(SpinnerColumn(), TextColumn("{task.description}"), transient=True) as progress:
            for name, check_func in checks:
                task = progress.add_task(f"Vérification: {name}", total=1)
                result = check_func()
                results.append((name, result))
                progress.update(task, completed=1)
                time.sleep(0.5)  # Visual feedback
        
        self._display_diagnostic_results(results)
        self._offer_fixes(results)
    
    def _check_dependencies(self) -> Dict[str, Any]:
        """Check Python dependencies."""
        try:
            import torch
            import rich
            import soundfile
            return {
                "status": "success",
                "message": "Toutes les dépendances critiques sont installées",
                "details": f"PyTorch: {torch.__version__}, Rich: {rich.__version__}"
            }
        except ImportError as e:
            return {
                "status": "error",
                "message": f"Dépendance manquante: {e}",
                "fix": "pip install -r requirements.txt"
            }
    
    def _check_gpu_setup(self) -> Dict[str, Any]:
        """Check GPU configuration."""
        try:
            import torch
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                gpu_names = [torch.cuda.get_device_name(i) for i in range(gpu_count)]
                memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                
                return {
                    "status": "success",
                    "message": f"{gpu_count} GPU(s) détecté(s)",
                    "details": f"GPU(s): {', '.join(gpu_names)}, Mémoire: {memory_gb:.1f}GB"
                }
            else:
                return {
                    "status": "warning",
                    "message": "Aucun GPU CUDA détecté",
                    "fix": "Vérifier l'installation CUDA/pilotes"
                }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Erreur GPU: {e}",
                "fix": "Réinstaller PyTorch avec support CUDA"
            }
    
    def _check_disk_space(self) -> Dict[str, Any]:
        """Check available disk space."""
        import shutil
        total, used, free = shutil.disk_usage('.')
        free_gb = free / (1024**3)
        
        if free_gb > 25:
            return {
                "status": "success",
                "message": f"Espace suffisant: {free_gb:.1f}GB libres"
            }
        elif free_gb > 10:
            return {
                "status": "warning",
                "message": f"Espace limité: {free_gb:.1f}GB libres",
                "fix": "Libérer de l'espace disque (recommandé: >25GB)"
            }
        else:
            return {
                "status": "error",
                "message": f"Espace insuffisant: {free_gb:.1f}GB libres",
                "fix": "Libérer au moins 25GB d'espace disque"
            }
    
    def _check_hf_token(self) -> Dict[str, Any]:
        """Check Hugging Face token."""
        token = os.getenv('HF_TOKEN') or os.getenv('HUGGING_FACE_HUB_TOKEN')
        if token:
            return {
                "status": "success",
                "message": "Token Hugging Face configuré"
            }
        else:
            return {
                "status": "error",
                "message": "Token Hugging Face manquant",
                "fix": "Configurer HF_TOKEN dans les variables d'environnement"
            }
    
    def _check_project_structure(self) -> Dict[str, Any]:
        """Check project file structure."""
        required_files = [
            "main.py", "config.py", "parallel_processor.py",
            "utils/model_utils.py", "utils/gpu_utils.py"
        ]
        
        missing = [f for f in required_files if not Path(f).exists()]
        
        if not missing:
            return {
                "status": "success",
                "message": "Structure projet complète"
            }
        else:
            return {
                "status": "error",
                "message": f"Fichiers manquants: {', '.join(missing)}",
                "fix": "Télécharger le projet complet depuis le repository"
            }
    
    def _check_models(self) -> Dict[str, Any]:
        """Check available models."""
        # This would check if models are cached/available
        return {
            "status": "info",
            "message": "Modèles seront téléchargés à la première utilisation",
            "details": "Voxtral-Small-24B, Voxtral-Mini-3B"
        }
    
    def _check_performance(self) -> Dict[str, Any]:
        """Check system performance indicators."""
        import psutil
        
        cpu_count = psutil.cpu_count()
        memory_gb = psutil.virtual_memory().total / (1024**3)
        
        return {
            "status": "info",
            "message": f"Système: {cpu_count} CPU, {memory_gb:.1f}GB RAM",
            "details": "Performance optimale avec 28+ vCPU, 188+ GB RAM"
        }
    
    def _display_diagnostic_results(self, results: List[Tuple[str, Dict]]):
        """Display diagnostic results in a table."""
        table = Table(title="📊 Résultats du Diagnostic", show_header=True, header_style="bold magenta")
        table.add_column("Composant", style="cyan", width=20)
        table.add_column("Statut", justify="center", width=10)
        table.add_column("Message", width=40)
        table.add_column("Détails", style="dim", width=30)
        
        for name, result in results:
            status = result["status"]
            if status == "success":
                status_icon = "[green]✅ OK[/green]"
            elif status == "warning":
                status_icon = "[yellow]⚠️  WARN[/yellow]"
            elif status == "error":
                status_icon = "[red]❌ ERR[/red]"
            else:
                status_icon = "[blue]ℹ️  INFO[/blue]"
            
            table.add_row(
                name,
                status_icon,
                result["message"],
                result.get("details", "")
            )
        
        console.print(table)
    
    def _offer_fixes(self, results: List[Tuple[str, Dict]]):
        """Offer to fix detected issues."""
        issues = [(name, result) for name, result in results if result["status"] in ["error", "warning"] and "fix" in result]
        
        if not issues:
            console.print("\n[green]✨ Aucun problème détecté ! Votre système est prêt.[/green]")
            return
        
        console.print(f"\n[yellow]🔧 {len(issues)} problème(s) détecté(s) avec des solutions disponibles:[/yellow]")
        
        for name, result in issues:
            console.print(f"\n[bold]{name}:[/bold]")
            console.print(f"  Problème: {result['message']}")
            console.print(f"  Solution: [cyan]{result['fix']}[/cyan]")
            
            if Confirm.ask(f"  Voulez-vous que je tente de résoudre ce problème automatiquement ?"):
                self._auto_fix_issue(name, result)
    
    def _auto_fix_issue(self, name: str, result: Dict):
        """Attempt to automatically fix an issue."""
        fix_command = result.get("fix", "")
        
        if "pip install" in fix_command:
            console.print("🔄 Installation des dépendances...")
            try:
                subprocess.run(fix_command.split(), check=True, capture_output=True)
                console.print("[green]✅ Dépendances installées avec succès![/green]")
            except subprocess.CalledProcessError as e:
                console.print(f"[red]❌ Échec de l'installation: {e}[/red]")
        
        elif "HF_TOKEN" in fix_command:
            console.print("🔑 Configuration du token Hugging Face...")
            self.setup_hf_token_interactive()
        
        else:
            console.print(f"[yellow]Solution manuelle requise: {fix_command}[/yellow]")
    
    def setup_hf_token_interactive(self):
        """Interactive Hugging Face token setup."""
        console.print(Panel("""
[bold]Configuration du Token Hugging Face[/bold]

Pour utiliser les modèles Voxtral, vous devez:
1. Créer un compte sur https://huggingface.co
2. Aller dans Settings → Access Tokens
3. Créer un nouveau token avec permissions de lecture
4. Copier le token ici

[dim]Le token ne sera pas affiché pendant la saisie pour des raisons de sécurité.[/dim]
        """, style="blue"))
        
        if Confirm.ask("Avez-vous déjà un token Hugging Face ?"):
            token = getpass.getpass("Entrez votre token HF (invisible): ")
            
            if token.strip():
                # Save to .env file with encryption for security
                env_file = Path(".env")
                try:
                    from utils.auth_manager import TokenManager
                    token_manager = TokenManager()
                    encrypted_token = token_manager._encrypt_token(token.strip())
                    
                    with open(env_file, "a") as f:
                        f.write(f"\nHF_TOKEN={encrypted_token}\n")
                    
                    console.print("[green]✅ Token chiffré et sauvegardé dans .env[/green]")
                except ImportError:
                    # Fallback to plain text if encryption not available
                    with open(env_file, "a") as f:
                        f.write(f"\nHF_TOKEN={token.strip()}\n")
                    
                    console.print("[green]✅ Token sauvegardé dans .env[/green]")
                    console.print("[yellow]⚠️  Installez 'cryptography' pour un stockage sécurisé[/yellow]")
                console.print("[yellow]ℹ️  Redémarrez l'application pour prendre en compte le token[/yellow]")
            else:
                console.print("[red]❌ Token vide, configuration annulée[/red]")
        else:
            console.print("Visitez https://huggingface.co pour créer un compte et obtenir un token.")
    
    def show_tutorial(self):
        """Show comprehensive tutorial."""
        console.print(Panel("📚 [bold]Guide d'Utilisation Complet[/bold]", style="green"))
        
        sections = [
            ("🎯 Vue d'ensemble", self._tutorial_overview),
            ("⚙️ Configuration initiale", self._tutorial_setup),
            ("🎬 Traitement simple", self._tutorial_simple),
            ("📦 Traitement en lot", self._tutorial_batch),
            ("🔧 Options avancées", self._tutorial_advanced),
            ("🚨 Résolution de problèmes", self._tutorial_troubleshooting)
        ]
        
        for title, content_func in sections:
            if Confirm.ask(f"\nAfficher: {title} ?", default=True):
                content_func()
                if not Confirm.ask("Continuer vers la section suivante ?", default=True):
                    break
    
    def _tutorial_overview(self):
        """Tutorial overview section."""
        content = """
[bold]EMANET VOXTRAL - Vue d'ensemble[/bold]

Ce logiciel transforme automatiquement vos vidéos/audios en sous-titres traduits de haute qualité.

[bold cyan]Processus typique :[/bold cyan]
1. 📥 Téléchargement/chargement du média (YouTube, fichier local)
2. 🎵 Extraction et traitement de l'audio
3. 🗣️  Détection des segments de parole (VAD)
4. 🤖 Transcription avec modèles Voxtral (Mistral AI)
5. 🌍 Traduction vers la langue cible
6. 💾 Génération du fichier SRT

[bold yellow]Formats supportés :[/bold yellow]
• Entrée: YouTube, MP4, MP3, WAV, M4A, et plus
• Sortie: Fichiers SRT (sous-titres)

[bold green]Optimisations B200 :[/bold green]
• Traitement GPU parallèle haute performance
• Gestion intelligente de la mémoire
• Monitoring temps réel des ressources
        """
        console.print(Panel(content, title="Vue d'ensemble", border_style="cyan"))
    
    def _tutorial_setup(self):
        """Tutorial setup section."""
        content = """
[bold]Configuration Initiale[/bold]

[bold cyan]1. Token Hugging Face (OBLIGATOIRE)[/bold cyan]
   • Créer un compte sur https://huggingface.co
   • Générer un token d'accès
   • Le configurer via l'option 'Configuration'

[bold cyan]2. Vérification GPU[/bold cyan]
   • GPU B200 recommandé (180GB VRAM)
   • Pilotes CUDA à jour
   • PyTorch avec support GPU

[bold cyan]3. Espace disque[/bold cyan]
   • Minimum 25GB libres
   • Plus pour le traitement en lot

[bold cyan]4. Variables d'environnement[/bold cyan]
   • HF_TOKEN=votre_token_huggingface
   • Optionnel: CUDA_VISIBLE_DEVICES=0
        """
        console.print(Panel(content, title="Configuration", border_style="green"))
    
    def _tutorial_simple(self):
        """Tutorial simple processing section."""
        content = """
[bold]Traitement Simple - Mode Guidé[/bold]

[bold cyan]Commande de base :[/bold cyan]
python main.py --url "https://youtube.com/watch?v=..." --output sous_titres.srt

[bold cyan]Options courantes :[/bold cyan]
• --target-lang fr          : Langue de traduction (fr, en, es, de, it...)
• --model voxtral-small     : Modèle à utiliser (small/mini)
• --quality best            : Qualité (fast/balanced/best)
• --debug                   : Mode debug pour plus d'informations

[bold cyan]Exemple complet :[/bold cyan]
python main.py \\
  --url "https://youtube.com/watch?v=dQw4w9WgXcQ" \\
  --output "rick_roll_fr.srt" \\
  --target-lang fr \\
  --quality best

[bold yellow]Le système vous guidera à chaque étape ![/bold yellow]
        """
        console.print(Panel(content, title="Mode Simple", border_style="blue"))
    
    def _tutorial_batch(self):
        """Tutorial batch processing section."""
        content = """
[bold]Traitement en Lot[/bold]

[bold cyan]1. Créer un fichier de lot (batch.txt) :[/bold cyan]
https://youtube.com/watch?v=video1
https://youtube.com/watch?v=video2
/chemin/vers/fichier_local.mp4

[bold cyan]2. Lancer le traitement :[/bold cyan]
python main.py --batch-list batch.txt --output-dir resultats/

[bold cyan]3. Structure de sortie :[/bold cyan]
resultats/
├── video1_fr.srt
├── video2_fr.srt
└── fichier_local_fr.srt

[bold cyan]Options avancées :[/bold cyan]
• --max-workers 4           : Nombre de workers parallèles
• --skip-existing           : Ignorer les fichiers déjà traités
• --continue-on-error       : Continuer même en cas d'erreur
        """
        console.print(Panel(content, title="Mode Batch", border_style="yellow"))
    
    def _tutorial_advanced(self):
        """Tutorial advanced options section."""
        content = """
[bold]Options Avancées[/bold]

[bold cyan]Performance GPU :[/bold cyan]
• --gpu-memory-limit 0.9    : Limite mémoire GPU (0.1-0.95)
• --batch-size 32           : Taille des lots de traitement
• --precision bf16          : Précision (fp16/bf16/fp32)

[bold cyan]Paramètres audio :[/bold cyan]
• --vad-threshold 0.3       : Seuil de détection vocale
• --min-segment-duration 1.0: Durée minimale des segments
• --audio-quality high      : Qualité audio (low/medium/high)

[bold cyan]Contrôle de qualité :[/bold cyan]
• --min-quality-score 0.7   : Score qualité minimal
• --retry-failed-segments   : Réessayer les segments échoués
• --quality-check           : Validation qualité post-traitement

[bold cyan]Monitoring :[/bold cyan]
• --monitor                 : Interface monitoring temps réel
• --telemetry               : Collecte de métriques avancées
        """
        console.print(Panel(content, title="Options Avancées", border_style="magenta"))
    
    def _tutorial_troubleshooting(self):
        """Tutorial troubleshooting section."""
        content = """
[bold]Résolution de Problèmes[/bold]

[bold red]Erreurs courantes :[/bold red]

[bold cyan]1. "CUDA out of memory"[/bold cyan]
   → Réduire --batch-size ou --gpu-memory-limit
   → Fermer autres applications GPU

[bold cyan]2. "Token HF invalide"[/bold cyan]
   → Vérifier le token dans .env
   → Régénérer un nouveau token sur HF

[bold cyan]3. "Espace disque insuffisant"[/bold cyan]
   → Libérer au moins 25GB
   → Nettoyer le cache: rm -rf ~/.cache/huggingface/

[bold cyan]4. "Erreur de téléchargement YouTube"[/bold cyan]
   → Vérifier l'URL
   → Mettre à jour yt-dlp: pip install -U yt-dlp

[bold yellow]Outils de diagnostic :[/bold yellow]
• python main.py --validate  : Validation complète
• python validator.py        : Tests détaillés
• Option 6 du menu principal : Diagnostic interactif

[bold green]Support :[/bold green]
• Mode --debug pour plus d'infos
• Logs détaillés dans emanet.log
• Monitoring temps réel avec --monitor
        """
        console.print(Panel(content, title="Dépannage", border_style="red"))
    
    def configure_system_interactive(self):
        """Interactive system configuration."""
        console.print(Panel("⚙️ [bold]Configuration Système Interactive[/bold]", style="blue"))
        
        config_options = [
            ("🔑 Token Hugging Face", self.setup_hf_token_interactive),
            ("🎮 Configuration GPU", self._configure_gpu),
            ("📁 Répertoires de travail", self._configure_directories),
            ("🌍 Langue par défaut", self._configure_default_language),
            ("⚡ Paramètres performance", self._configure_performance),
            ("📊 Monitoring et logs", self._configure_monitoring),
            ("💾 Sauvegarder configuration", self._save_configuration)
        ]
        
        while True:
            console.print("\n[bold]Options de configuration :[/bold]")
            for i, (name, _) in enumerate(config_options, 1):
                console.print(f"  {i}. {name}")
            console.print("  0. Retour au menu principal")
            
            choice = IntPrompt.ask("Votre choix", choices=[str(i) for i in range(len(config_options) + 1)])
            
            if choice == 0:
                break
            elif 1 <= choice <= len(config_options):
                config_options[choice - 1][1]()
    
    def _configure_gpu(self):
        """Configure GPU settings."""
        console.print("🎮 [bold]Configuration GPU[/bold]")
        
        # Auto-detect GPU info
        try:
            import torch
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                for i in range(gpu_count):
                    props = torch.cuda.get_device_properties(i)
                    console.print(f"GPU {i}: {props.name} ({props.total_memory / 1024**3:.1f}GB)")
                
                # Memory limit configuration
                current_limit = self.user_config.get("gpu_memory_limit", 0.9)
                new_limit = Prompt.ask(
                    f"Limite mémoire GPU (0.1-0.95)",
                    default=str(current_limit)
                )
                
                try:
                    self.user_config["gpu_memory_limit"] = float(new_limit)
                    console.print(f"[green]✅ Limite mémoire GPU: {new_limit}[/green]")
                except ValueError:
                    console.print("[red]❌ Valeur invalide[/red]")
                
                # Batch size configuration
                current_batch = self.user_config.get("batch_size", 32)
                new_batch = IntPrompt.ask(
                    "Taille de lot par défaut",
                    default=current_batch
                )
                self.user_config["batch_size"] = new_batch
                
            else:
                console.print("[yellow]⚠️ Aucun GPU CUDA détecté[/yellow]")
                
        except ImportError:
            console.print("[red]❌ PyTorch non disponible[/red]")
    
    def _configure_directories(self):
        """Configure working directories."""
        console.print("📁 [bold]Configuration des Répertoires[/bold]")
        
        # Output directory
        current_output = self.user_config.get("default_output_dir", "./output")
        new_output = Prompt.ask("Répertoire de sortie par défaut", default=current_output)
        
        output_path = Path(new_output)
        output_path.mkdir(parents=True, exist_ok=True)
        self.user_config["default_output_dir"] = str(output_path)
        
        # Cache directory
        current_cache = self.user_config.get("cache_dir", "./cache")
        new_cache = Prompt.ask("Répertoire cache", default=current_cache)
        
        cache_path = Path(new_cache)
        cache_path.mkdir(parents=True, exist_ok=True)
        self.user_config["cache_dir"] = str(cache_path)
        
        console.print("[green]✅ Répertoires configurés[/green]")
    
    def _configure_default_language(self):
        """Configure default target language."""
        languages = {
            "fr": "Français",
            "en": "English", 
            "es": "Español",
            "de": "Deutsch",
            "it": "Italiano",
            "pt": "Português",
            "ru": "Русский",
            "zh": "中文",
            "ja": "日本語",
            "ar": "العربية"
        }
        
        console.print("🌍 [bold]Langue de traduction par défaut[/bold]")
        for code, name in languages.items():
            console.print(f"  {code}: {name}")
        
        current_lang = self.user_config.get("default_target_language", "fr")
        new_lang = Prompt.ask(
            "Code langue",
            choices=list(languages.keys()),
            default=current_lang
        )
        
        self.user_config["default_target_language"] = new_lang
        console.print(f"[green]✅ Langue par défaut: {languages[new_lang]}[/green]")
    
    def _configure_performance(self):
        """Configure performance settings."""
        console.print("⚡ [bold]Paramètres de Performance[/bold]")
        
        # Quality preset
        quality_presets = ["fast", "balanced", "best"]
        current_quality = self.user_config.get("default_quality", "balanced")
        new_quality = Prompt.ask(
            "Préréglage qualité par défaut",
            choices=quality_presets,
            default=current_quality
        )
        self.user_config["default_quality"] = new_quality
        
        # Parallel workers
        import psutil
        max_workers = psutil.cpu_count()
        current_workers = self.user_config.get("max_workers", min(4, max_workers))
        new_workers = IntPrompt.ask(
            f"Nombre de workers parallèles (1-{max_workers})",
            default=current_workers
        )
        if 1 <= new_workers <= max_workers:
            self.user_config["max_workers"] = new_workers
        
        console.print("[green]✅ Paramètres de performance mis à jour[/green]")
    
    def _configure_monitoring(self):
        """Configure monitoring and logging."""
        console.print("📊 [bold]Configuration Monitoring[/bold]")
        
        # Enable monitoring by default
        enable_monitoring = Confirm.ask("Activer le monitoring temps réel ?", default=True)
        self.user_config["enable_monitoring"] = enable_monitoring
        
        # Telemetry
        enable_telemetry = Confirm.ask("Activer la télémétrie (métriques avancées) ?", default=True)
        self.user_config["enable_telemetry"] = enable_telemetry
        
        # Log level
        log_levels = ["INFO", "DEBUG", "WARNING", "ERROR"]
        current_level = self.user_config.get("log_level", "INFO")
        new_level = Prompt.ask(
            "Niveau de logs",
            choices=log_levels,
            default=current_level
        )
        self.user_config["log_level"] = new_level
        
        console.print("[green]✅ Configuration monitoring mise à jour[/green]")
    
    def _save_configuration(self):
        """Save current configuration."""
        self.save_user_config()
        console.print("[green]✅ Configuration sauvegardée dans cli_config.json[/green]")
        
        # Display current config summary
        if self.user_config:
            table = Table(title="Configuration Actuelle", show_header=True)
            table.add_column("Paramètre", style="cyan")
            table.add_column("Valeur", style="green")
            
            for key, value in self.user_config.items():
                table.add_row(key, str(value))
            
            console.print(table)
    
    def get_user_preferences(self) -> Dict[str, Any]:
        """Get user preferences for command-line arguments."""
        prefs = {}
        
        # Apply saved configuration
        if "default_target_language" in self.user_config:
            prefs["target_lang"] = self.user_config["default_target_language"]
        
        if "default_quality" in self.user_config:
            prefs["quality"] = self.user_config["default_quality"]
        
        if "gpu_memory_limit" in self.user_config:
            prefs["gpu_memory_limit"] = self.user_config["gpu_memory_limit"]
        
        if "batch_size" in self.user_config:
            prefs["batch_size"] = self.user_config["batch_size"]
        
        if "max_workers" in self.user_config:
            prefs["max_workers"] = self.user_config["max_workers"]
        
        if "enable_monitoring" in self.user_config:
            prefs["monitor"] = self.user_config["enable_monitoring"]
        
        if "log_level" in self.user_config:
            prefs["log_level"] = self.user_config["log_level"]
        
        return prefs


def create_enhanced_cli() -> InteractiveCLI:
    """Create enhanced CLI instance."""
    return InteractiveCLI()


if __name__ == "__main__":
    # Demo CLI
    cli = InteractiveCLI()
    cli.display_welcome_banner()
    
    mode = cli.show_quick_start_menu()
    
    if mode == "diagnostic":
        cli.run_system_diagnostic()
    elif mode == "tutorial":
        cli.show_tutorial()
    elif mode == "config":
        cli.configure_system_interactive()
    else:
        console.print(f"Mode sélectionné: {mode}")
        console.print("Intégration avec main.py en cours de développement...")