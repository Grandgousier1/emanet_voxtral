#!/usr/bin/env python3
"""
cli_wizard.py - Interactive CLI Wizard for EMANET VOXTRAL
Step-by-step guided experience for all user scenarios
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, Confirm, IntPrompt
from rich.table import Table
from rich.text import Text
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()


class CLIWizard:
    """Interactive wizard for building command-line arguments."""
    
    def __init__(self):
        self.args = {}
        self.mode = None
    
    def run_wizard(self) -> Dict[str, Any]:
        """Run the complete wizard and return parsed arguments."""
        self.show_welcome()
        self.select_mode()
        
        if self.mode == "simple":
            return self.wizard_simple_mode()
        elif self.mode == "batch":
            return self.wizard_batch_mode()
        elif self.mode == "advanced":
            return self.wizard_advanced_mode()
        else:
            return {}
    
    def show_welcome(self):
        """Show welcome message."""
        welcome_text = """
[bold cyan]🧙‍♂️ Assistant de Configuration EMANET VOXTRAL[/bold cyan]

Je vais vous guider étape par étape pour configurer votre traitement
de sous-titres selon vos besoins spécifiques.

[dim]Vous pouvez annuler à tout moment avec Ctrl+C[/dim]
        """
        console.print(Panel.fit(welcome_text, border_style="cyan"))
    
    def select_mode(self):
        """Let user select processing mode."""
        console.print("\n[bold]🎯 Quel type de traitement souhaitez-vous faire ?[/bold]\n")
        
        modes = {
            "1": ("simple", "🎬 Traiter une seule vidéo/audio", "Idéal pour tester ou traiter un fichier unique"),
            "2": ("batch", "📦 Traiter plusieurs fichiers", "Traitement en lot pour plusieurs vidéos"),
            "3": ("advanced", "🔧 Configuration avancée", "Contrôle fin de tous les paramètres")
        }
        
        for key, (mode, title, desc) in modes.items():
            console.print(f"  [bold cyan]{key}[/bold cyan]. {title}")
            console.print(f"     [dim]{desc}[/dim]\n")
        
        choice = Prompt.ask("Votre choix", choices=["1", "2", "3"], default="1")
        self.mode = modes[choice][0]
        
        console.print(f"\n[green]✅ Mode sélectionné: {modes[choice][1]}[/green]")
    
    def wizard_simple_mode(self) -> Dict[str, Any]:
        """Wizard for simple single video processing."""
        console.print(Panel("🎬 [bold]Mode Simple - Traitement d'une vidéo/audio[/bold]", style="blue"))
        
        # Source input
        source_type = self._ask_source_type()
        
        if source_type == "youtube":
            self.args["url"] = self._ask_youtube_url()
        elif source_type == "local":
            self.args["url"] = self._ask_local_file()
        
        # Output file
        self.args["output"] = self._ask_output_file()
        
        # Target language
        self.args["target_lang"] = self._ask_target_language()
        
        # Quality level
        self.args["quality"] = self._ask_quality_level()
        
        # Optional advanced settings
        if Confirm.ask("\n🔧 Configurer des paramètres avancés ?", default=False):
            self._ask_advanced_options()
        
        return self._build_namespace()
    
    def wizard_batch_mode(self) -> Dict[str, Any]:
        """Wizard for batch processing."""
        console.print(Panel("📦 [bold]Mode Batch - Traitement en lot[/bold]", style="yellow"))
        
        # Batch file creation or selection
        batch_file = self._ask_batch_file()
        self.args["batch_list"] = batch_file
        
        # Output directory
        self.args["output_dir"] = self._ask_output_directory()
        
        # Common settings for all files
        self.args["target_lang"] = self._ask_target_language()
        self.args["quality"] = self._ask_quality_level()
        
        # Parallel processing
        self.args["max_workers"] = self._ask_parallel_workers()
        
        # Error handling
        self.args["continue_on_error"] = Confirm.ask("Continuer même en cas d'erreur sur un fichier ?", default=True)
        self.args["skip_existing"] = Confirm.ask("Ignorer les fichiers déjà traités ?", default=True)
        
        return self._build_namespace()
    
    def wizard_advanced_mode(self) -> Dict[str, Any]:
        """Wizard for advanced configuration."""
        console.print(Panel("🔧 [bold]Mode Avancé - Configuration experte[/bold]", style="magenta"))
        
        # Start with basic mode selection
        if Confirm.ask("Commencer par le mode simple puis ajouter des options ?", default=True):
            # Get basic settings first
            basic_args = self.wizard_simple_mode()
            self.args.update(basic_args.__dict__)
        
        console.print("\n[bold]🎛️  Configuration avancée:[/bold]")
        
        # GPU and performance settings
        self._configure_gpu_settings()
        
        # Audio processing settings
        self._configure_audio_settings()
        
        # Quality and validation settings
        self._configure_quality_settings()
        
        # Monitoring and debugging
        self._configure_monitoring_settings()
        
        return self._build_namespace()
    
    def _ask_source_type(self) -> str:
        """Ask for source type."""
        console.print("\n[bold]📥 Source de votre média:[/bold]")
        console.print("  1. YouTube (URL)")
        console.print("  2. Fichier local (MP4, MP3, WAV, etc.)")
        
        choice = Prompt.ask("Type de source", choices=["1", "2"], default="1")
        return "youtube" if choice == "1" else "local"
    
    def _ask_youtube_url(self) -> str:
        """Ask for YouTube URL with validation."""
        while True:
            url = Prompt.ask("\n🎬 URL YouTube")
            
            if "youtube.com" in url or "youtu.be" in url:
                # Basic URL validation
                console.print(f"[green]✅ URL acceptée: {url[:50]}...[/green]")
                return url
            else:
                console.print("[red]❌ URL YouTube invalide. Exemples valides:[/red]")
                console.print("   https://www.youtube.com/watch?v=...")
                console.print("   https://youtu.be/...")
    
    def _ask_local_file(self) -> str:
        """Ask for local file path with validation."""
        while True:
            file_path = Prompt.ask("\n📁 Chemin vers le fichier local")
            
            path = Path(file_path)
            if path.exists() and path.is_file():
                console.print(f"[green]✅ Fichier trouvé: {path.name}[/green]")
                return str(path.absolute())
            else:
                console.print(f"[red]❌ Fichier non trouvé: {file_path}[/red]")
                
                if Confirm.ask("Voulez-vous parcourir le répertoire actuel ?"):
                    self._show_current_directory_files()
    
    def _show_current_directory_files(self):
        """Show files in current directory."""
        console.print("\n[bold]📁 Fichiers dans le répertoire actuel:[/bold]")
        
        media_extensions = {'.mp4', '.mp3', '.wav', '.m4a', '.flac', '.ogg', '.webm', '.mkv', '.avi'}
        media_files = []
        
        for file_path in Path('.').iterdir():
            if file_path.is_file() and file_path.suffix.lower() in media_extensions:
                media_files.append(file_path)
        
        if media_files:
            table = Table(show_header=True, header_style="bold magenta")
            table.add_column("Fichier", style="cyan")
            table.add_column("Taille", style="green")
            table.add_column("Type", style="yellow")
            
            for file_path in sorted(media_files):
                size_mb = file_path.stat().st_size / (1024 * 1024)
                table.add_row(file_path.name, f"{size_mb:.1f} MB", file_path.suffix)
            
            console.print(table)
        else:
            console.print("[yellow]Aucun fichier média trouvé dans le répertoire actuel[/yellow]")
    
    def _ask_output_file(self) -> str:
        """Ask for output file name."""
        console.print("\n[bold]💾 Fichier de sortie:[/bold]")
        
        default_name = "sous_titres.srt"
        output_file = Prompt.ask("Nom du fichier SRT", default=default_name)
        
        # Ensure .srt extension
        if not output_file.endswith('.srt'):
            output_file += '.srt'
        
        # Check if file exists
        if Path(output_file).exists():
            if not Confirm.ask(f"Le fichier {output_file} existe. L'écraser ?"):
                return self._ask_output_file()  # Ask again
        
        return output_file
    
    def _ask_output_directory(self) -> str:
        """Ask for output directory."""
        console.print("\n[bold]📁 Répertoire de sortie:[/bold]")
        
        default_dir = "./output"
        output_dir = Prompt.ask("Répertoire de sortie", default=default_dir)
        
        # Create directory if it doesn't exist
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        console.print(f"[green]✅ Répertoire: {Path(output_dir).absolute()}[/green]")
        
        return output_dir
    
    def _ask_target_language(self) -> str:
        """Ask for target language."""
        languages = {
            "fr": "🇫🇷 Français",
            "en": "🇺🇸 English", 
            "es": "🇪🇸 Español",
            "de": "🇩🇪 Deutsch",
            "it": "🇮🇹 Italiano",
            "pt": "🇵🇹 Português",
            "ru": "🇷🇺 Русский",
            "zh": "🇨🇳 中文",
            "ja": "🇯🇵 日本語",
            "ar": "🇸🇦 العربية"
        }
        
        console.print("\n[bold]🌍 Langue de traduction:[/bold]")
        
        # Show in columns
        lang_list = list(languages.items())
        for i in range(0, len(lang_list), 2):
            left = f"  {lang_list[i][0]}: {lang_list[i][1]}"
            right = f"  {lang_list[i+1][0]}: {lang_list[i+1][1]}" if i+1 < len(lang_list) else ""
            console.print(f"{left:<25} {right}")
        
        lang_code = Prompt.ask(
            "Code langue",
            choices=list(languages.keys()),
            default="fr"
        )
        
        console.print(f"[green]✅ Langue sélectionnée: {languages[lang_code]}[/green]")
        return lang_code
    
    def _ask_quality_level(self) -> str:
        """Ask for quality level."""
        qualities = {
            "fast": ("⚡ Rapide", "Traitement rapide, qualité standard"),
            "balanced": ("⚖️  Équilibré", "Bon compromis vitesse/qualité (recommandé)"),
            "best": ("🏆 Meilleure", "Qualité maximale, traitement plus lent")
        }
        
        console.print("\n[bold]🎯 Niveau de qualité:[/bold]")
        for key, (title, desc) in qualities.items():
            console.print(f"  {key}: {title}")
            console.print(f"      [dim]{desc}[/dim]")
        
        quality = Prompt.ask(
            "Qualité",
            choices=list(qualities.keys()),
            default="balanced"
        )
        
        console.print(f"[green]✅ Qualité: {qualities[quality][0]}[/green]")
        return quality
    
    def _ask_batch_file(self) -> str:
        """Ask for batch file, with option to create one."""
        console.print("\n[bold]📄 Fichier de lot (liste des URLs/fichiers):[/bold]")
        
        if Confirm.ask("Avez-vous déjà un fichier de lot ?"):
            while True:
                batch_file = Prompt.ask("Chemin vers le fichier de lot")
                if Path(batch_file).exists():
                    console.print(f"[green]✅ Fichier de lot: {batch_file}[/green]")
                    return batch_file
                else:
                    console.print(f"[red]❌ Fichier non trouvé: {batch_file}[/red]")
        else:
            return self._create_batch_file()
    
    def _create_batch_file(self) -> str:
        """Interactive batch file creation."""
        console.print("\n[bold]📝 Création d'un fichier de lot:[/bold]")
        
        batch_file = Prompt.ask("Nom du fichier de lot", default="batch_list.txt")
        
        urls = []
        console.print("\nAjoutez vos URLs/fichiers (ligne vide pour terminer):")
        
        while True:
            url = Prompt.ask(f"  Entrée {len(urls)+1} (ou Entrée pour terminer)", default="")
            if not url:
                break
            urls.append(url)
            console.print(f"    [green]✅ Ajouté: {url[:50]}...[/green]")
        
        if urls:
            with open(batch_file, 'w', encoding='utf-8') as f:
                f.write('\n'.join(urls))
            
            console.print(f"[green]✅ Fichier de lot créé: {batch_file} ({len(urls)} entrées)[/green]")
            return batch_file
        else:
            console.print("[red]❌ Aucune entrée, création annulée[/red]")
            return self._ask_batch_file()
    
    def _ask_parallel_workers(self) -> int:
        """Ask for number of parallel workers."""
        import psutil
        max_workers = psutil.cpu_count()
        
        console.print(f"\n[bold]⚡ Traitement parallèle:[/bold]")
        console.print(f"CPU disponibles: {max_workers}")
        console.print("Recommandation: 2-4 workers pour éviter la surcharge")
        
        workers = IntPrompt.ask(
            "Nombre de workers",
            default=min(4, max_workers),
            choices=list(range(1, max_workers + 1))
        )
        
        console.print(f"[green]✅ Workers: {workers}[/green]")
        return workers
    
    def _ask_advanced_options(self):
        """Ask for basic advanced options."""
        console.print("\n[bold]🔧 Options avancées disponibles:[/bold]")
        
        # Model selection
        if Confirm.ask("Choisir le modèle de traduction ?", default=False):
            models = {
                "voxtral-small": "Voxtral Small (24B) - Qualité élevée",
                "voxtral-mini": "Voxtral Mini (3B) - Plus rapide"
            }
            
            console.print("Modèles disponibles:")
            for key, desc in models.items():
                console.print(f"  {key}: {desc}")
            
            model = Prompt.ask("Modèle", choices=list(models.keys()), default="voxtral-small")
            self.args["model"] = model
        
        # Debug mode
        self.args["debug"] = Confirm.ask("Activer le mode debug (plus d'informations) ?", default=False)
        
        # Monitoring
        self.args["monitor"] = Confirm.ask("Activer le monitoring temps réel ?", default=True)
    
    def _configure_gpu_settings(self):
        """Configure GPU-specific settings."""
        console.print("\n[bold]🎮 Configuration GPU:[/bold]")
        
        # GPU memory limit
        if Confirm.ask("Configurer la limite mémoire GPU ?", default=False):
            console.print("Limite mémoire GPU (0.1 = 10%, 0.9 = 90%)")
            memory_limit = Prompt.ask("Limite mémoire", default="0.9")
            try:
                self.args["gpu_memory_limit"] = float(memory_limit)
            except ValueError:
                console.print("[red]Valeur invalide, utilisation de la valeur par défaut[/red]")
        
        # Batch size
        if Confirm.ask("Configurer la taille de lot GPU ?", default=False):
            batch_size = IntPrompt.ask("Taille de lot", default=32)
            self.args["batch_size"] = batch_size
        
        # Precision
        if Confirm.ask("Configurer la précision GPU ?", default=False):
            precisions = ["fp16", "bf16", "fp32"]
            precision = Prompt.ask("Précision", choices=precisions, default="bf16")
            self.args["precision"] = precision
    
    def _configure_audio_settings(self):
        """Configure audio processing settings."""
        console.print("\n[bold]🎵 Configuration Audio:[/bold]")
        
        # VAD threshold
        if Confirm.ask("Configurer la détection de voix (VAD) ?", default=False):
            console.print("Seuil VAD (0.1-0.9, plus bas = plus sensible)")
            vad_threshold = Prompt.ask("Seuil VAD", default="0.3")
            try:
                self.args["vad_threshold"] = float(vad_threshold)
            except ValueError:
                pass
        
        # Minimum segment duration
        if Confirm.ask("Configurer la durée minimale des segments ?", default=False):
            min_duration = Prompt.ask("Durée minimale (secondes)", default="1.0")
            try:
                self.args["min_segment_duration"] = float(min_duration)
            except ValueError:
                pass
    
    def _configure_quality_settings(self):
        """Configure quality and validation settings."""
        console.print("\n[bold]🏆 Configuration Qualité:[/bold]")
        
        # Quality score threshold
        if Confirm.ask("Configurer le score qualité minimal ?", default=False):
            console.print("Score qualité (0.0-1.0, plus haut = plus strict)")
            quality_score = Prompt.ask("Score minimal", default="0.7")
            try:
                self.args["min_quality_score"] = float(quality_score)
            except ValueError:
                pass
        
        # Retry failed segments
        if Confirm.ask("Réessayer les segments échoués ?", default=True):
            self.args["retry_failed_segments"] = True
        
        # Quality validation
        if Confirm.ask("Activer la validation qualité post-traitement ?", default=False):
            self.args["quality_check"] = True
    
    def _configure_monitoring_settings(self):
        """Configure monitoring and debugging settings."""
        console.print("\n[bold]📊 Configuration Monitoring:[/bold]")
        
        # Real-time monitoring
        self.args["monitor"] = Confirm.ask("Interface monitoring temps réel ?", default=True)
        
        # Telemetry
        self.args["telemetry"] = Confirm.ask("Collecte de métriques avancées ?", default=True)
        
        # Debug level
        if Confirm.ask("Configurer le niveau de debug ?", default=False):
            debug_levels = ["INFO", "DEBUG", "WARNING", "ERROR"]
            debug_level = Prompt.ask("Niveau de log", choices=debug_levels, default="INFO")
            self.args["log_level"] = debug_level
        
        # Verbose output
        self.args["verbose"] = Confirm.ask("Sortie détaillée ?", default=False)
    
    def _build_namespace(self) -> argparse.Namespace:
        """Build argparse.Namespace from collected arguments."""
        # Set defaults
        defaults = {
            "target_lang": "fr",
            "quality": "balanced",
            "model": "voxtral-small",
            "debug": False,
            "monitor": True,
            "log_level": "INFO",
            "gpu_memory_limit": 0.9,
            "batch_size": 32,
            "max_workers": 4,
            "continue_on_error": True,
            "skip_existing": True,
            "verbose": False,
            "telemetry": True,
            "retry_failed_segments": True,
            "precision": "bf16",
            "vad_threshold": 0.3,
            "min_segment_duration": 1.0,
            "min_quality_score": 0.7,
            "quality_check": False,
            "dry_run": False,
            "force": False,
            "validate": False
        }
        
        # Update with user choices
        final_args = {**defaults, **self.args}
        
        # Show final configuration summary
        self._show_configuration_summary(final_args)
        
        return argparse.Namespace(**final_args)
    
    def _show_configuration_summary(self, args: Dict[str, Any]):
        """Show final configuration summary."""
        console.print("\n" + "="*60)
        console.print(Panel("📋 [bold]Résumé de la Configuration[/bold]", style="green"))
        
        # Create summary table
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Paramètre", style="cyan", width=25)
        table.add_column("Valeur", style="green", width=30)
        
        # Important settings first
        important_keys = ["url", "output", "batch_list", "output_dir", "target_lang", "quality", "model"]
        other_keys = [k for k in args.keys() if k not in important_keys and args[k] is not None]
        
        for key in important_keys:
            if key in args and args[key] is not None:
                table.add_row(key, str(args[key]))
        
        if other_keys:
            table.add_section()
            for key in other_keys:
                table.add_row(key, str(args[key]))
        
        console.print(table)
        
        # Confirm before proceeding
        if not Confirm.ask("\n✅ Lancer le traitement avec cette configuration ?", default=True):
            console.print("[yellow]Configuration annulée[/yellow]")
            sys.exit(0)


def run_cli_wizard() -> argparse.Namespace:
    """Run the CLI wizard and return configured arguments."""
    wizard = CLIWizard()
    return wizard.run_wizard()


if __name__ == "__main__":
    # Demo wizard
    try:
        args = run_cli_wizard()
        console.print(f"\n[green]Configuration terminée![/green]")
        console.print(f"Arguments: {args}")
    except KeyboardInterrupt:
        console.print("\n[yellow]Assistant annulé par l'utilisateur[/yellow]")
    except Exception as e:
        console.print(f"\n[red]Erreur: {e}[/red]")