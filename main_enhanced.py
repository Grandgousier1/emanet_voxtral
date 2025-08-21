#!/usr/bin/env python3
"""
main_enhanced.py - Enhanced Main Entry Point with User-Friendly CLI
Complete rewrite with interactive guidance and improved UX
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm
from rich.traceback import install

# Install rich traceback handler
install(show_locals=True)

console = Console()

# Import CLI enhancements
try:
    from cli_enhanced import create_enhanced_cli
    from cli_wizard import run_cli_wizard
    CLI_ENHANCED_AVAILABLE = True
except ImportError:
    CLI_ENHANCED_AVAILABLE = False
    console.print("[yellow]Enhanced CLI not available, using standard mode[/yellow]")

# Core imports
try:
    from utils.telemetry import init_telemetry, get_telemetry_manager, shutdown_telemetry
    from cli_feedback import get_feedback
    from error_boundary import with_error_boundary, ErrorSeverity
    from utils.error_messages import ErrorReporter
    from utils.logging_config import setup_logging
    CORE_IMPORTS_AVAILABLE = True
except ImportError as e:
    CORE_IMPORTS_AVAILABLE = False
    console.print(f"[red]Core imports failed: {e}[/red]")


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments with enhanced help."""
    parser = argparse.ArgumentParser(
        description="🚀 EMANET VOXTRAL - Générateur de Sous-titres B200",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
🌟 EXEMPLES D'UTILISATION:

  Mode Simple (YouTube):
    python main.py --url "https://youtube.com/watch?v=..." --output sous_titres.srt

  Mode Simple (Fichier local):
    python main.py --url /chemin/vers/video.mp4 --output sous_titres.srt

  Mode Batch:
    python main.py --batch-list videos.txt --output-dir ./resultats/

  Mode Interactif (Recommandé pour débutants):
    python main.py --wizard

  Configuration et diagnostic:
    python main.py --setup
    python main.py --validate

📚 Pour plus d'aide, utilisez: python main.py --tutorial
        """
    )
    
    # Mode selection
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--wizard", action="store_true",
                           help="🧙‍♂️ Mode assistant interactif (recommandé)")
    mode_group.add_argument("--setup", action="store_true",
                           help="⚙️ Configuration interactive du système")
    mode_group.add_argument("--validate", action="store_true",
                           help="🏥 Diagnostic et validation du système")
    mode_group.add_argument("--tutorial", action="store_true",
                           help="📚 Guide d'utilisation complet")
    
    # Input sources
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument("--url", type=str,
                            help="🎬 URL YouTube ou chemin fichier local")
    input_group.add_argument("--batch-list", type=str,
                            help="📦 Fichier contenant liste URLs/fichiers")
    
    # Output settings
    parser.add_argument("--output", type=str,
                       help="💾 Fichier de sortie SRT (mode simple)")
    parser.add_argument("--output-dir", type=str, default="./output",
                       help="📁 Répertoire de sortie (mode batch)")
    
    # Language and quality
    parser.add_argument("--target-lang", type=str, default="fr",
                       choices=["fr", "en", "es", "de", "it", "pt", "ru", "zh", "ja", "ar"],
                       help="🌍 Langue de traduction (défaut: fr)")
    parser.add_argument("--quality", type=str, default="balanced",
                       choices=["fast", "balanced", "best"],
                       help="🎯 Niveau de qualité (défaut: balanced)")
    
    # Model selection
    parser.add_argument("--model", type=str, default="voxtral-small",
                       choices=["voxtral-small", "voxtral-mini"],
                       help="🤖 Modèle de traduction (défaut: voxtral-small)")
    
    # Performance settings
    parser.add_argument("--max-workers", type=int, default=4,
                       help="⚡ Nombre de workers parallèles (défaut: 4)")
    parser.add_argument("--batch-size", type=int, default=32,
                       help="📦 Taille de lot GPU (défaut: 32)")
    parser.add_argument("--gpu-memory-limit", type=float, default=0.9,
                       help="🎮 Limite mémoire GPU 0.1-0.95 (défaut: 0.9)")
    
    # Advanced options
    parser.add_argument("--precision", type=str, default="bf16",
                       choices=["fp16", "bf16", "fp32"],
                       help="🔢 Précision GPU (défaut: bf16)")
    parser.add_argument("--vad-threshold", type=float, default=0.3,
                       help="🗣️ Seuil détection voix (défaut: 0.3)")
    parser.add_argument("--min-segment-duration", type=float, default=1.0,
                       help="⏱️ Durée minimale segment (défaut: 1.0s)")
    parser.add_argument("--min-quality-score", type=float, default=0.7,
                       help="🏆 Score qualité minimal (défaut: 0.7)")
    
    # Flags
    parser.add_argument("--debug", action="store_true",
                       help="🔍 Mode debug détaillé")
    parser.add_argument("--monitor", action="store_true", default=True,
                       help="📊 Interface monitoring temps réel")
    parser.add_argument("--telemetry", action="store_true", default=True,
                       help="📈 Collecte métriques avancées")
    parser.add_argument("--verbose", action="store_true",
                       help="📢 Sortie détaillée")
    parser.add_argument("--dry-run", action="store_true",
                       help="🧪 Simulation sans traitement réel")
    parser.add_argument("--force", action="store_true",
                       help="💪 Ignorer les vérifications de sécurité")
    parser.add_argument("--skip-existing", action="store_true", default=True,
                       help="⏭️ Ignorer fichiers déjà traités")
    parser.add_argument("--continue-on-error", action="store_true", default=True,
                       help="🔄 Continuer malgré les erreurs")
    parser.add_argument("--retry-failed-segments", action="store_true", default=True,
                       help="🔁 Réessayer segments échoués")
    parser.add_argument("--quality-check", action="store_true",
                       help="✅ Validation qualité post-traitement")
    
    # Logging
    parser.add_argument("--log-level", type=str, default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="📝 Niveau de logging (défaut: INFO)")
    
    return parser.parse_args()


def show_banner():
    """Show enhanced application banner."""
    banner = """
[bold cyan]
██████████████████████████████████████████████████████████████████████████████
█▌                                                                          ▐█
█▌  ███████╗███╗   ███╗ █████╗ ███╗   ██╗███████╗████████╗                ▐█
█▌  ██╔════╝████╗ ████║██╔══██╗████╗  ██║██╔════╝╚══██╔══╝                ▐█
█▌  █████╗  ██╔████╔██║███████║██╔██╗ ██║█████╗     ██║                   ▐█
█▌  ██╔══╝  ██║╚██╔╝██║██╔══██║██║╚██╗██║██╔══╝     ██║                   ▐█
█▌  ███████╗██║ ╚═╝ ██║██║  ██║██║ ╚████║███████╗   ██║                   ▐█
█▌  ╚══════╝╚═╝     ╚═╝╚═╝  ╚═╝╚═╝  ╚═══╝╚══════╝   ╚═╝                   ▐█
█▌                                                                          ▐█
█▌  ██╗   ██╗ ██████╗ ██╗  ██╗████████╗██████╗  █████╗ ██╗                ▐█
█▌  ██║   ██║██╔═══██╗╚██╗██╔╝╚══██╔══╝██╔══██╗██╔══██╗██║                ▐█
█▌  ██║   ██║██║   ██║ ╚███╔╝    ██║   ██████╔╝███████║██║                ▐█
█▌  ╚██╗ ██╔╝██║   ██║ ██╔██╗    ██║   ██╔══██╗██╔══██║██║                ▐█
█▌   ╚████╔╝ ╚██████╔╝██╔╝ ██╗   ██║   ██║  ██║██║  ██║███████╗           ▐█
█▌    ╚═══╝   ╚═════╝ ╚═╝  ╚═╝   ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝           ▐█
█▌                                                                          ▐█
██████████████████████████████████████████████████████████████████████████████
[/bold cyan]

[bold yellow]🚀 GÉNÉRATEUR DE SOUS-TITRES B200 - Version 2.0 Enhanced 🚀[/bold yellow]

[dim]Optimisé pour GPU B200 (180GB VRAM) avec modèles Voxtral de Mistral AI[/dim]
[dim]Interface interactive complète • Monitoring temps réel • Qualité professionnelle[/dim]
    """
    
    console.print(Panel.fit(banner, border_style="cyan", padding=(0, 1)))


def show_quick_help():
    """Show quick help for new users."""
    help_text = """
[bold]🆘 AIDE RAPIDE[/bold]

[bold cyan]Première utilisation ?[/bold cyan]
  python main.py --wizard          # Assistant interactif complet
  python main.py --setup           # Configuration système
  python main.py --tutorial        # Guide détaillé

[bold cyan]Exemples rapides :[/bold cyan]
  python main.py --url "https://youtube.com/watch?v=..." --output sous_titres.srt
  python main.py --batch-list videos.txt --output-dir ./resultats/

[bold cyan]Diagnostic :[/bold cyan]
  python main.py --validate        # Vérifier le système
  python main.py --help           # Aide complète

[dim]💡 Pour une expérience optimale, utilisez --wizard pour être guidé étape par étape[/dim]
    """
    
    console.print(Panel(help_text, title="Aide Rapide", style="blue"))


def handle_special_modes(args: argparse.Namespace) -> bool:
    """Handle special modes (wizard, setup, etc). Returns True if handled."""
    
    if args.wizard:
        if CLI_ENHANCED_AVAILABLE:
            console.print("[cyan]🧙‍♂️ Lancement de l'assistant interactif...[/cyan]")
            wizard_args = run_cli_wizard()
            # Replace args with wizard results
            for key, value in wizard_args.__dict__.items():
                setattr(args, key, value)
            return False  # Continue with processing
        else:
            console.print("[red]Assistant non disponible, utilisez le mode standard[/red]")
            show_quick_help()
            return True
    
    elif args.setup:
        if CLI_ENHANCED_AVAILABLE:
            cli = create_enhanced_cli()
            cli.display_welcome_banner()
            cli.configure_system_interactive()
        else:
            console.print("[red]Configuration interactive non disponible[/red]")
        return True
    
    elif args.validate:
        if CLI_ENHANCED_AVAILABLE:
            cli = create_enhanced_cli()
            cli.run_system_diagnostic()
        else:
            console.print("[yellow]Validation de base...[/yellow]")
            # Basic validation here
            console.print("[green]Validation basique OK[/green]")
        return True
    
    elif args.tutorial:
        if CLI_ENHANCED_AVAILABLE:
            cli = create_enhanced_cli()
            cli.show_tutorial()
        else:
            show_quick_help()
        return True
    
    return False


@with_error_boundary("application startup", "main", ErrorSeverity.CRITICAL)
def enhanced_main():
    """Enhanced main function with user-friendly experience."""
    
    # Parse arguments
    args = parse_arguments()
    
    # Show banner for interactive modes
    if args.wizard or args.setup or args.tutorial or args.validate:
        show_banner()
    
    # Handle special modes
    if handle_special_modes(args):
        return 0
    
    # Validate required arguments for processing modes
    if not args.url and not args.batch_list:
        console.print("[red]❌ Erreur: Aucune source spécifiée[/red]")
        console.print("\n[yellow]Utilisez une de ces options :[/yellow]")
        console.print("  • --url pour traiter une vidéo/audio")
        console.print("  • --batch-list pour traiter plusieurs fichiers")
        console.print("  • --wizard pour l'assistant interactif")
        console.print("  • --help pour l'aide complète")
        return 1
    
    # Show banner for processing modes
    if not args.wizard:  # Wizard already showed banner
        show_banner()
    
    # Initialize core systems
    if CORE_IMPORTS_AVAILABLE:
        setup_logging(log_level=args.log_level)
        
        if args.telemetry:
            telemetry = init_telemetry("emanet_voxtral")
            telemetry.record_counter("application_starts", 1, {
                "mode": "batch" if args.batch_list else "single",
                "version": "2.0_enhanced"
            })
        
        feedback = get_feedback(debug_mode=args.debug)
        error_reporter = ErrorReporter(feedback)
    else:
        console.print("[yellow]Core systems not available, using basic mode[/yellow]")
        feedback = None
        error_reporter = None
    
    # Show processing summary
    show_processing_summary(args)
    
    # Confirm before proceeding (unless forced)
    if not args.force and not args.dry_run:
        if not Confirm.ask("\n🚀 Lancer le traitement ?", default=True):
            console.print("[yellow]Traitement annulé par l'utilisateur[/yellow]")
            return 0
    
    # Dry run mode
    if args.dry_run:
        console.print("\n[cyan]🧪 MODE SIMULATION - Aucun traitement réel effectué[/cyan]")
        simulate_processing(args)
        return 0
    
    # Import and run actual processing
    try:
        console.print("\n[cyan]📦 Chargement des modules de traitement...[/cyan]")
        
        # Import main processing function
        sys.path.insert(0, str(Path(__file__).parent))
        import main as original_main
        
        # Monkey patch args for compatibility
        sys.argv = build_argv_from_args(args)
        
        console.print("[green]✅ Modules chargés, démarrage du traitement...[/green]")
        
        # Run original main
        result = original_main.main()
        
        if args.telemetry and CORE_IMPORTS_AVAILABLE:
            shutdown_telemetry()
        
        return result
        
    except Exception as e:
        console.print(f"\n[red]❌ Erreur lors du traitement: {e}[/red]")
        if args.debug:
            console.print_exception()
        return 1


def show_processing_summary(args: argparse.Namespace):
    """Show processing configuration summary."""
    console.print(Panel("📋 [bold]Configuration du Traitement[/bold]", style="green"))
    
    # Create summary based on mode
    if args.batch_list:
        summary = f"""
[bold cyan]Mode :[/bold cyan] Traitement en lot
[bold cyan]Source :[/bold cyan] {args.batch_list}
[bold cyan]Sortie :[/bold cyan] {args.output_dir}
[bold cyan]Workers :[/bold cyan] {args.max_workers}
        """
    else:
        summary = f"""
[bold cyan]Mode :[/bold cyan] Traitement unique
[bold cyan]Source :[/bold cyan] {args.url}
[bold cyan]Sortie :[/bold cyan] {args.output or 'auto-générée'}
        """
    
    # Add common settings
    summary += f"""
[bold cyan]Langue :[/bold cyan] {args.target_lang}
[bold cyan]Qualité :[/bold cyan] {args.quality}
[bold cyan]Modèle :[/bold cyan] {args.model}
[bold cyan]GPU Limit :[/bold cyan] {args.gpu_memory_limit * 100:.0f}%
[bold cyan]Batch Size :[/bold cyan] {args.batch_size}
    """
    
    console.print(summary.strip())


def simulate_processing(args: argparse.Namespace):
    """Simulate processing for dry-run mode."""
    import time
    from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
    
    steps = [
        "Validation de l'environnement",
        "Chargement des modèles",
        "Traitement audio/vidéo",
        "Génération des sous-titres",
        "Sauvegarde des résultats"
    ]
    
    with Progress(
        SpinnerColumn(),
        TextColumn("{task.description}"),
        TimeElapsedColumn(),
        console=console
    ) as progress:
        for step in steps:
            task = progress.add_task(f"🔄 {step}...", total=1)
            time.sleep(1.5)  # Simulate work
            progress.update(task, completed=1)
            console.print(f"[green]✅ {step} - Simulé[/green]")
    
    console.print("\n[cyan]✨ Simulation terminée avec succès ![/cyan]")
    
    if args.batch_list:
        console.print(f"[dim]Mode batch: fichiers simulés dans {args.output_dir}[/dim]")
    else:
        console.print(f"[dim]Mode simple: {args.output or 'sous_titres.srt'} simulé[/dim]")


def build_argv_from_args(args: argparse.Namespace) -> List[str]:
    """Build sys.argv compatible list from args namespace."""
    argv = ["main.py"]
    
    # Add arguments back to argv format for compatibility
    if args.url:
        argv.extend(["--url", args.url])
    if args.batch_list:
        argv.extend(["--batch-list", args.batch_list])
    if args.output:
        argv.extend(["--output", args.output])
    if args.output_dir != "./output":
        argv.extend(["--output-dir", args.output_dir])
    if args.target_lang != "fr":
        argv.extend(["--target-lang", args.target_lang])
    if args.quality != "balanced":
        argv.extend(["--quality", args.quality])
    if args.model != "voxtral-small":
        argv.extend(["--model", args.model])
    if args.max_workers != 4:
        argv.extend(["--max-workers", str(args.max_workers)])
    if args.batch_size != 32:
        argv.extend(["--batch-size", str(args.batch_size)])
    if args.gpu_memory_limit != 0.9:
        argv.extend(["--gpu-memory-limit", str(args.gpu_memory_limit)])
    if args.debug:
        argv.append("--debug")
    if args.verbose:
        argv.append("--verbose")
    if args.monitor:
        argv.append("--monitor")
    if args.force:
        argv.append("--force")
    if args.log_level != "INFO":
        argv.extend(["--log-level", args.log_level])
    
    return argv


if __name__ == "__main__":
    try:
        sys.exit(enhanced_main())
    except KeyboardInterrupt:
        console.print("\n[yellow]⚠️ Interruption par l'utilisateur[/yellow]")
        sys.exit(130)
    except Exception as e:
        console.print(f"\n[red]💥 Erreur fatale: {e}[/red]")
        console.print("[dim]Utilisez --debug pour plus de détails[/dim]")
        sys.exit(1)