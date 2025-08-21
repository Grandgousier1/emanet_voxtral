

import time
from typing import List, Any, Optional, Dict

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeRemainingColumn, DownloadColumn, TransferSpeedColumn
from rich.status import Status
from rich.syntax import Syntax
from rich.table import Table

console = Console()

class CLIFeedback:
    """A centralized class for providing rich, user-friendly CLI feedback."""

    def __init__(self, debug_mode: bool = False):
        self.debug_mode = debug_mode

    def _print_panel(self, content, title, style):
        """Helper to print a consistent panel."""
        console.print(
            Panel(
                content,
                title=title,
                title_align="left",
                border_style=style,
                expand=False
            )
        )

    def display_welcome_panel(self, args: Dict):
        """Displays a welcome panel summarizing the requested task."""
        summary = ""
        if args.get('url'):
            summary += f"[bold]▶ Tâche[/bold]      : Traitement d'une source unique\n"
            summary += f"[bold]▶ Source[/bold]     : {args['url']}\n"
            summary += f"[bold]▶ Sortie[/bold]     : {args['output']}"
        elif args.get('batch_list'):
            summary += f"[bold]▶ Tâche[/bold]      : Traitement par lot\n"
            summary += f"[bold]▶ Fichier lot[/bold]: {args['batch_list']}\n"
            summary += f"[bold]▶ Dossier[/bold]    : {args['output_dir']}"

        self._print_panel(summary, "🚀 Voxtral B200 - Générateur de Sous-titres", "cyan")

    def display_success_panel(self, time_taken: float, output_path: str, segments_count: int):
        """Displays a success panel upon completion."""
        summary = f"[bold]▶ Fichier de sortie[/bold] : {output_path}\n"
        summary += f"[bold]▶ Segments traités[/bold]  : {segments_count}\n"
        summary += f"[bold]▶ Temps total[/bold]       : {time_taken:.2f} secondes"
        self._print_panel(summary, "✅ Succès !", "green")

    def display_error_panel(self, what: str, why: str, how: str):
        """Displays a structured, user-friendly error panel."""
        content = f"[bold]QUOI ?[/bold]\n{what}\n\n"
        content += f"[bold]POURQUOI ?[/bold]\n{why}\n\n"
        content += f"[bold]COMMENT RÉSOUDRE ?[/bold]\n{how}"
        self._print_panel(content, "❌ ERREUR CRITIQUE", "red")

    def display_health_dashboard(self, check_results: List[Dict]):
        """Displays a system health dashboard table."""
        table = Table(title="Tableau de Bord de Santé du Système", show_header=True, header_style="bold magenta")
        table.add_column("Composant", style="cyan")
        table.add_column("Statut", justify="center")
        table.add_column("Détail", justify="left")

        for check in check_results:
            status_icon = "✅" if check['status'] else "❌"
            table.add_row(check['check'], status_icon, check['value'])
        
        console.print(table)

    def major_step(self, step_num: int, total_steps: int, description: str):
        """Prints a major step title."""
        console.print(f"\n[cyan bold][Étape {step_num}/{total_steps}] {description}...[/cyan]")

    def info(self, message: str):
        """Prints an informational message."""
        console.print(f"  [cyan]ℹ {message}[/cyan]")

    def success(self, message: str):
        """Prints a success message."""
        console.print(f"  [green]✔ {message}[/green]")

    def warning(self, message: str):
        """Prints a warning message."""
        console.print(f"  [yellow]⚠️ {message}[/yellow]")

    def debug(self, message: str):
        """Prints a debug message if debug mode is on."""
        if self.debug_mode:
            console.print(f"    [dim]🔍 {message}[/dim]")

    def download_progress(self) -> Progress:
        """Returns a rich Progress instance configured for downloads."""
        return Progress(
            TextColumn("[bold blue]{task.description}", justify="right"),
            BarColumn(bar_width=None),
            "[progress.percentage]{task.percentage:>3.0f}%",
            "•",
            DownloadColumn(),
            "•",
            TransferSpeedColumn(),
            "•",
            TimeRemainingColumn(),
            console=console
        )

    def processing_progress(self) -> Progress:
        """Returns a rich Progress instance configured for processing items."""
        return Progress(
            TextColumn("[bold blue]{task.description}", justify="right"),
            BarColumn(bar_width=None),
            "[progress.percentage]{task.percentage:>3.0f}%",
            "•",
            TextColumn("{task.completed}/{task.total} traités"),
            "•",
            TimeRemainingColumn(),
            console=console
        )

# Global feedback instance management
_global_feedback = None

def get_feedback(debug_mode: bool = False) -> CLIFeedback:
    """Get global feedback instance."""
    global _global_feedback
    if _global_feedback is None:
        _global_feedback = CLIFeedback(debug_mode=debug_mode)
    elif debug_mode:
        _global_feedback.debug_mode = True
    return _global_feedback
