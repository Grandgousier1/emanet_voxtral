
"""
run_b200_validation.py - Orchestrateur pour la suite de validation B200.

Ce script fournit une expérience utilisateur guidée pour l'ensemble des tests
de compatibilité, de fonctionnalité et de performance pour l'architecture NVIDIA B200.
"""

import subprocess
import sys
import json
import time
from typing import List, Dict, Tuple

from cli_feedback import get_feedback
from rich.table import Table


def run_command(command: List[str], title: str, feedback) -> Tuple[bool, str]:
    """Exécute une commande et stream sa sortie en temps réel."""
    feedback.info(title)
    try:
        process = subprocess.Popen(
            command, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT, 
            text=True, 
            encoding='utf-8'
        )

        output_lines = []
        while True:
            line = process.stdout.readline()
            if not line:
                break
            # Affiche la sortie en temps réel tout en la capturant
            print(f"    {line.strip()}", flush=True)
            output_lines.append(line.strip())
        
        process.wait()
        full_output = "\n".join(output_lines)

        if process.returncode == 0:
            feedback.success(f"{title} terminé avec succès.")
            return True, full_output
        else:
            feedback.display_error_panel(
                what=f"L'étape '{title}' a échoué.",
                why=f"La commande `{' '.join(command)}` a retourné un code d'erreur: {process.returncode}.",
                how="Consultez la sortie de la commande ci-dessus pour les détails de l'erreur."
            )
            return False, full_output

    except FileNotFoundError:
        feedback.display_error_panel(
            what=f"Commande introuvable pour l'étape '{title}'.",
            why=f"Le programme '{command[0]}' n'est probablement pas installé ou pas dans le PATH.",
            how="Assurez-vous que Python et les dépendances du projet sont correctement installés."
        )
        return False, ""
    except Exception as e:
        feedback.display_error_panel(
            what=f"Une exception inattendue est survenue durant '{title}'.",
            why=str(e),
            how="Consultez le traceback et les logs pour plus de détails."
        )
        return False, str(e)

def main():
    """Point d'entrée du script d'orchestration."""
    feedback = get_feedback()
    feedback.display_welcome_panel({
        'url': None,
        'batch_list': 'Suite de Validation B200',
        'output_dir': 'N/A'
    })

    results = {}
    all_passed = True

    # --- Étape 1: Tests Atomiques --- 
    feedback.major_step(1, 3, "Test de l'API matérielle atomique")
    success, _ = run_command([sys.executable, "test_b200_api_atomic.py"], "Test API B200", feedback)
    results['atomic_test'] = {'passed': success}
    if not success:
        all_passed = False

    # --- Étape 2: Tests Fonctionnels --- 
    if all_passed:
        feedback.major_step(2, 3, "Lancement des tests fonctionnels (pytest)")
        success, output = run_command([sys.executable, "-m", "pytest", "-m", "b200", "-vv", "--showlocals"], "Pytest B200", feedback)
        results['functional_tests'] = {'passed': success, 'output': output}
        if not success:
            all_passed = False

    # --- Étape 3: Benchmarks --- 
    if all_passed:
        benchmark_output_file = "b200_benchmark.json"
        feedback.major_step(3, 3, "Lancement des benchmarks de performance")
        success, _ = run_command([sys.executable, "benchmark.py", "--b200-only", "--output", benchmark_output_file], "Benchmark B200", feedback)
        results['benchmark'] = {'passed': success, 'file': benchmark_output_file}
        if not success:
            all_passed = False

    # --- Étape 4: Résumé Final --- 
    if all_passed:
        # Lire les résultats du benchmark
        try:
            with open(results['benchmark']['file']) as f:
                bench_data = json.load(f)
            # Supposons une structure simple pour l'exemple
            inference_speed = bench_data.get('inference_speed_tokens_per_sec', 'N/A')
            vram_used = bench_data.get('peak_vram_gb', 'N/A')
        except (FileNotFoundError, json.JSONDecodeError):
            inference_speed = 'N/A'
            vram_used = 'N/A'

        summary_table = Table(title="Synthèse de la Validation B200", show_header=False, box=None)
        summary_table.add_row("[bold]▶ Test API Matérielle[/bold]", "[green]✅ Passé[/green]")
        summary_table.add_row("[bold]▶ Tests Fonctionnels[/bold]", "[green]✅ Passés[/green]")
        summary_table.add_row("[bold]▶ Benchmark[/bold]", "[green]✅ Terminé[/green]")
        summary_table.add_row("    [dim]• Vitesse Inférence[/dim]", f"[bold cyan]{inference_speed} tokens/s[/bold cyan]")
        summary_table.add_row("    [dim]• Utilisation VRAM[/dim]", f"[bold cyan]{vram_used} Go[/bold cyan]")

        feedback._print_panel(
            summary_table,
            title="✅ Validation B200 Terminée",
            style="green"
        )
        feedback.info("Conclusion : Le projet est prêt et optimisé pour le matériel B200.")
        sys.exit(0)
    else:
        feedback.display_error_panel(
            what="La suite de validation B200 a échoué.",
            why="Au moins une étape critique n'a pas réussi.",
            how="Veuillez analyser les logs ci-dessus pour identifier la cause de l'échec."
        )
        sys.exit(1)

if __name__ == "__main__":
    main()
