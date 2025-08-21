#!/usr/bin/env python3
"""
Configuration applicative centralisée pour Voxtral
"""

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any

from domain_models import ProcessingConfig, ErrorSeverity
from cli_feedback import CLIFeedback, get_feedback  
from error_boundary import ErrorContext


@dataclass(frozen=True)
class AppConfig:
    """Configuration immutable de l'application."""
    args: argparse.Namespace
    feedback: CLIFeedback
    start_time: float
    processing_config: ProcessingConfig
    
    @property
    def is_validation_mode(self) -> bool:
        """Vérifie si on est en mode validation uniquement."""
        return self.args.validate_only or self.args.dry_run
    
    @property  
    def requires_processing(self) -> bool:
        """Vérifie si le traitement de média est requis."""
        return not self.is_validation_mode and (self.args.url or self.args.batch_list)
    
    @property
    def output_path(self) -> Optional[Path]:
        """Chemin de sortie calculé."""
        if not self.args.output:
            return None
        return Path(self.args.output)


def create_app_config() -> AppConfig:
    """
    Crée la configuration de l'application de manière déterministe.
    
    Returns:
        Configuration immutable de l'application
    """
    from utils.logging_config import setup_logging
    
    args = parse_args()
    
    # Configuration du logging avant tout
    setup_logging(log_level=args.log_level.upper())
    
    # Configuration du feedback
    quiet_mode = getattr(args, 'quiet', False)
    feedback = get_feedback(debug_mode=args.debug, quiet_mode=quiet_mode)
    
    # Configuration de traitement
    processing_config = ProcessingConfig(
        quality_level=getattr(args, 'quality', 'balanced'),
        target_language='French',  # Default pour Turkish->French
        max_workers=getattr(args, 'max_workers', 0) or detect_optimal_workers(),
        gpu_memory_limit=getattr(args, 'gpu_memory_limit', 0.0) or detect_optimal_gpu_limit(),
        timeout_seconds=getattr(args, 'timeout', 3600)
    )
    
    return AppConfig(
        args=args,
        feedback=feedback,
        start_time=time.time(),
        processing_config=processing_config
    )


def parse_args() -> argparse.Namespace:
    """Parse les arguments de ligne de commande avec validation."""
    parser = argparse.ArgumentParser(
        description="Emanet subtitle generator with Voxtral (B200 optimized)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --url "https://youtube.com/watch?v=abc123" --output "episode.srt"
  %(prog)s --batch-list "videos.txt" --output-dir "./subtitles/"
  %(prog)s --setup-auth  # Setup HuggingFace authentication
  %(prog)s --quality best --format vtt --url "video.mp4"
  %(prog)s --validate-only  # Run validation tests only

For detailed help and examples:
  python cli_help.py          # Interactive guide
  python cli_help.py auth     # Authentication help
  python cli_help.py b200     # B200 optimization guide
  python auth_setup.py        # Standalone auth setup
        """
    )
    
    # Arguments principaux
    parser.add_argument('--url', '--youtube-url', 
                       help='YouTube URL or local file path')
    parser.add_argument('--output', '--out', 
                       help='Output SRT file path')
    parser.add_argument('--batch-list', 
                       help='Text file with URLs/paths, one per line')
    parser.add_argument('--output-dir', '--out-dir', default='./output',
                       help='Output directory for batch processing')
    
    # Options de traitement
    parser.add_argument('--dry-run', action='store_true',
                       help='Run preflight checks only')
    parser.add_argument('--validate-only', action='store_true',
                       help='Run full validation suite only')
    parser.add_argument('--force', action='store_true',
                       help='Skip validation and proceed anyway (dangerous)')
    
    # Authentification
    parser.add_argument('--setup-auth', action='store_true',
                       help='Setup HuggingFace authentication (interactive)')
    parser.add_argument('--hf-token',
                       help='HuggingFace token (prefer --setup-auth for secure input)')
    
    # Logging et debug
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug logging')
    parser.add_argument('--log-level', default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                       help='Set the logging level')
    parser.add_argument('--quiet', '-q', action='store_true',
                       help='Minimal output (errors only)')
    
    # Options avancées
    parser.add_argument('--no-cache', action='store_true',
                       help='Disable model and audio caching')
    parser.add_argument('--max-workers', type=int,
                       help='Maximum number of parallel workers (auto-detected if not specified)')
    parser.add_argument('--gpu-memory-limit', type=float,
                       help='GPU memory utilization limit (0.1-0.95, default: auto)')
    parser.add_argument('--timeout', type=int, default=3600,
                       help='Overall processing timeout in seconds (default: 3600)')
    
    # Options de sortie
    parser.add_argument('--format', choices=['srt', 'vtt', 'json'], default='srt',
                       help='Output format')
    parser.add_argument('--quality', choices=['fast', 'balanced', 'best'], default='balanced',
                       help='Quality/speed tradeoff')
    
    # Options modèle
    parser.add_argument('--use-voxtral-mini', action='store_true',
                       help='Use Voxtral Mini instead of Small')
    
    # Options pour cookies anti-bot
    parser.add_argument('--cookies',
                       help='Browser cookies file for anti-bot protection')
    
    return parser.parse_args()


def detect_optimal_workers() -> int:
    """Détecte le nombre optimal de workers basé sur le hardware."""
    import os
    
    # Utilise les variables d'environnement ou détection automatique
    if env_workers := os.getenv('MAX_WORKERS'):
        return int(env_workers)
    
    try:
        from config import detect_hardware
        hw = detect_hardware()
        # B200 a 28 vCPU, utilise 75% pour éviter la surcharge
        return max(1, int(hw.get('cpu_count', 4) * 0.75))
    except ImportError:
        return 4  # Fallback conservateur


def detect_optimal_gpu_limit() -> float:
    """Détecte la limite GPU optimale basée sur le hardware."""
    import os
    
    if env_limit := os.getenv('GPU_MEMORY_LIMIT'):
        return float(env_limit)
    
    try:
        from config import detect_hardware
        hw = detect_hardware()
        gpu_memory_gb = hw.get('gpu_memory_gb', 0)
        
        # B200 avec 180GB peut utiliser 95%, autres GPUs plus conservateurs
        if gpu_memory_gb >= 100:  # B200
            return 0.95
        elif gpu_memory_gb >= 40:  # A100/H100
            return 0.85
        elif gpu_memory_gb >= 16:  # GPU moyens
            return 0.75
        else:
            return 0.65  # GPU plus petits
    except ImportError:
        return 0.75  # Fallback raisonnable


class ConfigValidationError(Exception):
    """Erreur de validation de configuration."""
    
    def __init__(self, message: str, context: Optional[ErrorContext] = None):
        super().__init__(message)
        self.context = context


def validate_config(config: AppConfig) -> None:
    """
    Valide la configuration de l'application.
    
    Args:
        config: Configuration à valider
        
    Raises:
        ConfigValidationError: Si la configuration est invalide
    """
    errors = []
    
    # Validation des arguments mutuellement exclusifs
    if config.args.url and config.args.batch_list:
        errors.append("--url et --batch-list sont mutuellement exclusifs")
    
    if not config.args.url and not config.args.batch_list and not config.args.setup_auth and not config.args.validate_only:
        errors.append("Une source d'entrée est requise: --url, --batch-list, --setup-auth ou --validate-only")
    
    # Validation des plages de valeurs
    if config.processing_config.gpu_memory_limit < 0.1 or config.processing_config.gpu_memory_limit > 0.95:
        errors.append(f"GPU memory limit doit être entre 0.1 et 0.95, reçu: {config.processing_config.gpu_memory_limit}")
    
    if config.processing_config.max_workers < 1:
        errors.append(f"Max workers doit être >= 1, reçu: {config.processing_config.max_workers}")
    
    if config.processing_config.timeout_seconds < 60:
        errors.append(f"Timeout doit être >= 60s, reçu: {config.processing_config.timeout_seconds}")
    
    # Validation des chemins si spécifiés
    if config.args.output_dir:
        output_dir = Path(config.args.output_dir)
        if output_dir.exists() and not output_dir.is_dir():
            errors.append(f"Output directory existe mais n'est pas un répertoire: {output_dir}")
    
    if errors:
        error_msg = "Erreurs de configuration:\n" + "\n".join(f"  - {error}" for error in errors)
        context = ErrorContext(
            operation="configuration validation",
            component="app_config",
            severity=ErrorSeverity.CRITICAL
        )
        raise ConfigValidationError(error_msg, context)