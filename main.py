# Standard library imports
import argparse
import asyncio
import gc
import hashlib
import json
import logging
import os
import shutil
import sqlite3
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Logging is configured via setup_logging function.
# Avoid global basicConfig.
logger = logging.getLogger(__name__)

# Third-party imports - with fallback protection
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

# Rich console - with fallback
try:
    from rich.console import Console
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    # Fallback Console class
    class Console:
        def print(self, *args, **kwargs):
            print(*args)
        def input(self, *args, **kwargs):
            return input(*args)

# Third-party imports - with fallbacks
try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

try:
    import yt_dlp
    YT_DLP_AVAILABLE = True
except ImportError:
    YT_DLP_AVAILABLE = False

try:
    import soundfile as sf
    SOUNDFILE_AVAILABLE = True
except ImportError:
    SOUNDFILE_AVAILABLE = False
    logger.warning("soundfile not available - install with: pip install soundfile>=0.12.0")

# Local imports - constants and domain models (always required)
from constants import CACHE_DB, CHANNELS, SAMPLE_RATE, VOXTRAL_MINI, VOXTRAL_SMALL
from domain_models import ErrorContext, ErrorSeverity, ModelConfig, ProcessingConfig
from error_boundary import error_boundary, with_error_boundary

# Critical local imports validation
CRITICAL_MODULES = [
    'cli_feedback', 'config', 'utils.gpu_utils', 'utils.error_messages'
]

for module_name in CRITICAL_MODULES:
    try:
        __import__(module_name)
    except ImportError as e:
        # Use lazy formatting for logger
        logger.critical("Missing required module %s: %s", module_name, e)
        sys.exit(1)

# Local imports - now safe to import
from cli_feedback import CLIFeedback, get_feedback
from config import detect_hardware, get_optimal_config, setup_runpod_environment
# Try to import auth_manager, fallback to simple version if issues
try:
    from utils.auth_manager import TokenManager, cli_auth_setup
except Exception:
    # Fallback to simple token management if auth_manager has issues
    from simple_token_manager import SimpleTokenManager as TokenManager
    cli_auth_setup = None
from utils.error_messages import ErrorReporter
from utils.gpu_utils import available_device, check_cuda_available, free_cuda_mem, gpu_mem_info
from utils.memory_manager import get_memory_manager
from utils.model_utils import ModelManager

# Optional local imports with graceful degradation
try:
    from parallel_processor import get_disk_manager, get_optimized_processor
    from utils.audio_utils import enhanced_download_audio, enhanced_vad_segments
    from utils.processing_utils import (
        _validate_local_path_security, enhanced_process_batch,
        enhanced_process_single_video, enhanced_voxtral_process,
        get_audio_path, process_audio
    )
    from utils.srt_utils import enhanced_generate_srt, format_srt_time
    from utils.validation_utils import enhanced_preflight_checks
    from validator import CodeValidator
    from voxtral_prompts import (
        TURKISH_DRAMA_PROMPTS, get_transformers_generation_params,
        get_vllm_generation_params, get_voxtral_prompt,
        optimize_subtitle_timing, validate_translation_quality
    )
    FULL_FUNCTIONALITY = True
except ImportError as e:
    logger.warning("Some functionality will be limited due to missing modules: %s", e)
    FULL_FUNCTIONALITY = False

# Remaining optional imports
try:
    from utils.logging_config import setup_logging
    from utils.reproducibility import ReproducibleSession, ensure_reproducible_environment
    from utils.security_utils import validate_hf_token
    from utils.system_utils import check_disk_space
    from utils.telemetry import get_telemetry_manager, init_telemetry, shutdown_telemetry, telemetry_decorator
except ImportError as e:
    # These are truly optional - provide minimal fallbacks
    def setup_logging(log_level: str = 'INFO', **kwargs) -> None: pass
    def validate_hf_token() -> bool: return True
    def check_disk_space(path: str, required_gb: int = 1) -> bool: return True
    def init_telemetry(name: str) -> Any: return type('MockTelemetry', (), {'record_counter': lambda *a, **k: None})()

console = Console()

# ... (rest of the file remains the same until main)

# The old main and run_processing functions will be replaced entirely.

@with_error_boundary("application startup", "main", ErrorSeverity.CRITICAL)
def setup_application(args: argparse.Namespace) -> Tuple[CLIFeedback, ErrorReporter]:
    """Setup application services and validate environment."""
    setup_logging(log_level=args.log_level.upper())
    
    # Initialize telemetry system
    telemetry = init_telemetry("emanet_voxtral")
    telemetry.record_counter("application_starts", 1, {"version": "1.0.0"})
    
    feedback = get_feedback(debug_mode=args.debug)
    error_reporter = ErrorReporter(feedback)
    
    # Attempt interactive setup if config is missing
    setup_interactive_configuration()
    feedback.display_welcome_panel(vars(args))
    
    return feedback, error_reporter


@with_error_boundary("environment validation", "preflight", ErrorSeverity.CRITICAL)
def validate_environment(args: argparse.Namespace, feedback: CLIFeedback) -> bool:
    """Validate system environment and dependencies."""
    if args.force:
        return True
        
    feedback.major_step(1, 4, "Validation de l'environnement")
    health_checks = []
    
    # Token validation
    token_ok = validate_hf_token()
    health_checks.append({
        'check': 'Token Hugging Face', 
        'status': token_ok, 
        'value': 'Valide' if token_ok else 'Invalide'
    })
    if not token_ok:
        raise ValueError("INVALID_HF_TOKEN")

    # Disk space validation  
    required_disk_gb = 25
    available_gb = (shutil.disk_usage('.').free / (1024**3))
    disk_ok = check_disk_space('.', required_gb=required_disk_gb)
    health_checks.append({
        'check': 'Espace Disque (>25 Go)', 
        'status': disk_ok, 
        'value': f'{available_gb:.2f} Go libres'
    })
    if not disk_ok:
        raise RuntimeError(f"INSUFFICIENT_DISK_SPACE: required={required_disk_gb}, available={available_gb}")

    # Dependencies validation
    preflight_ok = enhanced_preflight_checks(feedback)
    health_checks.append({
        'check': 'Dépendances & GPU', 
        'status': preflight_ok, 
        'value': 'Détecté et compatible' if preflight_ok else 'Problème détecté'
    })
    if not preflight_ok:
        raise RuntimeError("PREFLIGHT_CHECKS_FAILED")

    feedback.display_health_dashboard(health_checks)
    feedback.info("Validation de l'environnement terminée avec succès.")
    return True


@with_error_boundary("main processing", "pipeline", ErrorSeverity.ERROR)
def execute_processing_pipeline(args: argparse.Namespace, feedback: CLIFeedback) -> int:
    """Execute the main processing pipeline."""
    return run_processing(args, feedback)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
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
    
    parser.add_argument('--url', '--youtube-url', help='YouTube URL or local file path')
    parser.add_argument('--output', '--out', help='Output SRT file path')
    parser.add_argument('--batch-list', help='Text file with URLs/paths, one per line')
    parser.add_argument('--output-dir', '--out-dir', default='./output', help='Output directory for batch processing')
    parser.add_argument('--cookies', help='Browser cookies file for anti-bot protection (auto-detected if not specified)')
    parser.add_argument('--use-voxtral-mini', action='store_true', help='Use Voxtral Mini instead of Small')
    
    # Processing options
    parser.add_argument('--dry-run', action='store_true', help='Run preflight checks only')
    parser.add_argument('--validate-only', action='store_true', help='Run full validation suite only')
    parser.add_argument('--force', action='store_true', help='Skip validation and proceed anyway (dangerous)')
    
    # Authentication and setup
    parser.add_argument('--setup-auth', action='store_true', help='Setup HuggingFace authentication (interactive)')
    parser.add_argument('--hf-token', help='HuggingFace token (prefer --setup-auth for secure input)')
    
    # Logging and debugging
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    parser.add_argument('--log-level', default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], help='Set the logging level.')
    parser.add_argument('--quiet', '-q', action='store_true', help='Minimal output (errors only)')
    
    # Advanced options
    parser.add_argument('--no-cache', action='store_true', help='Disable model and audio caching')
    parser.add_argument('--max-workers', type=int, help='Maximum number of parallel workers (auto-detected if not specified)')
    parser.add_argument('--gpu-memory-limit', type=float, help='GPU memory utilization limit (0.1-0.95, default: auto)')
    parser.add_argument('--timeout', type=int, default=3600, help='Overall processing timeout in seconds (default: 3600)')
    
    # Output options
    parser.add_argument('--format', choices=['srt', 'vtt', 'json'], default='srt', help='Output format')
    parser.add_argument('--quality', choices=['fast', 'balanced', 'best'], default='balanced', help='Quality/speed tradeoff')
    
    return parser.parse_args()


def main() -> int:
    """Main entry point with clean separation of concerns."""
    start_time = time.time()
    # The main function should not handle argument parsing directly
    # to be more testable and reusable.
    args = parse_args()
    
    # Handle special commands first
    if args.setup_auth:
        if cli_auth_setup:
            cli_auth_setup()
        else:
            print("❌ Auth setup not available - using simple token manager")
            print("💡 Use environment variable HF_TOKEN or create .env file")
        return 0
    
    error_reporter = None
    try:
        # 1. Application Setup
        feedback, error_reporter = setup_application(args)
        
        # 2. Environment Validation
        validate_environment(args, feedback)
        
        # 3. Early Exit Conditions
        if args.validate_only or args.dry_run:
            feedback.display_success_panel(time.time() - start_time, "N/A", 0)
            return 0
        
        # 4. Main Processing Pipeline
        return execute_processing_pipeline(args, feedback)
        
    except KeyboardInterrupt:
        console.print("[yellow]Interrupted by user[/yellow]")
        return 1
    except (ValueError, RuntimeError, FileNotFoundError) as e:
        # These are expected, actionable errors for the user.
        console.print(f"[red]An error occurred: {e}[/red]")
        if error_reporter:
            # Optionally report to a more structured system
            error_reporter.report(e, "main_execution_flow")
        return 1
    except Exception:
        # These are unexpected errors, indicating a potential bug.
        console.print("[bold red]An unexpected critical error occurred.[/bold red]")
        # Log the full traceback for debugging purposes.
        logger.critical("Unhandled exception caught at top level", exc_info=True)
        console.print("[yellow]Please check the application logs for a detailed traceback.[/yellow]")
        return 1
    finally:
        # Ensure critical cleanup happens.
        free_cuda_mem()


class CacheDB:
    def __init__(self, path: Path = CACHE_DB, feedback: Optional[CLIFeedback] = None) -> None:
        self.path = path
        self.feedback = feedback or get_feedback()
        self._ensure()
    
    @contextmanager
    def _conn(self, commit: bool = False):
        """Context manager for SQLite connections.
        
        Args:
            commit: Whether to commit transaction on success
        """
        conn = sqlite3.connect(str(self.path), timeout=30.0)
        conn.row_factory = sqlite3.Row  # Enable dict-like access to rows
        conn.execute('PRAGMA journal_mode=WAL')  # Better concurrency
        conn.execute('PRAGMA synchronous=NORMAL')  # Balance safety/speed
        try:
            yield conn
            if commit:
                conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
    
    def _ensure(self) -> None:
        try:
            with self._conn(commit=True) as c:
                c.execute('''CREATE TABLE IF NOT EXISTS translations (
                    k TEXT PRIMARY KEY,
                    src TEXT NOT NULL,
                    trg TEXT NOT NULL,
                    model TEXT NOT NULL,
                    ts REAL NOT NULL DEFAULT (strftime('%s', 'now'))
                )''')
                c.execute('''CREATE TABLE IF NOT EXISTS videos (
                    vid TEXT PRIMARY KEY,
                    srt TEXT NOT NULL,
                    status TEXT NOT NULL,
                    ts REAL NOT NULL DEFAULT (strftime('%s', 'now'))
                )''')
                # Create indexes for better performance
                c.execute('CREATE INDEX IF NOT EXISTS idx_translations_model ON translations(model)')
                c.execute('CREATE INDEX IF NOT EXISTS idx_videos_status ON videos(status)')
        except (sqlite3.Error, OSError) as e:
            self.feedback.warning(f"Cache database setup issue: {e}")
    
    def get(self, k: str) -> Optional[str]:
        """Retrieve cached translation by key.
        
        Args:
            k: Cache key (typically hash of source text)
            
        Returns:
            Cached translation text or None if not found
        """
        try:
            with self._conn() as c:
                r = c.execute('SELECT trg FROM translations WHERE k=?', (k,)).fetchone()
                return r['trg'] if r else None
        except (sqlite3.Error, KeyError) as e:
            self.feedback.debug(f"Cache read error: {e}")
            return None
    
    def set(self, k: str, src: str, trg: str, model: str) -> None:
        """Store translation in cache.
        
        Args:
            k: Cache key (typically hash of source text)
            src: Source text
            trg: Translated text
            model: Model identifier used for translation
        """
        try:
            with self._conn(commit=True) as c:
                c.execute('INSERT OR REPLACE INTO translations (k,src,trg,model,ts) VALUES (?,?,?,?,?)', 
                         (k, src, trg, model, time.time()))
        except (sqlite3.Error, TypeError) as e:
            self.feedback.debug(f"Cache write error: {e}")
    
    def mark_done(self, vid: str, srt: str) -> None:
        """Mark video processing as completed.
        
        Args:
            vid: Video identifier
            srt: Path to generated SRT file
        """
        try:
            with self._conn(commit=True) as c:
                c.execute('INSERT OR REPLACE INTO videos (vid,srt,status,ts) VALUES (?,?,?,?)', 
                         (vid, srt, 'done', time.time()))
        except (sqlite3.Error, TypeError) as e:
            self.feedback.debug(f"Cache mark done error: {e}")






















@with_error_boundary("cookie setup", "auth", ErrorSeverity.WARNING, fallback_value=None)
def setup_cookie_file(args: argparse.Namespace, feedback) -> Optional[Path]:
    """Setup and validate cookie file for anti-bot protection."""
    if args.cookies:
        if not _validate_local_path_security(args.cookies):
            raise PermissionError(f"Security validation failed for cookie file: {args.cookies}")
        cookiefile = Path(args.cookies)
        if not cookiefile.exists():
            raise FileNotFoundError(f"Cookie file not found: {cookiefile}")
        return cookiefile
    else:
        # Auto-detect browser cookies
        from utils.antibot_utils import auto_detect_cookies
        auto_cookies = auto_detect_cookies()
        if auto_cookies:
            feedback.debug(f"Auto-detected browser cookies: {auto_cookies}")
            return auto_cookies
    return None


@with_error_boundary("batch processing", "batch_pipeline", ErrorSeverity.ERROR)
def process_batch_videos(args: argparse.Namespace, feedback, model_manager, 
                        disk_manager, cookiefile: Optional[Path]) -> int:
    """Process multiple videos from batch list."""
    # Security validation
    if not _validate_local_path_security(args.batch_list):
        raise PermissionError(f"Security validation failed for batch file: {args.batch_list}")
    if not _validate_local_path_security(args.output_dir):
        raise PermissionError(f"Security validation failed for output directory: {args.output_dir}")
        
    batch_file = Path(args.batch_list)
    output_dir = Path(args.output_dir)
    use_small_model = not args.use_voxtral_mini
    
    enhanced_process_batch(
        batch_file, output_dir, feedback, model_manager, disk_manager,
        detect_hardware, process_audio, enhanced_generate_srt, free_cuda_mem,
        use_small_model, cookiefile
    )
    return 0


@with_error_boundary("single video processing", "video_pipeline", ErrorSeverity.ERROR)
def process_single_video(args: argparse.Namespace, feedback, model_manager,
                        disk_manager, cookiefile: Optional[Path]) -> int:
    """Process a single video URL or file."""
    # Output path validation
    if args.output:
        if not _validate_local_path_security(args.output):
            raise PermissionError(f"Security validation failed for output file: {args.output}")
        output_path = Path(args.output)
    else:
        output_path = Path(f'output_{int(time.time())}.srt')
    
    use_small_model = not args.use_voxtral_mini
    
    success = enhanced_process_single_video(
        args.url, output_path, feedback, model_manager, disk_manager,
        detect_hardware, process_audio, enhanced_generate_srt, free_cuda_mem,
        use_small_model, cookiefile
    )
    
    if success:
        feedback.success(f"✅ Subtitle file created: {output_path.absolute()}")
        return 0
    else:
        raise RuntimeError("Processing failed")


def run_processing(args: argparse.Namespace, feedback) -> int:
    """Orchestrate processing workflow with proper error handling."""
    # Setup dependencies
    model_manager = ModelManager(feedback)
    disk_manager = get_disk_manager()
    cookiefile = setup_cookie_file(args, feedback)
    
    # Route to appropriate processing function
    if args.batch_list:
        return process_batch_videos(args, feedback, model_manager, disk_manager, cookiefile)
    elif args.url:
        return process_single_video(args, feedback, model_manager, disk_manager, cookiefile)
    else:
        raise ValueError("No input specified: provide --url or --batch-list")


def setup_interactive_configuration():
    """Setup interactive configuration if needed."""
    try:
        from cli_enhanced import create_enhanced_cli
        
        # Check if this is first run or needs configuration
        cli = create_enhanced_cli()
        
        # Only show interactive setup if first run or missing critical config
        if cli.session_state["first_run"] or not os.getenv('HF_TOKEN'):
            console.print("\n[yellow]🔧 Configuration initiale requise...[/yellow]")
            
            if not os.getenv('HF_TOKEN'):
                cli.setup_hf_token_interactive()
            
            # Mark as configured
            cli.user_config["setup_completed"] = True
            cli.save_user_config()
            
    except ImportError:
        # Fallback if enhanced CLI not available
        pass


if __name__ == '__main__':
    import traceback
    import logging
    logger = logging.getLogger(__name__)
    sys.exit(main())