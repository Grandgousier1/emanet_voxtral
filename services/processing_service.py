#!/usr/bin/env python3
"""
Service de traitement de médias pour Voxtral
"""

import time
from pathlib import Path
from typing import Optional, Union, Dict, Any
from dataclasses import dataclass

from domain_models import ProcessingResult, ErrorSeverity
from config.app_config import AppConfig
from error_boundary import with_error_boundary, ErrorContext
from cli_feedback import CLIFeedback


@dataclass
class ProcessingReport:
    """Rapport de traitement d'un média."""
    input_source: str
    output_path: Path
    processing_time: float
    segment_count: int
    success: bool
    error_message: Optional[str] = None


class MediaProcessingService:
    """Service de traitement de médias avec gestion d'erreurs robuste."""
    
    def __init__(self, config: AppConfig):
        self.config = config
        self.feedback = config.feedback
    
    @with_error_boundary(
        "authentication setup",
        "auth_manager",
        severity=ErrorSeverity.INFO,
        recovery_strategy="authentication configured"
    )
    def handle_auth_setup(self) -> int:
        """Gère la configuration d'authentification interactive."""
        from utils.auth_manager import cli_auth_setup
        
        self.feedback.info("🔐 Configuration d'authentification...")
        cli_auth_setup()
        self.feedback.success("✅ Configuration d'authentification terminée")
        return 0
    
    @with_error_boundary(
        "validation mode",
        "validation",
        severity=ErrorSeverity.INFO,
        recovery_strategy="validation completed"
    )
    def handle_validation_mode(self) -> int:
        """Gère les modes de validation uniquement."""
        if self.config.args.validate_only:
            self.feedback.success("✅ Validation complète - système prêt pour traitement")
        elif self.config.args.dry_run:
            self.feedback.success("✅ Test à sec terminé - tous les systèmes opérationnels")
        
        # Affichage du panneau de succès
        if hasattr(self.feedback, 'display_success_panel'):
            elapsed_time = time.time() - self.config.start_time
            self.feedback.display_success_panel(elapsed_time, "N/A", 0)
        
        return 0
    
    @with_error_boundary(
        "media processing orchestration",
        "processing_service",
        severity=ErrorSeverity.ERROR,
        recovery_strategy="check input source and try again"
    )
    def process_media(self) -> int:
        """
        Orchestre le traitement de médias selon la configuration.
        
        Returns:
            Code de retour (0 = succès, 1+ = erreur)
        """
        self.feedback.info("🎬 Début du traitement de médias...")
        
        try:
            if self.config.args.batch_list:
                return self._process_batch()
            elif self.config.args.url:
                return self._process_single_video()
            else:
                self.feedback.error("❌ Aucune source d'entrée spécifiée")
                return 1
                
        except Exception as e:
            self.feedback.error(f"💥 Erreur lors du traitement: {e}")
            return 1
        finally:
            # Nettoyage GPU garanti
            try:
                from utils.gpu_utils import free_cuda_mem
                free_cuda_mem()
                self.feedback.debug("🧹 Mémoire GPU nettoyée")
            except ImportError:
                pass
    
    @with_error_boundary(
        "batch processing",
        "batch_processor",
        severity=ErrorSeverity.ERROR,
        recovery_strategy="process videos individually"
    )
    def _process_batch(self) -> int:
        """Traite un lot de vidéos."""
        from utils.security_utils import validate_local_path_security, enhanced_process_batch
        from utils.error_messages import ErrorReporter
        
        error_reporter = ErrorReporter(self.feedback)
        
        # Validation sécurisée du batch file
        if not _validate_local_path_security(self.config.args.batch_list):
            error_reporter.report("SECURITY_PATH_NOT_ALLOWED", 
                                file_type="Batch file", 
                                path=self.config.args.batch_list)
            return 1
        
        batch_file = Path(self.config.args.batch_list)
        if not batch_file.exists():
            error_reporter.report("FILE_NOT_FOUND", 
                                file_type="Batch file", 
                                path=batch_file)
            return 1
        
        # Validation sécurisée du output dir
        if not _validate_local_path_security(self.config.args.output_dir):
            error_reporter.report("SECURITY_PATH_NOT_ALLOWED",
                                file_type="Output directory", 
                                path=self.config.args.output_dir)
            return 1
        
        output_dir = Path(self.config.args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Configuration du traitement
        use_small_model = not self.config.args.use_voxtral_mini
        cookiefile = self._get_cookies_file()
        
        # Orchestration avec dependency injection
        with self._create_processing_context() as context:
            try:
                enhanced_process_batch(
                    batch_file=batch_file,
                    output_dir=output_dir,
                    feedback=self.feedback,
                    model_manager=context['model_manager'],
                    disk_manager=context['disk_manager'],
                    hardware_detector=context['hardware_detector'],
                    audio_processor=context['audio_processor'],
                    srt_generator=context['srt_generator'],
                    cuda_mem_freeer=context['cuda_mem_freeer'],
                    use_small_model=use_small_model,
                    cookiefile=cookiefile
                )
                
                self.feedback.success(f"✅ Traitement batch terminé: {output_dir}")
                return 0
                
            except Exception as e:
                error_reporter.report("PROCESSING_FAILED", details=str(e))
                return 1
    
    @with_error_boundary(
        "single video processing",
        "video_processor", 
        severity=ErrorSeverity.ERROR,
        recovery_strategy="check video source and format"
    )
    def _process_single_video(self) -> int:
        """Traite une seule vidéo."""
        from utils.security_utils import validate_local_path_security, enhanced_process_single_video
        from utils.error_messages import ErrorReporter
        
        error_reporter = ErrorReporter(self.feedback)
        
        # Détermination du chemin de sortie
        if self.config.args.output:
            if not _validate_local_path_security(self.config.args.output):
                error_reporter.report("SECURITY_PATH_NOT_ALLOWED",
                                    file_type="Output file",
                                    path=self.config.args.output)
                return 1
            output_path = Path(self.config.args.output)
        else:
            # Génération automatique du nom de sortie
            timestamp = int(time.time())
            output_path = Path(f'voxtral_output_{timestamp}.srt')
        
        # Configuration du traitement
        use_small_model = not self.config.args.use_voxtral_mini
        cookiefile = self._get_cookies_file()
        
        # Orchestration avec dependency injection
        with self._create_processing_context() as context:
            try:
                success = enhanced_process_single_video(
                    url=self.config.args.url,
                    output_path=output_path,
                    feedback=self.feedback,
                    model_manager=context['model_manager'],
                    disk_manager=context['disk_manager'],
                    hardware_detector=context['hardware_detector'],
                    audio_processor=context['audio_processor'],
                    srt_generator=context['srt_generator'],
                    cuda_mem_freeer=context['cuda_mem_freeer'],
                    use_small_model=use_small_model,
                    cookiefile=cookiefile
                )
                
                if success:
                    # Affichage du panneau de succès
                    elapsed_time = time.time() - self.config.start_time
                    if hasattr(self.feedback, 'display_success_panel'):
                        self.feedback.display_success_panel(
                            total_time=elapsed_time,
                            output_file=str(output_path.absolute()),
                            segment_count=self._estimate_segment_count(output_path)
                        )
                    else:
                        self.feedback.success(f"✅ Fichier de sous-titres créé: {output_path.absolute()}")
                    return 0
                else:
                    error_reporter.report("PROCESSING_FAILED")
                    return 1
                    
            except Exception as e:
                error_reporter.report("PROCESSING_FAILED", details=str(e))
                return 1
    
    def _get_cookies_file(self) -> Optional[Path]:
        """Récupère le fichier de cookies avec auto-détection."""
        from utils.security_utils import validate_local_path_security
        
        if self.config.args.cookies:
            if not _validate_local_path_security(self.config.args.cookies):
                self.feedback.warning(f"Chemin cookies non sécurisé: {self.config.args.cookies}")
                return None
            
            cookiefile = Path(self.config.args.cookies)
            if not cookiefile.exists():
                self.feedback.warning(f"Fichier cookies non trouvé: {cookiefile}")
                return None
            
            self.feedback.debug(f"Utilisation cookies: {cookiefile}")
            return cookiefile
        
        # Auto-détection des cookies du navigateur
        try:
            from utils.antibot_utils import auto_detect_cookies
            auto_cookies = auto_detect_cookies()
            if auto_cookies:
                self.feedback.debug(f"Cookies auto-détectés: {auto_cookies}")
                return auto_cookies
        except Exception as e:
            self.feedback.debug(f"Auto-détection cookies échouée: {e}")
        
        return None
    
    def _create_processing_context(self):
        """Crée le contexte de traitement avec dependency injection."""
        from contextlib import contextmanager
        
        @contextmanager
        def processing_context():
            # Import des dépendances
            from utils.model_utils import create_model_manager
            from parallel_processor import get_disk_manager
            from config import detect_hardware
            from utils.processing_utils import process_audio
            from utils.srt_utils import enhanced_generate_srt
            from utils.gpu_utils import free_cuda_mem
            
            # Création des services avec cleanup garanti
            model_manager = None
            try:
                model_manager = create_model_manager(self.feedback)
                
                context = {
                    'model_manager': model_manager,
                    'disk_manager': get_disk_manager(),
                    'hardware_detector': detect_hardware,
                    'audio_processor': process_audio,
                    'srt_generator': enhanced_generate_srt,
                    'cuda_mem_freeer': free_cuda_mem
                }
                
                yield context
                
            finally:
                # Cleanup garanti des ressources
                if model_manager and hasattr(model_manager, '_cleanup_all_models'):
                    model_manager._cleanup_all_models()
                    self.feedback.debug("🧹 Modèles nettoyés")
        
        return processing_context()
    
    def _estimate_segment_count(self, output_path: Path) -> int:
        """Estime le nombre de segments dans le fichier SRT."""
        try:
            if not output_path.exists():
                return 0
            
            content = output_path.read_text(encoding='utf-8')
            # Compte les blocs de sous-titres (séparés par lignes vides)
            segments = [b for b in content.split('\n\n') if b.strip()]
            return len(segments)
            
        except Exception as e:
            self.feedback.debug(f"Erreur estimation segments: {e}")
            return 0


def create_processing_service(config: AppConfig) -> MediaProcessingService:
    """Factory pour créer le service de traitement."""
    return MediaProcessingService(config)