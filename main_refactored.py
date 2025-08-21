#!/usr/bin/env python3
"""
Main refactorisé pour Voxtral - Architecture élégante et maintenable
"""

import sys
from typing import NoReturn

# Domain models et configuration
from domain_models import ErrorSeverity
from config.app_config import (
    create_app_config, 
    validate_config, 
    ConfigValidationError,
    AppConfig
)

# Services métier
from services.validation_service import validate_environment
from services.processing_service import (
    create_processing_service,
    MediaProcessingService
)

# Error boundary pour gestion d'erreurs robuste
from error_boundary import with_error_boundary, ErrorContext


@with_error_boundary(
    "application configuration",
    "app_config", 
    severity=ErrorSeverity.CRITICAL,
    recovery_strategy="check command line arguments"
)
def setup_application() -> AppConfig:
    """
    Configure l'application de manière déterministe.
    
    Returns:
        Configuration immutable de l'application
        
    Raises:
        ConfigValidationError: Si la configuration est invalide
    """
    config = create_app_config()
    
    # Validation de la configuration
    validate_config(config)
    
    # Affichage du panneau de bienvenue
    if hasattr(config.feedback, 'display_welcome_panel'):
        config.feedback.display_welcome_panel(vars(config.args))
    else:
        config.feedback.info("🚀 EMANET VOXTRAL - B200 Optimized Subtitle Generator")
    
    return config


@with_error_boundary(
    "environment validation",
    "validation_service",
    severity=ErrorSeverity.CRITICAL, 
    recovery_strategy="use --force to skip validation (risky)"
)
def validate_runtime_environment(config: AppConfig) -> bool:
    """
    Valide l'environnement d'exécution.
    
    Args:
        config: Configuration de l'application
        
    Returns:
        True si l'environnement est valide, False sinon
    """
    config.feedback.info("🔍 Étape 1/4: Validation de l'environnement")
    
    return validate_environment(config)


@with_error_boundary(
    "processing orchestration",
    "processing_service",
    severity=ErrorSeverity.ERROR,
    recovery_strategy="check input sources and configuration"
)
def orchestrate_processing(config: AppConfig, service: MediaProcessingService) -> int:
    """
    Orchestre le traitement selon le mode demandé.
    
    Args:
        config: Configuration de l'application
        service: Service de traitement
        
    Returns:
        Code de retour de l'application
    """
    # Gestion des commandes spéciales
    if config.args.setup_auth:
        return service.handle_auth_setup()
    
    # Mode validation uniquement
    if config.is_validation_mode:
        return service.handle_validation_mode()
    
    # Traitement principal des médias
    if config.requires_processing:
        config.feedback.info("🎬 Étapes 2-4: Traitement des médias")
        return service.process_media()
    
    # Aucune action requise
    config.feedback.warning("⚠️ Aucune action spécifiée")
    config.feedback.info("💡 Utilisez --help pour voir les options disponibles")
    return 0


def handle_application_error(error: Exception, config: AppConfig) -> NoReturn:
    """
    Gère les erreurs de niveau application.
    
    Args:
        error: Exception capturée
        config: Configuration de l'application
    """
    if isinstance(error, ConfigValidationError):
        config.feedback.critical(f"❌ Configuration invalide:\n{error}")
        if error.context and error.context.recovery_strategy:
            config.feedback.info(f"💡 Solution: {error.context.recovery_strategy}")
        sys.exit(1)
    
    elif isinstance(error, KeyboardInterrupt):
        config.feedback.warning("⚠️ Interruption utilisateur")
        sys.exit(130)  # Code UNIX standard pour SIGINT
    
    elif isinstance(error, FileNotFoundError):
        config.feedback.error(f"📁 Fichier non trouvé: {error.filename}")
        sys.exit(2)
    
    elif isinstance(error, PermissionError):
        config.feedback.error(f"🔒 Permissions insuffisantes: {error}")
        sys.exit(13)  # Code UNIX pour permission denied
    
    else:
        # Erreur inattendue - affichage détaillé en mode debug
        config.feedback.critical(f"💥 Erreur inattendue: {type(error).__name__}")
        if config.args.debug:
            import traceback
            config.feedback.error(f"Traceback complet:\n{traceback.format_exc()}")
        else:
            config.feedback.error(str(error))
            config.feedback.info("💡 Utilisez --debug pour plus de détails")
        
        sys.exit(1)


def main() -> int:
    """
    Point d'entrée principal - orchestration pure et élégante.
    
    Cette fonction implémente le pattern "configure, validate, execute" 
    avec gestion d'erreurs robuste et responsabilités clairement séparées.
    
    Returns:
        Code de retour de l'application (0 = succès)
    """
    config = None
    
    try:
        # 1. Configuration de l'application
        config = setup_application()
        
        # 2. Validation de l'environnement
        if not validate_runtime_environment(config):
            config.feedback.error("❌ Validation d'environnement échouée")
            return 1
        
        # 3. Création du service de traitement
        processing_service = create_processing_service(config)
        
        # 4. Orchestration du traitement
        return orchestrate_processing(config, processing_service)
        
    except Exception as error:
        # Gestion centralisée des erreurs avec config si disponible
        if config:
            handle_application_error(error, config)
        else:
            # Fallback si config non disponible
            print(f"❌ Erreur critique lors de l'initialisation: {error}", file=sys.stderr)
            if "--debug" in sys.argv:
                import traceback
                print(f"Traceback: {traceback.format_exc()}", file=sys.stderr)
            sys.exit(1)


if __name__ == '__main__':
    """Point d'entrée du programme avec code de retour approprié."""
    sys.exit(main())