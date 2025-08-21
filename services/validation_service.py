#!/usr/bin/env python3
"""
Service de validation d'environnement pour Voxtral
"""

import shutil
from typing import List, Dict, Any, NamedTuple
from dataclasses import dataclass

from domain_models import ErrorSeverity
from config.app_config import AppConfig
from error_boundary import with_error_boundary, ErrorContext
from cli_feedback import CLIFeedback


class ValidationCheck(NamedTuple):
    """Résultat d'une vérification de validation."""
    name: str
    status: bool
    value: str
    severity: ErrorSeverity = ErrorSeverity.ERROR


@dataclass
class ValidationReport:
    """Rapport de validation complet."""
    checks: List[ValidationCheck]
    overall_status: bool
    critical_failures: List[str]
    
    @property
    def passed_count(self) -> int:
        """Nombre de vérifications réussies."""
        return sum(1 for check in self.checks if check.status)
    
    @property
    def failed_count(self) -> int:
        """Nombre de vérifications échouées."""
        return len(self.checks) - self.passed_count
    
    @property
    def success_rate(self) -> float:
        """Taux de réussite des vérifications."""
        if not self.checks:
            return 0.0
        return self.passed_count / len(self.checks)


class EnvironmentValidator:
    """Validateur d'environnement avec vérifications modulaires."""
    
    def __init__(self, feedback: CLIFeedback):
        self.feedback = feedback
        self.config = get_optimal_config()
        self.required_disk_gb = self.config.get('disk', {}).get('min_required_gb', 25)
    
    @with_error_boundary(
        "token validation", 
        "security",
        severity=ErrorSeverity.CRITICAL,
        recovery_strategy="use --setup-auth to configure token"
    )
    def validate_hf_token(self) -> ValidationCheck:
        """Valide le token HuggingFace."""
        try:
            from utils.security_utils import validate_hf_token
            is_valid = validate_hf_token()
            
            return ValidationCheck(
                name="Token Hugging Face",
                status=is_valid,
                value="Valide" if is_valid else "Manquant/Invalide",
                severity=ErrorSeverity.CRITICAL if not is_valid else ErrorSeverity.INFO
            )
        except ImportError:
            return ValidationCheck(
                name="Token Hugging Face",
                status=False,
                value="Module de validation manquant",
                severity=ErrorSeverity.CRITICAL
            )
    
    @with_error_boundary(
        "disk space check",
        "system",
        severity=ErrorSeverity.ERROR,
        recovery_strategy="free up disk space or reduce model cache"
    )
    def validate_disk_space(self) -> ValidationCheck:
        """Valide l'espace disque disponible."""
        try:
            from utils.system_utils import check_disk_space
            
            available_gb = shutil.disk_usage('.').free / (1024**3)
            is_sufficient = check_disk_space('.', required_gb=self.required_disk_gb)
            
            return ValidationCheck(
                name=f"Espace Disque (≥{self.required_disk_gb} Go)",
                status=is_sufficient,
                value=f"{available_gb:.1f} Go libres",
                severity=ErrorSeverity.ERROR if not is_sufficient else ErrorSeverity.INFO
            )
        except Exception as e:
            self.feedback.warning(f"Erreur vérification disque: {e}")
            return ValidationCheck(
                name="Espace Disque",
                status=False,
                value="Erreur de vérification",
                severity=ErrorSeverity.WARNING
            )
    
    @with_error_boundary(
        "dependencies check",
        "dependencies", 
        severity=ErrorSeverity.CRITICAL,
        recovery_strategy="run: python validator.py to check dependencies"
    )
    def validate_dependencies(self) -> ValidationCheck:
        """Valide les dépendances et GPU."""
        try:
            from utils.validation_utils import enhanced_preflight_checks
            
            is_valid = enhanced_preflight_checks(self.feedback)
            
            return ValidationCheck(
                name="Dépendances & GPU",
                status=is_valid,
                value="Compatibles" if is_valid else "Problèmes détectés",
                severity=ErrorSeverity.CRITICAL if not is_valid else ErrorSeverity.INFO
            )
        except ImportError as e:
            self.feedback.error(f"Module de validation manquant: {e}")
            return ValidationCheck(
                name="Dépendances & GPU", 
                status=False,
                value="Module validation manquant",
                severity=ErrorSeverity.CRITICAL
            )
    
    @with_error_boundary(
        "cuda availability",
        "gpu",
        severity=ErrorSeverity.WARNING,
        recovery_strategy="install CUDA or use CPU fallback"
    )
    def validate_cuda(self) -> ValidationCheck:
        """Valide la disponibilité CUDA."""
        try:
            import torch
            
            cuda_available = torch.cuda.is_available()
            if cuda_available:
                device_count = torch.cuda.device_count()
                device_name = torch.cuda.get_device_name(0) if device_count > 0 else "Unknown"
                memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3) if device_count > 0 else 0
                
                value = f"{device_name} ({memory_gb:.0f}GB)"
                
                # Détection B200
                is_b200 = "B200" in device_name or memory_gb >= 180
                if is_b200:
                    value += " [B200 Optimized]"
            else:
                value = "Non disponible - mode CPU"
            
            return ValidationCheck(
                name="CUDA/GPU",
                status=cuda_available,
                value=value,
                severity=ErrorSeverity.WARNING if not cuda_available else ErrorSeverity.INFO
            )
        except Exception as e:
            return ValidationCheck(
                name="CUDA/GPU",
                status=False,
                value=f"Erreur: {e}",
                severity=ErrorSeverity.WARNING
            )
    
    @with_error_boundary(
        "model access",
        "models",
        severity=ErrorSeverity.WARNING,
        recovery_strategy="check internet connection and HF token"
    )
    def validate_model_access(self) -> ValidationCheck:
        """Valide l'accès aux modèles Voxtral."""
        try:
            from constants import VOXTRAL_SMALL
            from transformers import AutoConfig
            import os
            
            # Test access to model config (lightweight check)
            hf_token = os.getenv('HF_TOKEN') or os.getenv('HUGGINGFACE_HUB_TOKEN')
            auth_kwargs = {'token': hf_token} if hf_token else {}
            
            try:
                config = AutoConfig.from_pretrained(VOXTRAL_SMALL, **auth_kwargs)
                return ValidationCheck(
                    name="Accès Modèles Voxtral",
                    status=True,
                    value="Accessible",
                    severity=ErrorSeverity.INFO
                )
            except Exception as model_error:
                return ValidationCheck(
                    name="Accès Modèles Voxtral", 
                    status=False,
                    value=f"Inaccessible: {type(model_error).__name__}",
                    severity=ErrorSeverity.WARNING
                )
                
        except ImportError:
            return ValidationCheck(
                name="Accès Modèles Voxtral",
                status=False,
                value="Dépendances manquantes",
                severity=ErrorSeverity.WARNING
            )
    
    def run_all_validations(self) -> ValidationReport:
        """
        Lance toutes les validations et retourne un rapport complet.
        
        Returns:
            Rapport de validation avec statuts détaillés
        """
        self.feedback.info("🔍 Validation de l'environnement...")
        
        # Liste des vérifications à effectuer
        validation_methods = [
            self.validate_hf_token,
            self.validate_disk_space, 
            self.validate_dependencies,
            self.validate_cuda,
            self.validate_model_access
        ]
        
        checks = []
        critical_failures = []
        
        for validation_method in validation_methods:
            try:
                check = validation_method()
                checks.append(check)
                
                # Log du résultat
                if check.status:
                    self.feedback.debug(f"✅ {check.name}: {check.value}")
                else:
                    level = "critical" if check.severity == ErrorSeverity.CRITICAL else "warning"
                    getattr(self.feedback, level)(f"❌ {check.name}: {check.value}")
                    
                    if check.severity == ErrorSeverity.CRITICAL:
                        critical_failures.append(check.name)
                        
            except Exception as e:
                self.feedback.error(f"Erreur validation {validation_method.__name__}: {e}")
                checks.append(ValidationCheck(
                    name=validation_method.__name__.replace('validate_', '').title(),
                    status=False,
                    value=f"Erreur: {type(e).__name__}",
                    severity=ErrorSeverity.ERROR
                ))
        
        # Calcul du statut global
        has_critical_failures = len(critical_failures) > 0
        overall_status = not has_critical_failures
        
        report = ValidationReport(
            checks=checks,
            overall_status=overall_status,
            critical_failures=critical_failures
        )
        
        # Log du résumé
        self.feedback.info(f"📊 Validation terminée: {report.passed_count}/{len(checks)} réussies ({report.success_rate:.0%})")
        
        if critical_failures:
            self.feedback.error(f"💥 Échecs critiques: {', '.join(critical_failures)}")
        elif report.failed_count > 0:
            self.feedback.warning(f"⚠️ {report.failed_count} avertissements détectés")
        else:
            self.feedback.success("🎉 Tous les tests de validation réussis!")
            
        return report


def validate_environment(config: AppConfig) -> bool:
    """
    Fonction utilitaire pour valider l'environnement.
    
    Args:
        config: Configuration de l'application
        
    Returns:
        True si l'environnement est valide, False sinon
    """
    if config.args.force:
        config.feedback.warning("🚨 Mode --force activé - validation ignorée")
        return True
    
    validator = EnvironmentValidator(config.feedback)
    report = validator.run_all_validations()
    
    # Affichage du dashboard de santé
    if hasattr(config.feedback, 'display_health_dashboard'):
        health_checks = [
            {
                'check': check.name,
                'status': check.status, 
                'value': check.value
            }
            for check in report.checks
        ]
        config.feedback.display_health_dashboard(health_checks)
    
    # Décision finale
    if not report.overall_status:
        config.feedback.error("❌ Validation d'environnement échouée")
        if report.critical_failures:
            config.feedback.info("💡 Utilisez --force pour ignorer (non recommandé)")
        return False
    
    config.feedback.success("✅ Environnement validé avec succès")
    return True