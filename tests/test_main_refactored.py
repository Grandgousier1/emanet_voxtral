#!/usr/bin/env python3
"""
Tests unitaires pour le main.py refactorisé
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import argparse
from pathlib import Path
import tempfile
import sys
import os

# Ajout du répertoire parent au path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import (
    setup_application, validate_environment, execute_processing_pipeline,
    setup_cookie_file, process_batch_videos, process_single_video,
    run_processing, main
)
from domain_models import ErrorSeverity
from error_boundary import error_boundary


class TestSetupApplication:
    """Tests pour setup_application."""
    
    def test_setup_application_success(self):
        """Test setup normal de l'application."""
        args = argparse.Namespace(
            log_level='INFO',
            debug=False
        )
        
        with patch('main.setup_logging') as mock_setup_logging, \
             patch('main.get_feedback') as mock_get_feedback, \
             patch('main.ErrorReporter') as mock_error_reporter, \
             patch('main.setup_interactive_configuration'):
            
            mock_feedback = Mock()
            mock_get_feedback.return_value = mock_feedback
            mock_reporter = Mock()
            mock_error_reporter.return_value = mock_reporter
            
            feedback, error_reporter = setup_application(args)
            
            mock_setup_logging.assert_called_once_with(log_level='INFO')
            mock_get_feedback.assert_called_once_with(debug_mode=False)
            mock_feedback.display_welcome_panel.assert_called_once()
            assert feedback == mock_feedback
            assert error_reporter == mock_reporter

    def test_setup_application_with_debug(self):
        """Test setup avec mode debug."""
        args = argparse.Namespace(
            log_level='DEBUG',
            debug=True
        )
        
        with patch('main.setup_logging'), \
             patch('main.get_feedback') as mock_get_feedback, \
             patch('main.ErrorReporter'), \
             patch('main.setup_interactive_configuration'):
            
            setup_application(args)
            mock_get_feedback.assert_called_once_with(debug_mode=True)


class TestValidateEnvironment:
    """Tests pour validate_environment."""
    
    def test_validate_environment_force_mode(self):
        """Test avec --force, validation bypassed."""
        args = argparse.Namespace(force=True)
        mock_feedback = Mock()
        
        result = validate_environment(args, mock_feedback)
        
        assert result is True
        # Pas d'appel aux vérifications
        mock_feedback.major_step.assert_not_called()

    def test_validate_environment_success(self):
        """Test validation réussie."""
        args = argparse.Namespace(force=False)
        mock_feedback = Mock()
        
        with patch('main.validate_hf_token', return_value=True), \
             patch('main.check_disk_space', return_value=True), \
             patch('main.enhanced_preflight_checks', return_value=True), \
             patch('main.shutil.disk_usage') as mock_disk_usage:
            
            mock_disk_usage.return_value = Mock(free=50 * 1024**3)  # 50GB free
            
            result = validate_environment(args, mock_feedback)
            
            assert result is True
            mock_feedback.major_step.assert_called_once()
            mock_feedback.display_health_dashboard.assert_called_once()
            mock_feedback.info.assert_called_with("Validation de l'environnement terminée avec succès.")

    def test_validate_environment_token_failure(self):
        """Test échec validation token."""
        args = argparse.Namespace(force=False)
        mock_feedback = Mock()
        
        with patch('main.validate_hf_token', return_value=False):
            with pytest.raises(ValueError, match="INVALID_HF_TOKEN"):
                validate_environment(args, mock_feedback)

    def test_validate_environment_disk_failure(self):
        """Test échec validation espace disque."""
        args = argparse.Namespace(force=False)
        mock_feedback = Mock()
        
        with patch('main.validate_hf_token', return_value=True), \
             patch('main.check_disk_space', return_value=False), \
             patch('main.shutil.disk_usage') as mock_disk_usage:
            
            mock_disk_usage.return_value = Mock(free=10 * 1024**3)  # 10GB free (insufficient)
            
            with pytest.raises(RuntimeError, match="INSUFFICIENT_DISK_SPACE"):
                validate_environment(args, mock_feedback)

    def test_validate_environment_preflight_failure(self):
        """Test échec validation preflight."""
        args = argparse.Namespace(force=False)
        mock_feedback = Mock()
        
        with patch('main.validate_hf_token', return_value=True), \
             patch('main.check_disk_space', return_value=True), \
             patch('main.enhanced_preflight_checks', return_value=False), \
             patch('main.shutil.disk_usage') as mock_disk_usage:
            
            mock_disk_usage.return_value = Mock(free=50 * 1024**3)
            
            with pytest.raises(RuntimeError, match="PREFLIGHT_CHECKS_FAILED"):
                validate_environment(args, mock_feedback)


class TestCookieFileSetup:
    """Tests pour setup_cookie_file."""
    
    def test_setup_cookie_file_explicit_path(self):
        """Test avec chemin de cookie explicite."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            cookie_path = tmp_file.name
            
        try:
            args = argparse.Namespace(cookies=cookie_path)
            mock_feedback = Mock()
            
            with patch('main._validate_local_path_security', return_value=True):
                result = setup_cookie_file(args, mock_feedback)
                
                assert result == Path(cookie_path)
        finally:
            os.unlink(cookie_path)

    def test_setup_cookie_file_auto_detect(self):
        """Test avec auto-détection des cookies."""
        args = argparse.Namespace(cookies=None)
        mock_feedback = Mock()
        mock_auto_cookies = Path("/tmp/test_cookies")
        
        with patch('main.auto_detect_cookies', return_value=mock_auto_cookies):
            result = setup_cookie_file(args, mock_feedback)
            
            assert result == mock_auto_cookies
            mock_feedback.debug.assert_called()

    def test_setup_cookie_file_security_failure(self):
        """Test échec validation sécurité."""
        args = argparse.Namespace(cookies="/dangerous/path")
        mock_feedback = Mock()
        
        with patch('main._validate_local_path_security', return_value=False):
            with pytest.raises(PermissionError):
                setup_cookie_file(args, mock_feedback)

    def test_setup_cookie_file_not_found(self):
        """Test fichier cookie non trouvé."""
        args = argparse.Namespace(cookies="/nonexistent/path")
        mock_feedback = Mock()
        
        with patch('main._validate_local_path_security', return_value=True):
            with pytest.raises(FileNotFoundError):
                setup_cookie_file(args, mock_feedback)


class TestProcessingFunctions:
    """Tests pour les fonctions de traitement."""
    
    def test_process_batch_videos_success(self):
        """Test traitement batch réussi."""
        args = argparse.Namespace(
            batch_list="/tmp/batch.txt",
            output_dir="/tmp/output",
            use_voxtral_mini=False
        )
        mock_feedback = Mock()
        mock_model_manager = Mock()
        mock_disk_manager = Mock()
        mock_cookiefile = Path("/tmp/cookies")
        
        with patch('main._validate_local_path_security', return_value=True), \
             patch('main.enhanced_process_batch') as mock_process_batch, \
             patch('main.detect_hardware'), \
             patch('main.process_audio'), \
             patch('main.enhanced_generate_srt'), \
             patch('main.free_cuda_mem'):
            
            result = process_batch_videos(
                args, mock_feedback, mock_model_manager, 
                mock_disk_manager, mock_cookiefile
            )
            
            assert result == 0
            mock_process_batch.assert_called_once()

    def test_process_single_video_success(self):
        """Test traitement vidéo unique réussi."""
        args = argparse.Namespace(
            url="https://youtube.com/watch?v=test",
            output="/tmp/output.srt",
            use_voxtral_mini=False
        )
        mock_feedback = Mock()
        mock_model_manager = Mock()
        mock_disk_manager = Mock()
        
        with patch('main._validate_local_path_security', return_value=True), \
             patch('main.enhanced_process_single_video', return_value=True) as mock_process_video:
            
            result = process_single_video(
                args, mock_feedback, mock_model_manager, 
                mock_disk_manager, None
            )
            
            assert result == 0
            mock_process_video.assert_called_once()
            mock_feedback.success.assert_called()

    def test_process_single_video_failure(self):
        """Test échec traitement vidéo unique."""
        args = argparse.Namespace(
            url="https://youtube.com/watch?v=test",
            output="/tmp/output.srt",
            use_voxtral_mini=False
        )
        mock_feedback = Mock()
        mock_model_manager = Mock()
        mock_disk_manager = Mock()
        
        with patch('main._validate_local_path_security', return_value=True), \
             patch('main.enhanced_process_single_video', return_value=False):
            
            with pytest.raises(RuntimeError, match="Processing failed"):
                process_single_video(
                    args, mock_feedback, mock_model_manager, 
                    mock_disk_manager, None
                )


class TestRunProcessing:
    """Tests pour run_processing."""
    
    def test_run_processing_batch(self):
        """Test routing vers traitement batch."""
        args = argparse.Namespace(
            batch_list="/tmp/batch.txt",
            url=None
        )
        mock_feedback = Mock()
        
        with patch('main.ModelManager') as mock_model_cls, \
             patch('main.get_disk_manager') as mock_disk_mgr, \
             patch('main.setup_cookie_file', return_value=None), \
             patch('main.process_batch_videos', return_value=0) as mock_process_batch:
            
            result = run_processing(args, mock_feedback)
            
            assert result == 0
            mock_process_batch.assert_called_once()

    def test_run_processing_single_video(self):
        """Test routing vers traitement vidéo unique."""
        args = argparse.Namespace(
            batch_list=None,
            url="https://youtube.com/watch?v=test"
        )
        mock_feedback = Mock()
        
        with patch('main.ModelManager'), \
             patch('main.get_disk_manager'), \
             patch('main.setup_cookie_file', return_value=None), \
             patch('main.process_single_video', return_value=0) as mock_process_video:
            
            result = run_processing(args, mock_feedback)
            
            assert result == 0
            mock_process_video.assert_called_once()

    def test_run_processing_no_input(self):
        """Test sans input spécifié."""
        args = argparse.Namespace(
            batch_list=None,
            url=None
        )
        mock_feedback = Mock()
        
        with patch('main.ModelManager'), \
             patch('main.get_disk_manager'), \
             patch('main.setup_cookie_file', return_value=None):
            
            with pytest.raises(ValueError, match="No input specified"):
                run_processing(args, mock_feedback)


class TestMainFunction:
    """Tests pour la fonction main."""
    
    def test_main_auth_setup(self):
        """Test commande --setup-auth."""
        test_args = ['main.py', '--setup-auth']
        
        with patch('sys.argv', test_args), \
             patch('main.parse_args') as mock_parse, \
             patch('main.cli_auth_setup') as mock_auth_setup:
            
            mock_parse.return_value = argparse.Namespace(setup_auth=True)
            
            result = main()
            
            assert result == 0
            mock_auth_setup.assert_called_once()

    def test_main_validate_only(self):
        """Test mode --validate-only."""
        mock_args = argparse.Namespace(
            setup_auth=False,
            validate_only=True,
            dry_run=False
        )
        
        with patch('main.parse_args', return_value=mock_args), \
             patch('main.setup_application') as mock_setup, \
             patch('main.validate_environment'), \
             patch('main.time.time', side_effect=[100, 105]):  # 5 secondes écoulées
            
            mock_feedback = Mock()
            mock_setup.return_value = (mock_feedback, Mock())
            
            result = main()
            
            assert result == 0
            mock_feedback.display_success_panel.assert_called_with(5, "N/A", 0)

    def test_main_processing_success(self):
        """Test traitement réussi complet."""
        mock_args = argparse.Namespace(
            setup_auth=False,
            validate_only=False,
            dry_run=False
        )
        
        with patch('main.parse_args', return_value=mock_args), \
             patch('main.setup_application') as mock_setup, \
             patch('main.validate_environment'), \
             patch('main.execute_processing_pipeline', return_value=0) as mock_exec:
            
            mock_feedback = Mock()
            mock_setup.return_value = (mock_feedback, Mock())
            
            result = main()
            
            assert result == 0
            mock_exec.assert_called_once()

    def test_main_keyboard_interrupt(self):
        """Test interruption utilisateur."""
        mock_args = argparse.Namespace(setup_auth=False)
        
        with patch('main.parse_args', return_value=mock_args), \
             patch('main.setup_application', side_effect=KeyboardInterrupt), \
             patch('main.console') as mock_console:
            
            result = main()
            
            assert result == 1
            mock_console.print.assert_called_with("[yellow]Interrupted by user[/yellow]")

    def test_main_critical_error(self):
        """Test erreur critique."""
        mock_args = argparse.Namespace(setup_auth=False)
        
        with patch('main.parse_args', return_value=mock_args), \
             patch('main.setup_application', side_effect=RuntimeError("Critical error")), \
             patch('main.console') as mock_console:
            
            result = main()
            
            assert result == 1
            mock_console.print.assert_called_with("[red]Critical error: Critical error[/red]")

    def test_main_cleanup_always_called(self):
        """Test que le cleanup est toujours appelé."""
        mock_args = argparse.Namespace(setup_auth=False)
        
        with patch('main.parse_args', return_value=mock_args), \
             patch('main.setup_application', side_effect=Exception("Test error")), \
             patch('main.free_cuda_mem') as mock_cleanup, \
             patch('main.console'):
            
            result = main()
            
            assert result == 1
            mock_cleanup.assert_called_once()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])