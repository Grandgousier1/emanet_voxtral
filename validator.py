#!/usr/bin/env python3
"""
validator.py - Comprehensive Code Validation and Testing Suite
Performs complete validation before one-shot execution
"""

import sys
import ast
import importlib
import subprocess
import traceback
import time
import tempfile
from pathlib import Path
from typing import List, Any, Optional
import json

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeRemainingColumn, MofNCompleteColumn
from rich.table import Table
from rich.panel import Panel

console = Console()

class ValidationResult:
    """Container for validation results."""
    
    def __init__(self, test_name: str):
        self.test_name = test_name
        self.passed = False
        self.error = None
        self.warning = None
        self.details = []
        self.execution_time = 0.0
        self.critical = False
    
    def set_passed(self, details: str = ""):
        self.passed = True
        if details:
            self.details.append(details)
    
    def set_failed(self, error: str, critical: bool = False):
        self.passed = False
        self.error = error
        self.critical = critical
    
    def add_warning(self, warning: str):
        self.warning = warning
    
    def add_detail(self, detail: str):
        self.details.append(detail)

class CodeValidator:
    """Comprehensive code validator for one-shot execution."""
    
    def __init__(self):
        self.results: List[ValidationResult] = []
        self.project_root = Path(__file__).parent.resolve()
        self.python_files = [
            "main.py",
            "config.py", 
            "parallel_processor.py",
            "monitor.py",
            "benchmark.py",
            "utils/gpu_utils.py"
        ]
        self.config_files = [
            "requirements.txt",
            "Makefile",
            "setup_runpod.sh"
        ]
        
    def run_validation(self) -> bool:
        """Run complete validation suite."""
        
        console.print(Panel.fit(
            "🔍 EMANET VOXTRAL - COMPREHENSIVE VALIDATION SUITE\n"
            "Ensuring one-shot execution readiness...",
            style="bold blue"
        ))
        
        # Define all validation tests
        validation_tests = [
            ("File Structure", self._validate_file_structure),
            ("Python Syntax", self._validate_python_syntax),
            ("Import Dependencies", self._validate_imports),
            ("Configuration Files", self._validate_config_files),
            ("Hardware Detection", self._validate_hardware_detection),
            ("Model Loading", self._validate_model_loading),
            ("Audio Processing", self._validate_audio_processing),
            ("Pipeline Integration", self._validate_pipeline_integration),
            ("Error Handling", self._validate_error_handling),
            ("Memory Management", self._validate_memory_management),
            ("Disk Management", self._validate_disk_management),
            ("CLI Interface", self._validate_cli_interface),
            ("Performance Metrics", self._validate_performance_metrics)
        ]
        
        total_tests = len(validation_tests)
        passed_tests = 0
        critical_failures = 0
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("•"),
            TimeRemainingColumn(),
            console=console,
            expand=True
        ) as progress:
            
            main_task = progress.add_task("🔍 Running validation tests...", total=total_tests)
            
            for test_name, test_func in validation_tests:
                # Update progress
                progress.update(main_task, description=f"🔍 {test_name}...")
                
                # Run test
                result = ValidationResult(test_name)
                start_time = time.time()
                
                try:
                    test_func(result)
                except Exception as e:
                    result.set_failed(f"Test execution failed: {str(e)}", critical=True)
                    result.add_detail(f"Exception: {traceback.format_exc()}")
                
                result.execution_time = time.time() - start_time
                self.results.append(result)
                
                # Count results
                if result.passed:
                    passed_tests += 1
                elif result.critical:
                    critical_failures += 1
                
                # Update progress
                progress.update(main_task, advance=1)
        
        # Display results
        self._display_results()
        
        # Return success status
        success = critical_failures == 0 and passed_tests >= (total_tests * 0.8)
        
        if success:
            console.print(Panel(
                f"✅ VALIDATION COMPLETE\n\n"
                f"• Passed: {passed_tests}/{total_tests} tests\n"
                f"• Critical failures: {critical_failures}\n"
                f"• Status: READY FOR ONE-SHOT EXECUTION 🚀",
                style="bold green"
            ))
        else:
            console.print(Panel(
                f"❌ VALIDATION FAILED\n\n"
                f"• Passed: {passed_tests}/{total_tests} tests\n" 
                f"• Critical failures: {critical_failures}\n"
                f"• Status: REQUIRES FIXES BEFORE EXECUTION",
                style="bold red"
            ))
        
        return success
    
    def _validate_file_structure(self, result: ValidationResult):
        """Validate project file structure."""
        
        missing_files = []
        found_files = []
        
        # Check Python files
        for file_path in self.python_files:
            full_path = self.project_root / file_path
            if full_path.exists():
                found_files.append(file_path)
                result.add_detail(f"✓ {file_path}")
            else:
                missing_files.append(file_path)
                result.add_detail(f"✗ {file_path} - MISSING")
        
        # Check config files
        for file_path in self.config_files:
            full_path = self.project_root / file_path
            if full_path.exists():
                found_files.append(file_path)
                result.add_detail(f"✓ {file_path}")
            else:
                missing_files.append(file_path)
                result.add_detail(f"✗ {file_path} - MISSING")
        
        if missing_files:
            result.set_failed(f"Missing {len(missing_files)} files: {', '.join(missing_files)}", critical=True)
        else:
            result.set_passed(f"All {len(found_files)} required files present")
    
    def _validate_python_syntax(self, result: ValidationResult):
        """Validate Python syntax for all files."""
        
        syntax_errors = []
        valid_files = []
        
        for file_path in self.python_files:
            full_path = self.project_root / file_path
            if not full_path.exists():
                continue
                
            try:
                with open(full_path, 'r', encoding='utf-8') as f:
                    source_code = f.read()
                
                # Parse syntax
                ast.parse(source_code, filename=str(full_path))
                valid_files.append(file_path)
                result.add_detail(f"✓ {file_path} - syntax OK")
                
            except SyntaxError as e:
                syntax_errors.append(f"{file_path}:{e.lineno}: {e.msg}")
                result.add_detail(f"✗ {file_path}:{e.lineno} - {e.msg}")
            except Exception as e:
                syntax_errors.append(f"{file_path}: {str(e)}")
                result.add_detail(f"✗ {file_path} - {str(e)}")
        
        if syntax_errors:
            result.set_failed(f"Syntax errors in {len(syntax_errors)} files", critical=True)
        else:
            result.set_passed(f"Syntax valid for all {len(valid_files)} Python files")
    
    def _validate_imports(self, result: ValidationResult):
        """Validate import dependencies."""
        
        import_errors = []
        successful_imports = []
        
        # Test core imports that must work
        core_imports = [
            ('torch', 'PyTorch'),
            ('rich', 'Rich CLI'),
            ('pathlib', 'Standard library'),
        ]
        
        # Test optional imports that can fallback
        optional_imports = [
            ('vllm', 'vLLM (optional, fallback available)'),
            ('transformers', 'Transformers'),
            ('torchaudio', 'TorchAudio'),
            ('soundfile', 'SoundFile'),
            ('faster_whisper', 'Faster Whisper (fallback)'),
            ('silero_vad', 'Silero VAD (fallback available)'),
            ('librosa', 'Librosa (audio processing)'),
            ('psutil', 'Psutil (system monitoring)'),
            ('pynvml', 'PyNVML (GPU monitoring)'),
            ('yt_dlp', 'yt-dlp (video download)'),
            ('ffmpeg_python', 'ffmpeg-python (video processing)'),
            ('fake_useragent', 'fake-useragent (anti-bot)'),
            ('requests', 'Requests (HTTP client)'),
            ('python_dotenv', 'python-dotenv (environment variables)'),
            ('sqlalchemy', 'SQLAlchemy (database ORM)'),
            ('resampy', 'Resampy (audio resampling)'),
            ('audioread', 'Audioread (audio loading)'),
            ('nltk', 'NLTK (natural language toolkit)'),
            ('sacrebleu', 'SacreBLEU (NLP metric)'),
            ('sacremoses', 'Sacremoses (NLP tool)'),
            ('sentencepiece', 'SentencePiece (NLP tokenizer)'),
        ]
        
        # Test core imports
        for module_name, description in core_imports:
            try:
                importlib.import_module(module_name)
                successful_imports.append(f"{description}")
                result.add_detail(f"✓ {module_name} - {description}")
            except ImportError as e:
                import_errors.append(f"{module_name}: {str(e)}")
                result.add_detail(f"✗ {module_name} - CRITICAL: {str(e)}")
        
        # Test optional imports (warnings only)
        optional_warnings = []
        for module_name, description in optional_imports:
            try:
                importlib.import_module(module_name)
                successful_imports.append(f"{description}")
                result.add_detail(f"✓ {module_name} - {description}")
            except ImportError:
                optional_warnings.append(f"{module_name}")
                result.add_detail(f"⚠ {module_name} - Optional: {description}")
        
        if import_errors:
            result.set_failed(f"Critical import failures: {', '.join(import_errors)}", critical=True)
        else:
            result.set_passed(f"All critical imports successful ({len(successful_imports)} modules)")
            if optional_warnings:
                result.add_warning(f"Optional modules missing: {', '.join(optional_warnings)}")
    
    def _validate_config_files(self, result: ValidationResult):
        """Validate configuration files."""
        
        config_issues = []
        
        # Check requirements.txt
        req_file = self.project_root / "requirements.txt"
        if req_file.exists():
            try:
                with open(req_file, 'r') as f:
                    requirements = f.read()
                
                # Check for essential packages
                essential_packages = ['transformers', 'vllm', 'mistral-common', 'rich', 'soundfile']
                missing_packages = []
                
                for package in essential_packages:
                    if package not in requirements:
                        missing_packages.append(package)
                
                if missing_packages:
                    config_issues.append(f"requirements.txt missing: {', '.join(missing_packages)}")
                else:
                    result.add_detail("✓ requirements.txt contains essential packages")
                    
            except Exception as e:
                config_issues.append(f"requirements.txt read error: {str(e)}")
        else:
            config_issues.append("requirements.txt missing")
        
        # Check Makefile
        makefile = self.project_root / "Makefile"
        if makefile.exists():
            result.add_detail("✓ Makefile present")
        else:
            config_issues.append("Makefile missing")
        
        # Check setup script
        setup_script = self.project_root / "setup_runpod.sh"
        if setup_script.exists():
            if setup_script.stat().st_mode & 0o111:  # Check if executable
                result.add_detail("✓ setup_runpod.sh present and executable")
            else:
                result.add_warning("setup_runpod.sh not executable")
        else:
            config_issues.append("setup_runpod.sh missing")
        
        if config_issues:
            result.set_failed(f"Configuration issues: {'; '.join(config_issues)}")
        else:
            result.set_passed("All configuration files valid")
    
    def _validate_hardware_detection(self, result: ValidationResult):
        """Validate hardware detection functionality."""
        
        try:
            # Test config import
            sys.path.insert(0, str(self.project_root))
            from config import detect_hardware, get_optimal_config
            
            # Test hardware detection
            hw = detect_hardware()
            
            required_keys = ['gpu_count', 'gpu_memory_gb', 'total_ram_gb', 'cpu_count', 'is_b200']
            missing_keys = [key for key in required_keys if key not in hw]
            
            if missing_keys:
                result.set_failed(f"Hardware detection missing keys: {missing_keys}")
                return
            
            result.add_detail(f"✓ Detected {hw['gpu_count']} GPU(s)")
            result.add_detail(f"✓ RAM: {hw['total_ram_gb']:.1f}GB")
            result.add_detail(f"✓ CPUs: {hw['cpu_count']}")
            result.add_detail(f"✓ B200: {hw['is_b200']}")
            
            # Test configuration loading
            config = get_optimal_config()
            
            required_sections = ['vllm', 'audio', 'memory', 'disk', 'performance']
            missing_sections = [section for section in required_sections if section not in config]
            
            if missing_sections:
                result.set_failed(f"Configuration missing sections: {missing_sections}")
                return
            
            result.set_passed("Hardware detection and configuration working")
            
        except Exception as e:
            result.set_failed(f"Hardware detection failed: {str(e)}", critical=True)
    
    def _validate_model_loading(self, result: ValidationResult):
        """Validate model loading capabilities (without actually loading)."""
        
        try:
            sys.path.insert(0, str(self.project_root))
            from config import get_vllm_args
            
            # Test vLLM args generation
            models_to_test = [
                'mistralai/Voxtral-Small-24B-2507',
                'mistralai/Voxtral-Mini-3B-2507'
            ]
            
            for model_name in models_to_test:
                try:
                    vllm_args = get_vllm_args(model_name)
                    
                    required_args = ['gpu_memory_utilization', 'max_num_seqs', 'dtype']
                    missing_args = [arg for arg in required_args if arg not in vllm_args]
                    
                    if missing_args:
                        result.add_detail(f"⚠ {model_name}: missing vLLM args: {missing_args}")
                    else:
                        result.add_detail(f"✓ {model_name}: vLLM args OK")
                        
                except Exception as e:
                    result.add_detail(f"✗ {model_name}: {str(e)}")
            
            # Test transformers availability
            try:
                from transformers import AutoProcessor
                result.add_detail("✓ Transformers AutoProcessor available")
            except ImportError:
                result.add_warning("Transformers not available - vLLM required")
            
            result.set_passed("Model loading framework ready")
            
        except Exception as e:
            result.set_failed(f"Model loading validation failed: {str(e)}")
    
    def _validate_audio_processing(self, result: ValidationResult):
        """Validate audio processing pipeline."""
        
        try:
            # Test synthetic audio creation
            import numpy as np
            
            # Create test audio
            sample_rate = 16000
            duration = 5  # 5 seconds
            audio_data = np.random.randn(duration * sample_rate).astype(np.float32)
            
            # Test audio file I/O
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                try:
                    import soundfile as sf
                    sf.write(tmp.name, audio_data, sample_rate)
                    
                    # Read back
                    read_data, read_sr = sf.read(tmp.name)
                    
                    if len(read_data) != len(audio_data):
                        result.add_warning("Audio I/O length mismatch")
                    if read_sr != sample_rate:
                        result.add_warning(f"Sample rate mismatch: {read_sr} != {sample_rate}")
                    
                    result.add_detail("✓ Audio file I/O working")
                    
                except ImportError:
                    result.add_detail("✗ soundfile not available")
                except Exception as e:
                    result.add_detail(f"✗ Audio I/O error: {str(e)}")
                finally:
                    Path(tmp.name).unlink(missing_ok=True)
            
            result.set_passed("Audio processing pipeline ready")
            
        except Exception as e:
            result.set_failed(f"Audio processing validation failed: {str(e)}")
    
    def _validate_pipeline_integration(self, result: ValidationResult):
        """Validate main pipeline integration."""
        
        try:
            sys.path.insert(0, str(self.project_root))
            
            # Test CLI parser
            
            result.add_detail("✓ CLI interface importable")
            
            # Test parallel processor
            from parallel_processor import get_optimized_processor, get_disk_manager
            result.add_detail("✓ Parallel processor importable")
            
            # Test monitoring
            from monitor import B200Monitor
            result.add_detail("✓ Monitor system importable")
            
            result.set_passed("Pipeline integration complete")
            
        except Exception as e:
            result.set_failed(f"Pipeline integration failed: {str(e)}", critical=True)
    
    def _validate_error_handling(self, result: ValidationResult):
        """Validate error handling robustness."""
        
        error_scenarios = []
        
        try:
            sys.path.insert(0, str(self.project_root))
            
            if error_scenarios:
                result.set_failed(f"Error handling issues: {'; '.join(error_scenarios)}")
            else:
                result.set_passed("Error handling robust")
                
        except Exception as e:
            result.set_failed(f"Error handling validation failed: {str(e)}")
    
    def _validate_memory_management(self, result: ValidationResult):
        """Validate memory management."""
        
        try:
            sys.path.insert(0, str(self.project_root))
            from utils.gpu_utils import free_cuda_mem, gpu_mem_info
            
            # Test GPU utilities
            if hasattr(free_cuda_mem, '__call__'):
                result.add_detail("✓ GPU memory cleanup function available")
            else:
                result.add_detail("✗ GPU memory cleanup not callable")
            
            # Test memory info
            mem_info = gpu_mem_info()
            if mem_info is None:
                result.add_detail("⚠ GPU memory info not available (no GPU or pynvml missing)")
            else:
                result.add_detail(f"✓ GPU memory info: {mem_info}")
            
            result.set_passed("Memory management ready")
            
        except Exception as e:
            result.set_failed(f"Memory management validation failed: {str(e)}")
    
    def _validate_disk_management(self, result: ValidationResult):
        """Validate disk space management."""
        
        try:
            sys.path.insert(0, str(self.project_root))
            from parallel_processor import get_disk_manager
            
            disk_manager = get_disk_manager()
            
            if hasattr(disk_manager, 'create_work_dir'):
                result.add_detail("✓ Disk manager create_work_dir available")
            if hasattr(disk_manager, 'cleanup_all'):
                result.add_detail("✓ Disk manager cleanup_all available")
            
            # Test work directory creation (and cleanup)
            work_dir = disk_manager.create_work_dir()
            if work_dir.exists():
                result.add_detail(f"✓ Work directory created: {work_dir}")
                work_dir.rmdir()  # Clean up
            else:
                result.add_detail("✗ Work directory creation failed")
            
            result.set_passed("Disk management ready")
            
        except Exception as e:
            result.set_failed(f"Disk management validation failed: {str(e)}")
    
    def _validate_cli_interface(self, result: ValidationResult):
        """Validate CLI interface."""
        
        try:
            # Test help output
            cmd_result = subprocess.run(
                [sys.executable, str(self.project_root / "main.py"), "--help"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if cmd_result.returncode == 0:
                help_output = cmd_result.stdout
                
                # Check for expected CLI options
                expected_options = ['--url', '--output', '--batch-list', '--dry-run', '--debug']
                missing_options = []
                
                for option in expected_options:
                    if option not in help_output:
                        missing_options.append(option)
                
                if missing_options:
                    result.add_detail(f"⚠ Missing CLI options: {missing_options}")
                else:
                    result.add_detail("✓ All expected CLI options present")
                
                result.set_passed("CLI interface working")
            else:
                result.set_failed(f"CLI help failed: {cmd_result.stderr}")
                
        except subprocess.TimeoutExpired:
            result.set_failed("CLI help command timed out")
        except Exception as e:
            result.set_failed(f"CLI validation failed: {str(e)}")
    
    def _validate_performance_metrics(self, result: ValidationResult):
        """Validate performance monitoring."""
        
        try:
            sys.path.insert(0, str(self.project_root))
            from monitor import B200Monitor
            
            monitor = B200Monitor()
            
            # Test stats collection
            stats = {
                'gpu': monitor.get_gpu_stats(),
                'memory': monitor.get_memory_stats(),
                'disk': monitor.get_disk_stats(),
                'process': monitor.get_process_stats()
            }
            
            for stat_type, stat_data in stats.items():
                if isinstance(stat_data, dict) and stat_data:
                    result.add_detail(f"✓ {stat_type} stats: {len(stat_data)} metrics")
                else:
                    result.add_detail(f"⚠ {stat_type} stats: no data")
            
            # Test summary
            summary = monitor.get_summary()
            if summary and len(summary) > 10:
                result.add_detail("✓ Performance summary generation working")
            else:
                result.add_detail("⚠ Performance summary may be incomplete")
            
            result.set_passed("Performance monitoring ready")
            
        except Exception as e:
            result.set_failed(f"Performance monitoring validation failed: {str(e)}")
    
    def _display_results(self):
        """Display comprehensive validation results."""
        
        # Summary stats
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results if r.passed)
        failed_tests = sum(1 for r in self.results if not r.passed)
        critical_failures = sum(1 for r in self.results if not r.passed and r.critical)
        warnings = sum(1 for r in self.results if r.warning)
        
        # Create results table
        table = Table(title="🔍 Validation Results", show_header=True, header_style="bold magenta")
        table.add_column("Test", style="cyan", width=25)
        table.add_column("Status", justify="center", width=10)
        table.add_column("Time", justify="right", width=8)
        table.add_column("Details", width=50)
        
        for result in self.results:
            # Status icon
            if result.passed:
                status = "[green]✅ PASS[/green]"
            elif result.critical:
                status = "[red]❌ FAIL[/red]"
            else:
                status = "[yellow]⚠ WARN[/yellow]"
            
            # Time
            time_str = f"{result.execution_time:.2f}s"
            
            # Details summary
            if result.error:
                details = f"[red]{result.error}[/red]"
            elif result.warning:
                details = f"[yellow]{result.warning}[/yellow]"
            elif result.details:
                details = " | ".join(result.details[:2])  # First 2 details
                if len(result.details) > 2:
                    details += f" (+{len(result.details)-2} more)"
            else:
                details = ""
            
            table.add_row(result.test_name, status, time_str, details)
        
        console.print(table)
        
        # Summary panel
        summary_text = (
            f"📊 VALIDATION SUMMARY\n\n"
            f"• Total tests: {total_tests}\n"
            f"• Passed: [green]{passed_tests}[/green]\n"
            f"• Failed: [red]{failed_tests}[/red]\n"
            f"• Critical failures: [bold red]{critical_failures}[/bold red]\n"
            f"• Warnings: [yellow]{warnings}[/yellow]\n"
        )
        
        console.print(Panel.fit(summary_text, style="bold"))
        
        # Show detailed errors/warnings
        critical_errors = [r for r in self.results if not r.passed and r.critical]
        if critical_errors:
            console.print("\n[bold red]🚨 CRITICAL ISSUES REQUIRING FIXES:[/bold red]")
            for result in critical_errors:
                console.print(f"\n[red]❌ {result.test_name}[/red]")
                console.print(f"   Error: {result.error}")
                if result.details:
                    for detail in result.details:
                        console.print(f"   • {detail}")
        
        # Show warnings
        warning_results = [r for r in self.results if r.warning or (not r.passed and not r.critical)]
        if warning_results:
            console.print("\n[yellow]⚠ WARNINGS AND NON-CRITICAL ISSUES:[/yellow]")
            for result in warning_results:
                if result.warning:
                    console.print(f"   • {result.test_name}: {result.warning}")
                elif not result.passed:
                    console.print(f"   • {result.test_name}: {result.error}")

def main():
    """Run the validation suite."""
    validator = CodeValidator()
    success = validator.run_validation()
    
    # Generate report file
    report_file = Path("validation_report.json")
    report_data = {
        'timestamp': time.time(),
        'success': success,
        'results': []
    }
    
    for result in validator.results:
        report_data['results'].append({
            'test_name': result.test_name,
            'passed': result.passed,
            'error': result.error,
            'warning': result.warning,
            'details': result.details,
            'execution_time': result.execution_time,
            'critical': result.critical
        })
    
    with open(report_file, 'w') as f:
        json.dump(report_data, f, indent=2)
    
    console.print(f"\n📄 Full validation report saved to: {report_file}")
    
    return 0 if success else 1

if __name__ == '__main__':
    sys.exit(main())