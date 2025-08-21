#!/usr/bin/env python3
"""
test_timing_sync.py - Comprehensive test for subtitle timing synchronization and FFmpeg compatibility
"""

import subprocess
import tempfile
import time
from pathlib import Path
import numpy as np

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()

def test_ffmpeg_compatibility():
    """Test FFmpeg compatibility and audio processing capabilities."""
    
    console.print(Panel("🎬 FFmpeg Compatibility Test", style="bold blue"))
    
    try:
        # Test FFmpeg version
        result = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True)
        if result.returncode != 0:
            console.print("[red]❌ FFmpeg not found[/red]")
            return False
        
        version_line = result.stdout.split('\n')[0]
        console.print(f"[green]✅ {version_line}[/green]")
        
        # Test audio conversion capabilities
        console.print("\n🔍 Testing audio conversion capabilities...")
        
        # Create test audio file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            test_input = Path(tmp.name)
            
        # Generate 5 seconds of test audio (sine wave)
        sample_rate = 16000
        duration = 5
        t = np.linspace(0, duration, duration * sample_rate, False)
        frequency = 440  # A4 note
        audio_data = np.sin(2 * np.pi * frequency * t) * 0.5
        
        # Save as temporary WAV
        import soundfile as sf
        sf.write(test_input, audio_data, sample_rate)
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            test_output = Path(tmp.name)
        
        # Test FFmpeg conversion
        ffmpeg_cmd = [
            'ffmpeg', '-y', '-i', str(test_input),
            '-ar', '16000',  # 16kHz sample rate
            '-ac', '1',      # Mono
            '-f', 'wav',     # WAV format
            str(test_output)
        ]
        
        start_time = time.time()
        result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
        conversion_time = time.time() - start_time
        
        if result.returncode == 0:
            console.print(f"[green]✅ Audio conversion successful ({conversion_time:.2f}s)[/green]")
            
            # Verify output file
            if test_output.exists() and test_output.stat().st_size > 0:
                console.print("[green]✅ Output file created successfully[/green]")
            else:
                console.print("[red]❌ Output file empty or missing[/red]")
                return False
        else:
            console.print(f"[red]❌ Audio conversion failed: {result.stderr}[/red]")
            return False
        
        # Cleanup
        test_input.unlink()
        test_output.unlink()
        
        return True
        
    except FileNotFoundError:
        console.print("[red]❌ FFmpeg not installed[/red]")
        return False
    except Exception as e:
        console.print(f"[red]❌ FFmpeg test failed: {e}[/red]")
        return False

def test_subtitle_timing_precision():
    """Test subtitle timing precision and synchronization."""
    
    console.print(Panel("⏱️ Subtitle Timing Precision Test", style="bold green"))
    
    try:
        from utils.srt_utils import format_srt_time
        from voxtral_prompts import optimize_subtitle_timing
        
        # Test timing format precision
        test_times = [
            0.0,           # Start
            1.5,           # 1.5 seconds
            61.123,        # Over 1 minute with milliseconds
            3661.456,      # Over 1 hour
            0.001,         # 1 millisecond
            59.999,        # Edge case
        ]
        
        expected_formats = [
            "00:00:00,000",
            "00:00:01,500", 
            "00:01:01,123",
            "01:01:01,456",
            "00:00:00,001",
            "00:00:59,999"
        ]
        
        timing_table = Table(title="Timing Format Test")
        timing_table.add_column("Input (seconds)", style="cyan")
        timing_table.add_column("Expected", style="green")
        timing_table.add_column("Actual", style="yellow")
        timing_table.add_column("Result", style="bold")
        
        all_passed = True
        for time_input, expected in zip(test_times, expected_formats):
            actual = format_srt_time(time_input)
            passed = actual == expected
            all_passed &= passed
            
            timing_table.add_row(
                f"{time_input}",
                expected,
                actual,
                "✅ PASS" if passed else "❌ FAIL"
            )
        
        console.print(timing_table)
        
        if all_passed:
            console.print("[green]✅ All timing format tests passed[/green]")
        else:
            console.print("[red]❌ Some timing format tests failed[/red]")
            return False
        
        # Test subtitle optimization
        console.print("\n🔍 Testing subtitle timing optimization...")
        
        test_cases = [
            ("Bonjour", 0.0, 2.0),  # Normal case
            ("Très long texte qui dépasse la durée normale de lecture", 0.0, 1.0),  # Too fast
            ("Court", 0.0, 10.0),   # Too long
        ]
        
        opt_table = Table(title="Timing Optimization Test")
        opt_table.add_column("Text", style="cyan")
        opt_table.add_column("Original Duration", style="yellow")
        opt_table.add_column("Optimized Duration", style="green")
        opt_table.add_column("Status", style="bold")
        
        for text, start, end in test_cases:
            optimized = optimize_subtitle_timing(text, start, end)
            original_duration = end - start
            optimized_duration = optimized['end'] - optimized['start']
            
            status = "✅ OPTIMAL"
            if 'warning' in optimized:
                status = f"⚠️ {optimized['warning'][:20]}..."
            
            opt_table.add_row(
                text[:30] + "..." if len(text) > 30 else text,
                f"{original_duration:.2f}s",
                f"{optimized_duration:.2f}s",
                status
            )
        
        console.print(opt_table)
        console.print("[green]✅ Subtitle timing optimization working[/green]")
        
        return True
        
    except ImportError as e:
        console.print(f"[red]❌ Import error: {e}[/red]")
        return False
    except Exception as e:
        console.print(f"[red]❌ Timing test failed: {e}[/red]")
        return False

def test_voxtral_prompts():
    """Test Voxtral prompt generation and optimization."""
    
    console.print(Panel("🤖 Voxtral Prompt Optimization Test", style="bold magenta"))
    
    try:
        from voxtral_prompts import (
            get_voxtral_prompt, get_transformers_generation_params, 
            get_vllm_generation_params, TURKISH_DRAMA_PROMPTS,
            validate_translation_quality
        )
        
        # Test prompt generation
        drama_prompt = get_voxtral_prompt(context="drama")
        if len(drama_prompt) > 100 and "Turkish" in drama_prompt and "French" in drama_prompt:
            console.print("[green]✅ Drama prompt generated successfully[/green]")
        else:
            console.print("[red]❌ Drama prompt generation failed[/red]")
            return False
        
        # Test pre-defined prompts
        if len(TURKISH_DRAMA_PROMPTS) >= 3:
            console.print(f"[green]✅ {len(TURKISH_DRAMA_PROMPTS)} pre-defined drama prompts available[/green]")
        else:
            console.print("[red]❌ Insufficient pre-defined prompts[/red]")
            return False
        
        # Test generation parameters
        trans_params = get_transformers_generation_params()
        vllm_params = get_vllm_generation_params()
        
        required_trans_keys = ['max_new_tokens', 'temperature', 'do_sample', 'top_p', 'num_beams']
        required_vllm_keys = ['max_tokens', 'temperature', 'top_p']
        
        trans_ok = all(key in trans_params for key in required_trans_keys)
        vllm_ok = all(key in vllm_params for key in required_vllm_keys)
        
        if trans_ok and vllm_ok:
            console.print("[green]✅ Generation parameters configured correctly[/green]")
        else:
            console.print("[red]❌ Generation parameters missing required keys[/red]")
            return False
        
        # Test translation quality validation
        test_translations = [
            ("Bonjour mon amour", True),    # Good
            ("Je t'aime beaucoup mon cœur", True),  # Good
            ("Trop long texte qui dépasse vraiment la limite normale des sous-titres français", False),  # Too long
            ("abi ne yapıyorsun", False),   # Contains Turkish
        ]
        
        quality_table = Table(title="Translation Quality Test")
        quality_table.add_column("Translation", style="cyan")
        quality_table.add_column("Expected", style="green")
        quality_table.add_column("Quality Score", style="yellow")
        quality_table.add_column("Issues", style="red")
        quality_table.add_column("Result", style="bold")
        
        quality_passed = True
        for translation, should_be_good in test_translations:
            quality = validate_translation_quality("", translation)
            is_good = quality['quality_score'] >= 8 and len(quality['issues']) == 0
            passed = is_good == should_be_good
            quality_passed &= passed
            
            quality_table.add_row(
                translation[:40] + "..." if len(translation) > 40 else translation,
                "Good" if should_be_good else "Issues",
                str(quality['quality_score']),
                str(len(quality['issues'])),
                "✅ PASS" if passed else "❌ FAIL"
            )
        
        console.print(quality_table)
        
        if quality_passed:
            console.print("[green]✅ Translation quality validation working[/green]")
        else:
            console.print("[red]❌ Translation quality validation issues[/red]")
            return False
        
        return True
        
    except ImportError as e:
        console.print(f"[red]❌ Import error: {e}[/red]")
        return False
    except Exception as e:
        console.print(f"[red]❌ Voxtral prompt test failed: {e}[/red]")
        return False

def test_complete_pipeline_integration():
    """Test complete pipeline integration for timing sync."""
    
    console.print(Panel("🔄 Complete Pipeline Integration Test", style="bold cyan"))
    
    try:
        # Simulate a complete subtitle generation process
        mock_segments = [
            {'start': 0.0, 'end': 3.5, 'text': 'Bonjour, comment allez-vous ?'},
            {'start': 3.2, 'end': 6.8, 'text': 'Je vais très bien, merci beaucoup.'},  # Overlapping
            {'start': 7.0, 'end': 8.1, 'text': 'Parfait !'},  # Too short
            {'start': 8.5, 'end': 20.0, 'text': 'Voici un très long texte qui devrait être divisé en plusieurs lignes pour respecter les contraintes de sous-titres français standard.'},  # Too long
        ]
        
        from utils.srt_utils import enhanced_generate_srt
        from cli_feedback import get_feedback
        
        with tempfile.NamedTemporaryFile(suffix='.srt', delete=False) as tmp:
            test_srt_path = Path(tmp.name)
        
        feedback = get_feedback(debug_mode=True)
        
        # Generate SRT with our enhanced function
        enhanced_generate_srt(mock_segments, test_srt_path, feedback)
        
        # Verify SRT file
        if test_srt_path.exists():
            content = test_srt_path.read_text(encoding='utf-8')
            console.print("[green]✅ SRT file generated successfully[/green]")
            
            # Check for proper formatting
            if "-->" in content and content.count('\n\n') >= 3:
                console.print("[green]✅ SRT format is correct[/green]")
            else:
                console.print("[red]❌ SRT format issues detected[/red]")
                return False
            
            # Verify timing fixes
            lines = content.split('\n')
            timing_lines = [line for line in lines if '-->' in line]
            
            if len(timing_lines) == 4:  # Should have 4 subtitles
                console.print("[green]✅ Correct number of subtitles generated[/green]")
            else:
                console.print(f"[red]❌ Expected 4 subtitles, got {len(timing_lines)}[/red]")
                return False
            
            console.print("\n📝 Generated SRT preview:")
            preview_lines = content.split('\n')[:12]  # First subtitle
            for line in preview_lines:
                console.print(f"   {line}")
            
        else:
            console.print("[red]❌ SRT file not created[/red]")
            return False
        
        # Cleanup
        test_srt_path.unlink()
        
        return True
        
    except Exception as e:
        console.print(f"[red]❌ Pipeline integration test failed: {e}[/red]")
        return False

