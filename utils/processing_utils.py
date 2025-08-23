#!/usr/bin/env python3
"""
utils/processing_utils.py - Audio processing utilities
"""

import urllib.parse
from pathlib import Path
from typing import Optional, List, Dict, Any, Callable
import asyncio
import gc
import hashlib
import os
import shutil
import torch
import soundfile as sf

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

from cli_feedback import CLIFeedback, get_feedback
from utils.audio_utils import enhanced_download_audio, enhanced_vad_segments
from utils.model_utils import ModelManager
from config import detect_hardware
from constants import VOXTRAL_MINI, VOXTRAL_SMALL, SAMPLE_RATE
from parallel_processor import get_optimized_processor, get_disk_manager, DiskSpaceManager
from utils.error_messages import ErrorReporter
from utils.gpu_utils import free_cuda_mem
from utils.memory_manager import get_memory_manager
from utils.srt_utils import enhanced_generate_srt
from utils.validation_utils import get_file_validator, get_timeout_manager, get_rate_limiter, ValidationError, TimeoutError
from voxtral_prompts import get_transformers_generation_params, optimize_subtitle_timing, validate_translation_quality


def _validate_url_security(url: str) -> bool:
    """Enhanced URL security validation with comprehensive protections."""
    try:
        # Length protection against DoS
        if len(url) > 2048:  # RFC 2616 recommends 2048 max
            return False
        
        parsed = urllib.parse.urlparse(url)
        
        # CRITICAL: Block file:// and other dangerous schemes
        if parsed.scheme not in ['https', 'http']:
            return False
        
        # Block dangerous schemes explicitly
        dangerous_schemes = ['file', 'ftp', 'data', 'javascript', 'vbscript']
        if any(scheme in url.lower() for scheme in dangerous_schemes):
            return False
        
        # Enhanced localhost restrictions for production
        domain = parsed.netloc.lower()
        if 'localhost' in domain:
            # Only allow localhost in development/test environments
            if os.getenv('ENVIRONMENT', 'production') == 'production':
                return False
        
        # Whitelist des domaines autorisés pour download audio
        allowed_domains = [
            'youtube.com', 'www.youtube.com', 'm.youtube.com',
            'youtu.be', 'music.youtube.com',
            'soundcloud.com', 'www.soundcloud.com',
            'vimeo.com', 'www.vimeo.com',
            'dailymotion.com', 'www.dailymotion.com',
        ]
        
        # Production environment: strict domain validation
        if not any(domain == allowed or domain.endswith('.' + allowed) for allowed in allowed_domains):
            # Allow localhost only in dev/test
            if 'localhost' not in domain and '127.0.0.1' not in domain:
                return False
        
        # Enhanced path traversal protection
        dangerous_patterns = ['..', '//', '\\\\', '%2e%2e', '%2f%2f', '....']
        if any(pattern in parsed.path.lower() for pattern in dangerous_patterns):
            return False
        
        # Query parameter validation
        if parsed.query:
            # Block common injection patterns
            dangerous_query_patterns = ['javascript:', 'data:', 'file:', '<script', 'eval(']
            if any(pattern in parsed.query.lower() for pattern in dangerous_query_patterns):
                return False
        
        # Enhanced port validation
        if parsed.port:
            # Block system ports and dangerous services
            blocked_ports = list(range(1, 1024)) + [1080, 3128, 8080]  # Add common proxy ports
            allowed_ports = [80, 443, 8443]  # Standard web ports
            if parsed.port in blocked_ports and parsed.port not in allowed_ports:
                return False
        
        # IP address validation (block private networks in production)
        import ipaddress
        try:
            ip = ipaddress.ip_address(domain)
            if ip.is_private and os.getenv('ENVIRONMENT', 'production') == 'production':
                return False
        except ValueError:
            pass  # Not an IP address, continue with domain validation
        
        return True
        
    except Exception as e:
        # Log security validation failure for debugging
        return False


def _validate_local_path_security(path_str: str) -> bool:
    """Enhanced local file path security validation."""
    try:
        path = Path(path_str)
        
        # Enhanced path traversal protection
        dangerous_patterns = [
            '..', '..\\', '../', '..\\\\',
            '%2e%2e', '%2e%2e%2f', '%2e%2e%5c',
            '....', '..../', '....\\\\',
            '.\\\\..', './/..',
        ]
        
        path_str_lower = path_str.lower()
        for pattern in dangerous_patterns:
            if pattern in path_str_lower:
                return False
        
        # Block access to sensitive system directories
        sensitive_dirs = [
            '/etc', '/proc', '/sys', '/dev', '/root',
            '/boot', '/usr/bin', '/bin', '/sbin',
            'c:\\\\windows', 'c:\\\\program files', 'c:\\\\users\\\\administrator'
        ]
        
        resolved_str = str(path.resolve()).lower()
        for sensitive in sensitive_dirs:
            if resolved_str.startswith(sensitive):
                return False
        
        # Enhanced absolute path validation
        if path.is_absolute():
            # Allow only specific safe absolute paths
            safe_absolute_prefixes = [
                '/tmp', '/var/tmp', '/home', '/workspace',
                str(Path.cwd()), str(Path.home()),
            ]
            
            import os
            # Add current working directory and common safe paths
            safe_absolute_prefixes.extend([
                os.getcwd(),
                os.path.expanduser('~'),
                '/app',  # Common container path
                '/data', # Common data mount
            ])
            
            if not any(resolved_str.startswith(safe.lower()) for safe in safe_absolute_prefixes):
                return False
        
        # Validate file name for dangerous characters
        if path.name and any(char in path.name for char in ['<', '>', ':', '"', '|', '?', '*']):
            return False
        
        # Block access to hidden files in sensitive locations
        if path.name.startswith('.') and 'ssh' in path.name.lower():
            return False
        
        return True
            
    except Exception as e:
        # Log path validation failure for debugging
        return False


def get_audio_path(url_or_path: str, work_dir: Path, feedback: CLIFeedback, cookiefile: Optional[Path] = None) -> Optional[Path]:
    """Gets the audio path, downloading if necessary with security and size validation."""
    error_reporter = ErrorReporter(feedback)
    file_validator = get_file_validator(feedback)
    timeout_manager = get_timeout_manager(feedback)
    
    if url_or_path.startswith(('http://', 'https://')):
        # Validation sécurisée des URLs
        if not _validate_url_security(url_or_path):
            error_reporter.report("SECURITY_PATH_NOT_ALLOWED", file_type="URL", path=url_or_path)
            return None
        
        # Download with timeout protection
        try:
            with timeout_manager.timeout_context(
                timeout_manager.get_download_timeout(), 
                f"download from {url_or_path[:50]}..."
            ):
                audio_path = enhanced_download_audio(url_or_path, work_dir, feedback, cookiefile)
                
                # Validate downloaded file size
                if audio_path and not file_validator.validate_audio_file_size(audio_path):
                    if audio_path.exists():
                        audio_path.unlink()  # Clean up oversized file
                    return None
                
                return audio_path
                
        except TimeoutError as e:
            error_reporter.report("DOWNLOAD_TIMEOUT", url=url_or_path, details=str(e))
            return None
        except Exception as e:
            error_reporter.report("DOWNLOAD_FAILED", url=url_or_path, details=str(e))
            return None
    else:
        # Validation sécurisée des chemins locaux
        if not _validate_local_path_security(url_or_path):
            error_reporter.report("SECURITY_PATH_NOT_ALLOWED", file_type="File path", path=url_or_path)
            return None
        
        feedback.substep(f"Processing local file: {url_or_path}")
        audio_path = Path(url_or_path)
        
        if not audio_path.exists():
            error_reporter.report("FILE_NOT_FOUND", file_type="Audio file", path=audio_path)
            return None
        
        # Validate local file size
        try:
            if not file_validator.validate_audio_file_size(audio_path):
                return None
        except ValidationError as e:
            error_reporter.report("FILE_SIZE_EXCEEDED", file_type="Audio file", path=audio_path, details=str(e))
            return None
        
        return audio_path


def process_audio(audio_path: Path, feedback: CLIFeedback, model_manager: ModelManager, use_small_model: bool, hw_info: Dict[str, Any] = None) -> Optional[List[Dict]]:
    """Performs VAD and translation with validation and timeout protection."""
    error_reporter = ErrorReporter(feedback)
    file_validator = get_file_validator(feedback)
    timeout_manager = get_timeout_manager(feedback)
    
    try:
        # Get audio duration for validation
        import soundfile as sf
        audio_data, sr = sf.read(str(audio_path))
        duration_seconds = len(audio_data) / sr
        
        # Validate audio duration
        if not file_validator.validate_audio_duration(duration_seconds):
            return None
        
        feedback.step("Voice Activity Detection", 3, 6)
        
        # VAD with timeout protection
        with timeout_manager.timeout_context(
            timeout_manager.get_vad_timeout(),
            f"VAD processing for {audio_path.name}"
        ):
            segments = enhanced_vad_segments(audio_path, feedback)
            
        if not segments:
            error_reporter.report("No speech segments found")
            return None
        
        # Validate segment count
        if not file_validator.validate_segment_count(len(segments)):
            return None
        
        feedback.step("Voxtral Transcription & Translation", 4, 6)
        
        # Translation with timeout protection
        with timeout_manager.timeout_context(
            timeout_manager.get_processing_timeout(),
            f"Translation processing for {len(segments)} segments"
        ):
            # Optimisation N+1: passe hw_info pour éviter detect_hardware() multiple
            translated_segments = enhanced_voxtral_process(
                audio_path, segments, feedback, model_manager, "French", use_small_model, hw_info
            )
            
        if not translated_segments:
            error_reporter.report("No translations produced")
            return None
        
        return translated_segments
        
    except TimeoutError as e:
        error_reporter.report("PROCESSING_TIMEOUT", details=str(e))
        return None
    except ValidationError as e:
        error_reporter.report("VALIDATION_FAILED", details=str(e))
        return None
    except Exception as e:
        error_reporter.report("AUDIO_PROCESSING_FAILED", details=str(e))
        return None

def enhanced_voxtral_process(audio_path: Path, segments: List[Dict], feedback, model_manager: ModelManager,
                           target_lang: str = "French", use_small: bool = True, hw_info: Dict[str, Any] = None) -> List[Dict]:
    """Enhanced Voxtral processing with B200 optimization and detailed feedback."""
    
    model_name = VOXTRAL_SMALL if use_small else VOXTRAL_MINI
    error_reporter = ErrorReporter(feedback)
    memory_manager = get_memory_manager(feedback)
    
    feedback.substep(f"Initializing Voxtral {model_name.split('/')[-1]} for {len(segments)} segments")
    
    processor, model = model_manager.load_voxtral_model(model_name)
    if not model:
        error_reporter.report("MODEL_LOAD_FAILED")
        return []
    
    # Check for B200 optimization (utilise hw_info passé en paramètre ou cache)
    hw = hw_info if hw_info else detect_hardware()
    if hw['is_b200']:
        feedback.info("🚀 Using B200 optimized parallel processing")
        
        try:
            processor_opt = get_optimized_processor()
            return asyncio.run(
                processor_opt.process_audio_segments_parallel(
                    audio_path, segments, model, processor, target_lang
                )
            )
        except Exception as e:
            feedback.warning(f"B200 optimization failed: {e}", solution="Falling back to standard processing")
    
    # Standard processing
    feedback.info("Using standard sequential processing")
    results = []
    
    try:
        audio_data, sr = sf.read(str(audio_path))
        if sr != SAMPLE_RATE:
            if LIBROSA_AVAILABLE:
                audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=SAMPLE_RATE)
            else:
                feedback.warning("Librosa not available, keeping original sample rate")
    except Exception as e:
        error_reporter.report("FILE_NOT_FOUND", file_type="Audio", path=audio_path, details=str(e))
        return []
    
    # Process segments with progress bar and interruption handling
    try:
        for i, segment in enumerate(feedback.progress_bar(segments, "Processing segments")):
            try:
                # Use cached audio data if available from segments processing
                if 'audio_data' in segment:
                    segment_audio = segment['audio_data']
                else:
                    # Protection contre données corrompues dans segments
                    try:
                        start_time = float(segment.get('start', 0))
                        end_time = float(segment.get('end', 0))
                        
                        if start_time < 0 or end_time < 0 or end_time < start_time:
                            feedback.warning(f"Invalid segment timing: {start_time}s-{end_time}s, skipping")
                            continue
                        
                    start_sample = int(start_time * SAMPLE_RATE)
                    end_sample = int(end_time * SAMPLE_RATE)
                    
                    # Protection contre dépassement array bounds
                    if start_sample >= len(audio_data) or end_sample > len(audio_data):
                        feedback.warning(f"Segment beyond audio bounds: {start_sample}-{end_sample} > {len(audio_data)}, skipping")
                        continue
                        
                    segment_audio = audio_data[start_sample:end_sample]
                    
                except (ValueError, TypeError, KeyError) as e:
                    feedback.warning(f"Invalid segment data: {e}, skipping segment {i+1}")
                    continue
            
            if len(segment_audio) == 0:
                feedback.debug(f"Skipping empty segment {i+1}")
                continue
            
            if processor and hasattr(model, 'generate'):
                inputs = processor(
                    segment_audio, 
                    sampling_rate=SAMPLE_RATE, 
                    return_tensors="pt"
                )
                
                inputs = {k: v.to(model.device) for k, v in inputs.items()}
                
                # Use optimized generation parameters for Turkish drama
                gen_params = get_transformers_generation_params()
                
                with torch.no_grad():
                    generated_ids = model.generate(
                        **inputs,
                        **gen_params,
                        pad_token_id=processor.tokenizer.eos_token_id,
                    )
                
                decoded = processor.batch_decode(generated_ids, skip_special_tokens=True)
                text = decoded[0] if decoded else ""
                
                if text.strip():
                    # Optimize subtitle timing based on text length
                    optimized_timing = optimize_subtitle_timing(text.strip(), segment['start'], segment['end'])
                    
                    # Validate translation quality
                    quality = validate_translation_quality("", text.strip())
                    if quality['issues']:
                        feedback.debug(f"Translation quality issues: {', '.join(quality['issues'])}")
                    
                    results.append({
                        'text': text.strip(),
                        'start': optimized_timing['start'],
                        'end': optimized_timing['end'],
                        'quality_score': quality['quality_score']
                    })
                    
                    # Warn about timing issues
                    if 'warning' in optimized_timing:
                        feedback.warning(f"Segment {i+1}: {optimized_timing['warning']}")
            
            # Unified memory management
            memory_manager.on_segment_processed()
            
            # Check for high memory pressure and auto-cleanup if needed
            if memory_manager.auto_cleanup_if_needed():
                feedback.warning(f"Auto-cleanup triggered due to high memory pressure")
                
        except Exception as e:
            error_reporter.report("GENERIC_PROCESSING_ERROR", segment_index=i + 1, details=str(e))
            results.append({
                'text': '[processing error]',
                'start': segment['start'],
                'end': segment['end']
            })
    except KeyboardInterrupt:
        feedback.warning("Processing interrupted by user")
        feedback.info(f"Partial results: {len(results)} segments processed")
        # Cleanup any resources
        memory_manager.cleanup()
        return results
    
    feedback.success(f"Processed {len(results)} segments successfully")
    return results


def enhanced_process_single_video(
    url_or_path: str, 
    output_path: Path, 
    feedback: CLIFeedback, 
    model_manager: ModelManager, 
    disk_manager: "DiskSpaceManager", 
    hardware_detector: Callable[[], Dict[str, Any]],
    audio_processor: Callable[..., Optional[List[Dict]]],
    srt_generator: Callable[[List[Dict], Path, CLIFeedback], None],
    cuda_mem_freeer: Callable[[], None],
    use_small_model: bool = True, 
    cookiefile: Optional[Path] = None
) -> bool:
    """Enhanced single video processing with comprehensive error handling."""
    error_reporter = ErrorReporter(feedback)
    try:
        feedback.step(f"Processing: {Path(url_or_path).name if not url_or_path.startswith('http') else 'YouTube Video'}", 2, 6)

        work_dir = disk_manager.create_work_dir()
        feedback.debug(f"Work directory: {work_dir}")

        audio_path = get_audio_path(url_or_path, work_dir, feedback, cookiefile)
        if not audio_path:
            return False

        # Optimisation N+1: récupère hw_info une seule fois pour tout le processing
        hw_info = hardware_detector()
        translated_segments = audio_processor(audio_path, feedback, model_manager, use_small_model, hw_info)
        if not translated_segments:
            return False

        feedback.step("SRT Generation", 5, 6)
        srt_generator(translated_segments, output_path, feedback)

        feedback.step("Cleanup", 6, 6)
        if work_dir.exists() and url_or_path.startswith(('http://', 'https://')):
            shutil.rmtree(work_dir)
            feedback.debug("Work directory cleaned up")

        cuda_mem_freeer()
        feedback.success(f"✅ Processing complete: {output_path}")
        return True

    except Exception as e:
        feedback.exception(e, "Video processing")
        return False


def enhanced_process_batch(
    batch_file: Path, 
    output_dir: Path, 
    feedback: CLIFeedback, 
    model_manager: ModelManager, 
    disk_manager: "DiskSpaceManager", 
    hardware_detector: Callable[[], Dict[str, Any]],
    audio_processor: Callable[..., Optional[List[Dict]]],
    srt_generator: Callable[[List[Dict], Path, CLIFeedback], None],
    cuda_mem_freeer: Callable[[], None],
    use_small_model: bool = True, 
    cookiefile: Optional[Path] = None
):
    """Enhanced batch processing with validation and detailed progress tracking."""
    
    error_reporter = ErrorReporter(feedback)
    file_validator = get_file_validator(feedback)
    rate_limiter = get_rate_limiter(feedback)
    
    feedback.step("Batch Processing Setup", 2, 6)
    
    if not batch_file.exists():
        error_reporter.report("FILE_NOT_FOUND", file_type="Batch file", path=batch_file)
        return
    
    # Validate batch file size
    try:
        if not file_validator.validate_batch_file_size(batch_file):
            return
    except ValidationError as e:
        error_reporter.report("BATCH_FILE_TOO_LARGE", path=batch_file, details=str(e))
        return
    
    # Read URLs/paths with async I/O for better performance
    urls = []
    try:
        # Use buffered reading for large batch files
        with open(batch_file, 'r', encoding='utf-8', buffering=8192) as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line and not line.startswith('#'):
                    # Validate line length to prevent DoS
                    if len(line) > 2048:
                        feedback.warning(f"Line {line_num} too long, skipping")
                        continue
                    urls.append(line)
                elif line.startswith('#'):
                    feedback.debug(f"Skipping comment line {line_num}: {line}")
                
                # Prevent memory exhaustion from huge batch files
                if len(urls) > 10000:  # Reasonable limit
                    feedback.warning(f"Batch file too large, processing first 10000 entries")
                    break
    except UnicodeDecodeError as e:
        error_reporter.report("BATCH_FILE_ENCODING_ERROR", path=batch_file, details=str(e))
        return
    except IOError as e:
        error_reporter.report("BATCH_FILE_READ_ERROR", path=batch_file, details=str(e))
        return
    
    if not urls:
        error_reporter.report("BATCH_FILE_EMPTY")
        return
    
    feedback.info(f"Found {len(urls)} items to process")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    success_count = 0
    
    # Process in smaller batches for better memory management
    batch_size = min(50, len(urls))  # Process in chunks of 50
    
    for batch_start in range(0, len(urls), batch_size):
        batch_end = min(batch_start + batch_size, len(urls))
        batch_urls = urls[batch_start:batch_end]
        
        feedback.info(f"Processing batch {batch_start//batch_size + 1}/{(len(urls)-1)//batch_size + 1}")
        
        for i, url in enumerate(batch_urls, batch_start + 1):
            feedback.step(f"Item {i}/{len(urls)}: {url[:50]}...", i+1, len(urls)+1)
            
            # Apply rate limiting between processing items
            rate_limiter.wait_if_needed()
        
        # Generate output filename
        if url.startswith(('http://', 'https://')):
            try:
                if YT_DLP_AVAILABLE:
                    with yt_dlp.YoutubeDL({'quiet': True, 'extract_flat': True}) as ydl:
                        info = ydl.extract_info(url, download=False)
                        video_id = info.get('id', hashlib.sha256(url.encode()).hexdigest()[:12])
                        title = info.get('title', video_id)
                else:
                    video_id = hashlib.sha256(url.encode()).hexdigest()[:12]
                    title = video_id
            except Exception as e:
                error_reporter.report("YT_DLP_INFO_EXTRACT_FAILED", url=url, details=str(e))
                video_id = hashlib.sha256(url.encode()).hexdigest()[:12]
                title = video_id
            
            safe_title = "".join(c for c in title if c.isalnum() or c in (' ', '-', '_')).strip()
            output_file = output_dir / f"{safe_title}_{video_id}.srt"
        else:
            input_path = Path(url)
            output_file = output_dir / f"{input_path.stem}.srt"
        
        if enhanced_process_single_video(
            url, 
            output_file, 
            feedback, 
            model_manager, 
            disk_manager,
            hardware_detector,
            audio_processor,
            srt_generator,
            cuda_mem_freeer,
            use_small_model, 
            cookiefile
        ):
            success_count += 1
        
        # Memory cleanup between videos
        cuda_mem_freeer()
        gc.collect()
    
    feedback.step("Batch Complete", len(urls)+1, len(urls)+1)
    feedback.success(f"Batch processing complete: {success_count}/{len(urls)} successful")
