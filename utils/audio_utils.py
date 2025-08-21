#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
utils/audio_utils.py - Audio processing utilities
"""

import subprocess
import time
from pathlib import Path
from typing import List, Dict, Optional

# Audio processing imports at top level for performance
import soundfile as sf
try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

try:
    import torch
    import torchaudio
    from silero_vad import load_silero_vad, get_speech_timestamps
    from .tensor_validation import validate_audio_tensor, check_tensor_health, validate_tensor_device
    SILERO_AVAILABLE = True
except ImportError:
    SILERO_AVAILABLE = False

from cli_feedback import ErrorHandler, get_feedback
from .security_utils import SecureSubprocess
from .audio_cache import get_audio_cache
from utils.antibot_utils import (
    find_browser_cookies,
    get_antibot_download_args,
    check_antibot_error,
    validate_yt_dlp_version,
)

# Constants
SAMPLE_RATE = 16000
CHANNELS = 1


def enhanced_download_audio(
    url: str, workdir: Path, feedback, cookiefile: Optional[Path] = None
) -> Path:
    """Enhanced audio download with robust anti-bot protection and adaptive timeouts."""

    error_handler = ErrorHandler(feedback)
    secure_proc = SecureSubprocess(feedback)

    feedback.substep(f"Preparing download directory: {workdir}")
    workdir.mkdir(parents=True, exist_ok=True)
    out_template = str(workdir / "%(id)s.%(ext)s")

    # Check yt-dlp version first
    is_valid_version, version_msg = validate_yt_dlp_version()
    if not is_valid_version:
        feedback.warning(version_msg)
        feedback.info("Consider updating: pip install -U yt-dlp")

    # Try to find cookies automatically
    if not cookiefile:
        auto_cookies = find_browser_cookies()
        if auto_cookies:
            cookiefile = auto_cookies
            feedback.info(f"Found browser cookies: {cookiefile}")

    # Anti-bot protection strategies (ordered by effectiveness)
    download_strategies = []
    for attempt in range(3):
        antibot_args = get_antibot_download_args(url, attempt)
        strategy_name = ["Standard protection", "Enhanced stealth", "Maximum stealth"][
            attempt
        ]
        download_strategies.append({"name": strategy_name, "cmd_extra": antibot_args})

    timeout = get_adaptive_timeout(url)
    feedback.debug(f"Using adaptive timeout: {timeout}s")

    # Try each strategy with exponential backoff
    for attempt, strategy in enumerate(download_strategies, 1):
        feedback.substep(f"Download attempt {attempt}/3: {strategy['name']}")

        cmd = [
            "yt-dlp",
            "-f",
            "bestaudio[ext=m4a]/bestaudio/best",
            "--no-playlist",
            "--extract-flat",
            "false",
            "-o",
            out_template,
        ] + strategy["cmd_extra"] + [url]

        # Add cookies if available
        if cookiefile and cookiefile.exists():
            cmd.insert(-1, "--cookies")
            cmd.insert(-1, str(cookiefile))
            feedback.debug(f"Using cookies file: {cookiefile}")

        try:
            with feedback.status(f"Downloading with {strategy['name']}..."):
                result = secure_proc.run_secure(
                    cmd, check=True, timeout=timeout
                )
            feedback.success("Audio download completed")
            break  # Success, exit retry loop

        except subprocess.TimeoutExpired:
            if attempt == len(download_strategies):
                error_handler.handle_network_error(
                    TimeoutError(f"Download timed out after {timeout}s"), url
                )
                raise
            feedback.warning(f"Download timed out, trying next strategy...")

        except subprocess.CalledProcessError as e:
            error_output = e.stderr if e.stderr else str(e)

            # Check for specific anti-bot errors using utility
            is_antibot, error_description = check_antibot_error(error_output)

            if is_antibot:
                if attempt < len(download_strategies):
                    feedback.warning(
                        f"{error_description} - trying strategy {attempt + 1}..."
                    )
                    time.sleep(2**attempt)  # Exponential backoff: 2s, 4s, 8s
                    continue
                else:
                    from utils.antibot_utils import create_cookie_instructions
                    feedback.error("All anti-bot bypass strategies failed")
                    feedback.info("💡 Try exporting browser cookies:")
                    feedback.info(create_cookie_instructions())

            if attempt == len(download_strategies):
                error_handler.handle_network_error(e, url)
                raise
            feedback.warning(
                f"Download failed: {error_output[:100]}..., trying next strategy..."
            )
            time.sleep(1)

    # Find downloaded file
    files = sorted(
        list(workdir.glob("*")), key=lambda p: p.stat().st_mtime, reverse=True
    )
    if not files:
        feedback.error(
            "No files were downloaded",
            solution="Check URL validity and network connection",
        )
        raise RuntimeError("yt-dlp did not produce any files")

    downloaded = files[0]
    wav = workdir / f"{downloaded.stem}.wav"

    feedback.substep(f"Converting to WAV: {downloaded.name} → {wav.name}")
    ffmpeg_cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(downloaded),
        "-ar",
        str(SAMPLE_RATE),
        "-ac",
        str(CHANNELS),
        "-f",
        "wav",
        str(wav),
    ]

    # Adaptive timeout for conversion based on file size
    try:
        file_size_mb = downloaded.stat().st_size / (1024 * 1024)
        conversion_timeout = max(
            120, int(file_size_mb * 2)
        )  # 2 seconds per MB, minimum 120s
        feedback.debug(
            f"File size: {file_size_mb:.1f}MB, conversion timeout: {conversion_timeout}s"
        )

        with feedback.status("Converting audio format..."):
            secure_proc.run_secure(
                ffmpeg_cmd, check=True, capture_output=True, text=True, timeout=conversion_timeout
            )
        feedback.success(f"Audio converted: {wav.name}")
    except subprocess.CalledProcessError as e:
        error_handler.handle_file_error(wav, e, "conversion")
        raise

    return wav

def get_adaptive_timeout(url: str) -> int:
    """Calculate adaptive timeout based on content type."""
    if "youtube.com" in url or "youtu.be" in url:
        return 600  # 10 minutes for YouTube (can be large)
    else:
        return 300  # 5 minutes for other sources


def enhanced_vad_segments(
    audio_path: Path, feedback, sr=SAMPLE_RATE, min_s: float = 0.5
) -> List[Dict[str, float]]:
    """Enhanced VAD with multiple fallbacks and detailed feedback."""

    error_handler = ErrorHandler(feedback)
    secure_proc = SecureSubprocess(feedback)
    audio_cache = get_audio_cache()
    audio_cache.set_feedback(feedback)

    feedback.substep(f"Voice Activity Detection on: {audio_path.name}")

    # Try to get audio from cache first
    cached_audio = audio_cache.get(audio_path)
    if cached_audio is not None:
        audio_array, cached_sr = cached_audio
        if cached_sr == sr:
            feedback.debug(f"Using cached audio data for {audio_path.name}")
        else:
            # Resample cached audio if needed
            if LIBROSA_AVAILABLE:
                audio_array = librosa.resample(audio_array, orig_sr=cached_sr, target_sr=sr)
            else:
                feedback.warning("Librosa not available for resampling cached audio")
    else:
        # Load audio normally
        if SILERO_AVAILABLE:
            feedback.debug("Loading audio with torchaudio")
            waveform, orig_sr = torchaudio.load(str(audio_path))
            
            # Move to CUDA if available for optimal processing
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            waveform = waveform.to(device)
            
            if orig_sr != sr:
                resampler = torchaudio.transforms.Resample(orig_sr, sr).to(device)
                waveform = resampler(waveform)

            # B200 tensor validation for audio processing
            try:
                validate_tensor_device(waveform, require_cuda=torch.cuda.is_available(), name="audio_waveform")
                validate_audio_tensor(waveform, sample_rate=sr, name="loaded_audio")
                check_tensor_health(waveform, name="audio_waveform", 
                                  check_range=(-1.0, 1.0))  # Audio should be normalized
            except ValueError as e:
                feedback.warning(f"Audio tensor validation failed: {e}")
                feedback.info("Attempting to fix audio tensor issues...")
                
                # Attempt basic fixes
                if torch.isnan(waveform).any():
                    waveform = torch.nan_to_num(waveform, nan=0.0)
                    feedback.debug("Fixed NaN values in audio")
                
                if torch.isinf(waveform).any():
                    waveform = torch.nan_to_num(waveform, posinf=1.0, neginf=-1.0)
                    feedback.debug("Fixed Inf values in audio")
                
                # Normalize if outside expected range
                if waveform.abs().max() > 1.0:
                    waveform = waveform / waveform.abs().max()
                    feedback.debug("Normalized audio to [-1, 1] range")

            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)

            # Only transfer to CPU when needed for caching (minimal transfer)
            audio_array = waveform.squeeze().cpu().numpy()
            
            # Cache the audio for future use
            audio_cache.put(audio_path, audio_array, sr)
        else:
            # Fallback to soundfile
            feedback.debug("Loading audio with soundfile (torchaudio not available)")
            audio_array, orig_sr = sf.read(str(audio_path))
            if orig_sr != sr:
                if LIBROSA_AVAILABLE:
                    audio_array = librosa.resample(audio_array, orig_sr=orig_sr, target_sr=sr)
                else:
                    feedback.warning("Cannot resample audio: librosa not available")
            
            # Cache the audio
            audio_cache.put(audio_path, audio_array, sr)

    # Now run VAD on the audio array (whether cached or freshly loaded)
    if SILERO_AVAILABLE:
        try:
            feedback.debug("Running Silero VAD")
            with feedback.status("Detecting speech segments..."):
                model = load_silero_vad()
                timestamps = get_speech_timestamps(audio_array, model, sampling_rate=sr)

            segments = []
            for ts in timestamps:
                start = ts["start"] / sr
                end = ts["end"] / sr
                if (end - start) >= min_s:
                    segments.append({"start": float(start), "end": float(end)})

            feedback.success(f"Silero VAD found {len(segments)} speech segments")
            return segments

        except Exception as e:
            feedback.warning(
                f"Silero VAD failed: {e}", solution="Falling back to energy-based VAD"
            )

    # Fallback: energy-based VAD
    feedback.substep("Using fallback energy-based VAD")
    try:
        audio_array, _ = sf.read(str(audio_path))

        frame_length = int(sr * 0.025)  # 25ms frames
        hop_length = int(sr * 0.010)  # 10ms hop

        energy = []
        for i in range(0, len(audio_array) - frame_length, hop_length):
            frame = audio_array[i : i + frame_length]
            energy.append(float((frame**2).mean()))

        if not energy:
            feedback.warning("No energy detected in audio")
            return []

        mean_energy = sum(energy) / len(energy)
        threshold = mean_energy * 2.0

        segments = []
        in_speech = False
        start_frame = 0

        for i, e in enumerate(energy):
            time_sec = i * hop_length / sr

            if e > threshold and not in_speech:
                in_speech = True
                start_frame = i
            elif e <= threshold and in_speech:
                in_speech = False
                start_time = start_frame * hop_length / sr
                end_time = time_sec

                if (end_time - start_time) >= min_s:
                    segments.append({"start": float(start_time), "end": float(end_time)})

        if in_speech:
            start_time = start_frame * hop_length / sr
            end_time = len(audio_array) / sr
            if (end_time - start_time) >= min_s:
                segments.append({"start": float(start_time), "end": float(end_time)})

        feedback.success(f"Energy-based VAD found {len(segments)} segments")
        return segments

    except Exception as e:
        error_handler.handle_file_error(audio_path, e, "VAD processing")
        feedback.warning(
            "All VAD methods failed, using full audio as single segment"
        )
        return [{"start": 0.0, "end": 60.0}]  # Default fallback
