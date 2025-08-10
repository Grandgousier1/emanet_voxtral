#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main.py — Emanet Runpod final
Pipeline local full: download -> VAD -> ASR (Voxtral small / mini fallback or faster-whisper) -> translation (mistral-small local) -> .srt
Supports single URL/file and batch list. Includes dry-run smoke tests and sequential batch mode (safe for single GPU).
"""

import argparse
import os
import sys
import time
import json
import sqlite3
import hashlib
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Optional

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeRemainingColumn

console = Console()

# local helpers
from utils.gpu_utils import check_cuda_available, available_device, free_cuda_mem, gpu_mem_info

# constants
SAMPLE_RATE = 16000
CHANNELS = 1
CACHE_DB = Path('.emanet_cache.db')
VOXTRAL_SMALL = 'mistralai/Voxtral-Small-24B-2507'
VOXTRAL_MINI = 'mistralai/Voxtral-Mini-3B-2507'
MISTRAL_SMALL = 'mistralai/mistral-small'

# ---------------- Cache DB -----------------
class CacheDB:
    def __init__(self, path=CACHE_DB):
        self.path = path
        self._ensure()
    def _conn(self):
        return sqlite3.connect(str(self.path))
    def _ensure(self):
        with self._conn() as c:
            c.execute('''CREATE TABLE IF NOT EXISTS translations (k TEXT PRIMARY KEY, src TEXT, trg TEXT, model TEXT, ts REAL)''')
            c.execute('''CREATE TABLE IF NOT EXISTS videos (vid TEXT PRIMARY KEY, srt TEXT, status TEXT, ts REAL)''')
    def get(self,k):
        with self._conn() as c:
            r=c.execute('SELECT trg FROM translations WHERE k=?',(k,)).fetchone()
            return r[0] if r else None
    def set(self,k,src,trg,model):
        with self._conn() as c:
            c.execute('INSERT OR REPLACE INTO translations (k,src,trg,model,ts) VALUES (?,?,?,?,?)',(k,src,trg,model,time.time()))
    def mark_done(self,vid,srt):
        with self._conn() as c:
            c.execute('INSERT OR REPLACE INTO videos (vid,srt,status,ts) VALUES (?,?,?,?)',(vid,srt,'done',time.time()))

# ---------------- Preflight -----------------
def smoke_tests():
    console.rule('[cyan]Preflight checks')
    errors = []
    console.log('Python: ' + sys.version.splitlines()[0])
    try:
        import torch
        console.log('Torch: ' + torch.__version__)
    except Exception:
        errors.append('torch not importable')
    gpu_ok = check_cuda_available()
    console.log('CUDA available: ' + str(gpu_ok))
    for exe in ('ffmpeg','yt-dlp'):
        if subprocess.call(['which', exe], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) != 0:
            errors.append(f'{exe} not found')
    if not errors:
        console.log('[green]Preflight OK')
    else:
        for e in errors:
            console.log(f'[red]{e}[/red]')
        raise RuntimeError('Preflight failed')

# ---------------- Download audio -----------------
def download_audio_from_url(url: str, workdir: Path, cookiefile: Optional[Path]=None) -> Path:
    workdir.mkdir(parents=True, exist_ok=True)
    out_template = str(workdir / '%(id)s.%(ext)s')
    cmd = ['yt-dlp', '-f', 'bestaudio[ext=m4a]/bestaudio', '--no-playlist', '-o', out_template, url]
    if cookiefile:
        cmd += ['--cookies', str(cookiefile)]
    subprocess.run(cmd, check=True)
    files = sorted(list(workdir.glob('*')), key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        raise RuntimeError('yt-dlp did not produce a file')
    downloaded = files[0]
    wav = workdir / f'{downloaded.stem}.wav'
    subprocess.run(['ffmpeg','-y','-i',str(downloaded),'-ar',str(SAMPLE_RATE),'-ac',str(CHANNELS),str(wav)], check=True)
    return wav

# ---------------- VAD -----------------
def vad_segments(audio_path: Path, sr=SAMPLE_RATE, min_s: float = 0.3) -> List[Dict[str, float]]:
    try:
        import torchaudio
    except Exception as e:
        raise RuntimeError('Please install torchaudio') from e
    waveform, orig_sr = torchaudio.load(str(audio_path))
    if orig_sr != sr:
        waveform = torchaudio.transforms.Resample(orig_sr, sr)(waveform)
    arr = waveform.mean(dim=0).cpu().numpy()
    try:
        if True:
            from silero_vad import get_speech_timestamps
            timestamps = get_speech_timestamps(arr, sampling_rate=sr)
            segs = []
            for t in timestamps:
                s = t.get('start', 0); e = t.get('end', 0)
                if s > 1000:
                    s /= 1000.0; e /= 1000.0
                if (e - s) >= min_s:
                    segs.append({'start': float(s), 'end': float(e)})
            return segs
    except Exception:
        # fallback energy-based
        frame_ms = 30
        frame_size = int(sr * frame_ms / 1000)
        energy = []
        for i in range(0, len(arr), frame_size):
            frame = arr[i:i+frame_size]
            energy.append(float((frame**2).mean()) if frame.size else 0.0)
        th = (sum(energy) / len(energy)) * 1.5
        segs = []
        start = None
        for idx, e in enumerate(energy):
            t0 = idx * frame_size / sr
            t1 = (idx + 1) * frame_size / sr
            if e > th:
                if start is None:
                    start = t0
                end = t1
            else:
                if start is not None and (end - start) >= min_s:
                    segs.append({'start': start, 'end': end})
                start = None
        if start is not None and (end - start) >= min_s:
            segs.append({'start': start, 'end': end})
        return segs

# ---------------- ASR Voxtral local (with fallback) -----------------
def try_load_voxtral(model_name: str):
    try:
        from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
        proc = AutoProcessor.from_pretrained(model_name)
        model = AutoModelForSpeechSeq2Seq.from_pretrained(model_name, torch_dtype=torch.float16, device_map='auto')
        return proc, model
    except Exception as e:
        console.log(f'[yellow]Voxtral load failed for {model_name}: {e}[/yellow]')
        return None, None

def asr_with_voxtral(audio_wav: Path, segments: List[Dict[str, Any]], prefer_small: bool = True):
    order = [VOXTRAL_SMALL, VOXTRAL_MINI] if prefer_small else [VOXTRAL_MINI, VOXTRAL_SMALL]
    for model_id in order:
        proc, model = try_load_voxtral(model_id)
        if proc and model:
            console.log(f'[green]Using Voxtral model {model_id}[/green]')
            out = []
            import soundfile as sf
            for idx, s in enumerate(segments, start=1):
                tmp = audio_wav.parent / f'vox_seg_{idx:04d}.wav'
                dur = s['end'] - s['start']
                subprocess.run(['ffmpeg','-y','-i',str(audio_wav),'-ss',str(s['start']),'-t',str(dur),'-ar',str(SAMPLE_RATE),'-ac',str(CHANNELS),str(tmp)], check=True)
                arr, sr = sf.read(str(tmp))
                inputs = proc(arr, sampling_rate=sr, return_tensors='pt')
                inputs = {k: v.to(model.device) for k, v in inputs.items()}
                with torch.no_grad():
                    gen = model.generate(**inputs)
                try:
                    decoded = proc.batch_decode(gen, skip_sp
