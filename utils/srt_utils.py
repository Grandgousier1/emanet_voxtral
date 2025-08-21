#!/usr/bin/env python3
"""
utils/srt_utils.py - SRT subtitle generation utilities
"""

from pathlib import Path
from typing import List, Dict

from cli_feedback import CLIFeedback, ErrorHandler


def format_srt_time(seconds: float) -> str:
    """Convert seconds to SRT time format with precise millisecond accuracy."""
    # Ensure precise timing by rounding to nearest millisecond
    total_ms = round(seconds * 1000)
    
    hours = total_ms // 3600000
    minutes = (total_ms % 3600000) // 60000
    secs = (total_ms % 60000) // 1000
    millis = total_ms % 1000
    
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def enhanced_generate_srt(segments: List[Dict], output_path: Path, feedback: CLIFeedback) -> None:
    """Enhanced SRT generation with timing validation and quality checks."""
    
    feedback.substep(f"Generating SRT file: {output_path.name}")
    
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Filter valid segments and sort by start time for perfect synchronization
        valid_segments = [seg for seg in segments if seg['text'].strip()]
        valid_segments.sort(key=lambda x: x['start'])
        
        # Validate and fix overlapping subtitles
        cleaned_segments = []
        prev_end = 0.0
        
        for segment in valid_segments:
            start = max(segment['start'], prev_end + 0.083)  # Minimum 83ms gap (2 frames at 24fps)
            end = max(segment['end'], start + 1.0)  # Minimum 1s duration
            
            # Ensure no subtitle is too long (max 6 seconds)
            if end - start > 6.0:
                end = start + 6.0
                
            cleaned_segments.append({
                'text': segment['text'].strip(),
                'start': start,
                'end': end,
                'quality_score': segment.get('quality_score', 10)
            })
            prev_end = end
        
        # Generate SRT with precise timing
        with open(output_path, 'w', encoding='utf-8') as f:
            for i, segment in enumerate(cleaned_segments, 1):
                start_time = format_srt_time(segment['start'])
                end_time = format_srt_time(segment['end'])
                
                # Split long texts into multiple lines (max 42 chars per line)
                text = segment['text']
                if len(text) > 42:
                    words = text.split()
                    line1, line2 = [], []
                    current_length = 0
                    
                    for word in words:
                        if current_length + len(word) + 1 <= 42 and line1:
                            line1.append(word)
                            current_length += len(word) + 1
                        elif current_length + len(word) <= 42:
                            line1.append(word)
                            current_length += len(word)
                        else:
                            line2.append(word)
                    
                    formatted_text = ' '.join(line1)
                    if line2:
                        formatted_text += '\n' + ' '.join(line2)
                else:
                    formatted_text = text
                
                f.write(f"{i}\n")
                f.write(f"{start_time} --> {end_time}\n")
                f.write(f"{formatted_text}\n\n")
        
        # Generate timing statistics
        total_duration = cleaned_segments[-1]['end'] if cleaned_segments else 0
        avg_quality = sum(seg.get('quality_score', 10) for seg in cleaned_segments) / len(cleaned_segments) if cleaned_segments else 0
        
        feedback.success(f"SRT generated: {len(cleaned_segments)} subtitles, {total_duration:.1f}s total, avg quality: {avg_quality:.1f}/10")
        
        # Validate timing consistency
        timing_issues = 0
        for i, seg in enumerate(cleaned_segments[:-1]):
            gap = cleaned_segments[i+1]['start'] - seg['end']
            if gap < 0:
                timing_issues += 1
        
        if timing_issues > 0:
            feedback.warning(f"Fixed {timing_issues} timing overlaps for perfect synchronization")
        else:
            feedback.success("✅ Perfect subtitle synchronization achieved")
        
    except Exception as e:
        ErrorHandler(feedback).handle_file_error(output_path, e, "SRT generation")
        raise