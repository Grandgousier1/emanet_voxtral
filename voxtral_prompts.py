#!/usr/bin/env python3
"""
voxtral_prompts.py - Optimized prompts for Voxtral translation
Specialized for Turkish drama series → French translation
"""

from typing import Dict, Optional, Any

def get_voxtral_prompt(
    source_lang: str = "Turkish", 
    target_lang: str = "French",
    context: str = "drama",
    speaker_context: Optional[str] = None
) -> str:
    """
    Generate optimized prompt for Voxtral transcription and translation.
    
    Args:
        source_lang: Source language (default: Turkish)
        target_lang: Target language (default: French) 
        context: Content type (drama, news, conversation, etc.)
        speaker_context: Optional speaker information for consistency
        
    Returns:
        Optimized prompt string for Voxtral
    """
    
    # Base system prompt for audio transcription + translation
    system_prompt = f"""You are an expert audio transcriber and translator specializing in {source_lang} to {target_lang} translation."""
    
    # Context-specific instructions
    context_instructions = {
        "drama": {
            "style": "dramatic and emotional",
            "tone": "Preserve the emotional intensity and dramatic tension",
            "specifics": [
                "Maintain character voice distinctions",
                "Preserve emotional undertones (anger, sadness, love, etc.)",
                "Keep dramatic pauses and emphasis", 
                "Translate Turkish cultural references appropriately for French audience",
                "Use natural French dialogue that sounds authentic in dubbed drama"
            ]
        },
        "conversation": {
            "style": "natural and conversational", 
            "tone": "Maintain casual speaking patterns",
            "specifics": [
                "Preserve speaking style and colloquialisms",
                "Keep natural speech patterns"
            ]
        },
        "news": {
            "style": "formal and precise",
            "tone": "Maintain professional tone",
            "specifics": [
                "Ensure factual accuracy",
                "Keep formal register"
            ]
        }
    }
    
    current_context = context_instructions.get(context, context_instructions["conversation"])
    
    # Construct detailed prompt
    prompt = f"""{system_prompt}

## Task: Transcribe and translate {source_lang} audio to {target_lang}

### Content Context: {context.title()}
- Style: {current_context['style']}
- Tone: {current_context['tone']}

### Translation Guidelines:
{chr(10).join(f"- {spec}" for spec in current_context['specifics'])}

### Quality Requirements:
- Accurate transcription of {source_lang} speech
- Natural, fluent {target_lang} translation
- Preserve speaker emotions and intentions
- Maintain subtitle-appropriate length (max 42 characters per line)
- Use proper {target_lang} grammar and syntax
- Keep cultural context relevant for {target_lang} audience

### Output Format:
Provide only the final {target_lang} translation, ready for subtitles."""

    if speaker_context:
        prompt += f"\n\n### Speaker Context:\n{speaker_context}"
    
    return prompt

def get_turkish_drama_specifics() -> Dict[str, str]:
    """
    Get specific translation guidance for Turkish drama series.
    
    Returns:
        Dictionary with Turkish drama translation specifics
    """
    return {
        "emotional_expressions": {
            "Turkish": ["Aşkım", "Canım", "Hayatım", "Kalbim"],
            "French_equivalents": ["Mon amour", "Mon cœur", "Ma vie", "Mon âme"],
            "context": "Use appropriate level of intimacy based on relationship"
        },
        "cultural_adaptations": {
            "family_titles": "Adapt Turkish family titles (Abla, Abi, Teyze) to French equivalents or explanatory terms",
            "religious_references": "Translate religious expressions naturally for French secular context",
            "social_customs": "Explain or adapt Turkish social customs for French understanding"
        },
        "dramatic_style": {
            "intensity": "Turkish dramas are highly emotional - maintain this intensity in French",
            "repetition": "Turkish uses repetition for emphasis - adapt to French rhetorical patterns",
            "formality": "Respect Turkish formal/informal speech levels in French translation"
        }
    }

def get_subtitle_constraints() -> Dict[str, int]:
    """
    Get optimal subtitle formatting constraints.
    
    Returns:
        Dictionary with subtitle timing and formatting constraints
    """
    return {
        "max_chars_per_line": 42,  # Standard for French subtitles
        "max_lines": 2,           # Maximum lines per subtitle
        "min_duration_ms": 1000,  # Minimum display time (1 second)
        "max_duration_ms": 6000,  # Maximum display time (6 seconds)
        "min_gap_ms": 83,         # Minimum gap between subtitles (2 frames at 24fps)
        "reading_speed_cps": 15,  # Characters per second reading speed
        "words_per_minute": 160   # Average speaking speed
    }

def optimize_subtitle_timing(text: str, start_time: float, end_time: float) -> Dict[str, float]:
    """
    Optimize subtitle timing based on text length and reading speed.
    
    Args:
        text: Subtitle text
        start_time: Original start time in seconds
        end_time: Original end time in seconds
        
    Returns:
        Dictionary with optimized start and end times
    """
    constraints = get_subtitle_constraints()
    
    # Calculate optimal duration based on text length
    char_count = len(text)
    optimal_duration = max(
        char_count / constraints["reading_speed_cps"],
        constraints["min_duration_ms"] / 1000
    )
    
    # Don't exceed maximum duration
    optimal_duration = min(optimal_duration, constraints["max_duration_ms"] / 1000)
    
    # Calculate available time
    available_time = end_time - start_time
    
    if available_time >= optimal_duration:
        # Use optimal duration, keep start time
        return {
            "start": start_time,
            "end": start_time + optimal_duration
        }
    else:
        # Use all available time but warn if too short
        # Protection contre division par zéro
        if available_time <= 0.001:  # Minimum 1ms pour éviter division par zéro
            return {
                "start": start_time,
                "end": start_time + 1.0,  # Durée minimale de 1 seconde
                "warning": f"Segment too short ({available_time:.3f}s), extended to 1s minimum"
            }
        
        chars_per_sec = char_count / available_time
        return {
            "start": start_time,
            "end": end_time,
            "warning": f"Subtitle too fast: {chars_per_sec:.1f} chars/sec (max {constraints['reading_speed_cps']})"
        }

def get_vllm_generation_params() -> Dict[str, Any]:
    """
    Get scientifically consistent generation parameters for vLLM with Voxtral.
    
    For subtitle generation, we prioritize deterministic behavior.
    Temperature=0 ensures completely deterministic output in vLLM.
    
    Returns:
        Dictionary with vLLM generation parameters
    """
    return {
        "max_tokens": 128,        # Shorter for subtitles
        "temperature": 0.0,       # Completely deterministic (vLLM supports this)
        "top_p": 1.0,            # Disabled when temperature=0
        "frequency_penalty": 0.1, # Reduce repetition
        "presence_penalty": 0.0,  # No penalty for presence
        "stop": ["\n\n", "###"], # Stop tokens
        "skip_special_tokens": True,
        "use_beam_search": True,  # Use beam search for quality
        "best_of": 3,            # Consider 3 candidates
    }

def get_transformers_generation_params() -> Dict[str, Any]:
    """
    Get scientifically consistent generation parameters for Transformers with Voxtral.
    
    For subtitle generation, we prioritize consistency and reproducibility over creativity.
    With low temperature (0.1), deterministic decoding is more appropriate than sampling.
    
    Returns:
        Dictionary with Transformers generation parameters
    """
    return {
        "max_new_tokens": 128,           # Shorter for subtitles
        "temperature": 1.0,              # Standard temp for deterministic mode
        "do_sample": False,              # Deterministic decoding for consistency
        "num_beams": 3,                  # Beam search for quality
        "early_stopping": True,          # Stop when done
        "repetition_penalty": 1.1,       # Slight penalty for repetition
        "length_penalty": 0.8,           # Prefer shorter outputs
        "no_repeat_ngram_size": 3,       # Avoid repeating trigrams
        "use_cache": True,               # Enable KV cache for consistency
    }



def validate_translation_quality(original_text: str, translated_text: str) -> Dict[str, Any]:
    """
    Basic validation of translation quality for Turkish drama.
    
    Args:
        original_text: Original Turkish text (if available)
        translated_text: Translated French text
        
    Returns:
        Dictionary with quality metrics and suggestions
    """
    issues = []
    suggestions = []
    
    # Check length for subtitles
    if len(translated_text) > 84:  # 2 lines * 42 chars
        issues.append("Translation too long for standard subtitles")
        suggestions.append("Consider splitting into multiple subtitles")
    
    # Check for common Turkish words that might not be translated
    turkish_words = ["abi", "abla", "efendi", "hanım", "bey"]
    for word in turkish_words:
        if word.lower() in translated_text.lower():
            issues.append(f"Possible untranslated Turkish word: {word}")
            suggestions.append(f"Ensure '{word}' is properly translated or adapted")
    
    # Check for emotional intensity markers
    emotional_markers = ["!", "...", "?", "...!"]
    has_emotion = any(marker in translated_text for marker in emotional_markers)
    
    return {
        "length_ok": len(translated_text) <= 84,
        "char_count": len(translated_text),
        "has_emotional_markers": has_emotion,
        "issues": issues,
        "suggestions": suggestions,
        "quality_score": max(0, 10 - len(issues))  # Simple scoring
    }

# Pre-defined prompts for common scenarios
TURKISH_DRAMA_PROMPTS = {
    "romantic_scene": get_voxtral_prompt(
        context="drama",
        speaker_context="Romantic dialogue between main characters. Maintain intimate and emotional tone."
    ),
    "family_conflict": get_voxtral_prompt(
        context="drama", 
        speaker_context="Family argument or emotional confrontation. Preserve anger and frustration."
    ),
    "dramatic_revelation": get_voxtral_prompt(
        context="drama",
        speaker_context="Major plot revelation or shocking discovery. Maintain suspense and drama."
    ),
    "general_dialogue": get_voxtral_prompt(context="drama")
}

if __name__ == "__main__":
    # Test prompt generation
    print("=== Turkish Drama Voxtral Prompt ===")
    print(get_voxtral_prompt())
    
    print("\n=== Generation Parameters ===")
    print("vLLM:", get_vllm_generation_params())
    print("Transformers:", get_transformers_generation_params())
    
    print("\n=== Subtitle Constraints ===")
    print(get_subtitle_constraints())