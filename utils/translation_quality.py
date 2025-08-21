#!/usr/bin/env python3
"""
utils/translation_quality.py - Advanced translation quality metrics
Scientific validation for Turkish→French subtitle translation
"""

import re
import math
from typing import Dict, List, Optional, Tuple, Any
from collections import Counter
import logging

logger = logging.getLogger(__name__)

class TranslationQualityValidator:
    """Advanced translation quality validation with scientific metrics."""
    
    def __init__(self):
        """Initialize translation quality validator."""
        # Turkish drama specific patterns
        self.turkish_patterns = {
            'emotional_markers': [
                r'ah\b', r'oh\b', r'ey\b', r'ya\b', r'işte\b', r'hadi\b',
                r'tamam\b', r'peki\b', r'hayır\b', r'evet\b'
            ],
            'cultural_terms': [
                r'abla\b', r'abi\b', r'teyze\b', r'amca\b', r'hoca\b',
                r'efendi\b', r'hanım\b', r'bey\b'
            ],
            'formality_markers': [
                r'siz\b', r'sizin\b', r'sizde\b', r'sizdeki\b'  # Formal you
            ]
        }
        
        # French drama equivalents and quality indicators
        self.french_patterns = {
            'emotional_markers': [
                r'ah\b', r'oh\b', r'eh\b', r'bon\b', r'alors\b', r'allez\b',
                r'oui\b', r'non\b', r'si\b', r'enfin\b'
            ],
            'formality_markers': [
                r'vous\b', r'votre\b', r'vos\b', r'monsieur\b', r'madame\b'
            ],
            'subtitle_quality': [
                r'^[A-Z]',  # Starts with capital
                r'[.!?]$',  # Ends with punctuation
                r'\b(le|la|les|un|une|des)\b'  # Articles present
            ]
        }
    
    def validate_translation_completeness(self, source: str, target: str) -> Dict[str, Any]:
        """
        Validate translation completeness and content preservation.
        
        Args:
            source: Source Turkish text
            target: Target French text
            
        Returns:
            Validation metrics and scores
        """
        metrics = {
            'completeness_score': 0.0,
            'length_ratio': 0.0,
            'content_preservation': 0.0,
            'issues': []
        }
        
        if not source or not target:
            metrics['issues'].append("Empty source or target text")
            return metrics
        
        # Length ratio analysis (French typically 15-20% longer than Turkish)
        source_len = len(source.strip())
        target_len = len(target.strip())
        
        if target_len == 0:
            metrics['issues'].append("Empty translation")
            return metrics
            
        length_ratio = target_len / source_len
        metrics['length_ratio'] = length_ratio
        
        # Expected ratio for Turkish→French: 1.1-1.3
        if length_ratio < 0.5:
            metrics['issues'].append(f"Translation suspiciously short (ratio: {length_ratio:.2f})")
        elif length_ratio > 2.0:
            metrics['issues'].append(f"Translation suspiciously long (ratio: {length_ratio:.2f})")
        
        # Completeness score based on length ratio
        if 0.8 <= length_ratio <= 1.5:
            metrics['completeness_score'] = 1.0
        elif 0.6 <= length_ratio <= 2.0:
            metrics['completeness_score'] = 0.7
        else:
            metrics['completeness_score'] = 0.3
            
        return metrics
    
    def validate_cultural_adaptation(self, source: str, target: str) -> Dict[str, Any]:
        """
        Validate cultural adaptation quality for Turkish drama.
        
        Args:
            source: Source Turkish text
            target: Target French text
            
        Returns:
            Cultural adaptation metrics
        """
        metrics = {
            'cultural_score': 0.0,
            'formality_preserved': False,
            'emotional_tone_preserved': False,
            'adaptations': []
        }
        
        source_lower = source.lower()
        target_lower = target.lower()
        
        # Check emotional markers preservation
        source_emotional = sum(1 for pattern in self.turkish_patterns['emotional_markers'] 
                             if re.search(pattern, source_lower))
        target_emotional = sum(1 for pattern in self.french_patterns['emotional_markers']
                             if re.search(pattern, target_lower))
        
        if source_emotional > 0:
            emotion_ratio = min(target_emotional / source_emotional, 1.0)
            metrics['emotional_tone_preserved'] = emotion_ratio > 0.5
            metrics['cultural_score'] += emotion_ratio * 0.4
        else:
            metrics['emotional_tone_preserved'] = True
            metrics['cultural_score'] += 0.4
        
        # Check formality preservation
        source_formal = sum(1 for pattern in self.turkish_patterns['formality_markers']
                          if re.search(pattern, source_lower))
        target_formal = sum(1 for pattern in self.french_patterns['formality_markers']
                          if re.search(pattern, target_lower))
        
        if source_formal > 0:
            formality_ratio = min(target_formal / source_formal, 1.0)
            metrics['formality_preserved'] = formality_ratio > 0.5
            metrics['cultural_score'] += formality_ratio * 0.3
        else:
            metrics['formality_preserved'] = True
            metrics['cultural_score'] += 0.3
            
        # Check subtitle quality indicators
        quality_score = sum(1 for pattern in self.french_patterns['subtitle_quality']
                           if re.search(pattern, target)) / len(self.french_patterns['subtitle_quality'])
        metrics['cultural_score'] += quality_score * 0.3
        
        return metrics
    
    def calculate_repetition_penalty(self, text: str) -> float:
        """
        Calculate repetition penalty for generated text.
        
        Args:
            text: Generated text to analyze
            
        Returns:
            Repetition penalty score (0.0 = high repetition, 1.0 = no repetition)
        """
        if not text:
            return 0.0
            
        words = text.lower().split()
        if len(words) < 2:
            return 1.0
            
        # Calculate unique word ratio
        unique_words = len(set(words))
        total_words = len(words)
        unique_ratio = unique_words / total_words
        
        # Check for immediate repetitions
        immediate_reps = sum(1 for word, next_word in zip(words[:-1], words[1:]) if word == next_word)
        immediate_penalty = max(0, 1.0 - (immediate_reps / max(len(words)-1, 1)))
        
        # Check for n-gram repetitions
        bigrams = [f"{word} {next_word}" for word, next_word in zip(words[:-1], words[1:])]
        bigram_counts = Counter(bigrams)
        bigram_reps = sum(count-1 for count in bigram_counts.values() if count > 1)
        bigram_penalty = max(0, 1.0 - (bigram_reps / max(len(bigrams), 1)))
        
        return (unique_ratio + immediate_penalty + bigram_penalty) / 3
    
    def validate_subtitle_constraints(self, text: str, duration: float) -> Dict[str, Any]:
        """
        Validate subtitle timing and readability constraints.
        
        Args:
            text: Subtitle text
            duration: Duration in seconds
            
        Returns:
            Subtitle constraint validation results
        """
        metrics = {
            'timing_score': 0.0,
            'readability_score': 0.0,
            'constraints_met': True,
            'violations': []
        }
        
        if not text or duration <= 0:
            metrics['constraints_met'] = False
            metrics['violations'].append("Invalid text or duration")
            return metrics
        
        # Character count and reading speed
        char_count = len(text)
        chars_per_second = char_count / duration
        
        # Subtitle constraints (industry standard)
        max_chars_per_second = 15  # Comfortable reading speed
        max_chars_total = 42       # Single line limit
        min_duration = 1.0         # Minimum subtitle duration
        
        # Timing validation
        if duration < min_duration:
            metrics['violations'].append(f"Duration too short: {duration:.1f}s < {min_duration}s")
            metrics['constraints_met'] = False
        
        if chars_per_second > max_chars_per_second:
            metrics['violations'].append(f"Reading speed too fast: {chars_per_second:.1f} > {max_chars_per_second} chars/sec")
            metrics['constraints_met'] = False
        
        if char_count > max_chars_total:
            metrics['violations'].append(f"Text too long: {char_count} > {max_chars_total} chars")
            metrics['constraints_met'] = False
        
        # Calculate scores
        timing_score = min(1.0, min_duration / duration) if duration > 0 else 0
        speed_score = min(1.0, max_chars_per_second / max(chars_per_second, 1))
        length_score = min(1.0, max_chars_total / max(char_count, 1))
        
        metrics['timing_score'] = timing_score
        metrics['readability_score'] = (speed_score + length_score) / 2
        
        return metrics
    
    def comprehensive_quality_assessment(
        self, 
        source: str, 
        target: str, 
        duration: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Comprehensive quality assessment for Turkish→French subtitle translation.
        
        Args:
            source: Source Turkish text
            target: Target French text
            duration: Subtitle duration in seconds (optional)
            
        Returns:
            Comprehensive quality metrics
        """
        assessment = {
            'overall_score': 0.0,
            'quality_level': 'poor',
            'detailed_metrics': {},
            'recommendations': []
        }
        
        # Completeness validation
        completeness = self.validate_translation_completeness(source, target)
        assessment['detailed_metrics']['completeness'] = completeness
        
        # Cultural adaptation
        cultural = self.validate_cultural_adaptation(source, target)
        assessment['detailed_metrics']['cultural'] = cultural
        
        # Repetition analysis
        repetition_score = self.calculate_repetition_penalty(target)
        assessment['detailed_metrics']['repetition_score'] = repetition_score
        
        # Subtitle constraints (if duration provided)
        if duration is not None:
            constraints = self.validate_subtitle_constraints(target, duration)
            assessment['detailed_metrics']['subtitle_constraints'] = constraints
        
        # Calculate overall score
        scores = [
            completeness['completeness_score'] * 0.3,
            cultural['cultural_score'] * 0.3,
            repetition_score * 0.2
        ]
        
        if duration is not None:
            subtitle_score = (
                assessment['detailed_metrics']['subtitle_constraints']['timing_score'] * 0.1 +
                assessment['detailed_metrics']['subtitle_constraints']['readability_score'] * 0.1
            )
            scores.append(subtitle_score)
        
        assessment['overall_score'] = sum(scores)
        
        # Quality level classification
        if assessment['overall_score'] >= 0.8:
            assessment['quality_level'] = 'excellent'
        elif assessment['overall_score'] >= 0.6:
            assessment['quality_level'] = 'good'
        elif assessment['overall_score'] >= 0.4:
            assessment['quality_level'] = 'fair'
        else:
            assessment['quality_level'] = 'poor'
        
        # Generate recommendations
        if completeness['completeness_score'] < 0.6:
            assessment['recommendations'].append("Improve translation completeness")
        if cultural['cultural_score'] < 0.6:
            assessment['recommendations'].append("Better cultural adaptation needed")
        if repetition_score < 0.7:
            assessment['recommendations'].append("Reduce repetitive language")
        if duration and not assessment['detailed_metrics']['subtitle_constraints']['constraints_met']:
            assessment['recommendations'].append("Adjust subtitle timing or length")
        
        logger.debug(f"Quality assessment: {assessment['quality_level']} ({assessment['overall_score']:.2f})")
        return assessment

# Convenience function for quick validation
def validate_translation_quality(source: str, target: str, duration: Optional[float] = None) -> Dict[str, Any]:
    """
    Quick translation quality validation.
    
    Args:
        source: Source text
        target: Target text  
        duration: Optional duration for subtitle constraints
        
    Returns:
        Quality assessment results
    """
    validator = TranslationQualityValidator()
    return validator.comprehensive_quality_assessment(source, target, duration)