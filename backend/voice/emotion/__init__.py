"""
AImpact Advanced Emotion Detection System

This module provides sophisticated emotion detection and analysis capabilities
with real-time processing, multi-modal fusion, and cultural awareness.

Features:
- Real-time emotion analysis
- Multi-modal detection (audio + text)
- Cultural context awareness
- Advanced emotional state recognition
- Confidence scoring and intensity measurement
- Temporal emotion tracking
"""

__version__ = "0.1.0"

from .models import EmotionCategory, EmotionIntensity, CulturalContext
from .analyzer import EmotionAnalyzer
from .acoustic_processor import AcousticEmotionProcessor
from .linguistic_processor import LinguisticEmotionProcessor
from .multimodal_fusion import MultimodalEmotionFusion

__all__ = [
    "EmotionCategory",
    "EmotionIntensity",
    "CulturalContext",
    "EmotionAnalyzer",
    "AcousticEmotionProcessor",
    "LinguisticEmotionProcessor",
    "MultimodalEmotionFusion"
]

