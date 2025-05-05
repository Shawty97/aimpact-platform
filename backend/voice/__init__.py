"""
AImpact Advanced Voice Processing System

This package provides advanced voice processing capabilities including:
- Speech-to-text and text-to-speech with multiple providers
- Emotion detection in voice
- Real-time voice cloning
- Multi-language support
- WebSocket streaming for real-time voice interaction
"""

__version__ = "0.1.0"

# Core components
from backend.voice.core.speech_engine import SpeechEngine
from backend.voice.core.emotion_detector import EmotionDetector
from backend.voice.core.voice_cloner import VoiceCloner

# Make key components available at package level
__all__ = [
    "SpeechEngine",
    "EmotionDetector",
    "VoiceCloner",
]

