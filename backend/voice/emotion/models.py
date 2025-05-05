"""
Emotion detection models and data structures.

This module defines the data models used for emotion detection and analysis.
"""

from enum import Enum
from typing import Dict, List, Optional, Union, Any, Set
from datetime import datetime
from pydantic import BaseModel, Field

class EmotionCategory(str, Enum):
    """Categories of emotions that can be detected."""
    # Basic emotions (Ekman's six basic emotions plus neutral)
    NEUTRAL = "neutral"
    HAPPY = "happy"
    SAD = "sad"
    ANGRY = "angry"
    FEARFUL = "fearful"
    DISGUSTED = "disgusted"
    SURPRISED = "surprised"
    
    # Complex emotional states
    STRESSED = "stressed"
    CONFIDENT = "confident"
    UNCERTAIN = "uncertain"
    INTERESTED = "interested"
    BORED = "bored"
    CONFUSED = "confused"
    ENTHUSIASTIC = "enthusiastic"
    DEFENSIVE = "defensive"
    EMPATHETIC = "empathetic"
    SARCASTIC = "sarcastic"
    FRUSTRATED = "frustrated"
    AMUSED = "amused"
    CONTEMPLATIVE = "contemplative"
    
    # Interpersonal emotions
    TRUSTING = "trusting"
    SUSPICIOUS = "suspicious"
    ADMIRING = "admiring"
    CRITICAL = "critical"
    
    @classmethod
    def basic_emotions(cls) -> Set["EmotionCategory"]:
        """Get the set of basic emotions."""
        return {
            cls.NEUTRAL, cls.HAPPY, cls.SAD, cls.ANGRY, 
            cls.FEARFUL, cls.DISGUSTED, cls.SURPRISED
        }
    
    @classmethod
    def complex_emotions(cls) -> Set["EmotionCategory"]:
        """Get the set of complex emotional states."""
        return {
            cls.STRESSED, cls.CONFIDENT, cls.UNCERTAIN, cls.INTERESTED,
            cls.BORED, cls.CONFUSED, cls.ENTHUSIASTIC, cls.DEFENSIVE,
            cls.EMPATHETIC, cls.SARCASTIC, cls.FRUSTRATED, cls.AMUSED,
            cls.CONTEMPLATIVE
        }
    
    @classmethod
    def interpersonal_emotions(cls) -> Set["EmotionCategory"]:
        """Get the set of interpersonal emotions."""
        return {
            cls.TRUSTING, cls.SUSPICIOUS, cls.ADMIRING, cls.CRITICAL
        }

class EmotionIntensity(str, Enum):
    """Intensity levels for detected emotions."""
    VERY_LOW = "very_low"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    VERY_HIGH = "very_high"
    
    @classmethod
    def from_score(cls, score: float) -> "EmotionIntensity":
        """Convert a numerical score (0-1) to an intensity level."""
        if score < 0.2:
            return cls.VERY_LOW
        elif score < 0.4:
            return cls.LOW
        elif score < 0.6:
            return cls.MODERATE
        elif score < 0.8:
            return cls.HIGH
        else:
            return cls.VERY_HIGH

class EmotionSource(str, Enum):
    """Source of the emotion detection."""
    ACOUSTIC = "acoustic"  # From audio features
    LINGUISTIC = "linguistic"  # From text content
    VISUAL = "visual"  # From facial expressions or body language
    CONTEXTUAL = "contextual"  # From conversation context
    MULTIMODAL = "multimodal"  # From multiple sources

class CulturalContext(str, Enum):
    """Cultural contexts for emotion interpretation."""
    GLOBAL = "global"  # Culture-agnostic interpretation
    WESTERN = "western"  # Western cultures
    EASTERN_ASIAN = "eastern_asian"  # East Asian cultures
    SOUTH_ASIAN = "south_asian"  # South Asian cultures
    MIDDLE_EASTERN = "middle_eastern"  # Middle Eastern cultures
    AFRICAN = "african"  # African cultures
    LATIN_AMERICAN = "latin_american"  # Latin American cultures

class EmotionFeatures(BaseModel):
    """Features extracted for emotion detection."""
    # Audio features
    mfcc_features: Optional[List[float]] = None
    pitch_features: Optional[List[float]] = None
    energy_features: Optional[List[float]] = None
    spectral_features: Optional[List[float]] = None
    voice_quality_features: Optional[List[float]] = None
    
    # Text features
    sentiment_scores: Optional[Dict[str, float]] = None
    emotional_words: Optional[List[str]] = None
    linguistic_patterns: Optional[Dict[str, Any]] = None
    
    # Context features
    conversation_context: Optional[Dict[str, Any]] = None
    user_history: Optional[Dict[str, Any]] = None
    
    # Metadata
    extraction_time: datetime = Field(default_factory=datetime.now)
    feature_quality: Optional[float] = None

class EmotionDetection(BaseModel):
    """A detected emotion with confidence and metadata."""
    category: EmotionCategory
    confidence: float = Field(0.0, ge=0.0, le=1.0)
    intensity: float = Field(0.0, ge=0.0, le=1.0)
    intensity_level: EmotionIntensity
    source: EmotionSource
    features_used: List[str] = Field(default_factory=list)
    temporal_stability: Optional[float] = None
    detection_time: datetime = Field(default_factory=datetime.now)

class EmotionAnalysisResult(BaseModel):
    """Comprehensive result of emotion analysis."""
    # Primary detected emotion
    primary_emotion: EmotionDetection
    
    # All detected emotions with confidence scores
    all_emotions: List[EmotionDetection] = Field(default_factory=list)
    
    # Complex emotional state analysis
    emotional_state: Dict[str, Any] = Field(default_factory=dict)
    
    # Cultural context
    cultural_context: CulturalContext = CulturalContext.GLOBAL
    cultural_adjustments: Optional[Dict[str, Any]] = None
    
    # Analysis metadata
    analysis_id: str
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)
    processing_time_ms: Optional[int] = None
    
    # Input information
    input_modalities: List[EmotionSource] = Field(default_factory=list)
    audio_duration: Optional[float] = None
    text_content: Optional[str] = None
    
    # Confidence and quality metrics
    overall_confidence: float = Field(0.0, ge=0.0, le=1.0)
    signal_quality: Optional[float] = None
    detection_quality: Optional[Dict[str, float]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert the result to a dictionary."""
        result = {
            "primary_emotion": {
                "category": self.primary_emotion.category.value,
                "confidence": round(self.primary_emotion.confidence, 3),
                "intensity": round(self.primary_emotion.intensity, 3),
                "intensity_level": self.primary_emotion.intensity_level.value,
                "source": self.primary_emotion.source.value
            },
            "all_emotions": [
                {
                    "category": emotion.category.value,
                    "confidence": round(emotion.confidence, 3),
                    "intensity": round(emotion.intensity, 3),
                    "intensity_level": emotion.intensity_level.value,
                    "source": emotion.source.value
                }
                for emotion in self.all_emotions
            ],
            "cultural_context": self.cultural_context.value,
            "timestamp": self.timestamp.isoformat(),
            "overall_confidence": round(self.overall_confidence, 3)
        }
        
        if self.emotional_state:
            result["emotional_state"] = self.emotional_state
            
        if self.cultural_adjustments:
            result["cultural_adjustments"] = self.cultural_adjustments
            
        if self.processing_time_ms is not None:
            result["processing_time_ms"] = self.processing_time_ms
            
        if self.signal_quality is not None:
            result["signal_quality"] = round(self.signal_quality, 3)
        
        return result

class EmotionStream(BaseModel):
    """Stream of emotion analysis results over time."""
    session_id: str
    user_id: Optional[str] = None
    results: List[EmotionAnalysisResult] = Field(default_factory=list)
    start_time: datetime = Field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    duration: Optional[float] = None
    
    # Aggregated emotion trends
    emotion_trends: Dict[EmotionCategory, List[float]] = Field(default_factory=dict)
    dominant_emotions: List[EmotionCategory] = Field(default_factory=list)
    
    # Stream metadata
    total_segments: int = 0
    cultural_context: CulturalContext = CulturalContext.GLOBAL
    metadata: Dict[str, Any] = Field(default_factory=dict)

class EmotionAnalyzerConfig(BaseModel):
    """Configuration for the emotion analyzer."""
    # General settings
    enabled: bool = True
    cultural_context: CulturalContext = CulturalContext.GLOBAL
    min_confidence_threshold: float = Field(0.5, ge=0.0, le=1.0)
    
    # Feature extraction
    extract_acoustic_features: bool = True
    extract_linguistic_features: bool = True
    extract_contextual_features: bool = True
    
    # Processing settings
    enable_streaming: bool = True
    stream_chunk_duration_ms: int = 500
    max_stream_history: int = 100
    
    # Model settings
    acoustic_model: str = "ensemble"
    linguistic_model: str = "transformer"
    fusion_strategy: str = "weighted"
    model_weights: Dict[str, float] = Field(
        default_factory=lambda: {
            "acoustic": 0.6,
            "linguistic": 0.3,
            "contextual": 0.1
        }
    )
    
    # Advanced settings
    enable_complex_emotions: bool = True
    enable_cultural_adaptation: bool = True
    enable_temporal_analysis: bool = True
    cache_results: bool = True
    cache_ttl_seconds: int = 3600
    
    # Logging and debugging
    log_level: str = "INFO"
    enable_feature_logging: bool = False
    save_audio_samples: bool = False

