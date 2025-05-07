"""
Models for the Adaptive Response Optimization System.

These models define the data structures used for tracking user feedback,
patterns, personalization profiles, and response adjustments.
"""

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Union, Any
from pydantic import BaseModel, Field


class FeedbackType(str, Enum):
    """Types of feedback that can be collected from users."""
    EXPLICIT = "explicit"  # Direct feedback (ratings, thumbs up/down)
    IMPLICIT = "implicit"  # Indirect feedback (time spent, engagement)
    EMOTIONAL = "emotional"  # Detected emotional response
    BEHAVIORAL = "behavioral"  # User behavioral patterns


class FeedbackSource(str, Enum):
    """Sources from which feedback was collected."""
    VOICE = "voice"
    TEXT = "text"
    INTERACTION = "interaction"
    WORKFLOW = "workflow"


class EmotionalState(BaseModel):
    """Model representing a detected emotional state."""
    primary_emotion: str
    confidence: float
    secondary_emotions: Dict[str, float] = {}
    valence: float = 0.0  # negative to positive scale
    arousal: float = 0.0  # low to high intensity scale


class UserFeedback(BaseModel):
    """Model for storing user feedback on responses."""
    id: Optional[str] = None
    user_id: str
    session_id: str
    response_id: str
    timestamp: datetime = Field(default_factory=datetime.now)
    feedback_type: FeedbackType
    feedback_source: FeedbackSource
    value: Union[int, float, str, Dict[str, Any]]
    emotional_context: Optional[EmotionalState] = None
    context_data: Dict[str, Any] = {}
    
    class Config:
        schema_extra = {
            "example": {
                "user_id": "user123",
                "session_id": "session456",
                "response_id": "resp789",
                "feedback_type": "explicit",
                "feedback_source": "text",
                "value": 4.5,
                "emotional_context": {
                    "primary_emotion": "satisfaction",
                    "confidence": 0.85,
                    "secondary_emotions": {"interest": 0.65},
                    "valence": 0.7,
                    "arousal": 0.3
                }
            }
        }


class InteractionPattern(BaseModel):
    """Model for identified user interaction patterns."""
    id: Optional[str] = None
    user_id: str
    pattern_type: str
    pattern_data: Dict[str, Any]
    confidence: float
    detected_count: int = 1
    last_detected: datetime = Field(default_factory=datetime.now)
    first_detected: datetime = Field(default_factory=datetime.now)
    
    class Config:
        schema_extra = {
            "example": {
                "user_id": "user123",
                "pattern_type": "clarification_requests",
                "pattern_data": {
                    "frequency": 0.23,
                    "trigger_contexts": ["technical_explanation", "complex_workflow"]
                },
                "confidence": 0.82,
                "detected_count": 5
            }
        }


class PersonalizationDimension(str, Enum):
    """Dimensions along which responses can be personalized."""
    VERBOSITY = "verbosity"
    FORMALITY = "formality"
    TECHNICAL_DEPTH = "technical_depth"
    EMOTIONAL_TONE = "emotional_tone"
    CULTURAL_CONTEXT = "cultural_context"
    RESPONSE_SPEED = "response_speed"
    PROACTIVITY = "proactivity"


class PersonalizationProfile(BaseModel):
    """User's personalization profile based on learned preferences."""
    id: Optional[str] = None
    user_id: str
    dimensions: Dict[PersonalizationDimension, float] = {}
    preferred_modalities: Dict[str, float] = {}
    topic_preferences: Dict[str, float] = {}
    context_specific_adjustments: Dict[str, Dict[str, Any]] = {}
    last_updated: datetime = Field(default_factory=datetime.now)
    confidence_scores: Dict[str, float] = {}
    
    class Config:
        schema_extra = {
            "example": {
                "user_id": "user123",
                "dimensions": {
                    "verbosity": 0.7,
                    "formality": 0.4,
                    "technical_depth": 0.85
                },
                "preferred_modalities": {
                    "voice": 0.8,
                    "text": 0.6
                },
                "topic_preferences": {
                    "technical": 0.9,
                    "procedural": 0.7,
                    "conceptual": 0.4
                }
            }
        }


class ResponseAdjustment(BaseModel):
    """Model for specific adjustments to be applied to responses."""
    id: Optional[str] = None
    user_id: str
    adjustment_type: str
    parameters: Dict[str, Any]
    priority: int = 1
    expiration: Optional[datetime] = None
    contexts: List[str] = []
    
    class Config:
        schema_extra = {
            "example": {
                "user_id": "user123",
                "adjustment_type": "simplify_language",
                "parameters": {
                    "grade_level": 8,
                    "avoid_jargon": True
                },
                "priority": 2,
                "contexts": ["technical_explanation", "error_handling"]
            }
        }


class OptimizationMetrics(BaseModel):
    """Metrics tracking the performance of the optimization system."""
    id: Optional[str] = None
    user_id: str
    session_id: Optional[str] = None
    period_start: datetime
    period_end: datetime
    feedback_counts: Dict[FeedbackType, int] = {}
    average_ratings: Dict[str, float] = {}
    adjustment_effectiveness: Dict[str, float] = {}
    personalization_impact: float = 0.0
    
    class Config:
        schema_extra = {
            "example": {
                "user_id": "user123",
                "period_start": "2025-05-01T00:00:00",
                "period_end": "2025-05-05T23:59:59",
                "feedback_counts": {
                    "explicit": 12,
                    "implicit": 45
                },
                "average_ratings": {
                    "helpfulness": 4.2,
                    "relevance": 3.9
                },
                "adjustment_effectiveness": {
                    "simplify_language": 0.75
                },
                "personalization_impact": 0.68
            }
        }

