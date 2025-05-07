"""
Cross-Modal Intelligence Models

This module defines the data models and structures for the cross-modal intelligence feature.
"""

from enum import Enum
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from pydantic import BaseModel, Field


class ModalityType(str, Enum):
    """Types of data modalities that can be processed."""
    TEXT = "text"
    VOICE = "voice"
    AUDIO = "audio"  # Non-speech audio
    STRUCTURED = "structured"  # Structured data
    WORKFLOW = "workflow"  # Workflow data
    CONTEXT = "context"  # Contextual information


class ModalityData(BaseModel):
    """Base class for modality data."""
    modality: ModalityType
    timestamp: datetime = Field(default_factory=datetime.now)
    confidence: float = Field(1.0, ge=0.0, le=1.0)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class TextModalityData(ModalityData):
    """Text modality data."""
    modality: ModalityType = ModalityType.TEXT
    text: str
    language: Optional[str] = None
    sentiment: Optional[float] = None


class VoiceModalityData(ModalityData):
    """Voice modality data."""
    modality: ModalityType = ModalityType.VOICE
    text: str  # Transcribed text
    audio_data: Optional[bytes] = None
    language: Optional[str] = None
    emotion: Optional[Dict[str, Any]] = None
    prosody: Optional[Dict[str, Any]] = None  # Pitch, tone, rhythm, etc.
    speaker_characteristics: Optional[Dict[str, Any]] = None


class StructuredModalityData(ModalityData):
    """Structured data modality."""
    modality: ModalityType = ModalityType.STRUCTURED
    data: Dict[str, Any]
    schema: Optional[str] = None


class WorkflowModalityData(ModalityData):
    """Workflow data modality."""
    modality: ModalityType = ModalityType.WORKFLOW
    workflow_id: str
    execution_id: Optional[str] = None
    node_id: Optional[str] = None
    data: Dict[str, Any] = Field(default_factory=dict)


class ContextualModalityData(ModalityData):
    """Contextual information modality."""
    modality: ModalityType = ModalityType.CONTEXT
    context_type: str  # Type of context (session, user, environment, etc.)
    data: Dict[str, Any]


class CrossModalInput(BaseModel):
    """Input for cross-modal processing with multiple modalities."""
    id: str = Field(default_factory=lambda: datetime.now().isoformat())
    modalities: List[Union[TextModalityData, VoiceModalityData, StructuredModalityData, 
                         WorkflowModalityData, ContextualModalityData]]
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ModalityAlignment(BaseModel):
    """Alignment between different modalities."""
    source_modality: ModalityType
    target_modality: ModalityType
    alignment_score: float = Field(1.0, ge=0.0, le=1.0)
    alignment_type: str  # Type of alignment (semantic, temporal, etc.)
    aligned_elements: Dict[str, Any] = Field(default_factory=dict)


class CrossModalUnderstanding(BaseModel):
    """Unified understanding derived from multiple modalities."""
    id: str = Field(default_factory=lambda: datetime.now().isoformat())
    input_id: str  # Reference to the CrossModalInput
    primary_intent: Optional[str] = None
    confidence: float = Field(0.0, ge=0.0, le=1.0)
    sentiment: Optional[float] = None
    emotion: Optional[Dict[str, Any]] = None
    entities: List[Dict[str, Any]] = Field(default_factory=list)
    modality_alignments: List[ModalityAlignment] = Field(default_factory=list)
    unified_representation: Dict[str, Any] = Field(default_factory=dict)
    context: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.now)
    processing_time_ms: Optional[int] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ModalityFusionMethod(str, Enum):
    """Methods for fusing multiple modalities."""
    EARLY_FUSION = "early_fusion"  # Combine features before processing
    LATE_FUSION = "late_fusion"  # Process modalities separately and combine results
    HYBRID_FUSION = "hybrid_fusion"  # Combine early and late fusion
    ATTENTION_FUSION = "attention_fusion"  # Use attention mechanisms to focus on important modalities
    ADAPTIVE_FUSION = "adaptive_fusion"  # Dynamically adjust fusion based on context


class CrossModalConfig(BaseModel):
    """Configuration for cross-modal intelligence."""
    enabled_modalities: List[ModalityType] = Field(default_factory=lambda: list(ModalityType))
    fusion_method: ModalityFusionMethod = ModalityFusionMethod.HYBRID_FUSION
    confidence_threshold: float = Field(0.6, ge=0.0, le=1.0)
    context_window_size: int = Field(5, ge=1)
    enable_advanced_reasoning: bool = True
    enable_pattern_recognition: bool = True
    enable_emotion_integration: bool = True
    enable_workflow_integration: bool = True
    custom_settings: Dict[str, Any] = Field(default_factory=dict)

