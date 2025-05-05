"""
Orchestration data models.

This module defines the data models used for cross-system orchestration between
the workflow engine, voice AI, and AutoPilot systems.
"""

from enum import Enum
from typing import Dict, List, Optional, Union, Any, Set
from datetime import datetime
from pydantic import BaseModel, Field
import uuid

class SystemComponent(str, Enum):
    """Major system components that can be orchestrated."""
    WORKFLOW_ENGINE = "workflow_engine"
    VOICE_AI = "voice_ai"
    AUTOPILOT = "autopilot"
    AGENT_STORE = "agent_store"
    KNOWLEDGE_BASE = "knowledge_base"
    ORCHESTRATOR = "orchestrator"

class EventType(str, Enum):
    """Types of events that can be routed between systems."""
    # Workflow events
    WORKFLOW_STARTED = "workflow_started"
    WORKFLOW_COMPLETED = "workflow_completed"
    WORKFLOW_FAILED = "workflow_failed"
    NODE_EXECUTION_STARTED = "node_execution_started"
    NODE_EXECUTION_COMPLETED = "node_execution_completed"
    NODE_EXECUTION_FAILED = "node_execution_failed"
    
    # Voice events
    VOICE_INTERACTION_STARTED = "voice_interaction_started"
    VOICE_INTERACTION_ENDED = "voice_interaction_ended"
    EMOTION_DETECTED = "emotion_detected"
    SPEECH_RECOGNIZED = "speech_recognized"
    SPEECH_SYNTHESIZED = "speech_synthesized"
    
    # AutoPilot events
    OPTIMIZATION_SUGGESTED = "optimization_suggested"
    OPTIMIZATION_APPLIED = "optimization_applied"
    ANOMALY_DETECTED = "anomaly_detected"
    EXPERIMENT_STARTED = "experiment_started"
    EXPERIMENT_COMPLETED = "experiment_completed"
    PATTERN_LEARNED = "pattern_learned"
    
    # Cross-system events
    SYSTEM_STARTED = "system_started"
    SYSTEM_STOPPED = "system_stopped"
    CONFIG_UPDATED = "config_updated"
    ERROR_OCCURRED = "error_occurred"
    FEEDBACK_RECEIVED = "feedback_received"

class EventPriority(str, Enum):
    """Priority levels for events."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class IntegrationPoint(str, Enum):
    """Integration points between systems."""
    # Workflow <-> Voice
    WORKFLOW_VOICE_NODE = "workflow_voice_node"  # Voice nodes in workflows
    EMOTION_ENHANCED_DECISION = "emotion_enhanced_decision"  # Decisions based on emotions
    
    # Workflow <-> AutoPilot
    WORKFLOW_OPTIMIZATION = "workflow_optimization"  # Optimizing workflows
    EXPERIMENT_EXECUTION = "experiment_execution"  # Running A/B tests
    
    # Voice <-> AutoPilot
    VOICE_QUALITY_MONITORING = "voice_quality_monitoring"  # Monitoring voice quality
    EMOTION_ANALYTICS = "emotion_analytics"  # Analyzing emotions
    
    # Three-way integration
    END_TO_END_OPTIMIZATION = "end_to_end_optimization"  # Holistic optimization
    USER_EXPERIENCE_ANALYTICS = "user_experience_analytics"  # UX analytics
    CROSS_MODAL_LEARNING = "cross_modal_learning"  # Learning across modalities

class OrchestrationEvent(BaseModel):
    """An event that can be routed between systems."""
    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    event_type: EventType
    source_component: SystemComponent
    target_components: List[SystemComponent] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=datetime.now)
    priority: EventPriority = EventPriority.MEDIUM
    payload: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    
    class Config:
        schema_extra = {
            "example": {
                "event_id": "e12b7e9a-0b9a-4e9e-8f1a-2b3c4d5e6f7g",
                "event_type": "emotion_detected",
                "source_component": "voice_ai",
                "target_components": ["workflow_engine", "autopilot"],
                "priority": "high",
                "payload": {
                    "primary_emotion": "happy",
                    "confidence": 0.85,
                    "intensity": 0.7
                },
                "session_id": "session-123",
                "user_id": "user-456"
            }
        }

class SystemState(BaseModel):
    """Current state of a system component."""
    component: SystemComponent
    status: str  # running, paused, stopped, error
    version: str
    uptime_seconds: int
    last_update: datetime = Field(default_factory=datetime.now)
    metrics: Dict[str, Any] = Field(default_factory=dict)
    config_hash: Optional[str] = None
    
    class Config:
        schema_extra = {
            "example": {
                "component": "voice_ai",
                "status": "running",
                "version": "0.1.0",
                "uptime_seconds": 3600,
                "metrics": {
                    "requests_processed": 150,
                    "average_response_time_ms": 250
                }
            }
        }

class IntegrationConfig(BaseModel):
    """Configuration for an integration point."""
    integration_point: IntegrationPoint
    enabled: bool = True
    components: List[SystemComponent] = Field(min_items=2)
    event_types: List[EventType] = Field(default_factory=list)
    polling_interval_seconds: Optional[int] = None
    config: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        schema_extra = {
            "example": {
                "integration_point": "emotion_enhanced_decision",
                "enabled": True,
                "components": ["voice_ai", "workflow_engine"],
                "event_types": ["emotion_detected", "node_execution_started"],
                "config": {
                    "emotion_threshold": 0.7,
                    "impact_weight": 0.5
                }
            }
        }

class OrchestratorConfig(BaseModel):
    """Configuration for the central orchestrator."""
    enabled: bool = True
    log_level: str = "INFO"
    event_retention_days: int = 7
    max_events_per_second: int = 1000
    enable_metrics: bool = True
    enable_tracing: bool = True
    integrations: List[IntegrationConfig] = Field(default_factory=list)
    default_event_priority: EventPriority = EventPriority.MEDIUM

