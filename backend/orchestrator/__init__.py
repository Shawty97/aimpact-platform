"""
AImpact Central Orchestrator

This module provides central coordination and integration between the
workflow engine, voice AI, and AutoPilot systems of the AImpact platform.

Features:
- Cross-system event routing
- Integrated state management
- Voice-enhanced workflow optimization
- Emotion-aware decision making
- End-to-end monitoring and analytics
- Unified configuration
- Holistic optimization
"""

__version__ = "0.1.0"

from .models import OrchestrationEvent, SystemComponent, IntegrationPoint
from .orchestrator import Orchestrator
from .event_bus import EventBus

__all__ = [
    "OrchestrationEvent",
    "SystemComponent",
    "IntegrationPoint",
    "Orchestrator",
    "EventBus"
]

