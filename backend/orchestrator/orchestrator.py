"""
Central Orchestrator

The main orchestrator that coordinates between the workflow engine, voice AI,
and AutoPilot systems of the AImpact platform.
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Union, Any, Callable, Set
from datetime import datetime
import uuid

from .models import (
    SystemComponent, EventType, EventPriority, IntegrationPoint,
    OrchestrationEvent, SystemState, OrchestratorConfig, IntegrationConfig
)
from .event_bus import EventBus
from .workflow_integration import WorkflowIntegration
from .voice_integration import VoiceIntegration
from .autopilot_integration import AutoPilotIntegration

logger = logging.getLogger("aimpact.orchestrator")

class Orchestrator:
    """
    Central orchestrator for the AImpact platform.
    
    This class integrates and coordinates the workflow engine, voice AI,
    and AutoPilot systems to create a cohesive platform with cross-component
    optimizations and unified performance.
    """
    
    def __init__(self, config: Optional[OrchestratorConfig] = None):
        """
        Initialize the orchestrator.
        
        Args:
            config: Configuration for the orchestrator
        """
        self.config = config or OrchestratorConfig()
        
        # Set up logging
        log_level = getattr(logging, self.config.log_level.upper(), logging.INFO)
        logging.basicConfig(level=log_level)
        
        # Initialize event bus
        self.event_bus = EventBus()
        
        # Initialize system state tracking
        self.system_states: Dict[SystemComponent, SystemState] = {}
        
        # Initialize integration handlers
        self.workflow_integration = WorkflowIntegration(self)
        self.voice_integration = VoiceIntegration(self)
        self.autopilot_integration = AutoPilotIntegration(self)
        
        # Integration registrations
        self.integration_handlers: Dict[IntegrationPoint, Callable] = {}
        self._register_integration_handlers()
        
        # Event subscriptions
        self.event_subscriptions: Dict[EventType, List[Callable]] = {}
        self._register_event_subscriptions()
        
        # Initialize metrics
        self.metrics = {
            "events_processed": 0,
            "events_by_type": {},
            "events_by_priority": {},
            "integration_invocations": {},
            "errors": 0,
            "start_time": time.time()
        }
        
        logger.info("Orchestrator initialized")
    
    def _register_integration_handlers(self):
        """Register handlers for each integration point."""
        # Workflow <-> Voice integrations
        self.register_integration_handler(
            IntegrationPoint.WORKFLOW_VOICE_NODE,
            self._handle_workflow_voice_integration
        )
        self.register_integration_handler(
            IntegrationPoint.EMOTION_ENHANCED_DECISION,
            self._handle_emotion_enhanced_decision
        )
        
        # Workflow <-> AutoPilot integrations
        self.register_integration_handler(
            IntegrationPoint.WORKFLOW_OPTIMIZATION,
            self._handle_workflow_optimization
        )
        self.register_integration_handler(
            IntegrationPoint.EXPERIMENT_EXECUTION,
            self._handle_experiment_execution
        )
        
        # Voice <-> AutoPilot integrations
        self.register_integration_handler(
            IntegrationPoint.VOICE_QUALITY_MONITORING,
            self._handle_voice_quality_monitoring
        )
        self.register_integration_handler(
            IntegrationPoint.EMOTION_ANALYTICS,
            self._handle_emotion_analytics
        )
        
        # Three-way integrations
        self.register_integration_handler(
            IntegrationPoint.END_TO_END_OPTIMIZATION,
            self._handle_end_to_end_optimization
        )
        self.register_integration_handler(
            IntegrationPoint.USER_EXPERIENCE_ANALYTICS,
            self._handle_user_experience_analytics
        )
        self.register_integration_handler(
            IntegrationPoint.CROSS_MODAL_LEARNING,
            self._handle_cross_modal_learning
        )
    
    def _register_event_subscriptions(self):
        """Register subscriptions for events."""
        # Workflow engine events
        self.subscribe_to_event(EventType.WORKFLOW_STARTED, self.workflow_integration.handle_workflow_started)
        self.subscribe_to_event(EventType.WORKFLOW_COMPLETED, self.workflow_integration.handle_workflow_completed)
        self.subscribe_to_event(EventType.NODE_EXECUTION_STARTED, self.workflow_integration.handle_node_execution_started)
        self.subscribe_to_event(EventType.NODE_EXECUTION_COMPLETED, self.workflow_integration.handle_node_execution_completed)
        
        # Voice AI events
        self.subscribe_to_event(EventType.VOICE_INTERACTION_STARTED, self.voice_integration.handle_voice_interaction_started)
        self.subscribe_to_event(EventType.VOICE_INTERACTION_ENDED, self.voice_integration.handle_voice_interaction_ended)
        self.subscribe_to_event(EventType.EMOTION_DETECTED, self.voice_integration.handle_emotion_detected)
        
        # AutoPilot events
        self.subscribe_to_event(EventType.OPTIMIZATION_SUGGESTED, self.autopilot_integration.handle_optimization_suggested)
        self.subscribe_to_event(EventType.ANOMALY_DETECTED, self.autopilot_integration.handle_anomaly_detected)
        self.subscribe_to_event(EventType.EXPERIMENT_COMPLETED, self.autopilot_integration.handle_experiment_completed)
        
        # Cross-system events
        self.subscribe_to_event(EventType.FEEDBACK_RECEIVED, self._handle_feedback_received)
        self.subscribe_to_event(EventType.ERROR_OCCURRED, self._handle_error_occurred)
    
    async def start(self):
        """Start the orchestrator."""
        logger.info("Starting orchestrator")
        
        # Initialize component states
        for component in SystemComponent:
            self.system_states[component] = SystemState(
                component=component,
                status="initializing",
                version=__version__,
                uptime_seconds=0
            )
        
        # Start the event bus
        await self.event_bus.start()
        
        # Start integration components
        await self.workflow_integration.start()
        await self.voice_integration.start()
        await self.autopilot_integration.start()
        
        # Update orchestrator state
        self.system_states[SystemComponent.ORCHESTRATOR] = SystemState(
            component=SystemComponent.ORCHESTRATOR,
            status="running",
            version=__version__,
            uptime_seconds=0
        )
        
        # Emit system started event
        await self.emit_event(
            

