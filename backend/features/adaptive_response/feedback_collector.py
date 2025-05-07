"""
Feedback Collector for the Adaptive Response Optimization System.

This module provides mechanisms for collecting explicit and implicit feedback
from various sources including voice interactions, text responses, and workflow
completions. It integrates with the emotion detection system to capture
emotional responses.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from .models import (
    EmotionalState,
    FeedbackSource,
    FeedbackType,
    UserFeedback
)

logger = logging.getLogger(__name__)


class FeedbackCollector:
    """Collects and processes user feedback from multiple sources."""
    
    def __init__(self, emotion_detector=None, db_connector=None):
        """
        Initialize the feedback collector.
        
        Args:
            emotion_detector: Optional emotion detector component
            db_connector: Database connector for storing feedback
        """
        self.emotion_detector = emotion_detector
        self.db_connector = db_connector
        self._feedback_processors = {}
        self._register_default_processors()
    
    def _register_default_processors(self):
        """Register default feedback processors for different types."""
        self._feedback_processors = {
            (FeedbackType.EXPLICIT, FeedbackSource.TEXT): self._process_explicit_text_feedback,
            (FeedbackType.EXPLICIT, FeedbackSource.VOICE): self._process_explicit_voice_feedback,
            (FeedbackType.IMPLICIT, FeedbackSource.TEXT): self._process_implicit_text_feedback,
            (FeedbackType.IMPLICIT, FeedbackSource.VOICE): self._process_implicit_voice_feedback,
            (FeedbackType.IMPLICIT, FeedbackSource.WORKFLOW): self._process_workflow_feedback,
            (FeedbackType.EMOTIONAL, FeedbackSource.VOICE): self._process_emotional_voice_feedback,
        }
    
    def register_processor(self, feedback_type: FeedbackType, source: FeedbackSource, processor_fn):
        """
        Register a custom feedback processor.
        
        Args:
            feedback_type: Type of feedback
            source: Source of feedback
            processor_fn: Function to process this type of feedback
        """
        self._feedback_processors[(feedback_type, source)] = processor_fn
    
    async def collect_feedback(
        self,
        user_id: str,
        session_id: str,
        response_id: str,
        feedback_type: FeedbackType,
        feedback_source: FeedbackSource,
        value: Union[int, float, str, Dict[str, Any]],
        context_data: Optional[Dict[str, Any]] = None
    ) -> UserFeedback:
        """
        Collect and process user feedback.
        
        Args:
            user_id: ID of the user providing feedback
            session_id: Current session ID
            response_id: ID of the response being rated
            feedback_type: Type of the feedback
            feedback_source: Source of the feedback
            value: Feedback value (rating, text, etc.)
            context_data: Additional contextual data
            
        Returns:
            Processed UserFeedback object
        """
        context_data = context_data or {}
        
        # Get emotional context if available
        emotional_context = None
        if self.emotion_detector and feedback_source == FeedbackSource.VOICE:
            try:
                emotion_data = await self.emotion_detector.detect_emotion(
                    context_data.get("audio_data")
                )
                emotional_context = EmotionalState(
                    primary_emotion=emotion_data["primary"],
                    confidence=emotion_data["confidence"],
                    secondary_emotions=emotion_data.get("secondary", {}),
                    valence=emotion_data.get("valence", 0.0),
                    arousal=emotion_data.get("arousal", 0.0)
                )
            except Exception as e:
                logger.warning(f"Failed to detect emotion: {e}")
        
        # Create feedback object
        feedback = UserFeedback(
            user_id=user_id,
            session_id=session_id,
            response_id=response_id,
            timestamp=datetime.now(),
            feedback_type=feedback_type,
            feedback_source=feedback_source,
            value=value,
            emotional_context=emotional_context,
            context_data=context_data
        )
        
        # Process the feedback based on type and source
        processor_key = (feedback_type, feedback_source)
        if processor_key in self._feedback_processors:
            feedback = await self._feedback_processors[processor_key](feedback)
        
        # Store the feedback if db connector is available
        if self.db_connector:
            try:
                await self.db_connector.store_feedback(feedback)
            except Exception as e:
                logger.error(f"Failed to store feedback: {e}")
        
        return feedback
    
    async def collect_implicit_feedback(
        self,
        user_id: str,
        session_id: str,
        response_id: str,
        interaction_data: Dict[str, Any],
        source: FeedbackSource
    ) -> UserFeedback:
        """
        Collect implicit feedback based on user interactions.
        
        Args:
            user_id: ID of the user
            session_id: Current session ID
            response_id: ID of the response
            interaction_data: Data about the interaction
            source: Source of the interaction
            
        Returns:
            Processed UserFeedback object
        """
        # Extract engagement metrics from interaction data
        engagement_metrics = self._calculate_engagement_metrics(interaction_data)
        
        return await self.collect_feedback(
            user_id=user_id,
            session_id=session_id,
            response_id=response_id,
            feedback_type=FeedbackType.IMPLICIT,
            feedback_source=source,
            value=engagement_metrics,
            context_data=interaction_data
        )
    
    def _calculate_engagement_metrics(self, interaction_data: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate engagement metrics from interaction data.
        
        Args:
            interaction_data: Raw interaction data
            
        Returns:
            Dictionary of engagement metrics
        """
        metrics = {}
        
        # Time-based metrics
        if "duration" in interaction_data:
            metrics["interaction_time"] = interaction_data["duration"]
        
        if "response_time" in interaction_data:
            metrics["response_time"] = interaction_data["response_time"]
        
        # Interaction-based metrics
        if "follow_up_count" in interaction_data:
            metrics["follow_up_engagement"] = min(1.0, interaction_data["follow_up_count"] / 5.0)
            
        if "interrupt_count" in interaction_data:
            metrics["interruption_rate"] = interaction_data["interrupt_count"] / max(1, interaction_data.get("duration", 60) / 60)
        
        # Content-based metrics
        if "content_views" in interaction_data:
            metrics["content_engagement"] = min(1.0, interaction_data["content_views"] / 3.0)
            
        # Calculate overall engagement score (0.0-1.0)
        if metrics:
            metrics["overall_engagement"] = sum(v for v in metrics.values() if isinstance(v, (int, float))) / len(metrics)
        else:
            metrics["overall_engagement"] = 0.5  # Neutral if no metrics available
            
        return metrics
    
    async def _process_explicit_text_feedback(self, feedback: UserFeedback) -> UserFeedback:
        """Process explicit text feedback (e.g., ratings, comments)."""
        # Additional processing could include sentiment analysis, etc.
        return feedback
    
    async def _process_explicit_voice_feedback(self, feedback: UserFeedback) -> UserFeedback:
        """Process explicit voice feedback."""
        # Additional processing could include tone analysis, etc.
        return feedback
    
    async def _process_implicit_text_feedback(self, feedback: UserFeedback) -> UserFeedback:
        """Process implicit text feedback (e.g., response time, engagement)."""
        return feedback
    
    async def _process_implicit_voice_feedback(self, feedback: UserFeedback) -> UserFeedback:
        """Process implicit voice feedback."""
        return feedback
    
    async def _process_workflow_feedback(self, feedback: UserFeedback) -> UserFeedback:
        """Process workflow-based feedback."""
        return feedback
    
    async def _process_emotional_voice_feedback(self, feedback: UserFeedback) -> UserFeedback:
        """Process emotional feedback detected from voice."""
        return feedback

    async def get_user_feedback_history(
        self, 
        user_id: str, 
        limit: int = 100, 
        feedback_type: Optional[FeedbackType] = None
    ) -> List[UserFeedback]:
        """
        Retrieve feedback history for a specific user.
        
        Args:
            user_id: ID of the user
            limit: Maximum number of feedback items to retrieve
            feedback_type: Optional filter by feedback type
            
        Returns:
            List of UserFeedback objects
        """
        if not self.db_connector:
            logger.warning("No DB connector available to retrieve feedback history")
            return []
        
        try:
            return await self.db_connector.get_user_feedback(
                user_id=user_id,
                limit=limit,
                feedback_type=feedback_type
            )
        except Exception as e:
            logger.error(f"Failed to retrieve user feedback history: {e}")
            return []
