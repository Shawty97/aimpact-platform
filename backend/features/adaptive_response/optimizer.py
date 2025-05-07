"""
Adaptive Response Optimizer for dynamic response adjustment.

This module implements the core optimization logic that applies personalization
profiles and interaction patterns to adjust AI responses for individual users.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from .feedback_collector import FeedbackCollector
from .models import (
    InteractionPattern,
    OptimizationMetrics,
    PersonalizationProfile,
    ResponseAdjustment,
    UserFeedback
)
from .personalization import PersonalizationEngine

logger = logging.getLogger(__name__)


class AdaptiveResponseOptimizer:
    """
    Core component of the Adaptive Response Optimization system.
    
    This class coordinates feedback collection, personalization,
    and response adjustments to optimize AI responses.
    """
    
    def __init__(
        self,
        db_connector=None,
        emotion_detector=None,
        feedback_collector=None,
        personalization_engine=None
    ):
        """
        Initialize the Adaptive Response Optimizer.
        
        Args:
            db_connector: Database connector for storing/retrieving data
            emotion_detector: Component for detecting emotions
            feedback_collector: Optional existing feedback collector
            personalization_engine: Optional existing personalization engine
        """
        self.db_connector = db_connector
        self.emotion_detector = emotion_detector
        
        # Initialize or use provided feedback collector
        self.feedback_collector = feedback_collector or FeedbackCollector(
            emotion_detector=emotion_detector,
            db_connector=db_connector
        )
        
        # Initialize or use provided personalization engine
        self.personalization_engine = personalization_engine or PersonalizationEngine(
            db_connector=db_connector
        )
        
        # Pattern recognition cache
        self._pattern_cache = {}
    
    async def optimize_response(
        self,
        user_id: str,
        session_id: str,
        response: Dict[str, Any],
        context_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Optimize a response for a specific user based on their profile and context.
        
        Args:
            user_id: ID of the user
            session_id: Current session ID
            response: The original response to optimize
            context_data: Contextual information about the current interaction
            
        Returns:
            Optimized response
        """
        # Get or create user personalization profile
        profile = await self.personalization_engine.get_or_create_profile(user_id)
        
        # Generate response adjustments based on profile and context
        adjustments = await self.personalization_engine.generate_response_adjustments(
            profile, context_data
        )
        
        # Look for interaction patterns that could trigger additional adjustments
        patterns = await self._identify_patterns(user_id, session_id, context_data)
        pattern_adjustments = self._generate_pattern_adjustments(patterns, user_id)
        adjustments.extend(pattern_adjustments)
        
        # Sort adjustments by priority
        adjustments.sort(key=lambda x: x.priority, reverse=True)
        
        # Apply adjustments to the response
        optimized_response = await self._apply_adjustments(response, adjustments, context_data)
        
        # Track the optimization
        await self._track_optimization(
            user_id=user_id,
            session_id=session_id,
            profile=profile,
            adjustments=adjustments,
            context_data=context_data
        )
        
        return optimized_response
    
    async def process_feedback(
        self,
        user_id: str,
        session_id: str,
        response_id: str,
        feedback_data: Dict[str, Any]
    ) -> None:
        """
        Process feedback and update the user's profile.
        
        Args:
            user_id: ID of the user
            session_id: Current session ID
            response_id: ID of the response being rated
            feedback_data: Feedback data including type, source, value, and context
        """
        # Collect and process the feedback
        feedback = await self.feedback_collector.collect_feedback(
            user_id=user_id,
            session_id=session_id,
            response_id=response_id,
            feedback_type=feedback_data["type"],
            feedback_source=feedback_data["source"],
            value=feedback_data["value"],
            context_data=feedback_data.get("context_data", {})
        )
        
        # Get user profile
        profile = await self.personalization_engine.get_or_create_profile(user_id)
        
        # Update profile with new feedback
        await self.personalization_engine.update_profile_with_feedback(profile, feedback)
        
        # Update pattern recognition
        await self._update_patterns(user_id, feedback)
    
    async def _identify_patterns(
        self,
        user_id: str,
        session_id: str,
        context_data: Dict[str, Any]
    ) -> List[InteractionPattern]:
        """
        Identify applicable interaction patterns for the current context.
        
        Args:
            user_id: ID of the user
            session_id: Current session ID
            context_data: Contextual information about the current interaction
            
        Returns:
            List of applicable interaction patterns
        """
        # Check cache first
        cache_key = f"{user_id}:{context_data.get('context_type', '')}"
        if cache_key in self._pattern_cache:
            return self._pattern_cache[cache_key]
        
        patterns = []
        
        if self.db_connector:
            try:
                # Retrieve patterns from database
                all_patterns = await self.db_connector.get_user_patterns(user_id)
                
                # Filter for patterns that apply to the current context
                for pattern in all_patterns:
                    if self._pattern_applies_to_context(pattern, context_data):
                        patterns.append(pattern)
                
                # Cache the result
                self._pattern_cache[cache_key] = patterns
            except Exception as e:
                logger.error(f"Failed to retrieve user patterns: {e}")
        
        return patterns
    
    def _pattern_applies_to_context(
        self,
        pattern: InteractionPattern,
        context_data: Dict[str, Any]
    ) -> bool:
        """
        Check if a pattern applies to the current context.
        
        Args:
            pattern: Interaction pattern to check
            context_data: Current context data
            
        Returns:
            True if pattern applies, False otherwise
        """
        # Pattern type-specific logic
        if pattern.pattern_type == "clarification_requests":
            return context_data.get("complexity", 0) > 0.7
        elif pattern.pattern_type == "follow_up_questions":
            return True  # Always applicable
        elif pattern.pattern_type == "emotional_triggers":
            return "emotional_context" in context_data
        
        return False
    
    def _generate_pattern_adjustments(
        self,
        patterns: List[InteractionPattern],
        user_id: str
    ) -> List[ResponseAdjustment]:
        """
        Generate response adjustments based on identified patterns.
        
        Args:
            patterns: List of applicable interaction patterns
            user_id: ID of the user
            
        Returns:
            List of generated ResponseAdjustment objects
        """
        adjustments = []
        
        for pattern in patterns:
            if pattern.pattern_type == "clarification_requests" and pattern.confidence > 0.6:
                # User often needs clarification, so preemptively make response clearer
                adjustment = ResponseAdjustment(
                    user_id=user_id,
                    adjustment_type="preemptive_clarification",
                    parameters={
                        "simplify_factor": min(0.8, pattern.confidence),
                        "add_definitions": True
                    },
                    priority=2
                )
                adjustments.append(adjustment)
            
            elif pattern.pattern_type == "follow_up_questions" and pattern.confidence > 0.7:
                # User often asks follow-up questions, so include additional info
                adjustment = ResponseAdjustment(
                    user_id=user_id,
                    adjustment_type="anticipate_questions",
                    parameters={
                        "add_related_info": True,
                        "suggestion_count": 2
                    },
                    priority=1
                )
                adjustments.append(adjustment)
            
            elif pattern.pattern_type == "emotional_triggers" and pattern.confidence > 0.5:
                # User has shown emotional responses to certain content
                adjustment = ResponseAdjustment(
                    user_id=user_id,
                    adjustment_type="emotional_adjustment",
                    parameters=pattern.pattern_data,
                    priority=3  # Higher priority for emotional adjustments
                )
                adjustments.append(adjustment)
        
        return adjustments

    async def _apply_adjustments(
        self,
        response: Dict[str, Any],
        adjustments: List[ResponseAdjustment],
        context_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Apply adjustments to modify the response.
        
        Args:
            response: Original response
            adjustments: List of adjustments to apply
            context_data: Current context data
            
        Returns:
            Adjusted response
        """
        if not adjustments:
            return response
            
        # Clone the response to avoid modifying the original
        optimized = response.copy()
        
        # Track which adjustments were applied
        applied_adjustments = []
        
        # Apply each adjustment
        for adjustment in adjustments:
            # Skip if this adjustment conflicts with previously applied ones
            if self._has_conflicts(adjustment, applied_adjustments):
                continue
                
            if adjustment.adjustment_type == "expand_response":
                optimized = self._expand_response(optimized, adjustment.parameters)
            elif adjustment.adjustment_type == "condense_response":
                optimized = self._condense_response(optimized, adjustment.parameters)
            elif adjustment.adjustment_type == "adjust_formality":
                optimized = self._adjust_formality(optimized, adjustment.parameters)
            elif adjustment.adjustment_type == "adjust_technical_depth":
                optimized = self._adjust_technical_depth(optimized, adjustment.parameters)
            elif adjustment.adjustment_type == "adjust_emotional_tone":
                optimized = self._adjust_emotional_tone(optimized, adjustment.parameters)
            elif adjustment.adjustment_type == "adjust_cultural_context":
                optimized = self._adjust_cultural_context(optimized, adjustment.parameters)
            elif adjustment.adjustment_type == "adjust_response_timing":
                optimized = self._adjust_response_timing(optimized, adjustment.parameters)
            elif adjustment.adjustment_type == "adjust_proactivity":
                optimized = self._adjust_proactivity(optimized, adjustment.parameters)
            elif adjustment.adjustment_type == "preemptive_clarification":
                optimized = self._apply_preemptive_clarification(optimized, adjustment.parameters)
            elif adjustment.adjustment_type == "anticipate_questions":
                optimized = self._apply_anticipate_questions(optimized, adjustment.parameters, context_data)
            elif adjustment.adjustment_type == "emotional_adjustment":
                optimized = self._apply_emotional_adjustment(optimized, adjustment.parameters)
            elif adjustment.adjustment_type == "expand_topic":
                optimized = self._expand_topic(optimized, adjustment.parameters)
            elif adjustment.adjustment_type == "summarize_topic":
                optimized = self._summarize_topic(optimized, adjustment.parameters)
            else:
                # Skip unknown adjustment types
                logger.warning(f"Unknown adjustment type: {adjustment.adjustment_type}")
                continue
                
            # Record that this adjustment was applied
            applied_adjustments.append(adjustment)
        
        # Add metadata about adjustments if requested in context
        if context_data.get("include_optimization_metadata", False):
            optimized["_optimization"] = {
                "adjustments_applied": [a.adjustment_type for a in applied_adjustments],
                "timestamp": datetime.now().isoformat()
            }
        
        return optimized
    
    def _has_conflicts(
        self,
        adjustment: ResponseAdjustment,
        applied_adjustments: List[ResponseAdjustment]
    ) -> bool:
        """Check if an adjustment conflicts with already applied adjustments."""
        # Define conflicting adjustment types
        conflicts = {
            "expand_response": ["condense_response"],
            "condense_response": ["expand_response"],
            "expand_topic": ["summarize_topic"],
            "summarize_topic": ["expand_topic"]
        }
        
        # Check for conflicts
        for applied in applied_adjustments:
            if adjustment.adjustment_type in conflicts.get(applied.adjustment_type, []):
                return True
            if applied.adjustment_type in conflicts.get(adjustment.adjustment_type, []):
                return True
        
        return False

    async def _update_patterns(self, user_id: str, feedback: UserFeedback) -> None:
        """Update pattern recognition based on new feedback."""
        if not self.db_connector:
            return
            
        try:
            # Detect patterns from feedback
            new_patterns = self._detect_patterns(feedback)
            
            for pattern_type, pattern_data in new_patterns.items():
                # Check if pattern already exists
                existing_patterns = await self.db_connector.get_user_patterns_by_type(
                    user_id, pattern_type
                )
                
                if existing_patterns:
                    # Update existing pattern
                    pattern = existing_patterns[0]
                    pattern.detected_count += 1
                    pattern.last_detected = datetime.now()
                    pattern.confidence = min(1.0, pattern.confidence + 0.05)
                    pattern.pattern_data.update(pattern_data)
                    
                    await self.db_connector.update_pattern(pattern)
                else:
                    # Create new pattern
                    pattern = InteractionPattern(
                        user_id=user_id,
                        pattern_type=pattern_type,
                        pattern_data=pattern_data,
                        confidence=0.3,  # Initial confidence
                        detected_count=1,
                        last_detected=datetime.now(),
                        first_detected=datetime.now()
                    )
                    
                    await self.db_connector.store_pattern(pattern)
                
                # Clear pattern cache
                for key in list(self._pattern_cache.keys()):
                    if key.startswith(f"{user_id}:"):
                        del self._pattern_cache[key]
                        
        except Exception as e:
            logger.error(f"Failed to update patterns: {e}")
    
    def _detect_patterns(self, feedback: UserFeedback) -> Dict[str, Dict[str, Any]]:
        """Detect patterns from feedback."""
        patterns = {}
        
        # Example pattern detection (simplified)
        if "follow_up_intent" in feedback.context_data and feedback.context_data["follow_up_intent"]:
            patterns["follow_up_questions"] = {"topic": feedback.context_data.get("topic", "general")}
        
        if "clarification_request" in feedback.context_data and feedback.context_data["clarification_request"]:
            patterns["clarification_requests"] = {"topic": feedback.context_data.get("topic", "general")}
        
        if feedback.emotional_context and feedback.emotional_context.primary_emotion in [
            "confusion", "frustration", "surprise"
        ]:
            patterns["emotional_triggers"] = {
                "emotion": feedback.emotional_context.primary_emotion,
                "trigger": feedback.context_data.get("last_topic", "unknown"),
                "intensity": feedback.emotional_context.arousal
            }
        
        return patterns

    async def _track_optimization(
        self,
        user_id: str,
        session_id: str,
        profile: PersonalizationProfile,
        adjustments: List[ResponseAdjustment],
        context_data: Dict[str, Any]
    ) -> None:
        """Track optimization for metrics and analysis."""
        if not self.db_connector:
            return
            
        try:
            # Get or create metrics for current day
            today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
            tomorrow = today + timedelta(days=1)
            
            metrics = await self.db_connector.get_optimization_metrics(
                user_id, today, tomorrow
            )
            
            if not metrics:
                metrics = OptimizationMetrics(
                    user_id=user_id,
                    period_start=today,
                    period_end=tomorrow,
                    session_id=session_id
                )
            
            # Update adjustment effectiveness tracking
            for adjustment in adjustments:
                adjustment_type = adjustment.adjustment_type
                metrics.adjustment_effectiveness[adjustment_type] = metrics.adjustment_effectiveness.get(
                    adjustment_type, 0.0
                )
            
            # Store updated metrics
            await self.db_connector.store_optimization_metrics(metrics)
            
        except Exception as e:
            logger.error(f"Failed to track optimization: {e}")
    
    # Response adjustment methods
    def _expand_response(self, response: Dict[str, Any], parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Expand a response with more details based on parameters."""
        # Implementation would integrate with the response generation system
        return response
        
    def _condense_response(self, response: Dict[str, Any], parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Condense a response to be more concise based on parameters."""
        # Implementation would integrate with the response generation system
        return response
        
    def _adjust_formality(self, response: Dict[str, Any], parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Adjust the formality level of a response."""
        # Implementation would integrate with the response generation system
        return response
        
    def _adjust_technical_depth(self, response: Dict[str, Any], parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Adjust the technical depth of a response."""
        # Implementation would integrate with the response generation system
        return response
        
    def _adjust_emotional_tone(self, response: Dict[str, Any], parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Adjust the emotional tone of a response."""
        # Implementation would integrate with the response generation system
        return response
        
    def _adjust_cultural_context(self, response: Dict[str, Any], parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Adjust for cultural context sensitivity."""
        # Implementation would integrate with the response generation system
        return response
        
    def _adjust_response_timing(self, response: Dict[str, Any], parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Adjust response timing characteristics."""
        # Implementation would integrate with the response generation system
        return response
        
    def _adjust_proactivity(self, response: Dict[str, Any], parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Adjust the proactivity level of a response."""
        # Implementation would integrate with the response generation system
        return response
        
    def _apply_preemptive_clarification(self, response: Dict[str, Any], parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Apply preemptive clarification to a response."""
        # Implementation would integrate with the response generation system
        return response
        
    def _apply_anticipate_questions(
        self, 
        response: Dict[str, Any], 
        parameters: Dict[str, Any],
        context_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Add anticipated questions and answers to a response."""
        # Implementation would integrate with the response generation system
        return response
        
    def _apply_emotional_adjustment(self, response: Dict[str, Any], parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Apply emotional adjustments to a response."""
        # Implementation would integrate with the response generation system
        return response
        
    def _expand_topic(self, response: Dict[str, Any], parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Expand coverage of a specific topic."""
        # Implementation would integrate with the response generation system
        return response
        
    def _summarize_topic(self, response: Dict[str, Any], parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Summarize a topic more concisely."""
        # Implementation would integrate with the response generation system
        return response
