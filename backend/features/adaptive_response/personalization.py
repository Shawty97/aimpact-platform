"""
Personalization Engine for the Adaptive Response Optimization System.

This module provides capabilities for building and maintaining user personalization
profiles, identifying preferences, and generating personalized response adjustments.
"""

import logging
import math
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Set, Union

from .models import (
    FeedbackType,
    InteractionPattern,
    PersonalizationDimension,
    PersonalizationProfile,
    ResponseAdjustment,
    UserFeedback
)

logger = logging.getLogger(__name__)


class PersonalizationEngine:
    """Engine for creating and managing personalization profiles."""
    
    def __init__(self, db_connector=None):
        """
        Initialize the personalization engine.
        
        Args:
            db_connector: Database connector for storing/retrieving profiles
        """
        self.db_connector = db_connector
        self.dimension_extractors = self._initialize_dimension_extractors()
        self.context_analyzers = self._initialize_context_analyzers()
    
    def _initialize_dimension_extractors(self) -> Dict[PersonalizationDimension, callable]:
        """Initialize the functions to extract dimension values from feedback."""
        return {
            PersonalizationDimension.VERBOSITY: self._extract_verbosity_preference,
            PersonalizationDimension.FORMALITY: self._extract_formality_preference,
            PersonalizationDimension.TECHNICAL_DEPTH: self._extract_technical_depth_preference,
            PersonalizationDimension.EMOTIONAL_TONE: self._extract_emotional_tone_preference,
            PersonalizationDimension.CULTURAL_CONTEXT: self._extract_cultural_context_preference,
            PersonalizationDimension.RESPONSE_SPEED: self._extract_response_speed_preference,
            PersonalizationDimension.PROACTIVITY: self._extract_proactivity_preference,
        }
    
    def _initialize_context_analyzers(self) -> Dict[str, callable]:
        """Initialize the functions to analyze specific context types."""
        return {
            "workflow": self._analyze_workflow_context,
            "technical": self._analyze_technical_context,
            "educational": self._analyze_educational_context,
            "social": self._analyze_social_context,
        }
    
    async def get_or_create_profile(self, user_id: str) -> PersonalizationProfile:
        """
        Retrieve existing profile or create a new one.
        
        Args:
            user_id: ID of the user
            
        Returns:
            PersonalizationProfile object
        """
        if self.db_connector:
            try:
                profile = await self.db_connector.get_personalization_profile(user_id)
                if profile:
                    return profile
            except Exception as e:
                logger.error(f"Failed to retrieve personalization profile: {e}")
        
        # Create a new profile with default values
        profile = PersonalizationProfile(
            user_id=user_id,
            dimensions={dim: 0.5 for dim in PersonalizationDimension},  # Neutral defaults
            preferred_modalities={"text": 0.5, "voice": 0.5},  # Equal preference by default
            confidence_scores={dim: 0.0 for dim in PersonalizationDimension}  # No confidence yet
        )
        
        # Store new profile if connector available
        if self.db_connector:
            try:
                await self.db_connector.store_personalization_profile(profile)
            except Exception as e:
                logger.error(f"Failed to store new personalization profile: {e}")
        
        return profile
    
    async def update_profile_with_feedback(
        self,
        profile: PersonalizationProfile,
        feedback: UserFeedback
    ) -> PersonalizationProfile:
        """
        Update a user's personalization profile based on new feedback.
        
        Args:
            profile: Current personalization profile
            feedback: New feedback to incorporate
            
        Returns:
            Updated PersonalizationProfile
        """
        # Extract dimension values from feedback
        extracted_values = {}
        confidence_deltas = {}
        
        for dimension, extractor in self.dimension_extractors.items():
            value, confidence = extractor(feedback)
            if value is not None:
                extracted_values[dimension] = value
                confidence_deltas[dimension] = confidence
        
        # Update dimensions with learning rate based on confidence
        for dimension, new_value in extracted_values.items():
            current_value = profile.dimensions.get(dimension, 0.5)
            current_confidence = profile.confidence_scores.get(dimension, 0.0)
            
            # Learning rate decreases as confidence increases
            learning_rate = max(0.05, 0.5 - current_confidence * 0.5)
            
            # Update value with weighted average
            updated_value = current_value * (1 - learning_rate) + new_value * learning_rate
            profile.dimensions[dimension] = max(0.0, min(1.0, updated_value))
            
            # Increase confidence (bounded by 1.0)
            profile.confidence_scores[dimension] = min(
                1.0, 
                current_confidence + confidence_deltas.get(dimension, 0.01)
            )
        
        # Update preferred modalities based on feedback source
        if feedback.feedback_type == FeedbackType.EXPLICIT:
            modality = feedback.feedback_source.value
            if modality in profile.preferred_modalities:
                # Positive feedback increases preference for modality
                if isinstance(feedback.value, (int, float)) and feedback.value > 3:
                    profile.preferred_modalities[modality] = min(
                        1.0, 
                        profile.preferred_modalities[modality] + 0.05
                    )
        
        # Update topic preferences based on context
        if "topic" in feedback.context_data:
            topic = feedback.context_data["topic"]
            current_preference = profile.topic_preferences.get(topic, 0.5)
            
            # Calculate preference adjustment based on feedback
            adjustment = 0.0
            if feedback.feedback_type == FeedbackType.EXPLICIT:
                if isinstance(feedback.value, (int, float)):
                    normalized_value = (feedback.value - 3) / 2  # Map 1-5 scale to -1.0 to +1.0
                    adjustment = normalized_value * 0.1  # Small incremental changes
            
            profile.topic_preferences[topic] = max(0.0, min(1.0, current_preference + adjustment))
        
        # Update context-specific adjustments
        if "context_type" in feedback.context_data:
            context_type = feedback.context_data["context_type"]
            if context_type in self.context_analyzers:
                context_adjustments = self.context_analyzers[context_type](feedback)
                if context_adjustments:
                    profile.context_specific_adjustments[context_type] = context_adjustments
        
        # Update timestamp
        profile.last_updated = datetime.now()
        
        # Store updated profile
        if self.db_connector:
            try:
                await self.db_connector.update_personalization_profile(profile)
            except Exception as e:
                logger.error(f"Failed to update personalization profile: {e}")
        
        return profile
    
    async def generate_response_adjustments(
        self,
        profile: PersonalizationProfile,
        context_data: Dict[str, Any]
    ) -> List[ResponseAdjustment]:
        """
        Generate response adjustments based on user's personalization profile.
        
        Args:
            profile: User's personalization profile
            context_data: Contextual information about the current interaction
            
        Returns:
            List of ResponseAdjustment objects to apply
        """
        adjustments = []
        
        # Generate base adjustments from profile dimensions
        for dimension, value in profile.dimensions.items():
            # Skip dimensions with low confidence
            if profile.confidence_scores.get(dimension, 0.0) < 0.2:
                continue
                
            # Convert dimension value to adjustment
            adjustment = self._dimension_to_adjustment(
                dimension, 
                value, 
                profile.user_id, 
                context_data
            )
            if adjustment:
                adjustments.append(adjustment)
        
        # Add context-specific adjustments if available
        context_type = context_data.get("context_type")
        if context_type and context_type in profile.context_specific_adjustments:
            context_params = profile.context_specific_adjustments[context_type]
            adjustment = ResponseAdjustment(
                user_id=profile.user_id,
                adjustment_type=f"context_{context_type}",
                parameters=context_params,
                priority=2,  # Context-specific adjustments have higher priority
                contexts=[context_type]
            )
            adjustments.append(adjustment)
        
        # Apply topic-specific adjustments
        topic = context_data.get("topic")
        if topic and topic in profile.topic_preferences:
            topic_preference = profile.topic_preferences[topic]
            if topic_preference > 0.7:  # Strong preference for topic
                adjustment = ResponseAdjustment(
                    user_id=profile.user_id,
                    adjustment_type="expand_topic",
                    parameters={"topic": topic, "expansion_factor": topic_preference},
                    priority=1
                )
                adjustments.append(adjustment)
            elif topic_preference < 0.3:  # Low interest in topic
                adjustment = ResponseAdjustment(
                    user_id=profile.user_id,
                    adjustment_type="summarize_topic",
                    parameters={"topic": topic, "brevity_factor": 1 - topic_preference},
                    priority=1
                )
                adjustments.append(adjustment)
        
        return adjustments
    
    def _dimension_to_adjustment(
        self,
        dimension: PersonalizationDimension,
        value: float,
        user_id: str,
        context_data: Dict[str, Any]
    ) -> Optional[ResponseAdjustment]:
        """Convert a dimension value to a concrete response adjustment."""
        adjustment_type = None
        parameters = {}
        
        if dimension == PersonalizationDimension.VERBOSITY:
            if value > 0.7:  # User prefers detailed responses
                adjustment_type = "expand_response"
                parameters = {"verbosity_level": value, "include_examples": True}
            elif value < 0.3:  # User prefers concise responses
                adjustment_type = "condense_response"
                parameters = {"brevity_level": 1.0 - value, "omit_examples": True}
        
        elif dimension == PersonalizationDimension.FORMALITY:
            adjustment_type = "adjust_formality"
            parameters = {"formality_level": value}
        
        elif dimension == PersonalizationDimension.TECHNICAL_DEPTH:
            adjustment_type = "adjust_technical_depth"
            parameters = {
                "technical_level": value,
                "include_jargon": value > 0.6,
                "simplify_concepts": value < 0.4
            }
        
        elif dimension == PersonalizationDimension.EMOTIONAL_TONE:
            adjustment_type = "adjust_emotional_tone"
            parameters = {"empathy_level": value}
        
        elif dimension == PersonalizationDimension.CULTURAL_CONTEXT:
            # Cultural context requires more specific information to be effective
            adjustment_type = "adjust_cultural_context"
            culture_info = context_data.get("user_culture", {})
            if culture_info:
                parameters = {
                    "culture_sensitivity": value,
                    "cultural_context": culture_info
                }
            else:
                # Skip adjustment if no culture information is available
                return None
        
        elif dimension == PersonalizationDimension.RESPONSE_SPEED:
            adjustment_type = "adjust_response_timing"
            parameters = {
                "detail_vs_speed": value,  # Higher value prioritizes detail over speed
                "chunk_responses": value < 0.3  # Chunk responses for users who prefer speed
            }
        
        elif dimension == PersonalizationDimension.PROACTIVITY:
            adjustment_type = "adjust_proactivity"
            parameters = {
                "proactivity_level": value,
                "suggest_next_actions": value > 0.6,
                "wait_for_explicit_requests": value < 0.4
            }
        
        if adjustment_type:
            return ResponseAdjustment(
                user_id=user_id,
                adjustment_type=adjustment_type,
                parameters=parameters,
                priority=1  # Default priority for dimension-based adjustments
            )
        
        return None
    
    # Dimension extractors
    def _extract_verbosity_preference(self, feedback: UserFeedback) -> Tuple[Optional[float], float]:
        """Extract verbosity preference from feedback."""
        value = None
        confidence = 0.0
        
        if feedback.feedback_type == FeedbackType.EXPLICIT and isinstance(feedback.value, (int, float)):
            if "verbosity" in feedback.context_data:
                # Direct feedback about verbosity
                current_verbosity = feedback.context_data["verbosity"]
                rating = feedback.value
                
                # High rating for verbose content suggests preference for verbosity
                if current_verbosity > 0.7 and rating > 3.5:
                    value = 0.8  # Strong preference for verbose
                    confidence = 0.2
                # High rating for concise content suggests preference for brevity
                elif current_verbosity < 0.3 and rating > 3.5:
                    value = 0.2  # Strong preference for concise
                    confidence = 0.2
        elif feedback.feedback_type == FeedbackType.IMPLICIT:
            if isinstance(feedback.value, dict) and "interaction_time" in feedback.value:
                # Analyze interaction time relative to content length
                if "content_length" in feedback.context_data:
                    time_per_unit = feedback.value["interaction_time"] / feedback.context_data["content_length"]
                    
                    # More time spent per unit of content might indicate preference for detailed content
                    if time_per_unit > 1.5:  # Threshold determined empirically
                        value = 0.7  # Preference for detailed content
                        confidence = 0.05
                    elif time_per_unit < 0.5:  # Quick skimming
                        value = 0.3  # Preference for concise content
                        confidence = 0.05
        
        return value, confidence

    def _extract_formality_preference(self, feedback: UserFeedback) -> Tuple[Optional[float], float]:
        """Extract formality preference from feedback."""
        # Implementation similar to verbosity preference
        return None, 0.0
        
    def _extract_technical_depth_preference(self, feedback: UserFeedback) -> Tuple[Optional[float], float]:
        """Extract technical depth preference from feedback."""
        # Implementation similar to verbosity preference
        return None, 0.0
        
    def _extract_emotional_tone_preference(self, feedback: UserFeedback) -> Tuple[Optional[float], float]:
        """Extract emotional tone preference from feedback."""
        # Implementation similar to verbosity preference
        return None, 0.0
        
    def _extract_cultural_context_preference(self, feedback: UserFeedback) -> Tuple[Optional[float], float]:
        """Extract cultural context preference from feedback."""
        # Implementation similar to verbosity preference
        return None, 0.0
        
    def _extract_response_speed_preference(self, feedback: UserFeedback) -> Tuple[Optional[float], float]:
        """Extract response speed preference from feedback."""
        # Implementation similar to verbosity preference
        return None, 0.0
        
    def _extract_proactivity_preference(self, feedback: UserFeedback) -> Tuple[Optional[float], float]:
        """Extract proactivity preference from feedback."""
        # Implementation similar to verbosity preference
        return None, 0.0
    
    # Context analyzers
    def _analyze_workflow_context(self, feedback: UserFeedback) -> Dict[str, Any]:
        """Analyze feedback in workflow context."""
        # Extract workflow-specific parameters
        return {}
        
    def _analyze_technical_context(self, feedback: UserFeedback) -> Dict[str, Any]:
        """Analyze feedback in technical context."""
        # Extract technical context parameters
        return {}
        
    def _analyze_educational_context(self, feedback: UserFeedback) -> Dict[str, Any]:
        """Analyze feedback in educational context."""
        # Extract educational context parameters
        return {}
        
    def _analyze_social_context(self, feedback: UserFeedback) -> Dict[str, Any]:
        """Analyze feedback in social context."""
        # Extract social context parameters
        return {}
