"""
API models for the Recommendations service.

This module defines Pydantic models for request and response objects
used in the Recommendations API.
"""

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field, root_validator

from ..models import (
    Recommendation, RecommendationType, RecommendationPriority, RecommendationImpact,
    ModuleSuggestion, PromptEnhancement, WorkflowOptimization,
    RecommendationRequest, RecommendationResponse
)


class GetRecommendationsRequest(BaseModel):
    """Request parameters for getting workflow recommendations."""
    focus_areas: Optional[List[RecommendationType]] = None
    node_ids: Optional[List[str]] = None
    max_suggestions: int = Field(default=5, description="Maximum number of suggestions to generate")
    min_confidence: float = Field(default=0.7, description="Minimum confidence threshold")
    include_reasoning: bool = Field(default=True, description="Whether to include detailed reasoning")


class ApplyRecommendationRequest(BaseModel):
    """Request for applying a recommendation."""
    customizations: Optional[Dict[str, Any]] = Field(None, description="Custom modifications to the recommendation")
    save_as_version: Optional[bool] = Field(False, description="Whether to save this as a new workflow version")
    version_name: Optional[str] = Field(None, description="Name for the new workflow version")


class RecommendationPreviewResponse(BaseModel):
    """Response for a recommendation preview."""
    recommendation_id: str
    workflow_id: str
    before: Dict[str, Any]
    after: Dict[str, Any]
    changes_summary: str
    can_apply: bool
    potential_issues: Optional[List[str]] = None


class RecommendationFeedbackRequest(BaseModel):
    """Feedback on a recommendation."""
    recommendation_id: str
    useful: bool = Field(..., description="Whether the recommendation was useful")
    applied: bool = Field(..., description="Whether the recommendation was applied")
    rating: Optional[int] = Field(None, description="Rating from 1-5")
    comments: Optional[str] = Field(None, description="Additional comments")
    user_id: Optional[str] = Field(None, description="ID of the user providing feedback")

