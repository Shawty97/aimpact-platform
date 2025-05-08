"""
Recommendation models for AImpact platform.

This module defines the data models for workflow recommendations,
prompt optimizations, and module suggestions.
"""
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field, root_validator


class RecommendationType(str, Enum):
    """Types of recommendations that can be provided."""
    WORKFLOW_OPTIMIZATION = "workflow_optimization"
    PROMPT_ENHANCEMENT = "prompt_enhancement"
    MODULE_SUGGESTION = "module_suggestion"
    ERROR_PREVENTION = "error_prevention"
    PERFORMANCE_IMPROVEMENT = "performance_improvement"


class RecommendationPriority(int, Enum):
    """Priority levels for recommendations."""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


class RecommendationImpact(str, Enum):
    """Impact level of applying a recommendation."""
    MINOR = "minor"
    MODERATE = "moderate"
    SIGNIFICANT = "significant"
    MAJOR = "major"


class ModuleSuggestion(BaseModel):
    """Suggestion for a module to add to a workflow."""
    module_id: str = Field(..., description="ID of the suggested module")
    module_name: str = Field(..., description="Name of the suggested module")
    module_type: str = Field(..., description="Type of module")
    description: str = Field(..., description="Description of what the module does")
    insertion_point: Optional[str] = Field(None, description="Suggested node ID after which to insert")
    compatibility_score: float = Field(..., description="Compatibility score (0-1)")
    configuration: Dict[str, Any] = Field(default_factory=dict, description="Suggested configuration")
    reasoning: str = Field(..., description="Reasoning behind this suggestion")


class PromptEnhancement(BaseModel):
    """Suggestion for enhancing a prompt in a workflow."""
    node_id: str = Field(..., description="ID of the node containing the prompt")
    original_prompt: str = Field(..., description="Original prompt text")
    enhanced_prompt: str = Field(..., description="Enhanced prompt text")
    improvements: List[str] = Field(..., description="List of improvements made")
    expected_benefits: List[str] = Field(..., description="Expected benefits of the enhancement")
    reasoning: str = Field(..., description="Reasoning behind this enhancement")
    before_after_comparison: Optional[Dict[str, Any]] = Field(None, description="Comparison of before and after")


class WorkflowOptimization(BaseModel):
    """Suggestion for optimizing workflow structure."""
    optimization_type: str = Field(..., description="Type of optimization")
    affected_nodes: List[str] = Field(..., description="IDs of affected nodes")
    description: str = Field(..., description="Description of the optimization")
    expected_benefits: List[str] = Field(..., description="Expected benefits")
    implementation_complexity: str = Field(..., description="Complexity to implement (easy, medium, hard)")
    before_diagram: Optional[str] = Field(None, description="Diagram of current structure")
    after_diagram: Optional[str] = Field(None, description="Diagram of optimized structure")
    reasoning: str = Field(..., description="Reasoning behind this optimization")


class Recommendation(BaseModel):
    """A recommendation for improving a workflow."""
    id: str = Field(..., description="Unique identifier for this recommendation")
    workflow_id: str = Field(..., description="ID of the workflow this recommendation is for")
    type: RecommendationType = Field(..., description="Type of recommendation")
    title: str = Field(..., description="Short title describing the recommendation")
    description: str = Field(..., description="Detailed description of the recommendation")
    priority: RecommendationPriority = Field(default=RecommendationPriority.MEDIUM, description="Priority level")
    impact: RecommendationImpact = Field(default=RecommendationImpact.MODERATE, description="Impact level")
    module_suggestion: Optional[ModuleSuggestion] = Field(None, description="Module suggestion details")
    prompt_enhancement: Optional[PromptEnhancement] = Field(None, description="Prompt enhancement details")
    workflow_optimization: Optional[WorkflowOptimization] = Field(None, description="Workflow optimization details")
    confidence: float = Field(..., description="Confidence in this recommendation (0-1)")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="When this recommendation was created")
    created_by: Optional[str] = Field(None, description="ID of the user/system that created this recommendation")
    applied: bool = Field(default=False, description="Whether this recommendation has been applied")
    applied_at: Optional[datetime] = Field(None, description="When this recommendation was applied")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class RecommendationRequest(BaseModel):
    """Request for generating recommendations."""
    workflow_id: str = Field(..., description="ID of the workflow to analyze")
    focus_areas: Optional[List[RecommendationType]] = Field(None, description="Areas to focus recommendations on")
    node_ids: Optional[List[str]] = Field(None, description="Specific nodes to analyze")
    max_suggestions: int = Field(default=5, description="Maximum number of suggestions to generate")
    min_confidence: float = Field(default=0.7, description="Minimum confidence threshold")
    include_reasoning: bool = Field(default=True, description="Whether to include detailed reasoning")
    user_id: Optional[str] = Field(None, description="ID of user requesting recommendations")


class RecommendationResponse(BaseModel):
    """Response containing workflow recommendations."""
    workflow_id: str = Field(..., description="ID of the analyzed workflow")
    recommendations: List[Recommendation] = Field(..., description="Generated recommendations")
    analysis_summary: str = Field(..., description="Summary of the workflow analysis")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="When recommendations were generated")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

