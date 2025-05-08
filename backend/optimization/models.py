"""
Optimization models for AImpact platform.

This module defines the data models for agent optimization,
feedback processing, and training jobs.
"""
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field, root_validator


class FeedbackType(str, Enum):
    """Types of feedback that can be collected."""
    HELPFULNESS = "helpfulness"  # Was the agent's response helpful?
    CORRECTNESS = "correctness"  # Was the agent's response correct?
    RELEVANCE = "relevance"      # Was the agent's response relevant?
    CLARITY = "clarity"          # Was the agent's response clear?
    COMPLETENESS = "completeness"  # Was the agent's response complete?
    CUSTOM = "custom"            # Custom feedback type


class FeedbackValue(BaseModel):
    """Value of feedback, which can be numeric, boolean, or text."""
    score: Optional[float] = Field(None, description="Numeric score (typically 1-5)")
    binary: Optional[bool] = Field(None, description="Boolean feedback (yes/no)")
    text: Optional[str] = Field(None, description="Text feedback")
    
    @root_validator
    def check_at_least_one_field(cls, values):
        """Ensure at least one field is set."""
        if not any(values.get(field) is not None for field in ['score', 'binary', 'text']):
            raise ValueError("At least one feedback value must be provided (score, binary, or text)")
        return values


class AgentFeedback(BaseModel):
    """Feedback for an agent's performance."""
    id: str = Field(..., description="Unique identifier for this feedback")
    agent_id: str = Field(..., description="ID of the agent this feedback is for")
    session_id: Optional[str] = Field(None, description="Session ID this feedback is related to")
    interaction_id: Optional[str] = Field(None, description="Specific interaction this feedback is about")
    user_id: Optional[str] = Field(None, description="User who provided the feedback")
    feedback_type: FeedbackType = Field(..., description="Type of feedback")
    value: FeedbackValue = Field(..., description="Value of the feedback")
    context: Dict[str, Any] = Field(default_factory=dict, description="Context information for this feedback")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="When this feedback was created")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class OptimizationTarget(str, Enum):
    """Target aspects of an agent to optimize."""
    RESPONSE_QUALITY = "response_quality"
    PROMPT_TEMPLATE = "prompt_template"
    RAG_RETRIEVAL = "rag_retrieval"
    WORKFLOW_STEPS = "workflow_steps"
    PARAMETERS = "parameters"
    ALL = "all"


class OptimizationStatus(str, Enum):
    """Status of an optimization job."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELED = "canceled"


class OptimizationJob(BaseModel):
    """Job for optimizing an agent."""
    id: str = Field(..., description="Unique identifier for this job")
    agent_id: str = Field(..., description="ID of the agent to optimize")
    status: OptimizationStatus = Field(default=OptimizationStatus.PENDING)
    targets: List[OptimizationTarget] = Field(..., description="Aspects to optimize")
    config: Dict[str, Any] = Field(default_factory=dict, description="Configuration for this job")
    metrics: Dict[str, float] = Field(default_factory=dict, description="Performance metrics")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    created_by: Optional[str] = None
    result_model_version: Optional[str] = None
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class AgentModelVersion(BaseModel):
    """Version of an agent's model."""
    id: str = Field(..., description="Unique identifier for this model version")
    agent_id: str = Field(..., description="ID of the agent")
    version: str = Field(..., description="Version string")
    model_path: str = Field(..., description="Path to the model files")
    params: Dict[str, Any] = Field(default_factory=dict, description="Model parameters")
    metrics: Dict[str, float] = Field(default_factory=dict, description="Performance metrics")
    is_active: bool = Field(default=False, description="Whether this version is currently active")
    training_job_id: Optional[str] = Field(None, description="ID of the training job that created this version")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class TrainingConfig(BaseModel):
    """Configuration for PPO training."""
    # General training settings
    learning_rate: float = Field(default=3e-4, description="Learning rate")
    batch_size: int = Field(default=64, description="Batch size")
    epochs: int = Field(default=10, description="Number of epochs")
    iterations: int = Field(default=100, description="Number of training iterations")
    max_training_time: int = Field(default=3600, description="Maximum training time in seconds")
    
    # PPO-specific settings
    clip_range: float = Field(default=0.2, description="PPO clip range")
    value_loss_coef: float = Field(default=0.5, description="Value loss coefficient")
    entropy_coef: float = Field(default=0.01, description="Entropy coefficient")
    gamma: float = Field(default=0.99, description="Discount factor")
    gae_lambda: float = Field(default=0.95, description="GAE lambda parameter")
    
    # Model-specific settings
    model_type: str = Field(default="peft", description="Type of model adaptation")
    freeze_base_model: bool = Field(default=True, description="Whether to freeze the base model")
    lora_rank: int = Field(default=8, description="LoRA rank for parameter-efficient fine-tuning")
    lora_alpha: float = Field(default=16.0, description="LoRA alpha")
    
    # Data settings
    min_feedback_count: int = Field(default=100, description="Minimum number of feedback items needed")
    max_samples: int = Field(default=10000, description="Maximum number of training samples")
    validation_split: float = Field(default=0.1, description="Validation split ratio")
    
    # Reward settings
    reward_weights: Dict[str, float] = Field(
        default_factory=lambda: {
            "helpfulness": 1.0,
            "correctness": 1.0,
            "relevance": 0.8,
            "clarity": 0.6,
            "completeness": 0.6,
        },
        description="Weights for different feedback types in reward calculation"
    )

