"""
AutoPilot system data models.

This module defines the data models used by the AutoPilot system for performance
monitoring, optimization, and continuous improvement.
"""

from enum import Enum
from typing import Dict, List, Optional, Union, Any, Set
from datetime import datetime, timedelta
from pydantic import BaseModel, Field

class OptimizationMetric(str, Enum):
    """Metrics that can be optimized by AutoPilot."""
    LATENCY = "latency"  # Response time
    THROUGHPUT = "throughput"  # Requests per minute
    COST = "cost"  # Operating cost
    SUCCESS_RATE = "success_rate"  # Success percentage
    USER_SATISFACTION = "user_satisfaction"  # User rating/feedback
    EMOTION_POSITIVITY = "emotion_positivity"  # Positive emotional responses
    CONVERSATION_LENGTH = "conversation_length"  # Length of conversations
    GOAL_COMPLETION = "goal_completion"  # Task completion rate
    TOKEN_EFFICIENCY = "token_efficiency"  # Tokens used per task
    ERROR_RATE = "error_rate"  # Error frequency
    HALLUCINATION_RATE = "hallucination_rate"  # Frequency of hallucinations
    FIRST_RESPONSE_TIME = "first_response_time"  # Time to first response
    ENGAGEMENT = "engagement"  # User engagement metrics

class OptimizationStrategy(str, Enum):
    """Strategies for optimizing workflows."""
    MINIMIZE = "minimize"  # Minimize the metric (e.g., latency, cost)
    MAXIMIZE = "maximize"  # Maximize the metric (e.g., success rate)
    TARGET = "target"  # Target a specific value
    BALANCED = "balanced"  # Balance multiple metrics

class OptimizationLevel(str, Enum):
    """Levels at which optimization can be applied."""
    NODE = "node"  # Individual node in a workflow
    EDGE = "edge"  # Connections between nodes
    WORKFLOW = "workflow"  # Entire workflow
    SYSTEM = "system"  # Entire system/platform
    LLM_PROMPT = "llm_prompt"  # LLM prompt optimization
    MODEL_SELECTION = "model_selection"  # LLM model selection

class AutomationLevel(str, Enum):
    """Levels of automation for applying changes."""
    SUGGEST = "suggest"  # Suggest changes for human approval
    AUTO_MINOR = "auto_minor"  # Automatically apply minor changes
    AUTO_MAJOR = "auto_major"  # Automatically apply major changes
    FULLY_AUTOMATIC = "fully_automatic"  # Apply all changes automatically

class MetricValue(BaseModel):
    """Value of a metric with metadata."""
    metric: OptimizationMetric
    value: float
    unit: str
    timestamp: datetime = Field(default_factory=datetime.now)
    context: Dict[str, Any] = Field(default_factory=dict)

class PerformanceSnapshot(BaseModel):
    """Snapshot of performance metrics at a point in time."""
    snapshot_id: str
    timestamp: datetime = Field(default_factory=datetime.now)
    metrics: Dict[OptimizationMetric, MetricValue] = Field(default_factory=dict)
    workflow_id: Optional[str] = None
    node_id: Optional[str] = None
    context: Dict[str, Any] = Field(default_factory=dict)

class OptimizationSuggestion(BaseModel):
    """Suggestion for optimizing a workflow."""
    suggestion_id: str
    workflow_id: str
    target_element_id: Optional[str] = None
    element_type: OptimizationLevel
    metrics_improved: List[OptimizationMetric] = Field(default_factory=list)
    estimated_improvement: Dict[OptimizationMetric, float] = Field(default_factory=dict)
    confidence: float = Field(0.0, ge=0.0, le=1.0)
    description: str
    implementation_details: str
    creation_time: datetime = Field(default_factory=datetime.now)
    status: str = "pending"  # pending, approved, rejected, implemented
    applied_at: Optional[datetime] = None
    results: Optional[Dict[str, Any]] = None

class LearningPattern(BaseModel):
    """Pattern learned from successful workflows."""
    pattern_id: str
    pattern_type: str  # node, edge, prompt, etc.
    pattern_data: Dict[str, Any]
    discovered_at: datetime = Field(default_factory=datetime.now)
    source_workflows: List[str] = Field(default_factory=list)
    success_metrics: Dict[OptimizationMetric, float] = Field(default_factory=dict)
    confidence: float = Field(0.0, ge=0.0, le=1.0)
    applications_count: int = 0
    application_success_rate: float = 0.0

class Experiment(BaseModel):
    """A/B test experiment configuration and results."""
    experiment_id: str
    name: str
    description: str
    workflow_id: str
    variant_a_id: str
    variant_b_id: str
    metrics_tracked: List[OptimizationMetric] = Field(default_factory=list)
    start_time: datetime = Field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    status: str = "running"  # setup, running, completed, analyzed
    traffic_split: Dict[str, float] = Field(default_factory=lambda: {"a": 0.5, "b": 0.5})
    results: Optional[Dict[str, Any]] = None
    winning_variant: Optional[str] = None
    confidence_level: Optional[float] = None

class AnomalyDetection(BaseModel):
    """Detection of anomalies in performance metrics."""
    anomaly_id: str
    workflow_id: str
    node_id: Optional[str] = None
    metric: OptimizationMetric
    expected_value: float
    actual_value: float
    deviation_percentage: float
    timestamp: datetime = Field(default_factory=datetime.now)
    severity: str  # low, medium, high, critical
    description: str
    possible_causes: List[str] = Field(default_factory=list)
    recommended_actions: List[str] = Field(default_factory=list)

class CostAnalysis(BaseModel):
    """Analysis of operational costs."""
    analysis_id: str
    workflow_id: Optional[str] = None
    time_period: str  # daily, weekly, monthly
    start_time: datetime
    end_time: datetime
    total_cost: float
    cost_breakdown: Dict[str, float] = Field(default_factory=dict)
    cost_per_execution: float
    cost_trend: str  # increasing, decreasing, stable
    optimization_opportunities: List[str] = Field(default_factory=list)
    estimated_savings: Optional[float] = None

class AutoPilotConfig(BaseModel):
    """Configuration for the AutoPilot system."""
    # General settings
    enabled: bool = True
    automation_level: AutomationLevel = AutomationLevel.SUGGEST
    
    # Monitoring settings
    monitoring_interval_seconds: int = 60
    metrics_to_track: List[OptimizationMetric] = Field(
        default_factory=lambda: [
            OptimizationMetric.LATENCY,
            OptimizationMetric.SUCCESS_RATE,
            OptimizationMetric.USER_SATISFACTION,
            OptimizationMetric.COST,
            OptimizationMetric.TOKEN_EFFICIENCY
        ]
    )
    
    # Optimization settings
    optimization_interval_minutes: int = 60
    optimization_strategies: Dict[OptimizationMetric, OptimizationStrategy] = Field(
        default_factory=lambda: {
            OptimizationMetric.LATENCY: OptimizationStrategy.MINIMIZE,
            OptimizationMetric.SUCCESS_RATE: OptimizationStrategy.MAXIMIZE,
            OptimizationMetric.USER_SATISFACTION: OptimizationStrategy.MAXIMIZE,
            OptimizationMetric.COST: OptimizationStrategy.MINIMIZE,
            OptimizationMetric.TOKEN_EFFICIENCY: OptimizationStrategy.MAXIMIZE
        }
    )
    minimum_data_points: int = 100
    confidence_threshold: float = Field(0.7, ge=0.0, le=1.0)
    
    # Learning settings
    enable_pattern_learning: bool = True
    learning_update_frequency_hours: int = 24
    minimum_success_samples: int = 50
    
    # Experiment settings
    enable_experiments: bool = True
    max_concurrent_experiments: int = 5
    default_experiment_duration_days: int = 7
    minimum_experiment_traffic: int = 1000
    
    # Cost optimization
    enable_cost_optimization: bool = True
    cost_check_frequency_hours: int = 24
    cost_alert_threshold_percentage: float = 20.0
    
    # Anomaly detection
    enable_anomaly_detection: bool = True
    anomaly_check_frequency_minutes: int = 15
    anomaly_deviation_threshold: float = 30.0
    
    # Notifications
    send_email_notifications: bool = True
    send_slack_notifications: bool = False
    notification_recipients: List[str] = Field(default_factory=list)
    
    # Advanced settings
    data_retention_days: int = 90
    enable_reinforcement_learning: bool = True
    backup_before_changes: bool = True

