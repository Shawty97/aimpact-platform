"""
Workflow execution models.

These models define the runtime state of workflow executions and execution history.
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Set
from enum import Enum
from datetime import datetime
import uuid

from .workflow import NodeType, EdgeType


class ExecutionStatus(str, Enum):
    """Status of a workflow execution."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"
    CANCELLED = "cancelled"
    WAITING = "waiting"


class NodeExecutionStatus(str, Enum):
    """Status of a node execution."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    WAITING = "waiting"


class ExperimentVariant(str, Enum):
    """Which variant of an experiment a user was assigned to."""
    A = "a"
    B = "b"
    NONE = "none"


class NodeExecution(BaseModel):
    """The execution of a single node in a workflow."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    workflow_execution_id: str
    node_id: str
    node_type: NodeType
    status: NodeExecutionStatus = NodeExecutionStatus.PENDING
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    input_data: Dict[str, Any] = Field(default_factory=dict)
    output_data: Dict[str, Any] = Field(default_factory=dict)
    error_message: Optional[str] = None
    retry_count: int = 0
    execution_time_ms: Optional[int] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ExperimentExecution(BaseModel):
    """The execution details of an experiment node."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    workflow_execution_id: str
    node_execution_id: str
    experiment_id: str
    variant: ExperimentVariant
    assigned_at: datetime = Field(default_factory=datetime.now)
    metrics: Dict[str, Any] = Field(default_factory=dict)
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class FeedbackData(BaseModel):
    """Feedback collected during workflow execution."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    workflow_execution_id: str
    node_execution_id: str
    feedback_type: str
    feedback_value: Any
    collected_at: datetime = Field(default_factory=datetime.now)
    user_id: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class EdgeExecution(BaseModel):
    """The execution of an edge between nodes."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    workflow_execution_id: str
    edge_id: str
    source_node_execution_id: str
    target_node_execution_id: str
    traversed_at: datetime = Field(default_factory=datetime.now)
    condition_result: Optional[bool] = None  # For conditional edges
    metadata: Dict[str, Any] = Field(default_factory=dict)


class WorkflowExecution(BaseModel):
    """The execution of a workflow."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    workflow_id: str
    workflow_version: str
    status: ExecutionStatus = ExecutionStatus.PENDING
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    input_data: Dict[str, Any] = Field(default_factory=dict)
    output_data: Dict[str, Any] = Field(default_factory=dict)
    current_node_executions: List[str] = []  # IDs of currently executing nodes
    node_executions: List[NodeExecution] = []
    edge_executions: List[EdgeExecution] = []
    variables: Dict[str, Any] = Field(default_factory=dict)  # Runtime variables
    error_message: Optional[str] = None
    created_by: Optional[str] = None
    experiment_executions: List[ExperimentExecution] = []
    feedback_data: List[FeedbackData] = []
    execution_time_ms: Optional[int] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class WorkflowExecutionCreateRequest(BaseModel):
    """Request to start a new workflow execution."""
    workflow_id: str
    input_data: Dict[str, Any] = Field(default_factory=dict)
    variables: Dict[str, Any] = Field(default_factory=dict)

