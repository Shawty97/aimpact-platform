"""
Workflow definition models.

These models define the structure of workflows, nodes, edges, and conditions.
"""

from pydantic import BaseModel, Field, validator
from typing import List, Dict, Any, Optional, Union, Literal
from enum import Enum
from datetime import datetime
import uuid


class NodeType(str, Enum):
    """Types of nodes in a workflow."""
    START = "start"
    END = "end"
    TASK = "task"
    DECISION = "decision"
    FORK = "fork"
    JOIN = "join"
    LOOP = "loop"
    CALLBACK = "callback"
    WAIT = "wait"
    MESSAGE = "message"
    LLM_PROMPT = "llm_prompt"
    KNOWLEDGE_QUERY = "knowledge_query"
    FUNCTION_CALL = "function_call"
    API_CALL = "api_call"
    HUMAN_IN_LOOP = "human_in_loop"
    VOICE_INTERACTION = "voice_interaction"
    SUBPROCESS = "subprocess"
    EXPERIMENT = "experiment"  # For A/B testing


class EdgeType(str, Enum):
    """Types of edges connecting nodes."""
    STANDARD = "standard"
    CONDITIONAL = "conditional"
    DEFAULT = "default"
    TIMEOUT = "timeout"
    ERROR = "error"
    FEEDBACK = "feedback"
    EXPERIMENT_A = "experiment_a"
    EXPERIMENT_B = "experiment_b"


class ConditionOperator(str, Enum):
    """Operators for conditional expressions."""
    EQUALS = "equals"
    NOT_EQUALS = "not_equals"
    GREATER_THAN = "greater_than"
    LESS_THAN = "less_than"
    CONTAINS = "contains"
    NOT_CONTAINS = "not_contains"
    STARTS_WITH = "starts_with"
    ENDS_WITH = "ends_with"
    MATCHES_REGEX = "matches_regex"
    IS_EMPTY = "is_empty"
    IS_NOT_EMPTY = "is_not_empty"
    AND = "and"
    OR = "or"


class ExperimentStrategy(str, Enum):
    """Strategies for A/B testing experiments."""
    RANDOM = "random"  # Random assignment
    PERCENTAGE = "percentage"  # Assign based on percentage
    USER_COHORT = "user_cohort"  # Assign based on user properties
    TIME_BASED = "time_based"  # Switch between variants based on time


class FeedbackType(str, Enum):
    """Types of feedback that can be collected."""
    BINARY = "binary"  # Thumbs up/down
    RATING = "rating"  # Star rating (1-5)
    TEXT = "text"  # Free-text feedback
    MULTI_CHOICE = "multi_choice"  # Multiple choice options


class Condition(BaseModel):
    """A condition for conditional branching."""
    field: str
    operator: ConditionOperator
    value: Any = None
    sub_conditions: List['Condition'] = []

    class Config:
        arbitrary_types_allowed = True


class NodeConfig(BaseModel):
    """Base configuration for a workflow node."""
    type: NodeType
    config: Dict[str, Any] = Field(default_factory=dict)


class LLMPromptConfig(NodeConfig):
    """Configuration for an LLM prompt node."""
    type: Literal[NodeType.LLM_PROMPT] = NodeType.LLM_PROMPT
    config: Dict[str, Any] = Field(
        default_factory=lambda: {
            "prompt_template": "",
            "model": "gpt-4",
            "temperature": 0.7,
            "max_tokens": 500,
            "provider": "openai"
        }
    )


class DecisionConfig(NodeConfig):
    """Configuration for a decision node."""
    type: Literal[NodeType.DECISION] = NodeType.DECISION
    config: Dict[str, Any] = Field(
        default_factory=lambda: {
            "decision_type": "condition",  # condition or llm
            "conditions": [],
        }
    )


class ExperimentConfig(NodeConfig):
    """Configuration for an A/B testing experiment node."""
    type: Literal[NodeType.EXPERIMENT] = NodeType.EXPERIMENT
    config: Dict[str, Any] = Field(
        default_factory=lambda: {
            "strategy": ExperimentStrategy.RANDOM,
            "variant_a_weight": 50,
            "variant_b_weight": 50,
            "experiment_id": "",
            "description": "",
            "metrics": []
        }
    )


class FeedbackConfig(NodeConfig):
    """Configuration for a feedback collection node."""
    type: Literal[NodeType.HUMAN_IN_LOOP] = NodeType.HUMAN_IN_LOOP
    config: Dict[str, Any] = Field(
        default_factory=lambda: {
            "feedback_type": FeedbackType.BINARY,
            "question": "Was this response helpful?",
            "options": ["Yes", "No"],
            "timeout_seconds": 86400,  # 24 hours
            "required": False
        }
    )


class VoiceInteractionConfig(NodeConfig):
    """Configuration for a voice interaction node."""
    type: Literal[NodeType.VOICE_INTERACTION] = NodeType.VOICE_INTERACTION
    config: Dict[str, Any] = Field(
        default_factory=lambda: {
            "voice_id": "default",
            "speech_recognition": True,
            "text_to_speech": True,
            "emotion_detection": False,
            "language": "en-US",
            "timeout_seconds": 30
        }
    )


class Edge(BaseModel):
    """An edge connecting two nodes in a workflow."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    source_id: str
    target_id: str
    type: EdgeType = EdgeType.STANDARD
    condition: Optional[Condition] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @validator('condition', always=True)
    def validate_condition(cls, v, values):
        if values.get('type') == EdgeType.CONDITIONAL and v is None:
            raise ValueError("Conditional edges must have a condition")
        return v


class Node(BaseModel):
    """A node in a workflow."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    description: Optional[str] = None
    type: NodeType
    config: NodeConfig
    position_x: float = 0
    position_y: float = 0
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Workflow(BaseModel):
    """A workflow definition."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    description: Optional[str] = None
    version: str = "1.0.0"
    nodes: List[Node] = []
    edges: List[Edge] = []
    variables: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    created_by: str
    tags: List[str] = []
    is_template: bool = False
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @validator('nodes')
    def validate_nodes(cls, nodes):
        # Ensure there is exactly one start node
        start_nodes = [node for node in nodes if node.type == NodeType.START]
        if len(start_nodes) != 1:
            raise ValueError("Workflow must have exactly one start node")
        
        # Ensure there is at least one end node
        end_nodes = [node for node in nodes if node.type == NodeType.END]
        if not end_nodes:
            raise ValueError("Workflow must have at least one end node")
        
        return nodes

    @validator('edges')
    def validate_edges(cls, edges, values):
        if 'nodes' not in values:
            return edges
        
        nodes = values['nodes']
        node_ids = [node.id for node in nodes]
        
        # Check that edge endpoints refer to existing nodes
        for edge in edges:
            if edge.source_id not in node_ids:
                raise ValueError(f"Edge source node {edge.source_id} does not exist")
            if edge.target_id not in node_ids:
                raise ValueError(f"Edge target node {edge.target_id} does not exist")
        
        return edges


class WorkflowCreateRequest(BaseModel):
    """Request to create a new workflow."""
    name: str
    description: Optional[str] = None
    nodes: List[Node] = []
    edges: List[Edge] = []
    variables: Dict[str, Any] = Field(default_factory=dict)
    tags: List[str] = []
    is_template: bool = False
    metadata: Dict[str, Any] = Field(default_factory=dict)


class WorkflowUpdateRequest(BaseModel):
    """Request to update an existing workflow."""
    name: Optional[str] = None
    description: Optional[str] = None
    nodes: Optional[List[Node]] = None
    edges: Optional[List[Edge]] = None
    variables: Optional[Dict[str, Any]] = None
    tags: Optional[List[str]] = None
    is_template: Optional[bool] = None
    metadata: Optional[Dict[str, Any]] = None

