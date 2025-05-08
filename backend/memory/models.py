"""
Memory models for the AImpact platform.

This module defines the data models used for storing and retrieving
agent memory contexts.
"""
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field


class MemoryType(str, Enum):
    """Types of memories that can be stored."""
    CONVERSATION = "conversation"
    INTERACTION = "interaction"
    FEEDBACK = "feedback"
    WORKFLOW = "workflow"
    CONTEXT = "context"
    KNOWLEDGE = "knowledge"


class MemoryImportance(int, Enum):
    """Importance levels for memories."""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


class MemoryMetadata(BaseModel):
    """Metadata associated with a memory."""
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    workflow_id: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    source: Optional[str] = None
    context_type: Optional[str] = None
    importance: MemoryImportance = MemoryImportance.MEDIUM
    custom_metadata: Dict[str, Any] = Field(default_factory=dict)


class Memory(BaseModel):
    """A single memory item that can be stored in the memory service."""
    id: str = Field(..., description="Unique identifier for the memory")
    agent_id: str = Field(..., description="Agent ID this memory belongs to")
    type: MemoryType = Field(..., description="Type of memory")
    content: Union[str, Dict[str, Any]] = Field(..., description="Content of the memory")
    vector_embedding: Optional[List[float]] = Field(None, description="Vector embedding of the content")
    metadata: MemoryMetadata = Field(default_factory=MemoryMetadata, description="Metadata for the memory")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="When this memory was created")
    expires_at: Optional[datetime] = Field(None, description="When this memory expires")
    last_accessed: Optional[datetime] = Field(None, description="When this memory was last accessed")
    access_count: int = Field(default=0, description="Number of times this memory has been accessed")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class MemorySearchResult(BaseModel):
    """Result of a memory search operation."""
    memory: Memory
    score: float = Field(..., description="Relevance score (0.0 to 1.0)")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class MemorySearchQuery(BaseModel):
    """Query parameters for searching memories."""
    agent_id: str = Field(..., description="Agent ID to search memories for")
    query: str = Field(..., description="Query text to search for")
    memory_types: Optional[List[MemoryType]] = Field(None, description="Filter by memory types")
    session_id: Optional[str] = Field(None, description="Filter by session ID")
    user_id: Optional[str] = Field(None, description="Filter by user ID")
    workflow_id: Optional[str] = Field(None, description="Filter by workflow ID")
    tags: Optional[List[str]] = Field(None, description="Filter by tags")
    min_importance: Optional[MemoryImportance] = Field(None, description="Minimum importance level")
    time_range: Optional[Dict[str, datetime]] = Field(None, description="Time range to filter created_at")
    limit: int = Field(default=10, description="Maximum number of results to return")
    min_score: float = Field(default=0.5, description="Minimum similarity score threshold")


class MemoryBatchOperation(BaseModel):
    """Batch operation parameters for memory service."""
    agent_id: str
    memory_ids: List[str]

