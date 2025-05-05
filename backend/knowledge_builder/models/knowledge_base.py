"""
Knowledge Base models for the Knowledge Auto-Builder.

These models represent knowledge bases and related settings.
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from enum import Enum
from datetime import datetime
import uuid


class KnowledgeBaseStatus(str, Enum):
    """Status of a knowledge base."""
    ACTIVE = "active"
    BUILDING = "building"
    INACTIVE = "inactive"
    ERROR = "error"


class EmbeddingModel(str, Enum):
    """Supported embedding models."""
    OPENAI_ADA = "text-embedding-ada-002"
    OPENAI_3_SMALL = "text-embedding-3-small"
    OPENAI_3_LARGE = "text-embedding-3-large"
    COHERE_EMBED = "embed-english-v3.0"
    COHERE_MULTILINGUAL = "embed-multilingual-v3.0"


class VectorStoreType(str, Enum):
    """Supported vector store types."""
    POSTGRES = "postgres"
    PINECONE = "pinecone"
    MILVUS = "milvus"
    QDRANT = "qdrant"
    WEAVIATE = "weaviate"
    FAISS = "faiss"


class KnowledgeBase(BaseModel):
    """A knowledge base containing documents and their embeddings."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    description: Optional[str] = None
    owner_id: str
    status: KnowledgeBaseStatus = KnowledgeBaseStatus.ACTIVE
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    document_count: int = 0
    embedding_model: EmbeddingModel = EmbeddingModel.OPENAI_ADA
    vector_store_type: VectorStoreType = VectorStoreType.POSTGRES
    vector_store_config: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class KnowledgeBaseCreateRequest(BaseModel):
    """Request to create a new knowledge base."""
    name: str
    description: Optional[str] = None
    embedding_model: EmbeddingModel = EmbeddingModel.OPENAI_ADA
    vector_store_type: VectorStoreType = VectorStoreType.POSTGRES
    vector_store_config: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class KnowledgeBaseUpdateRequest(BaseModel):
    """Request to update a knowledge base."""
    name: Optional[str] = None
    description: Optional[str] = None
    status: Optional[KnowledgeBaseStatus] = None
    embedding_model: Optional[EmbeddingModel] = None
    vector_store_config: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None


class SearchQuery(BaseModel):
    """A search query for the knowledge base."""
    query: str
    knowledge_base_id: str
    top_k: int = 5
    similarity_threshold: float = 0.7
    include_content: bool = True
    filters: Dict[str, Any] = Field(default_factory=dict)


class SearchResult(BaseModel):
    """A search result from the knowledge base."""
    document_id: str
    chunk_id: str
    content: Optional[str] = None
    metadata: Dict[str, Any]
    similarity_score: float
    knowledge_base_id: str

