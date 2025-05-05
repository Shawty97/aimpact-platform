"""
Document models for the Knowledge Auto-Builder.

These models represent documents, chunks, and extracted information.
"""

from pydantic import BaseModel, Field, HttpUrl
from typing import List, Dict, Any, Optional
from enum import Enum
from datetime import datetime
import uuid


class DocumentStatus(str, Enum):
    """Status of a document in the processing pipeline."""
    PENDING = "pending"
    PROCESSING = "processing"
    PROCESSED = "processed"
    FAILED = "failed"
    INDEXED = "indexed"


class DocumentType(str, Enum):
    """Types of documents supported by the system."""
    PDF = "pdf"
    DOCX = "docx"
    TXT = "txt"
    HTML = "html"
    MD = "markdown"
    CSV = "csv"
    JSON = "json"
    XML = "xml"
    PPTX = "pptx"
    XLSX = "xlsx"
    URL = "url"


class DocumentChunk(BaseModel):
    """A chunk of text extracted from a document."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    document_id: str
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    embedding: Optional[List[float]] = None
    chunk_index: int
    page_number: Optional[int] = None
    token_count: int


class Document(BaseModel):
    """A document in the knowledge base."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    knowledge_base_id: str
    title: str
    description: Optional[str] = None
    file_path: Optional[str] = None
    url: Optional[HttpUrl] = None
    document_type: DocumentType
    status: DocumentStatus = DocumentStatus.PENDING
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    processed_at: Optional[datetime] = None
    chunk_count: int = 0
    total_tokens: int = 0
    error_message: Optional[str] = None


class DocumentCreateRequest(BaseModel):
    """Request to create a new document."""
    knowledge_base_id: str
    title: str
    description: Optional[str] = None
    url: Optional[HttpUrl] = None
    document_type: DocumentType = DocumentType.PDF
    metadata: Dict[str, Any] = Field(default_factory=dict)


class DocumentUploadResponse(BaseModel):
    """Response after a document is uploaded."""
    document_id: str
    status: DocumentStatus
    message: str


class DocumentProcessingOptions(BaseModel):
    """Options for document processing."""
    chunk_size: int = 1000
    chunk_overlap: int = 200
    include_metadata: bool = True
    extract_tables: bool = True
    ocr_enabled: bool = False
    language: str = "en"

