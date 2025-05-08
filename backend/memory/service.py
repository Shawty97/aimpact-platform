"""
Memory Service for the AImpact platform.

This service provides memory management capabilities for agents, including:
- Storage and retrieval of memories in Redis/Vector DB
- Session memory management
- Memory retrieval and ranking
- Integration with existing agent services
"""

import asyncio
import json
import logging
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple

import numpy as np
import redis
from redis.commands.search.field import TextField, VectorField, NumericField, TagField
from redis.commands.search.indexDefinition import IndexDefinition, IndexType
from redis.commands.search.query import Query

from .models import (
    Memory, MemoryType, MemoryImportance, MemoryMetadata,
    MemorySearchQuery, MemorySearchResult, MemoryBatchOperation
)

logger = logging.getLogger(__name__)


class MemoryService:
    """
    Service for managing agent memories using Redis and vector databases.
    
    This service provides methods to:
    - Store and retrieve agent memories
    - Search for relevant memories using vector similarity
    - Prune old or less relevant memories
    - Manage memory across sessions
    """
    
    def __init__(
        self,
        redis_host: str = "localhost",
        redis_port: int = 6379,
        redis_db: int = 0,
        redis_password: Optional[str] = None,
        embedding_dimension: int = 1536,  # Default for OpenAI embeddings
        embedding_provider: Optional[Any] = None,
        default_ttl: int = 86400 * 30,  # 30 days default TTL
    ):
        """
        Initialize the memory service.
        
        Args:
            redis_host: Redis host
            redis_port: Redis port
            redis_db: Redis database
            redis_password: Redis password
            embedding_dimension: Dimension of embedding vectors
            embedding_provider: Provider for generating embeddings
            default_ttl: Default time-to-live for memories in seconds
        """
        self.redis_client = redis.Redis(
            host=redis_host,
            port=redis_port,
            db=redis_db,
            password=redis_password,
            decode_responses=True
        )
        
        self.embedding_dimension = embedding_dimension
        self.embedding_provider = embedding_provider
        self.default_ttl = default_ttl
        
        # Initialize Redis search index if it doesn't exist
        self._initialize_search_index()
    
    def _initialize_search_index(self) -> None:
        """Initialize Redis search indices for memory storage."""
        try:
            # Define schema for the memory index
            schema = [
                TextField("$.agent_id", as_name="agent_id"),
                TextField("$.type", as_name="type"),
                TextField("$.metadata.session_id", as_name="session_id"),
                TextField("$.metadata.user_id", as_name="user_id"),
                TextField("$.metadata.workflow_id", as_name="workflow_id"),
                TagField("$.metadata.tags", as_name="tags", separator=","),
                NumericField("$.metadata.importance", as_name="importance"),
                NumericField("$.created_at", as_name="created_at"),
                NumericField("$.expires_at", as_name="expires_at"),
                NumericField("$.access_count", as_name="access_count"),
                VectorField("$.vector_embedding", as_name="vector_embedding", 
                            algorithm="HNSW", attributes={"TYPE": "FLOAT32", "DIM": self.embedding_dimension, "DISTANCE_METRIC": "COSINE"}),
                TextField("$.content", as_name="content"),
            ]
            
            # Create index definition
            index_def = IndexDefinition(prefix=["memory:"], index_type=IndexType.JSON)
            
            # Create index if it doesn't exist
            try:
                self.redis_client.ft("memory_idx").info()
                logger.info("Memory search index already exists")
            except redis.exceptions.ResponseError:
                self.redis_client.ft("memory_idx").create_index(schema, definition=index_def)
                logger.info("Created memory search index")
        
        except Exception as e:
            logger.error(f"Error initializing memory search index: {e}")
    
    async def _get_embedding(self, text: str) -> List[float]:
        """
        Get vector embedding for text using the configured embedding provider.
        
        Args:
            text: Text to embed
            
        Returns:
            Vector embedding of the text
        """
        if self.embedding_provider is None:
            # Generate random embedding for testing if no provider
            logger.warning("No embedding provider configured, using random embedding")
            return list(np.random.rand(self.embedding_dimension).astype(float))
        
        # Get embedding from provider
        try:
            if hasattr(self.embedding_provider, "aembed"):
                # Async embedding
                embedding = await self.embedding_provider.aembed(text)
            else:
                # Sync embedding
                embedding = self.embedding_provider.embed(text)
            
            return embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            # Return zero vector as fallback
            return [0.0] * self.embedding_dimension
    
    async def save_memory(
        self,
        agent_id: str,
        memory_type: MemoryType,
        content: Union[str, Dict[str, Any]],
        metadata: Optional[MemoryMetadata] = None,
        ttl: Optional[int] = None,
        memory_id: Optional[str] = None
    ) -> Memory:
        """
        Save a memory to the database.
        
        Args:
            agent_id: ID of the agent this memory belongs to
            memory_type: Type of memory
            content: Content of the memory (text or structured data)
            metadata: Additional metadata for the memory
            ttl: Time-to-live in seconds (None for default)
            memory_id: Custom memory ID (auto-generated if None)
            
        Returns:
            The saved memory object
        """
        # Generate memory ID if not provided
        if memory_id is None:
            memory_id = str(uuid.uuid4())
        
        # Create memory object
        memory = Memory(
            id=memory_id,
            agent_id=agent_id,
            type=memory_type,
            content=content,
            metadata=metadata or MemoryMetadata(),
            created_at=datetime.utcnow(),
            expires_at=datetime.utcnow() + timedelta(seconds=ttl or self.default_ttl) if ttl is not None else None,
        )
        
        # Generate embedding for text content
        if isinstance(content, str):
            memory.vector_embedding = await self._get_embedding(content)
        else:
            # For structured data, convert to string for embedding
            content_str = json.dumps(content)
            memory.vector_embedding = await self._get_embedding(content_str)
        
        # Save memory to Redis
        try:
            # Convert memory to JSON
            memory_json = memory.json()
            
            # Store in Redis
            key = f"memory:{memory_id}"
            if ttl is not None:
                self.redis_client.setex(key, ttl, memory_json)
            else:
                self.redis_client.set(key, memory_json)
            
            # Store in agent's memory index
            agent_memory_key = f"agent:{agent_id}:memories"
            self.redis_client.sadd(agent_memory_key, memory_id)
            
            # If session-specific, add to session index
            if memory.metadata.session_id:
                session_key = f"agent:{agent_id}:session:{memory.metadata.session_id}"
                self.redis_client.sadd(session_key, memory_id)
                # Set TTL on session key if not already set
                if not self.redis_client.exists(session_key):
                    self.redis_client.expire(session_key, ttl or self.default_ttl)
            
            logger.debug(f"Saved memory {memory_id} for agent {agent_id}")
            return memory
            
        except Exception as e:
            logger.error(f"Error saving memory: {e}")
            raise
    
    async def get_memory(self, memory_id: str, update_access: bool = True) -> Optional[Memory]:
        """
        Retrieve a memory by ID.
        
        Args:
            memory_id: ID of the memory to retrieve
            update_access: Whether to update last accessed time and count
            
        Returns:
            Memory object if found, None otherwise
        """
        try:
            # Get memory from Redis
            key = f"memory:{memory_id}"
            memory_json = self.redis_client.get(key)
            
            if not memory_json:
                return None
            
            # Parse memory from JSON
            memory = Memory.parse_raw(memory_json)
            
            # Update access time and count if requested
            if update_access:
                memory.last_accessed = datetime.utcnow()
                memory.access_count += 1
                
                # Save updated memory
                self.redis_client.set(key, memory.json())
            
            return memory
            
        except Exception as e:
            logger.error(f"Error retrieving memory {memory_id}: {e}")
            return None
    
    async def delete_memory(self, memory_id: str) -> bool:
        """
        Delete a memory.
        
        Args:
            memory_id: ID of the memory to delete
            
        Returns:
            True if deleted, False otherwise
        """
        try:
            # Get memory to find its agent_id and session_id
            memory = await self.get_memory(memory_id, update_access=False)
            if not memory:
                return False
            
            # Delete from Redis
            key = f"memory:{memory_id}"
            self.redis_client.delete(key)
            
            # Remove from agent's memory index
            agent_memory_key = f"agent:{memory.agent_id}:memories"
            self.redis_client.srem(agent_memory_key, memory_id)
            
            # If session-specific, remove from session index
            if memory.metadata.session_id:
                session_key = f"agent:{memory.agent_id}:session:{memory.metadata.session_id}"
                self.redis_client.srem(session_key, memory_id)
            
            logger.debug(f"Deleted memory {memory_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting memory {memory_id}: {e}")
            return False
    
    async def search_memories(self, query: MemorySearchQuery) -> List[MemorySearchResult]:
        """
        Search for memories based on vector similarity and filters.
        
        Args:
            query: Search query parameters
            
        Returns:
            List of memory search results sorted by relevance
        """
        try:
            # Get vector embedding for query
            query_embedding = await self._get_embedding(query.query)

