# Memory Service

The Memory Service is responsible for storing and retrieving context for intelligent agents within the AImPact platform.

## Overview

The Memory Service provides:
- Short-term session memory for ongoing conversations
- Long-term vector-based memory for persistent knowledge
- Multi-tenant isolation of memory contexts
- Memory retrieval with relevance scoring

## Architecture

The Memory Service uses a combination of Redis for session state and a vector database for semantic search of historical contexts.

## Core Components

### ContextManager

Handles the storage and retrieval of context information with TTL and priority management.

```python
class ContextManager:
    def store_context(tenant_id, session_id, context_type, data, ttl=None)
    def retrieve_context(tenant_id, session_id, context_type, k=5)
    def update_context(tenant_id, session_id, context_id, data)
    def delete_context(tenant_id, session_id, context_id)
```

### VectorManager 

Manages vector embeddings for semantic search functionality.

```python
class VectorManager:
    def store_embedding(tenant_id, text, metadata, embedding=None)
    def search(tenant_id, query, filters={}, k=5)
    def delete_embedding(tenant_id, embedding_id)
```

## API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/memory/context` | POST | Store new context |
| `/api/memory/context/{context_id}` | GET | Retrieve specific context |
| `/api/memory/context/{context_id}` | PUT | Update existing context |
| `/api/memory/context/{context_id}` | DELETE | Remove context |
| `/api/memory/search` | POST | Search for relevant contexts |

## Configuration

The Memory Service can be configured via environment variables:

```
REDIS_HOST=localhost
REDIS_PORT=6379
VECTOR_DB_CONNECTION_STRING=...
DEFAULT_CONTEXT_TTL=86400
MAX_CONTEXTS_PER_SESSION=100
```

## Integration Points

- Integrated with Agent Service for context-aware responses
- Used by Optimizer for learning from historical interactions
- Supports the Recommendation Engine for contextual suggestions

## Error Handling

The service implements proper error handling with standardized error codes:

| Error Code | Description |
|------------|-------------|
| `MEMORY_001` | Context not found |
| `MEMORY_002` | Invalid tenant ID |
| `MEMORY_003` | Storage limit exceeded |
| `MEMORY_004` | Vector database connection error |

