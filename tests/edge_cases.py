"""
Edge case tests for AImpact platform services.

This module contains tests for various edge cases and error conditions that might
occur in the Memory, Optimizer, and Recommendation services.
"""

import pytest
import asyncio
import json
import uuid
import random
import string
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from unittest.mock import patch, MagicMock

# Import the services that need to be tested
from backend.memory.service import MemoryService
from backend.memory.models import Memory, MemoryType, MemoryMetadata
from backend.optimization.service import OptimizerService
from backend.optimization.models import (
    AgentFeedback, FeedbackValue, FeedbackType, OptimizationJob,
    OptimizationStatus, OptimizationTarget, TrainingConfig
)
from backend.recommendations.service import RecommendationService
from backend.recommendations.models import (
    Recommendation, RecommendationType, RecommendationRequest,
    RecommendationResponse, RecommendationPriority, RecommendationImpact
)

# Utility functions
def generate_large_text(size_kb: int) -> str:
    """Generate a large random text of specified size in KB."""
    chars = string.ascii_letters + string.digits + string.punctuation + ' ' * 10
    return ''.join(random.choice(chars) for _ in range(size_kb * 1024))

# Fixtures for large datasets
@pytest.fixture
def large_memory_content():
    """Generate a large memory content (5MB)."""
    return {
        "text": generate_large_text(5 * 1024),  # 5MB of text
        "metadata": {
            "source": "edge_case_test",
            "timestamp": datetime.utcnow().isoformat()
        }
    }

@pytest.fixture
def complex_workflow():
    """Generate a complex workflow with many nodes and edges."""
    num_nodes = 50
    nodes = []
    edges = []
    
    # Create nodes
    for i in range(num_nodes):
        node_type = random.choice(["input", "processor", "prompt", "llm", "output"])
        config = {}
        
        if node_type == "prompt":
            config["prompt"] = generate_large_text(1)  # 1KB prompt
        
        nodes.append({
            "id": f"node{i}",
            "name": f"Node {i}",
            "type": node_type,
            "config": config
        })
    
    # Create edges (making sure it's a connected graph)
    for i in range(num_nodes - 1):
        edges.append({
            "source_id": f"node{i}",
            "target_id": f"node{i + 1}",
            "type": "standard"
        })
    
    # Add some random additional edges for complexity
    for _ in range(20):
        source = random.randint(0, num_nodes - 2)
        target = random.randint(source + 1, num_nodes - 1)
        edges.append({
            "source_id": f"node{source}",
            "target_id": f"node{target}",
            "type": random.choice(["standard", "conditional", "error"])
        })
    
    return {
        "id": str(uuid.uuid4()),
        "name": "Complex Test Workflow",
        "description": "A complex workflow for edge case testing",
        "version": "1.0",
        "nodes": nodes,
        "edges": edges
    }

# --------------------------------------
# Memory Service Edge Case Tests
# --------------------------------------

@pytest.mark.asyncio
async def test_memory_large_content(memory_service):
    """Test storing and retrieving large memory content."""
    agent_id = str(uuid.uuid4())
    content = generate_large_text(2 * 1024)  # 2MB content
    
    # Store large memory
    memory = await memory_service.save_memory(
        agent_id=agent_id,
        memory_type=MemoryType.CONTEXT,
        content=content,
        metadata=MemoryMetadata(
            tags=["large_content_test"]
        )
    )
    
    assert memory is not None
    assert memory.id is not None
    
    # Retrieve and verify
    retrieved = await memory_service.get_memory(memory.id)
    assert retrieved is not None
    assert retrieved.content == content
    assert len(retrieved.content) == len(content)

@pytest.mark.asyncio
async def test_memory_concurrent_access(memory_service):
    """Test concurrent access to memory service."""
    agent_id = str(uuid.uuid4())
    session_id = str(uuid.uuid4())
    num_concurrent = 20
    
    async def save_memory(i: int):
        return await memory_service.save_memory(
            agent_id=agent_id,
            memory_type=MemoryType.CONVERSATION,
            content=f"Concurrent memory content {i}",
            metadata=MemoryMetadata(
                session_id=session_id,
                tags=["concurrent_test"]
            )
        )
    
    # Concurrently save multiple memories
    tasks = [save_memory(i) for i in range(num_concurrent)]
    memories = await asyncio.gather(*tasks)
    
    # Verify all memories were stored
    assert len(memories) == num_concurrent
    
    # Now concurrently retrieve them
    retrieve_tasks = [memory_service.get_memory(memory.id) for memory in memories]
    retrieved = await asyncio.gather(*retrieve_tasks)
    
    # Verify all were retrieved successfully
    assert len(retrieved) == num_concurrent
    assert all(memory is not None for memory in retrieved)
    
    # Verify content integrity
    content_set = {memory.content for memory in retrieved}
    assert len(content_set) == num_concurrent

@pytest.mark.asyncio
async def test_memory_expiration(memory_service):
    """Test memory expiration and cleanup."""
    agent_id = str(uuid.uuid4())
    
    # Create a memory with short TTL (1 second)
    memory = await memory_service.save_memory(
        agent_id=agent_id,
        memory_type=MemoryType.CONVERSATION,
        content="This memory will expire quickly",
        metadata=MemoryMetadata(tags=["expiration_test"]),
        ttl=1  # 1 second TTL
    )
    
    # Verify it exists initially
    initial_retrieve = await memory_service.get_memory(memory.id)
    assert initial_retrieve is not None
    
    # Wait for expiration
    await asyncio.sleep(2)
    
    # Verify it's gone
    expired_retrieve = await memory_service.get_memory(memory.id)
    assert expired_retrieve is None

@pytest.mark.asyncio
async def test_memory_error_handling():
    """Test memory service error handling."""
    # Create memory service with a mock Redis that fails
    mock_redis = MagicMock()
    mock_redis.get.side_effect = Exception("Redis connection error")
    mock_redis.set.side_effect = Exception("Redis connection error")
    
    service = MemoryService(
        redis_host="localhost",
        redis_port=6379,
        redis_db=0,
        embedding_dimension=768
    )
    service.redis_client = mock_redis
    
    # Test save memory error handling
    with pytest.raises(Exception):
        await service.save_memory(
            agent_id=str(uuid.uuid4()),
            memory_type=MemoryType.CONVERSATION,
            content="Test content"
        )
    
    # Test get memory error handling
    result = await service.get_memory("some_id")
    assert result is None

# --------------------------------------
# Optimizer Edge Case Tests
# --------------------------------------

@pytest.mark.asyncio
async def test_optimizer_low_feedback(optimizer_service):
    """Test optimizer behavior with minimal feedback."""
    agent_id = str(uuid.uuid4())
    
    # Create a single feedback
    feedback = AgentFeedback(
        id=str(uuid.uuid4()),
        agent_id=agent_id,
        feedback_type=FeedbackType.HELPFULNESS,
        value=FeedbackValue(score=3.0),
        context={}
    )
    
    await optimizer_service.save_feedback(feedback)
    
    # Try to run an optimization job with low feedback
    job_id = str(uuid.uuid4())
    job = OptimizationJob(
        id=job_id,
        agent_id=agent_id,
        targets=[OptimizationTarget.RESPONSE_QUALITY],
        config={"min_feedback_count": 1}  # Set low threshold for testing
    )
    
    # Manually start the job for testing
    optimizer_service._job_store[job_id] = job
    await optimizer_service._process_job(job)
    
    # Check if job completed despite low feedback
    updated_job = optimizer_service._job_store[job_id]
    assert updated_job.status != OptimizationStatus.FAILED
    
    # In a real test, we'd check for warnings about low feedback in logs

@pytest.mark.asyncio
async def test_optimizer_conflicting_feedback(optimizer_service):
    """Test optimizer with conflicting feedback."""
    agent_id = str(uuid.uuid4())
    
    # Create contrasting feedback for the same interaction
    interaction_id = str(uuid.uuid4())
    
    # Positive feedback
    positive = AgentFeedback(
        id=str(uuid.uuid4()),
        agent_id=agent_id,
        interaction_id=interaction_id,
        feedback_type=FeedbackType.HELPFULNESS,
        value=FeedbackValue(score=5.0),
        context={"query": "test query"}
    )
    
    # Negative feedback
    negative = AgentFeedback(
        id=str(uuid.uuid4()),
        agent_id=agent_id,
        interaction_id=interaction_id,
        feedback_type=FeedbackType.HELPFULNESS,
        value=FeedbackValue(score=1.0),
        context={"query": "test query"}
    )
    
    await optimizer_service.save_feedback(positive)
    await optimizer_service.save_feedback(negative)
    
    # In real implementation, the optimizer might have logic to handle conflicting feedback
    # For this test, we just verify both are saved
    
    # Get all feedback for the interaction
    # This is a simplified approach - in a real system, we'd have a proper query
    interaction_feedback = [
        f for f in optimizer_service._feedback_store.values() 
        if f.interaction_id == interaction_id
    ]
    
    assert len(interaction_feedback) == 2
    scores = [f.value.score for f in interaction_feedback]
    assert 5.0 in scores
    assert 1.0 in scores

@pytest.mark.asyncio
async def test_optimizer_resource_limits():
    """Test optimizer behavior with resource limits."""
    # Mock a memory-constrained environment
    with patch('psutil.virtual_memory') as mock_memory:
        # Simulate low memory (10% available)
        mock_memory.return_value.percent = 90.0
        
        service = OptimizerService(
            model_storage_path="/tmp/models",
            enable_background_processing=False
        )
        
        # Create a job with high resource needs
        job = OptimizationJob(
            id=str(uuid.uuid4()),
            agent_id=str(uuid.uuid4()),
            targets=[OptimizationTarget.ALL],
            config={
                "batch_size": 1024,  # High batch size
                "max_samples": 100000  # Many samples
            }
        )
        
        # Expect job to be throttled or have reduced resources
        service._job_store[job.id] = job
        
        with patch.object(service, '_adjust_job_resources') as mock_adjust:
            await service._process_job(job)
            assert mock_adjust.called

@pytest.mark.asyncio
async def test_optimizer_training_interruption(optimizer_service):
    """Test optimizer behavior when training is interrupted."""
    
    # Create a job that will be running
    job = OptimizationJob(
        id=str(uuid.uuid4()),
        agent_id=str(uuid.uuid4()),
        status=OptimizationStatus.RUNNING,
        targets=[OptimizationTarget.RESPONSE_QUALITY],
        started_at=datetime.utcnow() - timedelta(hours=1)  # Started an hour ago
    )
    
    optimizer_service._job_store[job.id] = job
    
    # Simulate interruption (e.g. service restart)
    # In a real system, we'd have recovery logic
    # Here we just test that the optimizer handles it gracefully
    
    # This would be more complex in a real test, but we can simulate by
    # directly calling the method that would run on restart
    await optimizer_service._handle_interrupted_jobs()
    
    # The job should either be marked as failed or restarted
    updated_job = optimizer_service._job_store[job.id]
    assert updated_job.status != OptimizationStatus.RUNNING
    
    # In a real system, this would be recoverable or properly cleanup would happen

# --------------------------------------
# Recommendation Engine Edge Case Tests
# --------------------------------------

@pytest.mark.asyncio
async def test_recommendations_complex_workflow(recommendation_service, complex_workflow):
    """Test recommendation engine with a complex workflow."""
    
    # Mock services
    class MockWorkflowService:
        async def get_workflow(self, workflow_id):
            return complex_workflow
    
    recommendation_service.workflow_service = MockWorkflowService()
    
    class MockModuleRegistry:
        async def list_modules(self):
            return [
                {
                    "id": "module1",
                    "name": "Test Module",
                    "type": "processor",
                    "description": "A test module",
                    "capabilities": ["data_processing"]
                }
            ]
    
    recommendation_service.module_registry = MockModuleRegistry()
    
    # Set up the request
    request = RecommendationRequest(
        workflow_id=complex_workflow["id"],
        focus_areas=[
            RecommendationType.WORKFLOW_OPTIMIZATION,
            RecommendationType.PROMPT_ENHANCEMENT
        ],
        max_suggestions=10,
        min_confidence=0.6
    )
    
    # This would normally time out, but we can simulate by mocking
    with patch.object(recommendation_service.llm_provider, 'generate', 
                     side_effect=lambda prompt, json_mode: 
                        json.dumps([{"optimization_type": "test"}]) if json_mode else "test"):
        
        # Generate recommendations
        response = await recommendation_service.generate_recommendations(request)
        
        # Verify a reasonable result despite the complexity
        assert response is not None
        assert len(response.recommendations) > 0

@pytest.mark.asyncio
async def test_recommendations_invalid_config(recommendation_service):
    """Test recommendation engine with invalid configurations."""
    
    # Create a workflow with invalid node config
    invalid_workflow = {
        "id": str(uuid.uuid4()),
        "name": "Invalid Workflow",
        "nodes": [
            {
                "id": "node1",
                "name": "Invalid Node",
                "type": "unknown_type",
                "config": {"invalid_key": "value"}
            }
        ],
        "edges": []
    }
    
    # Mock services
    class MockWorkflowService:
        async def get_workflow(self,

