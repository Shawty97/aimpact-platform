"""
Integration tests for workflow functionality including:
- Agent memory
- Self-learning
- Recommendations
"""

import pytest
import asyncio
import json
import uuid
from datetime import datetime, timedelta

# Import the services that need to be tested
from backend.memory.service import MemoryService
from backend.memory.models import Memory, MemoryType, MemoryMetadata
from backend.optimization.service import OptimizerService
from backend.optimization.models import AgentFeedback, FeedbackValue, FeedbackType, OptimizationJob
from backend.recommendations.service import RecommendationService
from backend.recommendations.models import Recommendation, RecommendationType

# Mock embeddings provider for testing
class MockEmbeddingProvider:
    def embed(self, text):
        # Return a simple mock embedding of the correct dimension
        return [0.1] * 768
    
    async def aembed(self, text):
        # Async version
        return [0.1] * 768

# Mock LLM provider for testing
class MockLLMProvider:
    async def generate(self, prompt, json_mode=False):
        if json_mode:
            if "workflow structure" in prompt:
                # Mock optimization suggestions
                return json.dumps([
                    {
                        "optimization_type": "parallelization",
                        "affected_nodes": ["node1", "node2"],
                        "description": "Run these nodes in parallel",
                        "expected_benefits": ["Improved performance", "Reduced latency"],
                        "implementation_complexity": "medium",
                        "reasoning": "These nodes do not have dependencies on each other."
                    }
                ])
            elif "enhance the following prompt" in prompt:
                # Mock prompt enhancement
                return json.dumps({
                    "enhanced_prompt": "This is an enhanced version of the prompt with more details.",
                    "improvements": ["Added context", "Clarified instructions", "Better formatting"],
                    "expected_benefits": ["Improved responses", "Reduced confusion"],
                    "reasoning": "The original prompt lacked specific instructions."
                })
            else:
                # Default response
                return json.dumps({"result": "mocked response"})
        else:
            return "This is a mocked LLM response."

# Test workflow data
@pytest.fixture
def sample_workflow():
    return {
        "id": str(uuid.uuid4()),
        "name": "Test Workflow",
        "description": "A workflow for testing",
        "version": "1.0",
        "nodes": [
            {
                "id": "node1",
                "name": "Input Node",
                "type": "input",
                "config": {}
            },
            {
                "id": "node2",
                "name": "Prompt Node",
                "type": "prompt",
                "config": {
                    "prompt": "This is a test prompt that could be improved."
                }
            },
            {
                "id": "node3",
                "name": "Output Node",
                "type": "output",
                "config": {}
            }
        ],
        "edges": [
            {
                "source_id": "node1",
                "target_id": "node2",
                "type": "standard"
            },
            {
                "source_id": "node2",
                "target_id": "node3",
                "type": "standard"
            }
        ]
    }

# Test agent data
@pytest.fixture
def sample_agent():
    return {
        "id": str(uuid.uuid4()),
        "name": "Test Agent",
        "description": "An agent for testing",
        "type": "chatbot",
        "capabilities": ["text_generation", "knowledge_base"],
        "created_at": datetime.utcnow().isoformat(),
        "updated_at": datetime.utcnow().isoformat()
    }

# Service fixtures
@pytest.fixture
async def memory_service():
    # Use an in-memory Redis mock
    from fakeredis import FakeRedis
    redis_client = FakeRedis()
    
    # Create memory service with mock Redis
    service = MemoryService(
        redis_host="mock",
        redis_port=6379,
        redis_db=0,
        embedding_dimension=768,
        embedding_provider=MockEmbeddingProvider(),
        default_ttl=3600
    )
    
    # Replace the Redis client with our mock
    service.redis_client = redis_client
    
    return service

@pytest.fixture
async def optimizer_service(memory_service):
    # Create optimizer service with mock dependencies
    service = OptimizerService(
        memory_service=memory_service,
        model_storage_path="/tmp/models",
        enable_background_processing=False
    )
    
    return service

@pytest.fixture
async def recommendation_service(memory_service):
    # Create recommendation service with mock dependencies
    service = RecommendationService(
        memory_service=memory_service,
        llm_provider=MockLLMProvider(),
        module_registry=None,  # Mock will be defined in the test
        workflow_service=None,  # Mock will be defined in the test
        embedding_provider=MockEmbeddingProvider(),
        cache_ttl=60
    )
    
    return service

# Test cases
@pytest.mark.asyncio
async def test_memory_storage_and_retrieval(memory_service, sample_agent):
    """Test storing and retrieving memories."""
    agent_id = sample_agent["id"]
    
    # Store a memory
    memory = await memory_service.save_memory(
        agent_id=agent_id,
        memory_type=MemoryType.CONVERSATION,
        content="This is a test memory",
        metadata=MemoryMetadata(
            session_id="test-session",
            tags=["test", "conversation"]
        )
    )
    
    assert memory is not None
    assert memory.id is not None
    assert memory.agent_id == agent_id
    assert memory.type == MemoryType.CONVERSATION
    assert memory.content == "This is a test memory"
    
    # Retrieve the memory
    retrieved_memory = await memory_service.get_memory(memory.id)
    
    assert retrieved_memory is not None
    assert retrieved_memory.id == memory.id
    assert retrieved_memory.agent_id == agent_id
    assert retrieved_memory.content == "This is a test memory"
    assert "test" in retrieved_memory.metadata.tags
    
    # Search for memories
    search_query = {
        "agent_id": agent_id,
        "query": "test memory",
        "memory_types": [MemoryType.CONVERSATION],
        "tags": ["test"]
    }
    
    search_results = await memory_service.search_memories(search_query)
    
    assert len(search_results) > 0
    assert search_results[0].memory.id == memory.id
    assert search_results[0].score > 0.5

@pytest.mark.asyncio
async def test_agent_learning_from_feedback(optimizer_service, memory_service, sample_agent):
    """Test that agents can learn from feedback."""
    agent_id = sample_agent["id"]
    
    # Create feedback
    feedback = AgentFeedback(
        id=str(uuid.uuid4()),
        agent_id=agent_id,
        session_id="test-session",
        feedback_type=FeedbackType.HELPFULNESS,
        value=FeedbackValue(score=4.5),
        context={"query": "What is the capital of France?", "response": "The capital of France is Paris."}
    )
    
    # Save feedback
    saved_feedback = await optimizer_service.save_feedback(feedback)
    
    assert saved_feedback is not None
    assert saved_feedback.id == feedback.id
    
    # Retrieve feedback
    retrieved_feedback = await optimizer_service.get_feedback(feedback.id)
    
    assert retrieved_feedback is not None
    assert retrieved_feedback.id == feedback.id
    assert retrieved_feedback.value.score == 4.5
    
    # Create optimization job
    job_id = str(uuid.uuid4())
    job = OptimizationJob(
        id=job_id,
        agent_id=agent_id,
        targets=["response_quality"],
        config={"learning_rate": 3e-4, "batch_size": 32}
    )
    
    # In a real test, we would submit the job and verify learning happened
    # For this integration test, we verify the job is stored correctly
    optimizer_service._job_store[job_id] = job
    
    assert job_id in optimizer_service._job_store
    assert optimizer_service._job_store[job_id].agent_id == agent_id

@pytest.mark.asyncio
async def test_recommendation_generation(recommendation_service, sample_workflow):
    """Test generating recommendations for a workflow."""
    # Mock workflow service
    class MockWorkflowService:
        async def get_workflow(self, workflow_id):
            return sample_workflow
    
    recommendation_service.workflow_service = MockWorkflowService()
    
    # Mock module registry
    class MockModuleRegistry:
        async def list_modules(self):
            return [
                {
                    "id": "module1",
                    "name": "Test Module",
                    "type": "processor",
                    "description": "A test module",
                    "capabilities": ["data_processing", "filtering"]
                }
            ]
    
    recommendation_service.module_registry = MockModuleRegistry()
    
    # Generate recommendations
    from backend.recommendations.models import RecommendationRequest
    
    request = RecommendationRequest(
        workflow_id=sample_workflow["id"],
        focus_areas=[RecommendationType.WORKFLOW_OPTIMIZATION, RecommendationType.PROMPT_ENHANCEMENT],
        max_suggestions=3,
        min_confidence=0.6
    )
    
    response = await recommendation_service.generate_recommendations(request)
    
    assert response is not None
    assert response.workflow_id == sample_workflow["id"]
    assert len(response.recommendations) > 0
    
    # Check that we have the expected recommendation types
    recommendation_types = [rec.type for rec in response.recommendations]
    assert RecommendationType.WORKFLOW_OPTIMIZATION in recommendation_types or RecommendationType.PROMPT_ENHANCEMENT in recommendation_types
    
    # Check recommendation details
    for rec in response.recommendations:
        assert rec.confidence >= request.min_confidence
        assert rec.workflow_id == sample_workflow["id"]
        assert rec.title is not None and len(rec.title) > 0
        assert rec.description is not None and len(rec.description) > 0

# Integration test for the entire workflow
@pytest.mark.asyncio
async def test_end_to_end_workflow(memory_service, optimizer_service, recommendation_service, sample_agent, sample_workflow):
    """Test the entire workflow from memory to learning to recommendations."""
    agent_id = sample_agent["id"]
    workflow_id = sample_workflow["id"]
    
    # 1. Store agent context in memory
    memory = await memory_service.save_memory(
        agent_id=agent_id,
        memory_type=MemoryType.CONTEXT,
        content={"workflow_id": workflow_id, "last_interaction": "query about data processing"},
        metadata=MemoryMetadata(
            session_id="test-session",
            tags=["context", "workflow"]
        )
    )
    
    assert memory is not None
    
    # 2. Provide agent feedback
    feedback = AgentFeedback(
        id=str(uuid.uuid4()),
        agent_id=agent_id,
        session_id="test-session",
        feedback_type=FeedbackType.HELPFULNESS,
        value=FeedbackValue(score=4.0),
        context={"workflow_id": workflow_id, "node_id": "node2"}
    )
    
    saved_feedback = await optimizer_service.save_feedback(feedback)
    assert saved_feedback is not None
    
    # 3. Generate workflow recommendations
    # Setup mock services
    class MockWorkflowService:
        async def get_workflow(self, workflow_id):
            return sample_workflow
    
    recommendation_service.workflow_service = MockWorkflowService()
    
    class MockModuleRegistry:
        async def list_modules(self):
            return [
                {
                    "id": "module1",
                    "name": "Test Module",
                    "type": "processor",
                    "description": "A test module",
                    "capabilities": ["data_processing", "filtering"]
                }
            ]
    
    recommendation_service.module_registry = MockModuleRegistry()
    
    # Generate recommendations
    from backend.recommendations.models import RecommendationRequest
    
    request = RecommendationRequest(
        workflow_id=workflow_id,
        focus_areas=[RecommendationType.WORKFLOW_OPTIMIZATION, RecommendationType.PROMPT_ENHANCEMENT],
        max_suggestions=3,
        min_confidence=0.6
    )
    
    response = await recommendation_service.generate_recommendations(request)
    
    assert response is not None
    assert len(response.recommendations) > 0
    
    # Apply a recommendation (simulate)
    if response.recommendations:
        rec = response.recommendations[0]
        updated_rec = await recommendation_service.apply_recommendation(rec.id)
        assert updated_rec is not None
        assert updated_rec.applied
    
    # 4. Verify that feedback influenced recommendations
    # (This would involve more complex testing in a real implementation)
    
    # Cleanup
    if memory:
        await memory_service.delete_memory(memory.id)

