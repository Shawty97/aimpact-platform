import os
import uuid
import logging
from typing import List, Optional, Dict, Any, Union
from enum import Enum
from datetime import datetime

from fastapi import APIRouter, HTTPException, Depends, Body, Query, Path, status
from pydantic import BaseModel, Field, validator

# Configure logging
logger = logging.getLogger("aimpact_api.agents")

# Initialize router
router = APIRouter()

# -------------------- Pydantic Models --------------------

class LLMProvider(str, Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    MISTRAL = "mistral"
    LOCAL = "local"

class AgentCapability(str, Enum):
    """Capabilities that can be assigned to an agent."""
    TEXT_GENERATION = "text_generation"
    SUMMARIZATION = "summarization"
    TRANSLATION = "translation"
    QUESTION_ANSWERING = "question_answering"
    CODE_GENERATION = "code_generation"
    IMAGE_ANALYSIS = "image_analysis"
    DOCUMENT_PROCESSING = "document_processing"
    VOICE_INTERACTION = "voice_interaction"
    CUSTOM_KNOWLEDGE = "custom_knowledge"

class AgentCreate(BaseModel):
    """Schema for creating a new agent."""
    name: str = Field(..., description="User-friendly name for the agent")
    description: str = Field(..., description="Description of what this agent does")
    llm_provider: LLMProvider = Field(default=LLMProvider.OPENAI, description="LLM provider to use")
    model_name: str = Field(default="gpt-4", description="Specific model to use from the provider")
    capabilities: List[AgentCapability] = Field(default_factory=list, description="List of capabilities for this agent")
    system_prompt: str = Field(
        default="You are a helpful AI assistant.",
        description="System instructions for the agent"
    )
    tools: List[Dict[str, Any]] = Field(default_factory=list, description="External tools this agent can use")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Additional configuration parameters")
    
    @validator('name')
    def name_must_be_valid(cls, v):
        if len(v.strip()) < 3:
            raise ValueError('Name must be at least 3 characters')
        return v.strip()

class AgentUpdate(BaseModel):
    """Schema for updating an existing agent."""
    name: Optional[str] = None
    description: Optional[str] = None
    llm_provider: Optional[LLMProvider] = None
    model_name: Optional[str] = None
    capabilities: Optional[List[AgentCapability]] = None
    system_prompt: Optional[str] = None
    tools: Optional[List[Dict[str, Any]]] = None
    parameters: Optional[Dict[str, Any]] = None

class Agent(BaseModel):
    """Schema for an agent."""
    id: str = Field(..., description="Unique identifier for the agent")
    name: str = Field(..., description="User-friendly name for the agent")
    description: str = Field(..., description="Description of what this agent does")
    llm_provider: LLMProvider = Field(..., description="LLM provider used")
    model_name: str = Field(..., description="Specific model used from the provider")
    capabilities: List[AgentCapability] = Field(..., description="List of capabilities for this agent")
    system_prompt: str = Field(..., description="System instructions for the agent")
    tools: List[Dict[str, Any]] = Field(default_factory=list, description="External tools this agent can use")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Additional configuration parameters")
    created_at: datetime = Field(..., description="When this agent was created")
    updated_at: datetime = Field(..., description="When this agent was last updated")

class MessageRole(str, Enum):
    """Message roles in a conversation."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    FUNCTION = "function"
    TOOL = "tool"

class Message(BaseModel):
    """Schema for a message in a conversation."""
    role: MessageRole
    content: str
    name: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None

class AgentInteractionRequest(BaseModel):
    """Schema for interacting with an agent."""
    messages: List[Message] = Field(..., description="Conversation history")
    stream: bool = Field(default=False, description="Whether to stream the response")
    max_tokens: Optional[int] = Field(default=None, description="Maximum tokens to generate")
    temperature: float = Field(default=0.7, description="Temperature for generation randomness")
    tools: Optional[List[Dict[str, Any]]] = Field(default=None, description="Tools available for this interaction")

class AgentInteractionResponse(BaseModel):
    """Schema for an agent interaction response."""
    agent_id: str
    message: Message
    usage: Dict[str, int] = Field(default_factory=dict, description="Token usage information")
    finished: bool = True

# -------------------- Mock Database --------------------
# In a real implementation, this would be replaced with database connections

# Simple in-memory storage for demonstration
agent_db: Dict[str, Agent] = {}

# -------------------- Helper Functions --------------------

def get_agent_by_id(agent_id: str) -> Agent:
    """Retrieve an agent by its ID."""
    if agent_id not in agent_db:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Agent with ID {agent_id} not found"
        )
    return agent_db[agent_id]

# -------------------- API Endpoints --------------------

@router.post("/", response_model=Agent, status_code=status.HTTP_201_CREATED)
async def create_agent(agent_data: AgentCreate):
    """
    Create a new AI agent with specific capabilities.
    
    Returns the created agent with its generated ID.
    """
    agent_id = str(uuid.uuid4())
    now = datetime.now()
    
    new_agent = Agent(
        id=agent_id,
        name=agent_data.name,
        description=agent_data.description,
        llm_provider=agent_data.llm_provider,
        model_name=agent_data.model_name,
        capabilities=agent_data.capabilities,
        system_prompt=agent_data.system_prompt,
        tools=agent_data.tools,
        parameters=agent_data.parameters,
        created_at=now,
        updated_at=now
    )
    
    agent_db[agent_id] = new_agent
    logger.info(f"Created new agent: {agent_id} ({agent_data.name})")
    
    return new_agent

@router.get("/", response_model=List[Agent])
async def list_agents(
    capability: Optional[AgentCapability] = Query(None, description="Filter by capability"),
    provider: Optional[LLMProvider] = Query(None, description="Filter by LLM provider")
):
    """
    List all available agents, with optional filtering.
    
    Parameters:
    - capability: Filter agents by specific capability
    - provider: Filter agents by LLM provider
    """
    agents = list(agent_db.values())
    
    if capability:
        agents = [agent for agent in agents if capability in agent.capabilities]
    
    if provider:
        agents = [agent for agent in agents if agent.llm_provider == provider]
    
    return agents

@router.get("/{agent_id}", response_model=Agent)
async def get_agent(agent_id: str = Path(..., description="The ID of the agent to retrieve")):
    """
    Retrieve a specific agent by ID.
    """
    return get_agent_by_id(agent_id)

@router.put("/{agent_id}", response_model=Agent)
async def update_agent(
    agent_data: AgentUpdate,
    agent_id: str = Path(..., description="The ID of the agent to update")
):
    """
    Update an existing agent's configuration.
    
    Only fields provided in the request will be updated.
    """
    agent = get_agent_by_id(agent_id)
    
    # Update agent fields if provided in the request
    update_data = agent_data.dict(exclude_unset=True)
    for key, value in update_data.items():
        if value is not None:
            setattr(agent, key, value)
    
    agent.updated_at = datetime.now()
    agent_db[agent_id] = agent
    
    logger.info(f"Updated agent: {agent_id}")
    return agent

@router.delete("/{agent_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_agent(agent_id: str = Path(..., description="The ID of the agent to delete")):
    """
    Delete an agent by ID.
    """
    get_agent_by_id(agent_id)  # Check if agent exists
    del agent_db[agent_id]
    logger.info(f"Deleted agent: {agent_id}")
    return None

@router.post("/{agent_id}/interact", response_model=AgentInteractionResponse)
async def interact_with_agent(
    interaction: AgentInteractionRequest,
    agent_id: str = Path(..., description="The ID of the agent to interact with")
):
    """
    Interact with a specific agent.
    
    Send messages to the agent and receive a response.
    """
    agent = get_agent_by_id(agent_id)
    
    # In a real implementation, this would call the appropriate LLM service
    # For demonstration, we'll simulate a response
    
    logger.info(f"Processing interaction with agent: {agent_id}")
    
    # Simulate agent processing
    if not interaction.messages:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No messages provided for interaction"
        )
    
    last_message = interaction.messages[-1]
    
    # Simulate a simple response
    response_content = f"This is a simulated response from agent {agent.name} using {agent.llm_provider} ({agent.model_name}). "
    response_content += f"I received: '{last_message.content[:30]}...'"
    
    response = AgentInteractionResponse(
        agent_id=agent_id,
        message=Message(
            role=MessageRole.ASSISTANT,
            content=response_content
        ),
        usage={
            "prompt_tokens": 50,
            "completion_tokens": len(response_content.split()) * 2,  # Rough estimate
            "total_tokens": 50 + len(response_content.split()) * 2
        }
    )
    
    return response

