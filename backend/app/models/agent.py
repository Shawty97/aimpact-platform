from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from enum import Enum

class AgentType(str, Enum):
    CHATBOT = "chatbot"
    VOICE = "voice"
    WORKFLOW = "workflow"
    AUTOMATION = "automation"
    ANALYTICS = "analytics"

class AgentCapability(str, Enum):
    TEXT_GENERATION = "text_generation"
    VOICE_SYNTHESIS = "voice_synthesis"
    IMAGE_GENERATION = "image_generation"
    DATA_ANALYSIS = "data_analysis"
    WORKFLOW_AUTOMATION = "workflow_automation"
    KNOWLEDGE_BASE = "knowledge_base"

class Agent(BaseModel):
    id: str = Field(..., description="Unique identifier for the agent")
    name: str = Field(..., description="Name of the agent")
    description: str = Field(..., description="Description of the agent's purpose")
    type: AgentType = Field(..., description="Type of the agent")
    capabilities: List[AgentCapability] = Field(..., description="List of agent capabilities")
    configuration: Dict[str, Any] = Field(default_factory=dict, description="Agent configuration")
    is_active: bool = Field(default=True, description="Whether the agent is active")
    version: str = Field(default="1.0.0", description="Version of the agent")
    created_at: str = Field(..., description="Creation timestamp")
    updated_at: str = Field(..., description="Last update timestamp")
    
    class Config:
        json_schema_extra = {
            "example": {
                "id": "agent-123",
                "name": "Customer Support Bot",
                "description": "AI-powered customer support agent",
                "type": "chatbot",
                "capabilities": ["text_generation", "knowledge_base"],
                "configuration": {
                    "temperature": 0.7,
                    "max_tokens": 1000
                },
                "is_active": True,
                "version": "1.0.0",
                "created_at": "2024-01-01T00:00:00Z",
                "updated_at": "2024-01-01T00:00:00Z"
            }
        } 