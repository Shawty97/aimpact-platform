from fastapi import APIRouter, HTTPException, Depends
from typing import List, Optional
from app.models.agent import Agent, AgentType, AgentCapability
from app.services.agent_service import AgentService

router = APIRouter()
agent_service = AgentService()

@router.post("/", response_model=Agent)
async def create_agent(agent_data: dict):
    try:
        return await agent_service.create_agent(agent_data)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/{agent_id}", response_model=Agent)
async def get_agent(agent_id: str):
    agent = await agent_service.get_agent(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    return agent

@router.get("/", response_model=List[Agent])
async def list_agents(
    agent_type: Optional[AgentType] = None,
    capability: Optional[AgentCapability] = None
):
    return await agent_service.list_agents(agent_type, capability)

@router.put("/{agent_id}", response_model=Agent)
async def update_agent(agent_id: str, update_data: dict):
    agent = await agent_service.update_agent(agent_id, update_data)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    return agent

@router.delete("/{agent_id}")
async def delete_agent(agent_id: str):
    success = await agent_service.delete_agent(agent_id)
    if not success:
        raise HTTPException(status_code=404, detail="Agent not found")
    return {"message": "Agent deleted successfully"}

@router.post("/{agent_id}/activate", response_model=Agent)
async def activate_agent(agent_id: str):
    agent = await agent_service.activate_agent(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    return agent

@router.post("/{agent_id}/deactivate", response_model=Agent)
async def deactivate_agent(agent_id: str):
    agent = await agent_service.deactivate_agent(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    return agent 