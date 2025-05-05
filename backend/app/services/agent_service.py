from typing import List, Optional
from app.models.agent import Agent, AgentType, AgentCapability
from datetime import datetime
import uuid

class AgentService:
    def __init__(self):
        self.agents = {}  # In-memory storage, replace with database in production
    
    async def create_agent(self, agent_data: dict) -> Agent:
        agent_id = str(uuid.uuid4())
        now = datetime.utcnow().isoformat()
        
        agent = Agent(
            id=agent_id,
            name=agent_data["name"],
            description=agent_data["description"],
            type=agent_data["type"],
            capabilities=agent_data["capabilities"],
            configuration=agent_data.get("configuration", {}),
            created_at=now,
            updated_at=now
        )
        
        self.agents[agent_id] = agent
        return agent
    
    async def get_agent(self, agent_id: str) -> Optional[Agent]:
        return self.agents.get(agent_id)
    
    async def list_agents(
        self,
        agent_type: Optional[AgentType] = None,
        capability: Optional[AgentCapability] = None
    ) -> List[Agent]:
        agents = list(self.agents.values())
        
        if agent_type:
            agents = [a for a in agents if a.type == agent_type]
        
        if capability:
            agents = [a for a in agents if capability in a.capabilities]
        
        return agents
    
    async def update_agent(self, agent_id: str, update_data: dict) -> Optional[Agent]:
        if agent_id not in self.agents:
            return None
        
        agent = self.agents[agent_id]
        update_data["updated_at"] = datetime.utcnow().isoformat()
        
        for key, value in update_data.items():
            if hasattr(agent, key):
                setattr(agent, key, value)
        
        self.agents[agent_id] = agent
        return agent
    
    async def delete_agent(self, agent_id: str) -> bool:
        if agent_id in self.agents:
            del self.agents[agent_id]
            return True
        return False
    
    async def activate_agent(self, agent_id: str) -> Optional[Agent]:
        agent = await self.get_agent(agent_id)
        if agent:
            agent.is_active = True
            agent.updated_at = datetime.utcnow().isoformat()
            self.agents[agent_id] = agent
            return agent
        return None
    
    async def deactivate_agent(self, agent_id: str) -> Optional[Agent]:
        agent = await self.get_agent(agent_id)
        if agent:
            agent.is_active = False
            agent.updated_at = datetime.utcnow().isoformat()
            self.agents[agent_id] = agent
            return agent
        return None 