from typing import Dict, Any, List, Optional
from pydantic import BaseModel

class AgentConfig(BaseModel):
    role_id: str
    name: str
    system_prompt: str

class AgentCreate(BaseModel):
    config: AgentConfig

class AgentResponse(BaseModel):
    agent_id: str
    status: str

class AgentInfo(BaseModel):
    agent_id: str
    config: Dict[str, Any] 