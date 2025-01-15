from typing import Dict, Any, List, Optional
from pydantic import BaseModel

class AgentConfig(BaseModel):
    role_id: str
    name: str
    description: str
    personality: str
    system_prompt: str
    constraints: List[str]
    tools: List[str]
    memory_config: Dict[str, Any]

class AgentCreate(BaseModel):
    config: AgentConfig

class AgentResponse(BaseModel):
    agent_id: str
    status: str

class AgentInfo(BaseModel):
    agent_id: str
    config: Dict[str, Any] 