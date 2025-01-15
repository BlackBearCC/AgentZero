from typing import Dict, List, Optional
from pydantic import BaseModel

class RoleConfig(BaseModel):
    role_id: str
    name: str
    description: str
    personality: str
    system_prompt: str
    constraints: List[str]
    tools: List[str]
    memory_config: Dict[str, any]
    
    class Config:
        arbitrary_types_allowed = True 