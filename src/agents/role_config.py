from typing import Dict, List, Optional
from pydantic import BaseModel

class RoleConfig(BaseModel):
    role_id: str  # 角色唯一标识
    name: str     # 角色名称
    system_prompt: str  # 系统提示词
    
    class Config:
        arbitrary_types_allowed = True 