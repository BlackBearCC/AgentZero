from typing import Dict, List, Optional
from pydantic import BaseModel

class LLMConfig(BaseModel):
    model_type: str  # "doubao-pro" | "doubao" | "deepseek-chat"
    temperature: float = 0.7
    max_tokens: int = 4096

class RoleConfig(BaseModel):
    role_id: str  # 角色唯一标识
    name: str     # 角色名称
    system_prompt: str  # 系统提示词
    llm_config: LLMConfig  # 主要 LLM 配置
    memory_llm_config: Optional[LLMConfig] = None  # 记忆 LLM 配置，可选
    
    class Config:
        arbitrary_types_allowed = True 