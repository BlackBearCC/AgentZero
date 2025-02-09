from typing import Dict, Any, Optional
from pydantic import BaseModel, Field

class AgentConfig(BaseModel):
    """Agent配置模型"""
    use_memory_queue: bool = Field(default=True, description="是否使用记忆队列")
    use_combined_query: bool = Field(default=False, description="是否使用组合查询")
    memory_queue_limit: int = Field(default=15, description="记忆队列长度限制")
    llm_model: str = Field(default="doubao", description="LLM模型选择")
    llm_temperature: float = Field(default=0.7, description="温度参数")

class ChatRequest(BaseModel):
    message: str
    user_id: str = Field(..., description="用户ID,用于数据隔离")
    remark: Optional[str] = None
    config: Optional[AgentConfig] = None

class ChatResponse(BaseModel):
    response: str
    metadata: Optional[Dict[str, Any]] = None

class AgentResponse(BaseModel):
    agent_id: str
    status: str 