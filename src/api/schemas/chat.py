from typing import Dict, Any, Optional
from fastapi import UploadFile
from pydantic import BaseModel, Field

class AgentConfig(BaseModel):
    """Agent配置模型"""
    use_memory_queue: bool = Field(default=True, description="是否使用记忆队列")
    use_combined_query: bool = Field(default=False, description="是否使用组合查询")
    enable_memory_recall: bool = Field(default=True, description="是否启用记忆 Recall")
    memory_queue_limit: int = Field(default=15, description="记忆队列长度限制")
    llm_model: str = Field(default="doubao", description="LLM模型选择")
    llm_temperature: float = Field(default=0.7, description="温度参数")
    use_event_summary: bool = Field(default=True, description="是否启用事件概要")

class ChatRequest(BaseModel):
    message: str
    user_id: str = Field(..., description="用户ID,用于数据隔离")
    remark: Optional[str] = None
    config: Optional[AgentConfig] = None

class ChatResponse(BaseModel):
    response: str
    metadata: Optional[Dict[str, Any]] = Field(None, description="包含实体召回等中间数据")

class StreamChatResponse(BaseModel):
    content: str
    metadata: Optional[Dict[str, Any]] = Field(None, description="包含实体召回等中间数据")

class AgentResponse(BaseModel):
    agent_id: str
    status: str

class EvalRequest(BaseModel):
    """评估请求模型"""
    file: UploadFile = Field(..., description="待评估的文件")
    eval_type: str = Field(..., description="评估类型")
    user_id: str = Field(..., description="用户ID")
    config: Optional[AgentConfig] = None

class EvalResponse(BaseModel):
    """评估响应模型"""
    content: str
    metadata: Optional[Dict[str, Any]] = Field(None, description="评估元数据") 