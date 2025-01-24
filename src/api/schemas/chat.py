from typing import Dict, Any, Optional
from pydantic import BaseModel

class ChatRequest(BaseModel):
    message: str
    remark: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    metadata: Optional[Dict[str, Any]] = None

class AgentResponse(BaseModel):
    agent_id: str
    status: str 