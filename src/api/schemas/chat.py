from typing import Dict, Any, Optional
from pydantic import BaseModel

class ChatRequest(BaseModel):
    message: str
    context: Optional[Dict[str, Any]] = None

class ChatResponse(BaseModel):
    response: str
    metadata: Optional[Dict[str, Any]] = None

class AgentResponse(BaseModel):
    agent_id: str
    status: str 