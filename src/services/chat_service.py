from typing import Dict, Any
from src.agents.base_agent import BaseAgent

class ChatService:
    def __init__(self):
        self.agents: Dict[str, BaseAgent] = {}
    
    async def process_message(
        self,
        agent_id: str,
        message: str,
        context: Dict[str, Any]
    ) -> str:
        """处理聊天消息"""
        agent = self.agents.get(agent_id)
        if not agent:
            raise ValueError(f"Agent {agent_id} not found")
        
        return await agent.process(message) 