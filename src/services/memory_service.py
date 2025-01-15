from typing import List, Dict, Any
from src.core.memory.base_memory import BaseMemory

class MemoryService:
    def __init__(self, memory_store: BaseMemory):
        self.memory_store = memory_store
    
    async def store_interaction(
        self,
        agent_id: str,
        user_input: str,
        agent_response: str,
        metadata: Dict[str, Any]
    ) -> None:
        """存储交互记录"""
        memory_entry = {
            "user_input": user_input,
            "agent_response": agent_response,
            "metadata": metadata,
            "timestamp": "utc_timestamp"
        }
        await self.memory_store.add(f"{agent_id}:interaction", memory_entry)
    
    async def get_relevant_history(
        self,
        agent_id: str,
        query: str,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """获取相关历史记录"""
        return await self.memory_store.search(query, limit) 