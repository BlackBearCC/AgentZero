from abc import ABC, abstractmethod
from typing import List, Dict, Any

class BaseMemory(ABC):
    @abstractmethod
    async def add(self, key: str, content: Any) -> None:
        """添加记忆"""
        pass
    
    @abstractmethod
    async def get(self, key: str) -> Any:
        """获取记忆"""
        pass
    
    @abstractmethod
    async def search(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """搜索相关记忆"""
        pass 