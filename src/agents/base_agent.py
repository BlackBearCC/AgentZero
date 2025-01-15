from abc import ABC, abstractmethod
from typing import Dict, List, Any

class BaseAgent(ABC):
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.memory = None
        self.llm = None
        self.tools = []
    
    @abstractmethod
    async def process(self, input_text: str) -> str:
        """处理输入并生成响应"""
        pass
    
    @abstractmethod
    async def think(self, context: Dict[str, Any]) -> List[str]:
        """思考链路"""
        pass 