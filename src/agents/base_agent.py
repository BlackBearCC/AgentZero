from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, AsyncIterator, Tuple
from langchain.schema import (
    SystemMessage,
    HumanMessage,
    AIMessage,
    BaseMessage
)
from src.utils.logger import Logger
from src.llm.doubao import DoubaoLLM
from src.memory.memory import Memory
import asyncio
import json
from string import Template
from abc import ABC, abstractmethod
from datetime import datetime
from src.utils.redis_client import RedisClient
import uuid

class BaseAgent(ABC):
    def __init__(self, 
                 config: Dict[str, Any],
                 llm=None,
                 memory_llm=None,
                 tools=None):
        """初始化基础组件"""
        self.config = config
        self.name = config.get("name", "Assistant")
        self.role_id = config.get("role_id")
        self.variables = config.get("variables", {})
        
        self.llm = llm or DoubaoLLM(
            model_name="ep-20241113173739-b6v4g",
            temperature=0.7,
            max_tokens=4096
        )
        
        self.memory = Memory(memory_llm)
        self.tools = tools or []
        self._logger = Logger()
        self.messages: List[BaseMessage] = []
        self.redis = RedisClient()
        if not self.redis.test_connection():
            raise Exception("Redis connection failed!")
        self.chat_id = str(uuid.uuid4())
        
    def _process_template(self, template: str) -> str:
        """处理提示词模板，替换变量"""
        try:
            for key, value in self.variables.items():
                template = template.replace(f"{{{{{key}}}}}", value)
            return template
        except Exception as e:
            self._logger.error(f"Error processing template: {str(e)}")
            return template
        
    @abstractmethod
    async def load_prompt(self) -> str:
        """加载角色提示词"""
        pass
        
    @abstractmethod
    async def update_prompt(self, **kwargs) -> str:
        """更新角色提示词"""
        pass
        
    @abstractmethod
    async def think(self, context: Dict[str, Any]) -> List[str]:
        """思考是否需要调用工具"""
        pass
        
    @abstractmethod
    async def generate_response(self, input_text: str) -> str:
        """生成回复"""
        pass
        
    @abstractmethod
    async def astream_response(self, input_text: str) -> AsyncIterator[str]:
        """流式生成回复"""
        pass

