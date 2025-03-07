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

import uuid
from src.services.db_service import DBService

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
        self.chat_id = str(uuid.uuid4())
        
        # 移除直接初始化，改为异步获取
        self._db = None
        
    async def _ensure_db(self):
        """确保数据库服务可用"""
        if not self._db:
            self._db = await DBService.get_instance()
            
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

    async def _build_context(self, input_text: str, remark: str = '') -> Dict[str, Any]:
        """构建基础上下文
        
        Args:
            input_text: 用户输入文本
            remark: 备注信息
            
        Returns:
            包含基础上下文信息的字典
        """
        # 确保系统消息存在
        if not self.messages or not isinstance(self.messages[0], SystemMessage):
            system_prompt = await self.load_prompt()
            self.messages = [SystemMessage(content=system_prompt)] + self.messages

        # 添加用户输入
        self.messages.append(HumanMessage(content=input_text))
        
        # 构建基础上下文
        context = {
            "messages": [
                {"role": msg.type, "content": msg.content}
                for msg in self.messages
            ],
            "input_text": input_text,
            "remark": remark,
            "agent_info": {
                "name": self.name,
                "role_id": self.role_id
            }
        }
        
        return context

