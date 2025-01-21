from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, AsyncIterator, Tuple
from langchain.schema import (
    SystemMessage,
    HumanMessage,
    AIMessage,
    BaseMessage
)
from langchain.prompts import ChatPromptTemplate
from src.utils.logger import Logger
from src.llm.doubao import DoubaoLLM
import asyncio
import json
from string import Template
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime

@dataclass
class Message:
    role: str
    content: str
    timestamp: datetime = datetime.now()

class Memory:
    def __init__(self):
        self.chat_history: List[Message] = []  # 对话历史
        self.summary: str = ""  # 线性时间概要
        self.entity_memories: Dict[str, Any] = {}  # 实体记忆存储
        
    async def add_message(self, role: str, content: str):
        """添加新消息到历史记录，如果是 assistant 的回复则解析 JSON"""
        if role == "assistant":
            try:
                # 解析 JSON 响应
                response_json = json.loads(content)
                # 只保存实际内容
                content = response_json.get("content", content)
            except json.JSONDecodeError:
                pass  # 如果不是 JSON 格式则保持原样
                
        self.chat_history.append(Message(role=role, content=content))

        
    async def get_recent_messages(self, limit: int = 10) -> List[Message]:
        """获取最近的消息"""
        return self.chat_history[-limit:]
        
    async def update_summary(self, new_content: str):
        """更新对话概要"""
        # TODO: 实现对话概要更新逻辑
        pass
        
    async def query_entity_memory(self, query: str) -> List[Dict[str, Any]]:
        """查询实体记忆"""
        # TODO: 实现 RAG 检索逻辑
        pass

class BaseAgent(ABC):
    def __init__(self, 
                 config: Dict[str, Any],
                 llm=None,
                 memory=None,
                 tools=None):
        """
        初始化基础组件

        参数:
        - config: 一个字典，包含配置信息，如名称、角色ID和变量等
        - llm: 语言模型实例，默认为None
        - memory: 存储记忆的实例，默认为None
        - tools: 一个工具列表，默认为空列表
        """
        self.config = config
        self.name = config.get("name", "Assistant")
        self.role_id = config.get("role_id")
        self.variables = config.get("variables", {})
        
        self.llm = llm or DoubaoLLM(
            model_name="ep-20241113173739-b6v4g",
            temperature=0.7,
            max_tokens=4096
        )
        
        self.memory = memory or Memory()
        self.tools = tools or []
        self._logger = Logger()
        self.messages: List[BaseMessage] = []
        
        # 初始化系统提示词
        if config.get("system_prompt"):
            processed_prompt = self._process_template(config["system_prompt"])
            self.messages.append(SystemMessage(content=processed_prompt))
            self.config["system_prompt"] = processed_prompt
    
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

    async def _process_memory(self, input_text: str) -> Tuple[str, List[Dict[str, Any]]]:
        """处理记忆并生成上下文"""
        # 获取最近对话历史
        recent_messages = await self.memory.get_recent_messages()
        
        # 获取当前对话概要
        current_summary = self.memory.summary
        
        # 查询相关实体记忆
        relevant_memories = await self.memory.query_entity_memory(input_text)
        
        # 构建上下文
        context = {
            "recent_messages": recent_messages,
            "summary": current_summary,
            "relevant_memories": relevant_memories
        }
        
        return context