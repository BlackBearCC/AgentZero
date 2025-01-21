from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, AsyncIterator
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

class BaseAgent(ABC):
    def __init__(self, 
                 config: Dict[str, Any],
                 llm=None,
                 memory=None,
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
        
        self.memory = memory
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
            # 直接替换 {{.xxx}} 为实际值
            for key, value in self.variables.items():
                template = template.replace(f"{{{{{key}}}}}", value)
            return template
        except Exception as e:
            self._logger.error(f"Error processing template: {str(e)}")
            return template
    
    async def update_prompt(self, **kwargs) -> str:
        """更新角色提示词和变量，基类提供基础实现"""
        if kwargs:
            if "system_prompt" in kwargs:
                self.config["system_prompt"] = kwargs["system_prompt"]
            
            if "variables" in kwargs:
                self.variables.update(kwargs["variables"])
        
        # 处理模板
        processed_prompt = self._process_template(self.config["system_prompt"])
        self.config["system_prompt"] = processed_prompt
        
        # 更新系统消息
        for i, msg in enumerate(self.messages):
            if isinstance(msg, SystemMessage):
                self.messages[i] = SystemMessage(content=processed_prompt)
                break
        
        return processed_prompt

    @abstractmethod
    async def load_prompt(self):
        """加载角色提示词"""
        pass
        
    @abstractmethod
    async def think(self, context: Dict[str, Any]) -> List[str]:
        """思考是否需要调用工具"""
        pass

    async def generate_response(self, input_text: str) -> str:
        """生成回复"""
        try:
            # 先更新提示词
            await self.update_prompt()
            
            # 添加用户消息
            self.messages.append(HumanMessage(content=input_text))
            
            # 构建消息列表
            messages = [
                {
                    "role": "system",
                    "content": self.config["system_prompt"]
                },
                {
                    "role": "user", 
                    "content": input_text
                }
            ]
            
            # 生成回复
            response = await self.llm.agenerate(messages)
            return response
            
        except Exception as e:
            self._logger.error(f"Error generating response: {str(e)}")
            raise
            
    async def astream_response(self, input_text: str) -> AsyncIterator[str]:
        """流式生成回复"""
        try:
            # 先更新提示词
            await self.update_prompt()
            
            # 添加用户消息到历史
            self.messages.append(HumanMessage(content=input_text))
            
            # 构建消息列表
            messages = [
                {
                    "role": "system",
                    "content": self.config["system_prompt"]
                },
                {
                    "role": "user",
                    "content": input_text
                }
            ]
            
            # 直接使用 LLM 的流式接口
            async for chunk in self.llm.astream(messages):
                yield chunk
                
        except Exception as e:
            self._logger.error(f"Stream error: {str(e)}")
            raise