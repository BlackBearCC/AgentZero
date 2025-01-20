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

class BaseAgent:
    def __init__(self, 
                 config: Dict[str, Any],
                 llm=None,
                 memory=None,
                 tools=None):
        """
        统一的Agent类，通过配置来区分不同角色
        
        Args:
            config: 角色配置
            llm: LLM 实例
            memory: 记忆系统实例
            tools: 工具管理器实例
        """
        self.config = config
        self.name = config.get("name", "Assistant")
        self.role_id = config.get("role_id")
        
        # 使用默认的 DoubaoLLM
        self.llm = llm or DoubaoLLM(
            model_name="ep-20241113173739-b6v4g",
            temperature=0.7,
            max_tokens=4096
        )
        
        # 初始化组件
        self.memory = memory
        self.tools = tools or []
        self._logger = Logger()
        self.messages: List[BaseMessage] = []
        
        # 初始化系统提示词
        if config.get("system_prompt"):
            self.messages.append(SystemMessage(content=config["system_prompt"]))
            
    @property
    def logger(self):
        """获取 logger 实例"""
        return self._logger
        
    async def update_history(self, new_message: BaseMessage) -> None:
        """更新对话历史"""
        self.messages.append(new_message)
        
        # 同步到记忆系统
        if self.memory:
            await self.memory.add(
                f"{self.role_id}:history",
                new_message.dict()
            )
            
        # 控制历史长度
        max_history = self.config.get("max_history_length", 20)
        if len(self.messages) > max_history:
            # 保留系统消息
            system_messages = [m for m in self.messages if isinstance(m, SystemMessage)]
            other_messages = [m for m in self.messages if not isinstance(m, SystemMessage)]
            self.messages = system_messages + other_messages[-max_history:]
            
    async def generate_response(self, input_text: str) -> str:
        """生成回复"""
        try:
            # 添加用户消息
            await self.update_history(HumanMessage(content=input_text))
            
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
            
            # 处理结构化输出
            try:
                # 尝试解析 JSON 格式
                response_json = json.loads(response)
                # 如果是祁煜的格式
                if "content" in response_json:
                    content = response_json["content"]
                    # 保存状态和任务信息到上下文
                    self.config["last_state"] = response_json.get("state")
                    self.config["last_task"] = response_json.get("task")
                    return content
            except json.JSONDecodeError:
                # 如果不是 JSON 格式，直接返回原文
                pass
                
            return response
            
        except Exception as e:
            self._logger.error(f"Error generating response: {str(e)}")
            raise
            
    async def astream_response(self, input_text: str) -> AsyncIterator[str]:
        """流式生成回复"""
        try:
            # 添加用户消息到历史
            await self.update_history(HumanMessage(content=input_text))
            
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
                await asyncio.sleep(0)
                
        except Exception as e:
            self._logger.error(f"Stream error: {str(e)}")
            raise
            
    @abstractmethod
    async def load_prompt(self):
        """加载角色提示词"""
        pass
        
    @abstractmethod
    async def update_prompt(self, **kwargs):
        """更新角色提示词"""
        pass
        
    @abstractmethod
    async def think(self, context: Dict[str, Any]) -> List[str]:
        """思考是否需要调用工具"""
        pass 