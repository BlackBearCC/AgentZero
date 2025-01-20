from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, AsyncIterator
from langchain_community.chat_models import ChatDeepInfra
from langchain.schema import (
    SystemMessage,
    HumanMessage,
    AIMessage,
    BaseMessage
)
from langchain.prompts import ChatPromptTemplate
from src.utils.logger import Logger
from src.llm.deepseek import DeepSeekLLM
import asyncio

class BaseAgent(ABC):
    def __init__(self, 
                 config: Dict[str, Any],
                 llm=None,
                 memory=None,
                 tools=None):
        """
        基础Agent类
        
        Args:
            config: 角色配置
            llm: LangChain 聊天模型实例
            memory: 记忆系统实例
            tools: 工具管理器实例
        """
        self.config = config
        self.name = config.get("name", "Assistant")
        
        # 使用自定义的 DeepSeek LLM
        self.llm = llm or DeepSeekLLM(
            model_name=config.get("model_name", "deepseek-chat"),
            temperature=config.get("temperature", 0.7),
            top_p=config.get("top_p", 0.9),
            max_tokens=config.get("max_tokens", 4096),
            tensor_parallel_size=config.get("tensor_parallel_size", 1)
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
                f"{self.config['role_id']}:history",
                new_message.dict()
            )
            
        # 控制历史长度
        max_history = self.config.get("max_history_length", 20)
        if len(self.messages) > max_history:
            # 保留系统消息
            system_messages = [m for m in self.messages if isinstance(m, SystemMessage)]
            other_messages = [m for m in self.messages if not isinstance(m, SystemMessage)]
            self.messages = system_messages + other_messages[-max_history:]
            
    async def generate_response(self, 
                              input_text: str,
                              context: Optional[Dict] = None) -> str:
        """生成回复"""
        try:
            # 构建提示模板
            prompt = ChatPromptTemplate.from_messages([
                ("system", self.config.get("system_prompt", "")),
                ("human", "{input}")
            ])
            
            # 添加用户消息
            await self.update_history(HumanMessage(content=input_text))
            
            # 检查工具调用
            if self.tools:
                tool_calls = await self.think({"input": input_text, "context": context})
                if tool_calls:
                    tool_results = []
                    for tool_call in tool_calls:
                        result = await self.tools.execute(tool_call)
                        tool_results.append(result)
                    context = context or {}
                    context["tool_results"] = tool_results
            
            # 生成回复
            chain = prompt | self.llm
            response = await chain.ainvoke({
                "input": input_text,
                "context": context
            })
            
            # 直接返回生成的文本
            if isinstance(response, str):
                return response
            elif hasattr(response, "generations"):
                # 处理 LLMResult
                return response.generations[0][0].text
            else:
                # 其他情况，尝试获取内容
                return str(response)
            
        except Exception as e:
            # self.logger.error(f"Error generating response: {str(e)}")
            raise
            
    async def astream_response(self, 
                           input_text: str,
                           context: Optional[Dict] = None) -> AsyncIterator[str]:
        """流式生成回复"""
        try:
            # 构建消息列表
            system_prompt = await self.load_prompt()
            messages = [
                {
                    "role": "system",
                    "content": system_prompt  # 直接使用字符串
                },
                {
                    "role": "user",
                    "content": input_text  # 直接使用字符串
                }
            ]
            
            # 添加用户消息到历史
            await self.update_history(HumanMessage(content=input_text))
            
            # 检查工具调用
            if self.tools:
                tool_calls = await self.think({"input": input_text, "context": context})
                if tool_calls:
                    tool_results = []
                    for tool_call in tool_calls:
                        result = await self.tools.execute(tool_call)
                        tool_results.append(result)
                    context = context or {}
                    context["tool_results"] = tool_results
            
            self._logger.logger.debug(f"Starting stream with messages: {messages}")
            
            # 直接使用 LLM 的流式接口
            async for chunk in self.llm.astream(messages):
                self._logger.logger.debug(f"Received chunk: {chunk}")
                yield chunk
                await asyncio.sleep(0)  # 确保立即输出
                
        except Exception as e:
            self._logger.logger.error(f"Stream error: {str(e)}")
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