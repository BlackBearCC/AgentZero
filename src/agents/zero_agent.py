from typing import Dict, Any, List, Optional, AsyncIterator
from src.agents.base_agent import BaseAgent
from langchain.schema import (
    SystemMessage,
    HumanMessage,
    AIMessage,
    BaseMessage
)

class ZeroAgent(BaseAgent):
    def __init__(self, config: Dict[str, Any], llm=None, memory=None, tools=None):
        """
        初始化函数，用于设置配置和系统提示词。
        
        参数:
        - config: 包含系统配置的字典。
        - llm: 语言模型实例，默认为None。
        - memory: 存储记忆的实例，默认为None。
        - tools: 可用工具的列表，默认为None。
        """
        super().__init__(config, llm, memory, tools)
        # 初始化系统提示词
        if config.get("system_prompt"):
            processed_prompt = self._process_template(config["system_prompt"])
            self.messages.append(SystemMessage(content=processed_prompt))
            self.config["system_prompt"] = processed_prompt

    async def load_prompt(self) -> str:
        """加载角色提示词"""
        return self.config.get("system_prompt", "")

    async def update_prompt(self, **kwargs) -> str:
        """更新角色提示词和变量"""
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

        print(f"[ZeroAgent] Updated prompt: {processed_prompt}")  
        return processed_prompt

    async def think(self, context: Dict[str, Any]) -> List[str]:
        """思考是否需要调用工具"""
        return []  # Zero酱暂时不使用工具

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
            self._logger.error(f"ZeroAgent stream error: {str(e)}")
            raise