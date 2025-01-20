from typing import Dict, Any, List, Optional, AsyncIterator
from src.agents.base_agent import BaseAgent
import asyncio

class ZeroAgent(BaseAgent):
    async def load_prompt(self) -> str:
        """加载角色提示词"""
        return self.config.get("system_prompt", "")

    async def update_prompt(self, **kwargs) -> str:
        """更新角色提示词"""
        if "system_prompt" in kwargs:
            self.config["system_prompt"] = kwargs["system_prompt"]
        return self.config["system_prompt"]

    async def think(self, context: Dict[str, Any]) -> List[str]:
        """思考是否需要调用工具"""
        return []  # Zero酱暂时不使用工具

    async def generate_response(self, 
                              input_text: str) -> str:
        """生成回复"""
        try:
            # 构建完整的提示词
            system_prompt = await self.load_prompt()
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": input_text}
            ]
            
            # 使用父类的 generate_response 方法
            return await super().generate_response(input_text)
            
        except Exception as e:
            # self.logger.error(f"Error generating response: {str(e)}") 
            raise

    async def astream_response(self, 
                           input_text: str) -> AsyncIterator[str]:
        """流式生成回复"""
        try:
            # 构建完整的提示词
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
            
            # self._logger.logger.debug(f"ZeroAgent streaming with messages: {messages}")
            
            # 直接使用 LLM 的流式接口
            async for chunk in self.llm.astream(messages):
                # self._logger.logger.debug(f"ZeroAgent chunk: {chunk}")
                yield chunk

                
        except Exception as e:
            self._logger.logger.error(f"ZeroAgent stream error: {str(e)}")
            raise