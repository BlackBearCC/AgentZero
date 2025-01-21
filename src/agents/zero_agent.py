from typing import Dict, Any, List, Optional, AsyncIterator
from src.agents.base_agent import BaseAgent
from langchain.schema import (
    SystemMessage,
    HumanMessage,
    AIMessage,
    BaseMessage
)

class ZeroAgent(BaseAgent):
    async def load_prompt(self) -> str:
        """加载角色提示词"""
        return self.config.get("system_prompt", "")

    async def think(self, context: Dict[str, Any]) -> List[str]:
        """思考是否需要调用工具"""
        return []  # Zero酱暂时不使用工具

    async def generate_response(self, input_text: str) -> str:
        """生成回复"""
        try:
            # 使用父类的 generate_response 方法
            return  super().generate_response(input_text)
            
        except Exception as e:
            self._logger.error(f"Error generating response: {str(e)}")
            raise

    async def astream_response(self, input_text: str) -> AsyncIterator[str]:
        """流式生成回复"""
        try:
            # 使用父类的 astream_response 方法
            async for chunk in super().astream_response(input_text):
                yield chunk
                
        except Exception as e:
            self._logger.error(f"ZeroAgent stream error: {str(e)}")
            raise