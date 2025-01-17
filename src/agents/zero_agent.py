from typing import Dict, Any, List, Optional
from src.agents.base_agent import BaseAgent

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
                              input_text: str,
                              context: Optional[Dict] = None) -> str:
        """生成回复"""
        try:
            # 构建完整的提示词
            system_prompt = await self.load_prompt()
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": input_text}
            ]
            
            # 使用父类的 generate_response 方法
            return await super().generate_response(input_text, context)
            
        except Exception as e:
            # self.logger.error(f"Error generating response: {str(e)}") 
            raise