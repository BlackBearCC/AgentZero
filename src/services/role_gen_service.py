from src.agents.role_generation_agent import RoleGenerationAgent
from src.utils.logger import Logger
from src.llm.doubao import DoubaoLLM
import os
import json
from typing import AsyncIterator

class RoleGenService:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, 'initialized'):
            self.initialized = True
            self.logger = Logger()
            self.gen_agent = self._create_gen_agent()
    
    def _create_gen_agent(self) -> RoleGenerationAgent:
        """创建角色生成Agent"""
        llm = DoubaoLLM(
            model_name=os.getenv("DOUBAO_MODEL_PRO"),
            temperature=0.8,
            max_tokens=4096
        )
        
        config = {
            "name": "角色生成Agent",
            "generation_type": "full_role_config"
        }
        
        return RoleGenerationAgent(config=config, llm=llm)
    
    async def generate_role_config(
        self,
        reference: str,
        user_id: str
    ) -> AsyncIterator[str]:
        """流式生成角色配置"""
        try:
            async for chunk in self.gen_agent.astream_response(reference, user_id):
                yield chunk
        except Exception as e:
            self.logger.error(f"角色生成失败: {str(e)}")
            yield json.dumps({"error": str(e)})

# 依赖注入函数
async def get_role_gen_service() -> RoleGenService:
    return RoleGenService() 