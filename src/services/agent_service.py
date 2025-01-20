from typing import Dict, Optional
from fastapi import Depends


from src.agents.base_agent import BaseAgent
from src.agents.zero_agent import ZeroAgent
from src.agents.role_config import RoleConfig
from src.agents.templates.agent_templates import AgentTemplates
from src.utils.logger import Logger
from src.llm.deepseek import DeepSeekLLM
from src.llm.doubao import DoubaoLLM
import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

class AgentService:
    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.agents = {}
            cls._instance.logger = Logger()
        return cls._instance

    def __init__(self):
        if not self._initialized:
            self._initialized = True
            # 在这里直接创建默认角色，不使用异步
            self._create_default_agents()
    
    def _create_default_agents(self):
        """初始化默认角色"""
        try:
            # # 从环境变量获取 API key
            # deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
            # if not deepseek_api_key:
            #     raise ValueError("DEEPSEEK_API_KEY not found in environment variables")
            
            # 使用自定义的 DeepSeekLLM
            # llm = DeepSeekLLM(
            #     model_name="deepseek-chat",
            #     temperature=0.7,
            #     api_key=deepseek_api_key
            # )
            llm = DoubaoLLM(
                model_name="ep-20241113173739-b6v4g",
                temperature=0.7,
                max_tokens=4096
            )
            
            # 创建 Zero酱
            zero_config = AgentTemplates.get_zero_agent()
            agent = ZeroAgent(
                config=zero_config.dict(),
                llm=llm,
                memory=None,
                tools=None
            )
            self.agents[zero_config.role_id] = agent
            self.logger.info(f"Default agent {zero_config.name} initialized with DoubaoLLM")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize default agents: {str(e)}")
            raise

    async def create_agent(self, config: RoleConfig) -> str:
        """创建新的 Agent 实例"""
        try:
            agent = BaseAgent(
                config=config.dict(),
                llm=None,
                memory=None,
                tools=None
            )
            self.agents[config.role_id] = agent
            return config.role_id
        except Exception as e:
            self.logger.error(f"Failed to initialize default agents: {str(e)}")
            raise

    async def get_agent(self, agent_id: str) -> Optional[BaseAgent]:
        """获取指定的 Agent 实例"""
        return self.agents.get(agent_id)

    async def delete_agent(self, agent_id: str) -> bool:
        """删除 Agent 实例"""
        if agent_id in self.agents:
            del self.agents[agent_id]
            return True
        return False

# 单例模式的依赖注入函数
async def get_agent_service() -> AgentService:
    return AgentService() 