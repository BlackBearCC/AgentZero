from typing import Dict, Optional
from fastapi import Depends

from src.agents.base_agent import BaseAgent
from src.agents.zero_agent import ZeroAgent
from src.agents.role_config import RoleConfig
from src.agents.templates.agent_templates import AgentTemplates
from src.utils.logger import Logger
from langchain.chat_models import ChatOpenAI
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
            # 创建 LLM 实例
            llm = ChatOpenAI(
                temperature=0.7,
                model_name="gpt-3.5-turbo",
                # 如果环境变量中没有，可以直接设置
                # openai_api_key="你的API_KEY",
                # openai_api_base="你的代理地址"  # 可选
            )
            
            # 创建 Zero酱
            zero_config = AgentTemplates.get_zero_agent()
            self.logger.info(f"Creating default agent with config: {zero_config}")
            
            agent = ZeroAgent(
                config=zero_config.dict(),
                llm=llm,
                memory=None,
                tools=None
            )
            self.agents[zero_config.role_id] = agent
            self.logger.logger.info(f"Default agent {zero_config.name} initialized. Current agents: {list(self.agents.keys())}")
        except Exception as e:
            self.logger.error(f"Failed to initialize default agents: {str(e)}")
            raise  # 让错误显示在日志中

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