from typing import Dict, Optional
from fastapi import Depends
from src.agents.base_agent import BaseAgent
from src.agents.zero_agent import ZeroAgent
from src.agents.role_config import RoleConfig
from src.utils.logger import Logger
from src.llm.deepseek import DeepSeekLLM
from src.llm.doubao import DoubaoLLM
import os
from dotenv import load_dotenv
from pathlib import Path
from src.services.db_service import DBService

# 加载环境变量
load_dotenv()

class AgentService:
    _instance: Optional['AgentService'] = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
        
    def __init__(self):
        if not self._initialized:
            self._initialized = True
            self.prompts_dir = Path(__file__).parent.parent / "prompts"
            self.agents: Dict[str, BaseAgent] = {}
            self.logger = Logger()
            self._db_service = None
    
    @property
    async def db_service(self):
        """数据库服务的属性访问器"""
        if not self._db_service:
            self._db_service = await DBService.get_instance()
        return self._db_service
    
    def _load_system_prompt(self, prompt_name: str) -> str:
        """从文件加载系统提示词"""
        prompt_path = self.prompts_dir / "system" / f"{prompt_name}.txt"
        try:
            with open(prompt_path, "r", encoding="utf-8") as f:
                return f.read().strip()
        except Exception as e:
            self.logger.error(f"Failed to load prompt {prompt_name}: {str(e)}")
            raise
    
    @classmethod
    async def get_instance(cls) -> 'AgentService':
        """获取单例实例"""
        if not hasattr(cls, '_instance'):
            cls._instance = cls()
            await cls._instance.init()
        return cls._instance

    async def init(self):
        """初始化服务"""
        try:
            # 确保数据库服务已初始化
            await self.db_service
            # 只有在没有 agents 时才创建默认 agents
            if not self.agents:
                await self._create_default_agents()
        except Exception as e:
            self.logger.error(f"Failed to initialize AgentService: {str(e)}")
            raise
    
    async def _create_default_agents(self):
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
            # 获取数据库服务
            db = await self.db_service
            
            # 初始化 LLM
            llm = DoubaoLLM(
                model_name="ep-20241113173739-b6v4g",
                temperature=0.7,
                max_tokens=4096
            )
            
            # 从环境变量获取 API key
            deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
            if not deepseek_api_key:
                raise ValueError("DEEPSEEK_API_KEY not found in environment variables")
            
            # 使用自定义的 DeepSeekLLM
            memory_llm = DeepSeekLLM(
                model_name="deepseek-chat",
                temperature=0.7,
                api_key=deepseek_api_key
            )
            
            # 初始化所有默认角色
            default_roles = [
                {
                    "role_id": "zero_001",
                    "name": "Zero酱",
                    "prompt_file": "zero"
                },
                {
                    "role_id": "qiyu_001",
                    "name": "祁煜",
                    "prompt_file": "qiyu-20250120",
                    "variables": {
                        "user": "木木",
                        "scene": "",
                    }
                }
            ]
            
            for role in default_roles:
                system_prompt = self._load_system_prompt(role["prompt_file"])
                config = {
                    "role_id": role["role_id"],
                    "name": role["name"],
                    "system_prompt": system_prompt,
                    "variables": role.get("variables", {})
                }
                
                agent = ZeroAgent(
                    config=config,
                    llm=llm,
                    memory_llm=memory_llm,
                    tools=None
                )
                # 确保 agent 的数据库服务已初始化
                await agent._ensure_db()
                self.agents[role["role_id"]] = agent
                
            self.logger.info("Default agents initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize default agents: {str(e)}")
            raise

    async def create_agent(self, config: RoleConfig) -> str:
        """创建新的 Agent 实例"""
        try:
            # 使用与默认 agents 相同的 LLM 配置
            llm = DoubaoLLM(
                model_name=os.getenv("DOUBAO_MODEL_PRO"),
                temperature=0.7,
                max_tokens=4096
            )
            
            # 获取 DeepSeek API key
            deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
            if not deepseek_api_key:
                raise ValueError("DEEPSEEK_API_KEY not found")
                
            memory_llm = DeepSeekLLM(
                model_name="deepseek-chat",
                temperature=0.7,
                api_key=deepseek_api_key
            )
            
            agent = ZeroAgent(
                config=config.dict(),
                db_service=self.db_service,
                llm=llm,
                memory_llm=memory_llm,
                tools=None
            )
            self.agents[config.role_id] = agent
            return config.role_id
            
        except Exception as e:
            self.logger.error(f"Failed to create agent: {str(e)}")
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
    instance = AgentService()
    await instance.init()
    return instance 