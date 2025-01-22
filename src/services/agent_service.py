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
            self.prompts_dir = Path(__file__).parent.parent / "prompts"
            self._create_default_agents()
    
    def _load_system_prompt(self, prompt_name: str) -> str:
        """从文件加载系统提示词"""
        prompt_path = self.prompts_dir / "system" / f"{prompt_name}.txt"
        try:
            with open(prompt_path, "r", encoding="utf-8") as f:
                return f.read().strip()
        except Exception as e:
            self.logger.error(f"Failed to load prompt {prompt_name}: {str(e)}")
            raise
    
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
                        "user": "琦琦",
                        "scene": "在画廊里",
                        "examples": "示例对话1\n示例对话2"
                    }
                }
            ]
                    # 从环境变量获取 API key
            deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
            if not deepseek_api_key:
                raise ValueError("DEEPSEEK_API_KEY not found in environment variables")
            
            # 使用自定义的 DeepSeekLLM
            memery_llm = DeepSeekLLM(
                model_name="deepseek-chat",
                temperature=0.7,
                api_key=deepseek_api_key
            )            
            for role in default_roles:
                system_prompt = self._load_system_prompt(role["prompt_file"])
                
                # 创建配置对象
                config = {
                    "role_id": role["role_id"],
                    "name": role["name"],
                    "system_prompt": system_prompt,
                    "variables": role.get("variables", {})  # 确保变量被正确传递
                }
                
                agent = ZeroAgent(
                    config=config,  # 直接传递字典，而不是 RoleConfig.dict()
                    llm=llm,
                    memory=memery_llm,
                    tools=None
                )
                self.agents[role["role_id"]] = agent
                
            self.logger.info("Default agents initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize default agents: {str(e)}")
            raise

    async def create_agent(self, config: RoleConfig) -> str:
        """创建新的 Agent 实例"""
        try:
            agent = ZeroAgent(
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