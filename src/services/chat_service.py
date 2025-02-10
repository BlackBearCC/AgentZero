from typing import Optional, Dict, Any, AsyncIterator
from fastapi import Depends
from src.services.agent_service import AgentService, get_agent_service
from src.utils.logger import Logger
from src.api.schemas.chat import AgentConfig
import asyncio
import json

class ChatService:
    def __init__(self, agent_service: AgentService):
        self.agent_service = agent_service
        self.logger = Logger()

    async def _apply_temp_config(self, agent: Any, config: Optional[AgentConfig]) -> Dict[str, Any]:
        """临时应用配置并返回原始配置"""
        if not config:
            return {}
            
        # 保存原始配置
        original_config = {
            "use_memory_queue": agent.use_memory_queue,
            "use_combined_query": agent.use_combined_query,
            "memory_queue_limit": agent.memory_queue_limit,
        }
        
        # 临时应用新配置
        if hasattr(config, "use_memory_queue"):
            agent.use_memory_queue = config.use_memory_queue
        if hasattr(config, "use_combined_query"):
            agent.use_combined_query = config.use_combined_query
        if hasattr(config, "memory_queue_limit"):
            agent.memory_queue_limit = config.memory_queue_limit
            
        return original_config

    async def _restore_config(self, agent: Any, original_config: Dict[str, Any]):
        """恢复原始配置"""
        for key, value in original_config.items():
            if hasattr(agent, key):
                setattr(agent, key, value)

    async def process_message(
        self,
        agent_id: str,
        message: str,
        user_id: str,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """处理聊天消息"""
        try:
            # 获取 agent 实例
            config = context.get("config") if context else None
            agent = await self.agent_service.get_agent(
                agent_id=agent_id,
                user_id=user_id,
                config=config
            )
            if not agent:
                raise ValueError(f"Agent {agent_id} not found")

            # 确保 context 存在
            context = context or {}
            remark = context.get("remark", "")
            
            # 生成回复
            response = await agent.generate_response(
                message, 
                user_id=user_id,
                remark=remark
            )
            return response

        except Exception as e:
            self.logger.error(f"Error processing message: {str(e)}")
            raise

    async def stream_message(
        self,
        agent_id: str,
        user_id: str,
        message: str,
        remark: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ) -> AsyncIterator[str]:
        """流式处理聊天消息"""
        try:
            agent = await self.agent_service.get_agent(
                agent_id=agent_id,
                user_id=user_id,
                config=config
            )
            if not agent:
                raise ValueError(f"Agent {agent_id} not found")
            
            metadata_sent = False
            async for chunk in agent.astream_response(
                input_text=message,
                user_id=user_id,
                remark=remark or "",
                config=config
            ):
                if isinstance(chunk, dict):
                    # 元数据使用data字段
                    yield f"event: metadata\ndata: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                else:
                    # 文本内容使用data字段
                    yield f"data: {chunk}\n\n"

        except Exception as e:
            self.logger.error(f"Error streaming message: {str(e)}")
            yield f"event: error\ndata: {str(e)}\n\n"

# 依赖注入函数
async def get_chat_service(
    agent_service: AgentService = Depends(get_agent_service)
) -> ChatService:
    return ChatService(agent_service) 