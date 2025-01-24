from typing import Optional, Dict, Any, AsyncIterator
from fastapi import Depends
from src.services.agent_service import AgentService, get_agent_service
from src.utils.logger import Logger
import asyncio

class ChatService:
    def __init__(self, agent_service: AgentService):
        self.agent_service = agent_service
        self.logger = Logger()

    async def process_message(
        self,
        agent_id: str,
        message: str,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """处理聊天消息"""
        try:
            # 获取 agent 实例
            agent = await self.agent_service.get_agent(agent_id)
            if not agent:
                raise ValueError(f"Agent {agent_id} not found")

            # 确保 context 存在
            context = context or {}
            remark = context.get("remark", "")
            
            # 生成回复
            response = await agent.generate_response(message, remark=remark)
            return response

        except Exception as e:
            self.logger.error(f"Error processing message: {str(e)}")
            raise

    async def stream_message(
        self,
        agent_id: str,
        message: str,
        context: Optional[Dict[str, Any]] = None
    ) -> AsyncIterator[str]:
        """流式处理聊天消息"""
        try:
            # 获取 agent 实例
            agent = await self.agent_service.get_agent(agent_id)
            if not agent:
                raise ValueError(f"Agent {agent_id} not found")

            # 确保 context 存在
            context = context or {}
            remark = context.get("remark", "")
            
            # 生成流式回复
            async for chunk in agent.astream_response(message, remark=remark):
                yield chunk

        except Exception as e:
            self.logger.error(f"Error streaming message: {str(e)}")
            raise

# 依赖注入函数
async def get_chat_service(
    agent_service: AgentService = Depends(get_agent_service)
) -> ChatService:
    return ChatService(agent_service) 