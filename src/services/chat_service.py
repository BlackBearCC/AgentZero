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
            
            # 如果是字典类型且包含 pre_tool_message，直接返回
            if isinstance(response, dict) and "pre_tool_message" in response:
                return response
                
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

    async def process_telegram_message(
        self,
        agent_id: str,
        message: str,
        context: Optional[Dict[str, Any]] = None
    ) -> AsyncIterator[Dict[str, Any]]:
        """处理 Telegram 消息，支持阶段性返回
        
        Args:
            agent_id: Agent ID
            message: 用户消息
            context: 上下文信息
            
        Yields:
            Dict[str, Any]: 包含阶段信息的响应字典
        """
        try:
            # 获取 agent 实例
            agent = await self.agent_service.get_agent(agent_id)
            if not agent:
                raise ValueError(f"Agent {agent_id} not found")

            # 确保 context 存在
            context = context or {}
            remark = context.get("remark", "")
            
            # 使用新的阶段性生成方法
            async for response in agent.generate_response(message, remark=remark):
                yield response

        except Exception as e:
            self.logger.error(f"Error processing telegram message: {str(e)}")
            yield {
                "stage": "error",
                "error": str(e)
            }

# 依赖注入函数
async def get_chat_service(
    agent_service: AgentService = Depends(get_agent_service)
) -> ChatService:
    return ChatService(agent_service) 