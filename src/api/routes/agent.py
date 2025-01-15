from fastapi import APIRouter, Depends, HTTPException
from src.agents.role_config import RoleConfig
from src.services.agent_service import AgentService, get_agent_service
from src.api.schemas.agent import AgentCreate, AgentResponse, AgentInfo
from typing import List

router = APIRouter()

@router.post("/agents/create", response_model=AgentResponse)
async def create_agent(
    config: AgentCreate,
    agent_service: AgentService = Depends(get_agent_service)
):
    """
    创建新的 Agent
    
    Args:
        config (AgentCreate): Agent 配置信息
        agent_service (AgentService): Agent 服务实例
        
    Returns:
        AgentResponse: 包含 agent_id 和状态的响应
        
    Raises:
        HTTPException: 当创建失败时抛出 400 错误
    """
    try:
        agent_id = await agent_service.create_agent(config.config)
        return AgentResponse(agent_id=agent_id, status="created")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.delete("/agents/{agent_id}", response_model=AgentResponse)
async def delete_agent(
    agent_id: str,
    agent_service: AgentService = Depends(get_agent_service)
):
    """
    删除指定的 Agent
    
    Args:
        agent_id (str): 要删除的 Agent ID
        agent_service (AgentService): Agent 服务实例
        
    Returns:
        AgentResponse: 包含 agent_id 和删除状态的响应
        
    Raises:
        HTTPException: 当 Agent 不存在时抛出 404 错误
    """
    success = await agent_service.delete_agent(agent_id)
    if not success:
        raise HTTPException(status_code=404, detail="Agent not found")
    return AgentResponse(agent_id=agent_id, status="deleted")

@router.get("/agents/{agent_id}", response_model=AgentInfo)
async def get_agent(
    agent_id: str,
    agent_service: AgentService = Depends(get_agent_service)
):
    """
    获取 Agent 信息
    
    Args:
        agent_id (str): 要获取的 Agent ID
        agent_service (AgentService): Agent 服务实例
        
    Returns:
        AgentInfo: 包含 agent_id 和配置信息的响应
        
    Raises:
        HTTPException: 当 Agent 不存在时抛出 404 错误
    """
    agent = await agent_service.get_agent(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    return AgentInfo(agent_id=agent_id, config=agent.config)

@router.get("/agents", response_model=List[str])
async def list_agents(
    agent_service: AgentService = Depends(get_agent_service)
):
    """列出所有 Agent"""
    return list(agent_service.agents.keys())