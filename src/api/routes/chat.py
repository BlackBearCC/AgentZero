from fastapi import APIRouter, Depends
from typing import Dict

router = APIRouter()

@router.post("/chat/{agent_id}")
async def chat(
    agent_id: str,
    message: Dict[str, str],
    # 依赖注入
):
    """处理聊天请求"""
    pass 