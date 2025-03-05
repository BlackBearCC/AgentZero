from typing import List, Optional
from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import StreamingResponse
from src.services.role_gen_service import RoleGenService, get_role_gen_service
from pydantic import BaseModel
import json

router = APIRouter()

class RoleGenRequest(BaseModel):
    reference: str
    user_id: str = "web"
    categories: Optional[List[str]] = None

@router.post("/generate_role_config/stream")
async def stream_generate_role_config(
    request: RoleGenRequest,
    role_gen_service: RoleGenService = Depends(get_role_gen_service)
) -> StreamingResponse:
    """流式生成角色配置"""
    try:
        message_stream = role_gen_service.generate_role_config(request.reference, request.user_id,request.categories)
        return StreamingResponse(
            message_stream,
            media_type="text/event-stream"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 