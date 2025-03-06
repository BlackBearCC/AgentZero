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

class AttributeGenerateRequest(BaseModel):
    category: str
    existingAttributes: List[dict]
    reference: str
    user_id: str = "web"
class ContentOptimizeRequest(BaseModel):
    category: str
    content: str
    reference: str
    user_id: str = "web"

class KeywordsOptimizeRequest(BaseModel):
    category: str
    content: str
    keywords: List[str]
    reference: str
    user_id: str = "web"

@router.post("/optimize_content")
async def optimize_content(
    request: ContentOptimizeRequest,
    role_gen_service: RoleGenService = Depends(get_role_gen_service)
) -> dict:
    """优化属性内容"""
    print(request.dict())
    try:
        result = await role_gen_service.optimize_content(
            category=request.category,
            content=request.content,
            reference=request.reference,
            user_id=request.user_id
        )
        print(result)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/optimize_keywords")
async def optimize_keywords(
    request: KeywordsOptimizeRequest,
    role_gen_service: RoleGenService = Depends(get_role_gen_service)
) -> dict:
    """优化属性关键词"""
    print(request.dict())
    try:
        result = await role_gen_service.optimize_keywords(
            category=request.category,
            content=request.content,
            keywords=request.keywords,
            reference=request.reference,
            user_id=request.user_id
        )
        print(result)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/generate_new_attribute")
async def generate_new_attribute(
    request: AttributeGenerateRequest,
    role_gen_service: RoleGenService = Depends(get_role_gen_service)
) -> dict:
    """生成新属性"""
    print(request.dict())
    try:
        result = await role_gen_service.generate_new_attribute(
            category=request.category,
            existing_attributes=request.existingAttributes,
            reference=request.reference,
            user_id=request.user_id
        )
        print(result)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
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