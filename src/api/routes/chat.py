from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from src.api.schemas.chat import ChatRequest, ChatResponse
from src.services.chat_service import ChatService, get_chat_service
from typing import Optional, Dict, Any

router = APIRouter()

@router.post("/chat/{agent_id}", response_model=ChatResponse)
async def chat(
    agent_id: str,
    request: ChatRequest,
    chat_service: ChatService = Depends(get_chat_service)
):
    """处理聊天请求"""
    try:
        # 构建 context
        context = {
            "remark": request.remark,
            "config": request.config.dict() if request.config else None
        } if request.remark or request.config else None
        
        response = await chat_service.process_message(
            agent_id=agent_id,
            message=request.message,
            user_id=request.user_id,  # 传递 user_id
            context=context
        )
        return ChatResponse(response=response)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/chat/{agent_id}/stream")
async def stream_chat(
    agent_id: str,
    request: ChatRequest,
    chat_service: ChatService = Depends(get_chat_service)
) -> StreamingResponse:
    """流式聊天接口"""
    try:
        # 获取配置
        config = request.config.dict() if request.config else None
        
        # 调用服务层处理消息
        message_stream = chat_service.stream_message(
            agent_id=agent_id,
            user_id=request.user_id,
            message=request.message,
            remark=request.remark,
            config=config
        )
        
        return StreamingResponse(
            message_stream,
            media_type="text/event-stream"
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))