from fastapi import APIRouter, Depends, HTTPException
from src.api.schemas.chat import ChatRequest, ChatResponse
from src.services.chat_service import ChatService, get_chat_service

router = APIRouter()

@router.post("/chat/{agent_id}", response_model=ChatResponse)
async def chat(
    agent_id: str,
    request: ChatRequest,
    chat_service: ChatService = Depends(get_chat_service)
):
    """处理聊天请求"""
    try:
        response = await chat_service.process_message(
            agent_id=agent_id,
            message=request.message,
            context=request.context
        )
        return ChatResponse(response=response)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))