from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, Form
from src.services.eval_service import EvalService, get_eval_service
from src.utils.file_utils import read_data_file
from typing import List, Dict, Any
from fastapi.responses import StreamingResponse
import json
from src.api.schemas.chat import EvalRequest

router = APIRouter()

@router.post("/evaluate/stream")
async def stream_evaluate(
    file: UploadFile = File(...),
    eval_type: str = Form(...),
    user_id: str = Form(...),
    eval_service: EvalService = Depends(get_eval_service)
) -> StreamingResponse:
    """流式评估接口"""
    try:
        data = await read_data_file(file)
        

        message_stream = eval_service.evaluate_data(
            data=data,
            eval_type=eval_type,
            user_id=user_id
        )


        return StreamingResponse(
            message_stream,
            media_type="text/event-stream"
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 