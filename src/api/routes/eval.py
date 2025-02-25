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
    selected_fields: str = Form(...),
    eval_service: EvalService = Depends(get_eval_service)
) -> StreamingResponse:
    """流式评估接口"""
    try:
        file_data = await read_data_file(file)
        selected_fields_list = json.loads(selected_fields)
        
        # 只保留选中的字段
        filtered_data = []
        for item in file_data["data"]:
            filtered_item = {k: v for k, v in item.items() if k in selected_fields_list}
            filtered_data.append(filtered_item)

        message_stream = eval_service.evaluate_data(
            data=filtered_data,
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

@router.post("/file/columns")
async def get_file_columns(
    file: UploadFile = File(...)
) -> Dict[str, List[str]]:
    """获取文件列名"""
    try:
        file_data = await read_data_file(file)
        return {"columns": file_data["columns"]}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) 