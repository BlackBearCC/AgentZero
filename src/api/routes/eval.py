from fastapi import APIRouter, UploadFile, File, HTTPException
from src.agents.eval_agent import EvaluationAgent
from src.utils.file_utils import read_data_file
from typing import List, Dict, Any

router = APIRouter()

eval_agent = EvaluationAgent({
    "name": "质量评估Agent",
    "eval_type": "dialogue",
    "criteria": "基础对话质量评估标准"
})

@router.post("/evaluate")
async def evaluate_data(
    file: UploadFile = File(...),
    eval_type: str = "dialogue"
):
    try:
        # 读取上传文件
        data = await read_data_file(file)
        
        # 配置评估类型
        eval_agent.eval_type = eval_type
        if eval_type == "memory":
            await eval_agent.update_eval_criteria("记忆相关性评估标准")
        
        # 执行评估
        results = []
        async for result in eval_agent.astream_evaluate(data):
            results.append(result)
        
        return {"status": "completed", "results": results}
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 