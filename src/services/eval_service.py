from typing import Dict, Any, AsyncIterator, List, Optional

from src.agents.eval_agent import EvaluationAgent
from src.utils.logger import Logger
from src.llm.doubao import DoubaoLLM
import os
import json

class EvalService:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, 'initialized'):
            self.initialized = True
            self.logger = Logger()
            self.eval_agent = self._create_eval_agent()
    
    def _create_eval_agent(self) -> EvaluationAgent:
        """创建评估Agent"""
        llm = DoubaoLLM(
            model_name=os.getenv("DOUBAO_MODEL_PRO"),
            temperature=0.7,
            max_tokens=2048
        )
        
        config = {
            "name": "质量评估Agent",
            "eval_type": "dialogue",
            "criteria": "基础对话质量评估标准"
        }
        
        return EvaluationAgent(config=config, llm=llm)

    def _build_messages(self, item: Dict[str, Any], eval_type: str) -> List[Dict[str, str]]:
        """构建评估消息"""
        system_prompt = "你是一个专业的对话质量评估专家。请根据以下标准评估对话质量："
        if eval_type == "memory":
            system_prompt = "你是一个专业的记忆相关性评估专家。请根据以下标准评估记忆相关性："
            
        return [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": f"请评估以下对话:\n{json.dumps(item, ensure_ascii=False, indent=2)}"
            }
        ]
    
    async def evaluate_data(
        self,
        data: List[Dict[str, Any]],
        eval_type: str,
        user_id: str,
    ) -> AsyncIterator[str]:
        """评估数据"""
        try:
            self.eval_agent.eval_type = eval_type
            
            async for evaluation in self.eval_agent.evaluate_batch(data):
                yield evaluation
                
        except Exception as e:
            self.logger.error(f"评估失败: {str(e)}")
            yield f"data: {json.dumps({'error': str(e)}, ensure_ascii=False)}\n\n"

# 依赖注入函数
async def get_eval_service() -> EvalService:
    return EvalService() 