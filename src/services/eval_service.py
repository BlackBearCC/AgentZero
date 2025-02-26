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
    
    async def evaluate_data(
        self,
        data: List[Dict[str, Any]],
        eval_type: str,
        user_id: str,
        evaluation_code: Optional[str] = None,
        role_info: Optional[str] = None,
    ) -> AsyncIterator[str]:
        """评估数据"""
        try:
            self.eval_agent.eval_type = eval_type
            
            # 首先发送总数信息和评估信息
            yield f"data: {json.dumps({'total': len(data), 'evaluation_code': evaluation_code or '未命名评估'}, ensure_ascii=False)}\n\n"
            
            # 将人设信息传递给评估Agent
            self.eval_agent.role_info = role_info if role_info and role_info.strip() else None
                
            # 使用evaluate_batch方法进行批量评估
            async for result in self.eval_agent.evaluate_batch(data):
                yield result + "\n\n"
                
        except Exception as e:
            self.logger.error(f"评估失败: {str(e)}")
            yield f"data: {json.dumps({'error': str(e)}, ensure_ascii=False)}\n\n"

# 依赖注入函数
async def get_eval_service() -> EvalService:
    return EvalService() 