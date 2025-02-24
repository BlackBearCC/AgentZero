from string import Template
from src.agents.base_agent import BaseAgent
from typing import AsyncIterator, Dict, Any, List, Optional
import json

class EvaluationAgent(BaseAgent):
    def __init__(self, config: Dict[str, Any], llm=None):
        super().__init__(config, llm)
        self.eval_type = config.get("eval_type", "dialogue")
        self._load_eval_prompt()

    def _load_eval_prompt(self):
        """加载评估提示词模板"""
        self.eval_prompt = Template("""
        你是一个专业的AI对话质量评估专家，请根据以下评估标准对对话进行分析：
        
        # 评估标准
        $criteria
        
        # 待评估数据
        $eval_data
        
        请按照以下JSON格式输出评估结果：
        {
            "score": 0-100的评分,
            "reason": "详细的评估理由",
            "suggestions": ["改进建议1", "改进建议2"]
        }
        """)

    # 实现BaseAgent的抽象方法
    async def load_prompt(self) -> None:
        """加载提示词"""
        self._load_eval_prompt()

    async def update_prompt(self, prompt: str) -> None:
        """更新提示词"""
        pass  # 评估Agent不需要更新提示词

    async def think(self, input_text: str) -> str:
        """思考处理"""
        return input_text  # 评估Agent不需要思考处理

    async def generate_response(self, input_text: str, user_id: str, remark: str = '') -> str:
        """生成回复"""
        try:
            data = json.loads(input_text)
            result = await self._evaluate_single(data)
            return json.dumps(result, ensure_ascii=False)
        except Exception as e:
            return json.dumps({"error": str(e)}, ensure_ascii=False)

    async def astream_response(self, input_text: str, user_id: str, remark: str = '', context: Dict[str, Any] = None) -> AsyncIterator[str]:
        """流式生成回复"""
        try:
            data = json.loads(input_text)
            result = await self._evaluate_single(data)
            yield json.dumps(result, ensure_ascii=False)
        except Exception as e:
            yield json.dumps({"error": str(e)}, ensure_ascii=False)

    async def update_eval_criteria(self, criteria: str):
        """更新评估标准"""
        self.config["criteria"] = criteria
        self._load_eval_prompt()

    async def evaluate_batch(self, batch_data: List[Dict]) -> List[Dict]:
        """批量评估数据"""
        evaluations = []
        for data in batch_data:
            evaluation = await self._evaluate_single(data)
            evaluations.append({
                **data,
                "evaluation": evaluation
            })
        return evaluations

    async def _evaluate_single(self, data: Dict) -> Dict:
        """单条数据评估"""
        filled_prompt = self.eval_prompt.safe_substitute(
            criteria=self.config.get("criteria", ""),
            eval_data=json.dumps(data, ensure_ascii=False)
        )
        
        context = await self._build_eval_context(filled_prompt)
        response = await self.llm.agenerate(context["messages"])
        
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            return {"error": "评估结果解析失败"}

    async def _build_eval_context(self, prompt: str) -> Dict:
        """构建评估上下文"""
        return {
            "messages": [{
                "role": "system",
                "content": "你是一个严谨的质量评估专家，需要客观分析数据并给出改进建议"
            }, {
                "role": "user",
                "content": prompt
            }]
        }

    async def astream_evaluate(self, data: List[Dict]) -> AsyncIterator[Dict]:
        """流式评估"""
        for batch in self._chunk_data(data, batch_size=5):
            results = await self.evaluate_batch(batch)
            for result in results:
                yield result

    def _chunk_data(self, data: List[Dict], batch_size: int):
        """数据分块"""
        for i in range(0, len(data), batch_size):
            yield data[i:i + batch_size] 