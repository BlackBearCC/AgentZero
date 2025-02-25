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
        你是一个专业的AI对话质量评估专家，请根据以下评估标准对对话进行深入分析：
        
        # 评估标准
        $criteria
        
        # 评估维度
        ## 1. 角色扮演评估（占比50%）
        - 角色一致性：AI是否始终保持角色设定的一致性
        - 角色知识：AI是否展现了与角色相符的知识和背景
        - 语言风格：AI的语言风格是否符合角色特征
        - 情感表达：AI是否能适当展现角色的情感反应
        - 角色深度：AI是否能展现角色的深度和复杂性
        
        ## 2. 对话体验评估（占比50%）
        - 回应质量：AI回应是否准确、有用、相关
        - 交互流畅度：对话是否自然流畅，没有明显断层
        - 语言表达：语言是否清晰、连贯、易于理解
        - 情境适应性：是否能适应不同的对话情境和话题转换
        - 个性化体验：是否能记住用户信息并提供个性化体验
        
        # 待评估数据
        $eval_data
        
        # 思维链分析
        请首先逐步分析对话内容，指出关键点，然后对每个评估维度进行打分并给出理由。
        
        请按照以下JSON格式输出最终评估结果：
        {
            "role_play": {
                "consistency": {"score": 0-100, "comment": "评价理由"},
                "knowledge": {"score": 0-100, "comment": "评价理由"},
                "language_style": {"score": 0-100, "comment": "评价理由"},
                "emotional_expression": {"score": 0-100, "comment": "评价理由"},
                "character_depth": {"score": 0-100, "comment": "评价理由"},
                "role_score": 0-100
            },
            "dialogue_experience": {
                "response_quality": {"score": 0-100, "comment": "评价理由"},
                "interaction_fluency": {"score": 0-100, "comment": "评价理由"},
                "language_expression": {"score": 0-100, "comment": "评价理由"},
                "context_adaptation": {"score": 0-100, "comment": "评价理由"},
                "personalization": {"score": 0-100, "comment": "评价理由"},
                "dialogue_score": 0-100
            },
            "final_score": 0-100,
            "strengths": ["优势1", "优势2", "..."],
            "weaknesses": ["弱点1", "弱点2", "..."],
            "suggestions": ["改进建议1", "改进建议2", "..."]
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

    async def generate_response(self, input_text: str, user_id: str, remark: str = '', config: Optional[Dict[str, Any]] = None) -> str:
        """生成回复"""
        try:
            data = json.loads(input_text)
            result = await self._evaluate_single(data)
            return json.dumps(result, ensure_ascii=False)
        except Exception as e:
            return json.dumps({"error": str(e)}, ensure_ascii=False)

    async def astream_response(self, input_text: str, user_id: str, remark: str = '', config: Optional[Dict[str, Any]] = None, context: Dict[str, Any] = None) -> AsyncIterator[str]:
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

    async def evaluate_batch(self, batch_data: List[Dict]) -> AsyncIterator[str]:
        """批量评估数据"""
        # 用于保存所有评估结果的统计数据
        all_evaluations = []
        
        # 首先发送总数信息
        yield f"data: {json.dumps({'total': len(batch_data)}, ensure_ascii=False)}\n\n"
        
        for idx, data in enumerate(batch_data):
            try:
                # 构建提示词
                prompt = self.eval_prompt.safe_substitute(
                    criteria=self.config.get("criteria", ""),
                    eval_data=json.dumps(data, ensure_ascii=False, indent=2)
                )
                
                context = await self._build_eval_context(prompt)
                original_data = json.dumps(data, ensure_ascii=False, indent=2)
                # 发送评估项开始标记
                yield f"data: {json.dumps({'index': idx + 1, 'type': 'start', 'original_data': original_data}, ensure_ascii=False)}\n\n"
                
                # 收集完整响应
                full_response = ""
                # 流式返回每个chunk
                async for chunk in self.llm.astream(context["messages"]):
                    full_response += chunk
                    yield f"data: {json.dumps({'index': idx + 1, 'type': 'chunk', 'content': chunk}, ensure_ascii=False)}\n\n"
                
                # 尝试解析JSON响应
                try:
                    evaluation_result = json.loads(full_response)
                    all_evaluations.append(evaluation_result)
                except json.JSONDecodeError:
                    # 如果无法解析JSON，仍然保留原始响应
                    all_evaluations.append({"raw_response": full_response})
                
                # 发送评估项结束标记
                yield f"data: {json.dumps({'index': idx + 1, 'type': 'end'}, ensure_ascii=False)}\n\n"
                
            except Exception as e:
                yield f"data: {json.dumps({'index': idx + 1, 'type': 'error', 'error': str(e)}, ensure_ascii=False)}\n\n"
        
        # 生成统计报告
        stats = await self._generate_evaluation_stats(all_evaluations)
        
        # 发送完成标记和统计数据
        yield f"data: {json.dumps({'type': 'complete', 'stats': stats}, ensure_ascii=False)}\n\n"
        self._logger.info(f"评估完成，统计数据: {stats}")

    async def _generate_evaluation_stats(self, evaluations: List[Dict]) -> Dict:
        """生成评估统计数据"""
        valid_evaluations = [e for e in evaluations if isinstance(e, dict) and 'role_play' in e and 'dialogue_experience' in e]
        
        if not valid_evaluations:
            return {"error": "无有效评估数据"}
        
        # 初始化统计数据结构
        stats = {
            "overall_scores": {
                "role_score": 0,
                "dialogue_score": 0,
                "final_score": 0
            },
            "role_play": {
                "consistency": {"scores": [], "avg": 0},
                "knowledge": {"scores": [], "avg": 0},
                "language_style": {"scores": [], "avg": 0},
                "emotional_expression": {"scores": [], "avg": 0},
                "character_depth": {"scores": [], "avg": 0}
            },
            "dialogue_experience": {
                "response_quality": {"scores": [], "avg": 0},
                "interaction_fluency": {"scores": [], "avg": 0},
                "language_expression": {"scores": [], "avg": 0},
                "context_adaptation": {"scores": [], "avg": 0},
                "personalization": {"scores": [], "avg": 0}
            },
            "common_strengths": {},
            "common_weaknesses": {},
            "common_suggestions": {}
        }
        
        # 收集所有评分
        for eval_data in valid_evaluations:
            # 角色扮演维度
            role_play = eval_data.get("role_play", {})
            for key in stats["role_play"]:
                if key in role_play and "score" in role_play[key]:
                    stats["role_play"][key]["scores"].append(role_play[key]["score"])
            
            # 对话体验维度
            dialogue_exp = eval_data.get("dialogue_experience", {})
            for key in stats["dialogue_experience"]:
                if key in dialogue_exp and "score" in dialogue_exp[key]:
                    stats["dialogue_experience"][key]["scores"].append(dialogue_exp[key]["score"])
            
            # 总体评分
            if "role_score" in role_play:
                stats["overall_scores"]["role_score"] += role_play["role_score"]
            if "dialogue_score" in dialogue_exp:
                stats["overall_scores"]["dialogue_score"] += dialogue_exp["dialogue_score"]
            if "final_score" in eval_data:
                stats["overall_scores"]["final_score"] += eval_data["final_score"]
            
            # 收集优势、弱点和建议
            for strength in eval_data.get("strengths", []):
                stats["common_strengths"][strength] = stats["common_strengths"].get(strength, 0) + 1
            for weakness in eval_data.get("weaknesses", []):
                stats["common_weaknesses"][weakness] = stats["common_weaknesses"].get(weakness, 0) + 1
            for suggestion in eval_data.get("suggestions", []):
                stats["common_suggestions"][suggestion] = stats["common_suggestions"].get(suggestion, 0) + 1
        
        # 计算平均分
        num_evals = len(valid_evaluations)
        stats["overall_scores"]["role_score"] = round(stats["overall_scores"]["role_score"] / num_evals, 2)
        stats["overall_scores"]["dialogue_score"] = round(stats["overall_scores"]["dialogue_score"] / num_evals, 2)
        stats["overall_scores"]["final_score"] = round(stats["overall_scores"]["final_score"] / num_evals, 2)
        
        # 计算各维度平均分
        for category in ["role_play", "dialogue_experience"]:
            for key in stats[category]:
                scores = stats[category][key]["scores"]
                stats[category][key]["avg"] = round(sum(scores) / len(scores), 2) if scores else 0
                stats[category][key]["min"] = min(scores) if scores else 0
                stats[category][key]["max"] = max(scores) if scores else 0
        
        # 排序常见的优势、弱点和建议（取前5个）
        stats["common_strengths"] = dict(sorted(stats["common_strengths"].items(), key=lambda x: x[1], reverse=True)[:5])
        stats["common_weaknesses"] = dict(sorted(stats["common_weaknesses"].items(), key=lambda x: x[1], reverse=True)[:5])
        stats["common_suggestions"] = dict(sorted(stats["common_suggestions"].items(), key=lambda x: x[1], reverse=True)[:5])
        
        return stats

    async def _evaluate_single(self, data: Dict) -> AsyncIterator[str]:
        """单条数据评估"""
        # 构建提示词
        prompt = self.eval_prompt.safe_substitute(
            criteria=self.config.get("criteria", ""),
            eval_data=json.dumps(data, ensure_ascii=False, indent=2)
        )
        
        try:
            context = await self._build_eval_context(prompt)
            response = ""
            async for chunk in self.llm.astream(context["messages"]):
                response += chunk
            # print(response)
                yield f"data: {json.dumps({'result': response}, ensure_ascii=False)}\n\n"
            
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)}, ensure_ascii=False)}\n\n"


    async def _build_eval_context(self, prompt: str) -> Dict[str, Any]:
        """构建评估上下文"""
        return {
            "messages": [
                {
                    "role": "system",
                    "content": "你是一个专业的AI对话质量评估专家。"
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        } 