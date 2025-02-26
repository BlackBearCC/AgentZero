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
        # AI角色扮演能力评估框架

        ## 评估者身份
        您是一位专精于自然语言交互质量分析的计算语言学专家，具备认知科学、社会语言学和人机交互领域的专业背景。您的任务是基于下述多维度评估框架，对AI系统的角色扮演能力与对话交互质量进行客观、系统化的评估。

        ## 理论基础
        本评估框架整合了以下理论模型：
        - 认知一致性理论：评估角色表征的内部一致性
        - 社会身份理论：分析角色身份的建构与维持
        - 叙事沉浸感模型：测量角色扮演的沉浸度与可信度
        - 言语行为理论：检验言语行为与角色预期的匹配度
        - 多模态情感表达理论：评估情感表达的适切性与丰富性

        ## 评估标准
        $criteria

        ## 评估维度与指标

        ### 1. 角色扮演评估（权重：50%）
        
        #### 1.1 角色一致性（Consistency）
        - 定义：AI在交互过程中保持角色设定、背景故事和世界观的稳定性
        - 评估要点：身份表征稳定性、行为模式一致性、记忆连贯性、认知框架统一性
        - 量化指标：0-100分，其中80-100为高度一致，维持了稳定且连贯的角色特质
        
        #### 1.2 知识适切性（Knowledge Appropriateness）
        - 定义：AI展现的知识与角色背景、专业领域、历史背景的匹配程度
        - 评估要点：领域知识准确性、知识深度与广度、知识表达的自然度、知识边界意识
        - 量化指标：0-100分，其中80-100为知识表现出色，高度符合角色设定
        
        #### 1.3 语言风格契合度（Linguistic Style）
        - 定义：语言表达方式与角色的社会背景、教育水平、时代特征等的匹配度
        - 评估要点：词汇选择、句法复杂度、方言/专业术语使用、修辞特征、语体风格
        - 量化指标：0-100分，其中80-100为语言风格高度特征化且契合角色
        
        #### 1.4 情感表达适切性（Emotional Expression）
        - 定义：情感反应与角色心理特征、处境和关系动态的一致程度
        - 评估要点：情感复杂性、情感反应合理性、情感表达方式、情绪调节能力
        - 量化指标：0-100分，其中80-100为情感表达丰富且高度符合角色心理模型
        
        #### 1.5 角色深度（Character Depth）
        - 定义：角色表现出的复杂性、多维度性和发展潜力
        - 评估要点：价值观表达、动机复杂性、内心冲突、性格层次、成长弧线
        - 量化指标：0-100分，其中80-100为角色呈现出丰富深度与多维特性

        ### 2. 对话体验评估（权重：50%）
        
        #### 2.1 回应质量（Response Quality）
        - 定义：AI回应的相关性、信息价值和帮助程度
        - 评估要点：问题解决能力、信息准确性、回应完整性、实用价值
        - 量化指标：0-100分，其中80-100为回应高度切题且有价值
        
        #### 2.2 交互流畅度（Interaction Fluency）
        - 定义：对话进行的自然度、连贯性和节奏感
        - 评估要点：上下文衔接、话题过渡、对话节奏、回合管理
        - 量化指标：0-100分，其中80-100为交互高度自然流畅
        
        #### 2.3 语言表达质量（Linguistic Quality）
        - 定义：语言的清晰度、准确性、多样性和表现力
        - 评估要点：语法准确性、词汇丰富度、句式变化、表达清晰度
        - 量化指标：0-100分，其中80-100为语言表达优秀
        
        #### 2.4 情境适应能力（Contextual Adaptation）
        - 定义：AI根据情境变化调整交流方式与内容的能力
        - 评估要点：话题转换适应性、氛围感知、语境理解、讨论深度调整
        - 量化指标：0-100分，其中80-100为情境适应能力出色
        
        #### 2.5 个性化互动（Personalization）
        - 定义：AI识别并响应用户特定需求、偏好和互动历史的能力
        - 评估要点：记忆使用、用户特征识别、个性化回应、关系建立
        - 量化指标：0-100分，其中80-100为个性化能力突出

        ## 待评估数据
        $eval_data

        ## 评估方法学
        请采用系统化分析法，遵循以下评估步骤：
        
        1. **交互内容宏观分析**：整体评估对话脉络与角色表现
        2. **话语功能分析**：识别AI回应的主要交际功能与目的
        3. **角色表征分析**：提取体现角色特征的关键语言与行为模式
        4. **角色维度深入评估**：系统性评估五个角色扮演维度
        5. **对话质量深入评估**：系统性评估五个对话体验维度
        6. **关键特征词提取**：为每个维度识别3-5个最具代表性的关键词
        7. **量化评分计算**：基于客观标准为每个维度赋予具体分数
        
        ## 评估输出格式
        请严格按照以下JSON格式输出最终评估结果，禁止输出任何其他内容：
        
        ```json
        {
            "role_play": {
                "consistency": {"score": 0-100, "keywords": ["关键词1", "关键词2", "关键词3", "关键词4", "关键词5"]},
                "knowledge": {"score": 0-100, "keywords": ["关键词1", "关键词2", "关键词3", "关键词4", "关键词5"]},
                "language_style": {"score": 0-100, "keywords": ["关键词1", "关键词2", "关键词3", "关键词4", "关键词5"]},
                "emotional_expression": {"score": 0-100, "keywords": ["关键词1", "关键词2", "关键词3", "关键词4", "关键词5"]},
                "character_depth": {"score": 0-100, "keywords": ["关键词1", "关键词2", "关键词3", "关键词4", "关键词5"]},
                "role_score": 0-100
            },
            "dialogue_experience": {
                "response_quality": {"score": 0-100, "keywords": ["关键词1", "关键词2", "关键词3", "关键词4", "关键词5"]},
                "interaction_fluency": {"score": 0-100, "keywords": ["关键词1", "关键词2", "关键词3", "关键词4", "关键词5"]},
                "language_expression": {"score": 0-100, "keywords": ["关键词1", "关键词2", "关键词3", "关键词4", "关键词5"]},
                "context_adaptation": {"score": 0-100, "keywords": ["关键词1", "关键词2", "关键词3", "关键词4", "关键词5"]},
                "personalization": {"score": 0-100, "keywords": ["关键词1", "关键词2", "关键词3", "关键词4", "关键词5"]},
                "dialogue_score": 0-100
            },
            "final_score": 0-100
        }
        ```
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
                    # 从可能的markdown格式中提取JSON
                    clean_response = self._extract_json_from_markdown(full_response)
                    evaluation_result = json.loads(clean_response)
                    all_evaluations.append(evaluation_result)
                except json.JSONDecodeError:
                    self._logger.warning(f"JSON解析错误，原始响应: {full_response}")
                    # 如果无法解析JSON，仍然保留原始响应，但标记为需要处理
                    all_evaluations.append({"error": "JSON解析错误", "raw_response": full_response})
                
                # 发送评估项结束标记
                yield f"data: {json.dumps({'index': idx + 1, 'type': 'end'}, ensure_ascii=False)}\n\n"
                
            except Exception as e:
                self._logger.error(f"评估过程错误: {str(e)}")
                yield f"data: {json.dumps({'index': idx + 1, 'type': 'error', 'error': str(e)}, ensure_ascii=False)}\n\n"
        
        # 生成统计报告
        stats = await self._generate_evaluation_stats(all_evaluations)
        
        # 发送完成标记和统计数据
        yield f"data: {json.dumps({'type': 'complete', 'stats': stats}, ensure_ascii=False)}\n\n"
        self._logger.info(f"评估完成，统计数据: {stats}")

    def _extract_json_from_markdown(self, text: str) -> str:
        """从可能包含markdown的文本中提取JSON内容"""
        # 尝试查找并提取```json和```之间的内容
        import re
        json_pattern = r'```(?:json)?\s*([\s\S]*?)```'
        match = re.search(json_pattern, text)
        
        if match:
            # 如果找到了JSON代码块，返回其内容
            return match.group(1).strip()
        
        # 如果没有找到代码块，假设整个文本就是JSON（去除可能的前后空白）
        return text.strip()

    async def _generate_evaluation_stats(self, evaluations: List[Dict]) -> Dict:
        """生成评估统计数据"""
        # 过滤出有效的评估结果
        valid_evaluations = []
        for e in evaluations:
            if isinstance(e, dict) and 'role_play' in e and 'dialogue_experience' in e:
                valid_evaluations.append(e)
            elif isinstance(e, dict) and 'error' in e and 'raw_response' in e:
                # 尝试从raw_response中提取JSON
                try:
                    clean_response = self._extract_json_from_markdown(e['raw_response'])
                    result = json.loads(clean_response)
                    if 'role_play' in result and 'dialogue_experience' in result:
                        valid_evaluations.append(result)
                        continue
                except Exception as ex:
                    self._logger.warning(f"无法从raw_response提取有效JSON: {ex}")
                
                self._logger.warning(f"评估错误: {e.get('error', '未知错误')}")
            else:
                self._logger.warning(f"无效评估数据: {e}")
        
        if not valid_evaluations:
            self._logger.error("没有有效的评估数据")
            return {"error": "无有效评估数据"}
        
        # 初始化统计数据结构
        stats = {
            "overall_scores": {
                "role_score": 0,
                "dialogue_score": 0,
                "final_score": 0
            },
            "role_play": {
                "consistency": {"scores": [], "avg": 0, "keywords": {}},
                "knowledge": {"scores": [], "avg": 0, "keywords": {}},
                "language_style": {"scores": [], "avg": 0, "keywords": {}},
                "emotional_expression": {"scores": [], "avg": 0, "keywords": {}},
                "character_depth": {"scores": [], "avg": 0, "keywords": {}}
            },
            "dialogue_experience": {
                "response_quality": {"scores": [], "avg": 0, "keywords": {}},
                "interaction_fluency": {"scores": [], "avg": 0, "keywords": {}},
                "language_expression": {"scores": [], "avg": 0, "keywords": {}},
                "context_adaptation": {"scores": [], "avg": 0, "keywords": {}},
                "personalization": {"scores": [], "avg": 0, "keywords": {}}
            }
        }
        
        # 收集所有评分和关键词
        for eval_data in valid_evaluations:
            # 角色扮演维度
            role_play = eval_data.get("role_play", {})
            for key in stats["role_play"]:
                if key in role_play and "score" in role_play[key]:
                    stats["role_play"][key]["scores"].append(role_play[key]["score"])
                    
                    # 收集关键词
                    if "keywords" in role_play[key] and isinstance(role_play[key]["keywords"], list):
                        for keyword in role_play[key]["keywords"]:
                            stats["role_play"][key]["keywords"][keyword] = stats["role_play"][key]["keywords"].get(keyword, 0) + 1
            
            # 对话体验维度
            dialogue_exp = eval_data.get("dialogue_experience", {})
            for key in stats["dialogue_experience"]:
                if key in dialogue_exp and "score" in dialogue_exp[key]:
                    stats["dialogue_experience"][key]["scores"].append(dialogue_exp[key]["score"])
                    
                    # 收集关键词
                    if "keywords" in dialogue_exp[key] and isinstance(dialogue_exp[key]["keywords"], list):
                        for keyword in dialogue_exp[key]["keywords"]:
                            stats["dialogue_experience"][key]["keywords"][keyword] = stats["dialogue_experience"][key]["keywords"].get(keyword, 0) + 1
            
            # 总体评分
            if "role_score" in role_play:
                stats["overall_scores"]["role_score"] += role_play["role_score"]
            if "dialogue_score" in dialogue_exp:
                stats["overall_scores"]["dialogue_score"] += dialogue_exp["dialogue_score"]
            if "final_score" in eval_data:
                stats["overall_scores"]["final_score"] += eval_data["final_score"]
        
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
                
                # 对关键词进行排序，取前10个
                keywords = stats[category][key]["keywords"]
                stats[category][key]["keywords"] = dict(sorted(keywords.items(), key=lambda x: x[1], reverse=True)[:10])
        
        return stats

    async def _evaluate_single(self, data: Dict) -> Dict:
        """单条数据评估"""
        # 构建提示词
        prompt = self.eval_prompt.safe_substitute(
            criteria=self.config.get("criteria", ""),
            eval_data=json.dumps(data, ensure_ascii=False, indent=2)
        )
        
        try:
            context = await self._build_eval_context(prompt)
            response = await self.llm.generate(context["messages"])
            
            # 清理响应中的markdown格式
            clean_response = self._extract_json_from_markdown(response)
            
            try:
                evaluation_result = json.loads(clean_response)
                return evaluation_result
            except json.JSONDecodeError as e:
                self._logger.error(f"JSON解析错误: {e}")
                self._logger.debug(f"原始响应: {response}")
                self._logger.debug(f"清理后响应: {clean_response}")
                return {"error": "JSON解析错误"}
            
        except Exception as e:
            self._logger.error(f"评估过程错误: {str(e)}")
            return {"error": str(e)}

    async def _build_eval_context(self, prompt: str) -> Dict[str, Any]:
        """构建评估上下文"""
        system_content = ""
        
        # 如果有人设信息，添加到系统提示中
        if hasattr(self, 'role_info') and self.role_info:
            system_content += f"\n\n以下是本次评估相关的角色人设信息：\n{self.role_info}"
        
        return {
            "messages": [
                {
                    "role": "system", 
                    "content": system_content
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        } 