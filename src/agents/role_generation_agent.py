from src.agents.base_agent import BaseAgent
from string import Template
import json
from typing import AsyncIterator, Dict, Any, List

class RoleGenerationAgent(BaseAgent):
    def __init__(self, config: dict, llm=None):
        super().__init__(config, llm)
        # 定义公共的生成规范
        self.generation_rules = """
        属性类别：
        基础信息：角色最基本的设定信息，包括姓名、年龄、性别、工作职业等固有特征。
        性格特征：角色的个性特点和行为模式，如开朗、内向、谨慎、冲动等心理特质。
        关系定位：角色与user之间的互动模式和设定的关系。
        兴趣爱好：角色的个人喜好和日常活动倾向，展现角色的生活方式和价值取向。
        喜好厌恶：角色特别喜欢或讨厌的事物、场景、人物等。
        能力特征：角色所具备的特殊能力或技能，包括超能力、职业技能、天赋等。
        情感特质：角色在感情方面的特点，包括对待感情的态度、情感表达方式等。
        成长经历：角色的重要人生经历和背景故事，塑造其性格形成的原因。
        价值观念：角色的世界观、人生观等核心信念系统。
        社交关系：角色与其他人物之间的互动模式和关系网络。
        禁忌话题：角色极力回避或抗拒的话题和领域，往往与创伤经历相关。
        行为模式：角色在特定情境下的典型反应和习惯性行为。
        隐藏设定：不轻易展现但会影响角色行为的隐藏特质或秘密。
        目标动机：推动角色行动的内在驱动力和人生追求。
        弱点缺陷：角色的软肋和不足之处，使人物更立体真实。
        特殊习惯：角色独特的生活习惯和行为特征，增添个性化细节。
        语言风格：角色语言风格特征。

        生成规范：
        1. 内容结构：每条内容必须包含关键词数组、描述文本和重要程度。
           - 关键词：用于检索和匹配，包含10-50个关联词，覆盖内容的重要概念。
             * 关键词应该是联想触发词和对话触发词而不是和内容直接相关的
             * 联想触发词：通过联想和泛化可能触发该内容的词语，可以在语义上疏远但聊到某些内容时人类会联想到，用于提高检索召回率
             * 对话触发词：人类对话时，可能触发本内容的词语，可以包含一些话题、场景、人物或主谓宾词语
           - 内容：角色特征的具体描述，长度为10-30字，特征形成原因或具体细节可包含少量补充说明，使用()拼接在内容后。
             * 描述必须具体、清晰、可执行，避免模糊或抽象的表述
             * 补充说明应提供额外的上下文或条件信息，使描述更完整
             * 使用动词、形容词等描述性词语，让特征更生动
           - 强度：数值范围1-5，表示内容的重要性和优先级。
        2. 内容生成：确保描述的多样性和丰富性。
           - 内容之间应有逻辑关联，形成完整的特征体系
        3. 描述文本中可使用{{user}}和{{char}}占位符。
        """
        self._load_gen_prompt()
        
    def _load_gen_prompt(self):
        """加载生成提示词模板"""
        self.gen_prompt = Template(f"""
        你是一个专业的角色配置生成器。请基于用户输入，生成角色的 $category 相关配置。
        
        {self.generation_rules}
    
        用户输入：$reference
        
        请仅生成 $category 类别的内容，输出格式如下（严格按照JSON格式，不要有任何多余内容）：
        ```json
        {{
            "$category": [
                {{
                    "关键词": ["关键词1", "关键词2", "关键词3"],
                    "内容": "具体描述内容",
                    "强度": 5
                }}
                ...
            ]
        }}
        ```
        """)

    async def generate_category(self, category: str, input_text: str) -> AsyncIterator[str]:
        """生成指定类别的角色属性"""
        prompt = self.gen_prompt.safe_substitute(
            category=category,
            reference=input_text
        )
        
        messages = [
            {"role": "system", "content": "你是一个角色配置生成专家"},
            {"role": "user", "content": prompt}
        ]
        
        full_response = ""
        yield f"data: {json.dumps({'type': 'start', 'category': category, 'content': ''}, ensure_ascii=False)}\n\n"
        
        async for chunk in self.llm.astream(messages):
            full_response += chunk
            yield f"data: {json.dumps({'type': 'chunk', 'category': category, 'content': chunk}, ensure_ascii=False)}\n\n"
        
        yield f"data: {json.dumps({'type': 'end', 'category': category}, ensure_ascii=False)}\n\n"
        
        cleaned_response = full_response
        if "```json" in cleaned_response or "```josn" in cleaned_response:
            cleaned_response = cleaned_response.replace("```json", "").replace("```josn", "").replace("```", "")
        
        try:
            parsed_response = json.loads(cleaned_response)
            yield f"data: {json.dumps({'type': 'complete', 'category': category, 'content': parsed_response[category]}, ensure_ascii=False)}\n\n"
        except Exception as e:
            self._logger.warning(f"JSON解析失败: {str(e)}")
            yield f"data: {json.dumps({'type': 'error', 'category': category, 'content': cleaned_response}, ensure_ascii=False)}\n\n"

    # 实现基类要求的抽象方法
    async def load_prompt(self) -> str:
        """加载基础提示词"""
        return self.gen_prompt.template
    
    async def update_prompt(self, **kwargs) -> str:
        """更新生成提示词"""
        if 'template' in kwargs:
            self.gen_prompt = Template(kwargs['template'])
        return self.gen_prompt.template
    
    async def think(self, context: Dict[str, Any]) -> List[str]:
        """生成场景不需要调用工具"""
        return []
    
    async def generate_response(self, input_text: str, user_id: str, remark: str = '') -> str:
        """同步生成响应"""
        full_response = ""
        async for chunk in self.generate_role_config(input_text):
            full_response += chunk
        return full_response
    
    async def astream_response(self, input_text: str, user_id: str, remark: str = '', context: Dict[str, Any] = None) -> AsyncIterator[str]:
        """流式生成角色配置"""
        prompt = self.gen_prompt.safe_substitute(
            reference=input_text
        )
        
        messages = [
            {"role": "system", "content": "你是一个角色配置生成专家"},
            {"role": "user", "content": prompt}
        ]
        
        full_response = ""
        # 发送开始标记
        yield f"data: {json.dumps({'type': 'start','content':''}, ensure_ascii=False)}\n\n"
        
        async for chunk in self.llm.astream(messages):
            full_response += chunk
            # 发送数据块
            yield f"data: {json.dumps({'type': 'chunk', 'content': chunk}, ensure_ascii=False)}\n\n"
        
        # 发送结束标记
        yield f"data: {json.dumps({'type': 'end'}, ensure_ascii=False)}\n\n"
        
        # 处理可能的markdown格式
        cleaned_response = full_response
        # 移除可能的markdown代码块标记
        if "```json" in cleaned_response or "```josn" in cleaned_response:
            cleaned_response = cleaned_response.replace("```json", "").replace("```josn", "").replace("```", "")
        
        self._logger.info(f"角色生成完成: {cleaned_response}")
        
        try:
            # 尝试解析JSON
            parsed_response = json.loads(cleaned_response)
            yield f"data: {json.dumps({'type': 'complete', 'content': parsed_response}, ensure_ascii=False)}\n\n"
        except json.JSONDecodeError as e:
            self._logger.warning(f"JSON解析失败: {str(e)}")
            # 如果解析失败，发送原始响应
            yield f"data: {json.dumps({'type': 'complete', 'content': cleaned_response}, ensure_ascii=False)}\n\n"

    # 添加优化内容的提示词模板
    def _load_optimize_content_prompt(self):
        """加载优化内容提示词模板"""
        return Template(f"""
        你是一个专业的角色配置优化专家。请基于以下信息优化当前角色属性内容：

        {self.generation_rules}

        类别：$category
        当前内容：$content
        参考资料：$reference

        请按照以下步骤优化内容（只优化当前提供的内容）：

        1. 分析当前内容
           - 提取核心信息和关键概念
           - 识别内容的主要特征和表达方式
           - 评估内容的优缺点

        2. 分析参考资料
           - 提取与当前类别相关的信息
           - 寻找可以补充或完善的要素
           - 确保与参考资料保持一致性

        3. 优化内容描述
           - 保持核心信息不变
           - 使用更准确和生动的表达
           - 添加必要的补充说明
           - 确保长度在10-30字之间
        可使用{{{{user}}}}和{{{{char}}}}占位符。

        输出格式如下（严格按照以下JSON格式）：
        ```json
        {{
            "内容": "优化后的描述文本"
        }}
        ```
        """)

    # 添加优化关键词的提示词模板
    def _load_optimize_keywords_prompt(self):
        """加载优化关键词提示词模板"""
        return Template(f"""
        你是一个专业的关键词优化专家。请基于以下信息优化关键词：

        {self.generation_rules}

        类别：$category
        当前内容：$content
        当前关键词：$keywords

        请按照以下步骤优化关键词：

        1. 分析当前内容和关键词
           - 评估现有关键词的覆盖范围
           - 关键词需要用于辅助触发内容，但不是直接检索
           - 检查是否存在冗余或不相关的词语
           - 关键词应该是能触发该内容的广泛联想词，而非直接描述内容的词语
           - 关键词之间应保持语义多样性，覆盖不同维度和场景
           - 人类对话时，可能触发本内容的词语，可以包含一些话题、场景、人物或主谓宾词语

        2. 优化关键词列表
           - 删除不准确或冗余的词语
           - 添加更精准的描述词
           - 补充关键词
           - 确保数量在10-30个之间
           - 添加用户对话中可能出现的词语或短语，即使它们与内容的直接关联性不强
           - 包含不同语境下可能触发该内容的词语，扩大检索范围
           - 考虑多种可能的表达方式和同义词变体

        避免输出语意重复的内容，避免使用情感理解，情感沟通这样相同句式关键词.
        禁止输出抽象概念，要明确清晰的词语

        输出格式如下（严格按照JSON格式）：
        ```json
        {{
            "关键词": ["关键词1", "关键词2", "关键词3"]
        }}
        ```
        """)

    async def optimize_content(
        self,
        category: str,
        content: str,
        reference: str
    ) -> dict:
        """优化属性内容"""
        prompt_template = self._load_optimize_content_prompt()
        prompt = prompt_template.safe_substitute(
            category=category,
            content=content,
            reference=reference
        )
        
        messages = [
            {"role": "system", "content": "你是一个角色配置优化专家"},
            {"role": "user", "content": prompt}
        ]
        
        # 使用 astream 并拼接结果
        full_response = ""
        async for chunk in self.llm.astream(messages):
            full_response += chunk
        
        # 处理响应
        cleaned_response = full_response
        if "```json" in cleaned_response:
            cleaned_response = cleaned_response.replace("```json", "").replace("```", "")
        
        try:
            result = json.loads(cleaned_response)
            print(f"内容优化结果: {result}")
            return result
        except json.JSONDecodeError:
            self._logger.error(f"JSON解析失败: {cleaned_response}")
            raise ValueError("内容优化结果格式错误")

    async def optimize_keywords(
        self,
        category: str,
        content: str,
        keywords: List[str],
        reference: str
    ) -> dict:
        """优化属性关键词"""
        prompt_template = self._load_optimize_keywords_prompt()
        prompt = prompt_template.safe_substitute(
            category=category,
            content=content,
            keywords=json.dumps(keywords, ensure_ascii=False),
            reference=reference
        )
        
        messages = [
            {"role": "system", "content": "你是一个角色配置优化专家"},
            {"role": "user", "content": prompt}
        ]
        
        # 使用 astream 并拼接结果
        full_response = ""
        async for chunk in self.llm.astream(messages):
            full_response += chunk
        
        # 处理响应
        cleaned_response = full_response
        if "```json" in cleaned_response:
            cleaned_response = cleaned_response.replace("```json", "").replace("```", "")
        
        try:
            result = json.loads(cleaned_response)
            print(f"关键词优化结果: {result}")
            return result
        except json.JSONDecodeError:
            self._logger.error(f"JSON解析失败: {cleaned_response}")
            raise ValueError("关键词优化结果格式错误")

    def _load_new_attribute_prompt(self):
        """加载新属性生成提示词模板"""
        return Template(f"""
        你是一个专业的角色配置生成专家。请基于以下信息生成一个新的角色属性：

        {self.generation_rules}

        类别：$category
        已有属性：$existing_attributes
        参考资料：$reference
        请按照以下步骤生成一个新属性：

        1. 分析已有属性
           - 总结已覆盖的特征和维度
           - 识别潜在的空白点
           - 避免与现有属性重复或冲突

        2. 分析参考资料
           - 提取未被利用的信息
           - 寻找新的特征维度
           - 确保与整体设定协调

        3. 生成新属性内容
           - 选择合适的特征维度
           - 撰写具体的描述文本
           - 确保描述长度在10-30字
           - 适当使用补充说明和占位符

        4. 生成关键词
           - 提取核心概念
           - 添加相关联的词语
           - 确保覆盖完整的语义范围

        5. 确定强度等级
           - 评估特征的重要性
           - 考虑与其他属性的关系
           - 选择合适的强度值(1-5)

        6. 最终检查
           - 验证与已有属性的差异性
           - 确保格式符合要求
           - 检查内容的完整性和准确性
        
        若没有新属性，请返回 "该属性类别下内容已经全部覆盖"，格式还是下面的json。
        输出格式如下（严格按照JSON格式）：
        {{
            "内容": "新的描述文本",
            "关键词": ["关键词1", "关键词2", "关键词3"],
            "强度": 5
        }}
        """)


    async def generate_new_attribute(
        self,
        category: str,
        existing_attributes: List[dict],
        reference: str
    ) -> dict:
        """生成新属性"""
        prompt_template = self._load_new_attribute_prompt()
        prompt = prompt_template.safe_substitute(
            category=category,
            existing_attributes=json.dumps(existing_attributes, ensure_ascii=False),
            reference=reference
        )
        
        messages = [
            {"role": "system", "content": "你是一个角色配置生成专家"},
            {"role": "user", "content": prompt}
        ]
        
        # 使用 astream 替代 achat，并拼接结果
        full_response = ""
        async for chunk in self.llm.astream(messages):
            full_response += chunk
        
        # 处理响应
        cleaned_response = full_response
        if "```json" in cleaned_response:
            cleaned_response = cleaned_response.replace("```json", "").replace("```", "")
        
        try:
            result = json.loads(cleaned_response)
            print(f"生成结果: {result}")
            return result
        except json.JSONDecodeError:
            self._logger.error(f"JSON解析失败: {cleaned_response}")
            raise ValueError("生成结果格式错误")
