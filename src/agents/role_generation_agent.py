from src.agents.base_agent import BaseAgent
from string import Template
import json
from typing import AsyncIterator, Dict, Any, List

class RoleGenerationAgent(BaseAgent):
    def __init__(self, config: dict, llm=None):
        super().__init__(config, llm)
        # 定义公共的生成规范
        self.generation_rules = """
        生成规范：
        1. 内容结构：每条内容必须包含关键词数组、描述文本和重要程度。
           - 关键词：用于检索和匹配，包含3-30个关联词，覆盖内容的重要概念。
           - 内容：角色特征的具体描述，长度为10-30字，可包含补充说明，使用()标注。
           - 强度：数值范围1-5，表示内容的重要性和优先级。
        2. 内容生成：至少生成3条内容，确保描述的多样性和丰富性。
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
        你是一个专业的角色配置优化专家。请基于以下信息优化角色属性的内容：

        {self.generation_rules}

        类别：$category
        当前内容：$content
        参考资料：$reference

        请优化这个属性的内容描述，使其更加丰富和准确，但保持原有的关键信息，禁止长篇大论。
        内容长度应为10-30字，可包含补充说明，使用()标注。
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
        你是一个专业的角色配置优化专家。请基于以下信息优化角色属性的关键词：

        {self.generation_rules}

        类别：$category
        当前内容：$content
        当前关键词：$keywords
        参考资料：$reference

        请优化这个属性的关键词，使其更加精准和全面。
        关键词应包含3-30个关联词，覆盖内容的重要概念。

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
        你是一个专业的角色配置生成专家。请基于以下信息生成新的角色属性：

        {self.generation_rules}

        类别：$category
        已有属性：$existing_attributes
        参考资料：$reference

        请生成一个新的、不重复的属性，确保与已有属性不冲突。
        
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
