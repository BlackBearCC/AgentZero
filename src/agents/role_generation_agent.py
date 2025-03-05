from src.agents.base_agent import BaseAgent
from string import Template
import json
from typing import AsyncIterator, Dict, Any, List

class RoleGenerationAgent(BaseAgent):
    def __init__(self, config: dict, llm=None):
        super().__init__(config, llm)
        self._load_gen_prompt()
        
    def _load_gen_prompt(self):
        """加载生成提示词模板"""
        self.gen_prompt = Template("""
        你是一个专业的角色配置生成器。请基于用户输入，生成角色的 $category 相关配置。
        
        生成规范：
        1. 内容结构：每条内容必须包含关键词数组、描述文本和重要程度。
           - 关键词数组：用于检索和匹配，包含3-30个关联词，覆盖内容的重要概念。
           - 描述文本：角色特征的具体描述，长度为10-30字，可包含补充说明，使用()标注。
           - 重要程度：数值范围1-5，表示内容的重要性和优先级。
        2. 内容生成：至少生成3条内容，确保描述的多样性和丰富性。
        3. 描述文本中可使用{{user}}和{{char}}占位符。
    
        用户输入：$reference
        
        请仅生成 $category 类别的内容，输出格式如下（严格按照JSON格式，不要有任何多余内容）：
        {
            "$category": [
                {
                    "关键词": ["关键词1", "关键词2", "关键词3"],
                    "内容": "具体描述内容",
                    "强度": 5
                }
            ]
        }
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

    def _fix_json(self, text: str) -> str:
        """修复不完整的JSON"""
        # 移除可能的前缀和后缀
        text = text.strip()
        
        # 如果文本被转义了，需要还原
        if text.startswith('"') and text.endswith('"'):
            try:
                # 先解码转义的字符串
                text = json.loads(text)
            except:
                pass
        
        # 基础清理
        text = text.replace('\n', '')
        text = text.replace('\r', '')
        text = text.replace('\t', '')
        text = text.replace('，', ',')
        text = text.replace('：', ':')
        text = text.replace('"', '"')
        text = text.replace('"', '"')
        text = text.replace('\'', '"')
        
        # 确保 JSON 对象的完整性
        if not text.startswith('{'):
            text = '{' + text
        if not text.endswith('}'):
            text = text + '}'
        
        return text

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
