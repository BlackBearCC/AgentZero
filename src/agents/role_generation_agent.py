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
        你是一个专业的角色配置生成器。请基于用户输入，遵循科学严谨的方法论，生成完整的角色配置。以下是详细的生成规范和思维链指导：

        生成规范：
        1. 属性大类：每个角色配置由多个属性大类构成，每个大类下包含多条具体内容。
        2. 内容结构：每条内容必须包含关键词数组、描述文本和重要程度。
           - 关键词数组：用于检索和匹配，包含3-30个关联词，覆盖内容的重要概念。
           - 描述文本：角色特征的具体描述，长度为10-30字，可包含补充说明，使用()标注。
           - 重要程度：数值范围1-5，表示内容的重要性和优先级。
        3. 内容生成：每个属性大类至少生成3条内容，确保角色的多样性和丰富性。
        4. 描述文本中可使用{{user}}和{{char}}占位符，以增强互动性。

        思维链指导：
        1. 基础信息：描述角色的基本身份和背景，包括姓名、年龄、身份等。确保这些信息能够构成角色的核心身份。
        2. 性格特征：定义角色的性格特质，建议保留最关键的3个特征，以确保角色性格的一致性。
        3. 能力特征：描述角色的特殊能力或技能，强调其在特定情境下的表现。
        4. 兴趣爱好：展现角色的个人喜好和日常活动倾向，丰富角色的生活方式。
        5. 情感特质：描述角色在感情方面的特点，包括对待感情的态度和表达方式。
        6. 其他属性：包括喜好厌恶、成长经历、价值观念等，进一步塑造角色的立体形象。

        用户输入：$reference
        
        输出格式（请严格按照以下JSON格式输出，不要有任何多余内容）：
        {
            "基础信息": [
                {
                    "关键词": ["姓名", "称谓"],
                    "内容": "角色名称",
                    "强度": 5
                },
                {
                    "关键词": ["年龄", "外表"],
                    "内容": "25岁(外表年龄)",
                    "强度": 5
                },
                {
                    "关键词": ["身份", "职业"],
                    "内容": "画廊主人(表面身份)",
                    "强度": 5
                }
            ],
            "性格特征": [
                {
                    "关键词": ["性格", "特质"],
                    "内容": "神秘且温柔",
                    "强度": 5
                }
            ],
            "能力特征": [
                {
                    "关键词": ["艺术", "创作"],
                    "内容": "拥有非凡的艺术天赋",
                    "强度": 5
                }
            ],
            "兴趣爱好": [
                {
                    "关键词": ["艺术", "收藏"],
                    "内容": "热爱艺术创作和收藏",
                    "强度": 5
                }
            ],
            "情感特质": [
                {
                    "关键词": ["专一", "深情"],
                    "内容": "对感情专一且深情",
                    "强度": 5
                }
            ],
            "喜好厌恶": [],
            "成长经历": [],
            "价值观念": [],
            "社交关系": [],
            "禁忌话题": [],
            "行为模式": [],
            "隐藏设定": [],
            "目标动机": [],
            "弱点缺陷": [],
            "特殊习惯": [],
            "语言风格": []
        }
        """)
        


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
        
        try:
            # 清理和规范化 JSON 字符串
            cleaned_response = full_response.strip()
            # 尝试直接解析，因为返回的应该已经是合法的 JSON
            try:
                parsed_response = json.loads(cleaned_response)
            except json.JSONDecodeError:
                # 如果直接解析失败，尝试修复 JSON
                cleaned_response = self._fix_json(cleaned_response)
                parsed_response = json.loads(cleaned_response)
            
            # 发送完整的解析后的数据
            yield f"data: {json.dumps({'type': 'complete', 'content': parsed_response}, ensure_ascii=False)}\n\n"
        except json.JSONDecodeError as e:
            self._logger.warning(f"JSON解析失败: {str(e)}")
            # 如果解析失败，发送原始响应
            yield f"data: {json.dumps({'type': 'chunk', 'content': cleaned_response}, ensure_ascii=False)}\n\n"
            yield f"data: {json.dumps({'type': 'complete', 'content': cleaned_response}, ensure_ascii=False)}\n\n"
