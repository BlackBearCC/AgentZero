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
        你是一个专业的角色配置生成器。根据用户输入生成完整的角色配置。
        
        生成要求：
        1. 包含角色基本信息：名称、年龄、背景故事
        2. 定义角色性格特征（至少5个维度）
        3. 包含专业知识领域（至少3个）
        4. 语言风格描述（包含用词特点、句式结构）
        5. 行为模式特征
        
        用户输入：$reference
        
        输出格式：
        {
            "basic_info": {
                "name": "角色名称",
                "age": 25,
                "background": "背景故事"
            },
            "personality": {
                "traits": ["特质1", "特质2", "特质3", "特质4", "特质5"],
                "description": "综合性格描述"
            },
            "expertise": ["领域1", "领域2", "领域3"],
            "language_style": {
                "vocabulary": ["常用词1", "常用词2"],
                "sentence_structure": "句式结构特点"
            },
            "behavior_patterns": {
                "common_actions": ["行为1", "行为2"],
                "decision_making": "决策方式描述"
            }
        }
        """)
        


    def _fix_json(self, text: str) -> str:
        """修复不完整的JSON"""
        text = text.replace("'", '"')
        text = text.replace("，", ",")
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
        async for chunk in self.llm.astream(messages):
            full_response += chunk
            # 使用SSE格式包装响应
            try:
                # 尝试解析为JSON，如果成功则作为JSON数据返回
                json_obj = json.loads(chunk)
                yield f"data: {json.dumps(json_obj, ensure_ascii=False)}\n\n"
            except json.JSONDecodeError:
                # 如果不是有效JSON，则作为普通文本返回
                yield f"data: {chunk}\n\n"
        
        self._logger.info(f"角色生成完成: {full_response}")
        # 最终验证JSON格式
        try:
            json.loads(full_response)
        except json.JSONDecodeError:
            fixed_json = self._fix_json(full_response)
            yield f"data: {fixed_json}\n\n"