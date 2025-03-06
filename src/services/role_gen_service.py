import asyncio
from src.agents.role_generation_agent import RoleGenerationAgent
from src.utils.logger import Logger
from src.llm.doubao import DoubaoLLM
import os
import json
from typing import AsyncIterator, List, Dict, Any

class RoleGenService:
    _instance = None
    CATEGORY_MAP = {
        1: "基础信息",
        2: "性格特征",
        3: "能力特征", 
        4: "兴趣爱好",
        5: "情感特质",
        6: "喜好厌恶",
        7: "成长经历",
        8: "价值观念",
        9: "社交关系",
        10: "禁忌话题",
        11: "行为模式",
        12: "隐藏设定",
        13: "目标动机",
        14: "弱点缺陷",
        15: "特殊习惯",
        16: "语言风格"
    }    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, 'initialized'):
            self.initialized = True
            self.logger = Logger()
            self.gen_agent = self._create_gen_agent()
    
    def _create_gen_agent(self) -> RoleGenerationAgent:
        """创建角色生成Agent"""
        llm = DoubaoLLM(
            model_name=os.getenv("DOUBAO_MODEL_DEEPSEEK_V3"),
            temperature=0.8,
            # max_tokens=4096
        )
        
        config = {
            "name": "角色生成Agent",
            "generation_type": "full_role_config"
        }
        
        return RoleGenerationAgent(config=config, llm=llm)
    
    async def generate_category(
        self,
        category: str,
        reference: str,
        user_id: str
    ) -> AsyncIterator[str]:
        """生成单个类别的配置"""
        try:
            async for chunk in self.gen_agent.generate_category(category, reference):
                yield chunk
        except Exception as e:
            self.logger.error(f"类别 {category} 生成失败: {str(e)}")
            yield f"data: {json.dumps({'type': 'error', 'category': category, 'content': str(e)}, ensure_ascii=False)}\n\n"
    async def process_category(
        self,
        category: str,
        reference: str,
        user_id: str,
        idx: int
    ) -> List[str]:
        """处理单个类别并收集所有输出"""
        results = []
        async for chunk in self.generate_category(category, reference, user_id):
            # 添加索引信息
            if chunk.startswith('data: '):
                data = json.loads(chunk.replace('data: ', ''))
                data['index'] = idx
                results.append(f"data: {json.dumps(data, ensure_ascii=False)}\n\n")
            else:
                # 如果不是标准格式，添加索引并包装
                data = {'type': 'chunk', 'category': category, 'content': chunk, 'index': idx}
                results.append(f"data: {json.dumps(data, ensure_ascii=False)}\n\n")
        return results
    async def generate_role_config(
        self,
        reference: str,
        user_id: str,
        categories: List[str] = None
    ) -> AsyncIterator[str]:
        """串行生成所有类别的配置"""
        # 如果指定了类别，只生成这些类别
        if categories:
            category_items = [(idx, cat) for idx, cat in self.CATEGORY_MAP.items() if cat in categories]
        else:
            category_items = list(self.CATEGORY_MAP.items())
        
        # 发送开始信号
        yield f"data: {json.dumps({'type': 'start', 'content': '开始生成角色配置'}, ensure_ascii=False)}\n\n"
        
        # 串行处理每个类别
        for idx, category in category_items:
            self.logger.info(f"开始生成类别: {category}")
            try:
                # 创建协程并等待其完成
                results = await self.process_category(category, reference, user_id, idx)
                for result in results:
                    yield result
                print(result)
            except Exception as e:
                self.logger.error(f"任务 {idx} ({category}) 执行失败: {str(e)}")
                yield f"data: {json.dumps({'type': 'error', 'index': idx, 'category': category, 'content': str(e)}, ensure_ascii=False)}\n\n"
        
        # 发送结束信号
        yield f"data: {json.dumps({'type': 'all_complete', 'content': '所有类别生成完成'}, ensure_ascii=False)}\n\n"

    async def optimize_content(
        self,
        category: str,
        content: str,
        reference: str,
        user_id: str
    ) -> dict:
        """优化属性内容"""
        try:
            # 构建优化提示
            result = await self.gen_agent.optimize_content(
                category=category,
                content=content,
                reference=reference
            )
            return result
        except Exception as e:
            self.logger.error(f"内容优化失败: {str(e)}")
            raise

    async def optimize_keywords(
        self,
        category: str,
        content: str,
        keywords: List[str],
        reference: str,
        user_id: str
    ) -> dict:
        """优化属性关键词"""
        try:
            # 构建优化提示
            result = await self.gen_agent.optimize_keywords(
                category=category,
                content=content,
                keywords=keywords,
                reference=reference
            )
            return result
        except Exception as e:
            self.logger.error(f"关键词优化失败: {str(e)}")
            raise

    async def generate_new_attribute(
        self,
        category: str,
        existing_attributes: List[dict],
        reference: str,
        user_id: str
    ) -> dict:
        """生成新属性"""
        try:
            # 构建生成提示
            result = await self.gen_agent.generate_new_attribute(
                category=category,
                existing_attributes=existing_attributes,
                reference=reference
            )
            return result
        except Exception as e:
            self.logger.error(f"新属性生成失败: {str(e)}")
            raise
# 依赖注入函数
async def get_role_gen_service() -> RoleGenService:
    return RoleGenService()