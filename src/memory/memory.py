from typing import List, Dict, Any, Tuple, Optional
from src.types.message import Message
from src.utils.logger import Logger
import json
from datetime import datetime
import requests
import asyncio
from collections import deque

class Memory:
    def __init__(self, llm=None, max_history: int = 20, min_recent: int = 5):
        self.chat_history: List[Message] = []
        self.summary: str = ""  # 对话概要
        self.max_history = max_history
        self.min_recent = min_recent
        self.llm = llm
        self._logger = Logger()
        self.entity_api_url = "http://192.168.52.114:8014/query"
        self.entity_api_headers = {"Content-Type": "application/json"}
        
        # 新增：异步任务相关
        self._summary_task = None  # 当前运行的summary生成任务
        self._summary_queue = deque()  # 等待处理的消息队列
        self._is_generating = False  # 是否正在生成summary
        
    async def add_message(self, role: str, content: str):
        """添加新消息并在需要时触发异步生成概要"""
        if role == "assistant":
            try:
                response_json = json.loads(content)
                content = response_json.get("content", content)
            except json.JSONDecodeError:
                pass
                
        self.chat_history.append(Message(role=role, content=content))
        
        # 检查是否需要生成概要
        if len(self.chat_history) > self.max_history:
            # 将需要总结的消息添加到队列
            messages_to_summarize = self.chat_history[:-self.min_recent]
            self._summary_queue.append(messages_to_summarize)
            
            # 如果没有正在运行的任务，启动新任务
            if not self._is_generating:
                self._summary_task = asyncio.create_task(self._process_summary_queue())
            
    async def _process_summary_queue(self):
        """处理概要生成队列"""
        try:
            self._is_generating = True
            while self._summary_queue:
                messages = self._summary_queue.popleft()
                await self._generate_summary_for_messages(messages)
                
        finally:
            self._is_generating = False
            self._summary_task = None
            
    async def _generate_summary_for_messages(self, messages_to_summarize: List[Message]):
        """为指定消息生成概要"""
        try:
            # 构建对话内容
            new_lines = ""
            for msg in messages_to_summarize:
                new_lines += f"{msg.role}: {msg.content}\n"
                
            # 构建完整的提示词
            full_prompt = f"""你是对话概要记录员。请基于当前概要和新对话内容，生成更新后的对话概要。

要求：
1. 篇幅控制：
   - 总字数严格控制在200字以内
   - 新内容占比60-70%
   - 旧内容压缩至30-40%

2. 更新原则：
   - 新内容按时间顺序记录
   - 旧内容保留核心信息点
   - 相似主题内容合并
   - 远期内容逐步淡化
   - 保持时间连贯性

3. 记录重点：
   - 关键事件和行为
   - 明确的承诺或约定
   - 重要的情感转折
   - 具体的时间地点
   - 称呼的变化

4. 使用"USER"和"AI"指代对话双方
5. 删除以下内容:
   - 重复的对话内容
   - 日常问候客套
   - 无实质的对话
   - 模糊的描述
   - 过时的细节

当前概要：
{self.summary or '对话开始'}

新的对话内容：
{new_lines}

请直接输出更新后的对话概要，不需要包含具体时间："""

            response = ""
            async for chunk in self.llm.astream(full_prompt):
                response += chunk
            
            # 解析响应
            try:
                summary_json = json.loads(response)
                summary = summary_json.get("content", response)
            except json.JSONDecodeError:
                summary = response

            summary = summary.replace("USER", "木木").replace("AI", "祁煜")
            self.summary = summary
            
            # 更新历史记录
            self.chat_history = self.chat_history[-self.min_recent:]
                
        except Exception as e:
            self._logger.error(f"生成概要时出错: {str(e)}")
        
    async def get_full_memory(self) -> Tuple[str, List[Message]]:
        """获取完整上下文（概要 + 最近消息）"""
        return self.summary, self.chat_history
    
    async def get_summary(self) -> str:
        """获取对话概要"""
        return self.summary
        
    async def get_recent_messages(self, limit: int = 10) -> List[Message]:
        """获取最近的消息"""
        return self.chat_history[-limit:]
        
    async def query_entity_memory(self, query: str, limit: int = 5) -> Optional[List[Dict[str, Any]]]:
        """查询实体记忆
        
        Args:
            query: 查询文本
            limit: 返回结果数量限制
            
        Returns:
            List[Dict]: 实体记忆列表，每个实体包含相关信息
            None: 查询失败时返回
        """
        try:
            response = requests.post(
                url=self.entity_api_url,
                headers=self.entity_api_headers,
                json={"query": query, "limit": limit}
            )
            response.raise_for_status()
            return response.json()
            
        except Exception as e:
            self._logger.error(f"查询实体记忆失败: {str(e)}")
            return None
            
    async def add_entity_memory(self, entity_data: Dict[str, Any]) -> bool:
        """添加实体记忆（预留接口）
        
        Args:
            entity_data: 实体数据
            
        Returns:
            bool: 是否添加成功
        """
        # TODO: 实现实体记忆存储逻辑
        pass
        
    async def update_entity_memory(self, entity_id: str, entity_data: Dict[str, Any]) -> bool:
        """更新实体记忆（预留接口）
        
        Args:
            entity_id: 实体ID
            entity_data: 更新的实体数据
            
        Returns:
            bool: 是否更新成功
        """
        # TODO: 实现实体记忆更新逻辑
        pass 