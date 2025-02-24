from typing import List, Dict, Any, Tuple, Optional
from src.types.message import Message
from src.utils.logger import Logger
import json
from datetime import datetime
import requests
import asyncio
from collections import deque

class Memory:
    def __init__(self, llm=None, max_history: int = 20, min_recent: int = 5, enable_summary: bool = False):
        # 使用字典存储不同用户的对话历史
        self.chat_histories: Dict[str, List[Message]] = {}
        # 使用字典存储不同用户的对话概要
        self.summaries: Dict[str, str] = {}
        
        self.max_history = max_history
        self.min_recent = min_recent
        self.llm = llm
        self._logger = Logger()
        self.entity_api_url = "http://192.168.52.114:8016/query"
        self.entity_api_headers = {"Content-Type": "application/json"}
        
        # 新增：异步任务相关 - 按用户ID隔离
        self._summary_tasks: Dict[str, asyncio.Task] = {}  # 每个用户的summary生成任务
        self._summary_queues: Dict[str, deque] = {}  # 每个用户的消息队列
        self._is_generating: Dict[str, bool] = {}  # 每个用户的生成状态
        
        self.enable_summary = enable_summary  # 新增概要开关
        
        self._db = None  # 数据库服务实例
        
    def _ensure_user_state(self, user_id: str):
        """确保用户相关的状态已初始化"""
        if user_id not in self.chat_histories:
            self.chat_histories[user_id] = []
        if user_id not in self.summaries:
            self.summaries[user_id] = ""
        if user_id not in self._summary_queues:
            self._summary_queues[user_id] = deque()
        if user_id not in self._is_generating:
            self._is_generating[user_id] = False
            
    async def add_message(self, role: str, content: str, user_id: str):
        """添加新消息并在需要时触发异步生成概要"""
        self._ensure_user_state(user_id)
        
        if role == "assistant":
            try:
                # response_json = json.loads(content)
                # content = response_json.get("content", content)
                content = content
                # self._logger.debug(f"[Memory] 添加消息: {content}")
            except json.JSONDecodeError:
                pass
                
        self.chat_histories[user_id].append(Message(role=role, content=content))
        
        # 仅在启用概要时进行处理
        if self.enable_summary and len(self.chat_histories[user_id]) > self.max_history:
            messages_to_summarize = self.chat_histories[user_id][:-self.min_recent]
            self._summary_queues[user_id].append(messages_to_summarize)
            
            if not self._is_generating[user_id]:
                self._summary_tasks[user_id] = asyncio.create_task(
                    self._process_summary_queue(user_id)
                )
            
    async def _process_summary_queue(self, user_id: str):
        """处理特定用户的概要生成队列"""
        try:
            self._is_generating[user_id] = True
            self._logger.info(f"[Memory] 开始处理用户 {user_id} 的概要生成队列")
            
            while self._summary_queues[user_id]:
                messages = self._summary_queues[user_id].popleft()
                self._logger.info(f"[Memory] 正在为用户 {user_id} 生成对话概要，消息数量: {len(messages)}")
                await self._generate_summary_for_messages(messages, user_id)
                
        except Exception as e:
            self._logger.error(f"[Memory] 处理用户 {user_id} 的概要队列时出错: {str(e)}")
        finally:
            self._is_generating[user_id] = False
            self._summary_tasks[user_id] = None
            
    async def _ensure_db(self):
        """确保数据库服务已初始化"""
        if not self._db:
            from src.services.db_service import DBService
            self._db = await DBService.get_instance()

    async def _generate_summary_for_messages(self, messages_to_summarize: List[Message], user_id: str):
        """为指定用户的消息生成概要"""
        try:
            # 确保数据库连接已初始化
            await self._ensure_db()
            
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
{self.summaries[user_id] or '对话开始'}

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
            self._logger.info(f"[Memory] 处理后的概要: \n{summary}")
            
            # 保存到数据库
            save_result = await self._db.save_summary(user_id, summary)
            if save_result:
                self._logger.info(f"[Memory] 成功保存概要到数据库，用户ID: {user_id}")
            else:
                self._logger.error(f"[Memory] 保存概要到数据库失败，用户ID: {user_id}")
            
            # 更新内存中的概要
            self.summaries[user_id] = summary
            
            # 更新历史记录
            self.chat_histories[user_id] = self.chat_histories[user_id][-self.min_recent:]
            self._logger.info(f"[Memory] 更新历史记录完成，保留最近 {self.min_recent} 条消息")
                
        except Exception as e:
            self._logger.error(f"[Memory] 生成用户 {user_id} 的概要时出错: {str(e)}")
        
    async def get_full_memory(self, user_id: str) -> Tuple[str, List[Message]]:
        """获取指定用户的完整上下文"""
        self._ensure_user_state(user_id)
        return self.summaries[user_id], self.chat_histories[user_id]
    
    async def get_summary(self, user_id: str) -> str:
        """获取指定用户的对话概要"""
        self._ensure_user_state(user_id)
        return self.summaries[user_id]
        
    async def get_recent_messages(self, user_id: str, limit: int = 10) -> List[Message]:
        """获取指定用户的最近消息"""
        self._ensure_user_state(user_id)

        return self.chat_histories[user_id][-limit:]
    # async def get_recent_messages_with_format(self, user_id: str, limit: int = 10) -> List[Message]:
    #     """获取指定用户的最近消息(附带格式)"""
    #     self._ensure_user_state(user_id)
    #     normal_msg = self.chat_histories[user_id][-limit:]
    #     format_masg = []
    #     for msg in normal_msg
    #     return 
        
    async def query_entity_memory(self, query: str, limit: int = 5) -> Optional[Dict[str, Any]]:
        """查询指定用户的实体记忆"""
        try:
            response = requests.post(
                url=self.entity_api_url,
                headers=self.entity_api_headers,
                json={
                    "query": query,
                    "limit": limit,
                }
            )
            response.raise_for_status()
            return response.json()
            
        except Exception as e:
            self._logger.error(f"查询实体记忆失败: {str(e)}")
            return None
            
    async def add_entity_memory(self, entity_data: Dict[str, Any], user_id: str) -> bool:
        """添加指定用户的实体记忆"""
        # TODO: 实现实体记忆存储逻辑
        pass
        
    async def update_entity_memory(self, entity_id: str, entity_data: Dict[str, Any], user_id: str) -> bool:
        """更新指定用户的实体记忆"""
        # TODO: 实现实体记忆更新逻辑
        pass

    async def get_chat_history(self, user_id: str) -> List[Message]:
        """获取聊天历史"""
        self._ensure_user_state(user_id)
        return self.chat_histories[user_id]

    async def get_entity_memory(self, user_id: str) -> Dict[str, Any]:
        """获取实体记忆"""
        # 实现获取实体记忆的逻辑
        return {}

    async def get_processed_memory(self, user_id: str) -> str:
        """获取处理后的记忆"""
        self._ensure_user_state(user_id)
        return self.summaries[user_id] 