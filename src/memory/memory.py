from typing import List, Dict, Any, Tuple, Optional
from src.types.message import Message
from src.utils.logger import Logger
import json
from datetime import datetime
import requests

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
        
    async def add_message(self, role: str, content: str):
        """添加新消息并在需要时生成概要"""
        if role == "assistant":
            try:
                response_json = json.loads(content)
                content = response_json.get("content", content)
            except json.JSONDecodeError:
                pass
                
        self.chat_history.append(Message(role=role, content=content))
        
        # 检查是否需要生成概要
        if len(self.chat_history) > self.max_history:
            await self._generate_summary()
            
    async def _generate_summary(self):
        """生成对话概要并压缩历史记录"""
        recent_messages = self.chat_history[-self.min_recent:]
        messages_to_summarize = self.chat_history[:-self.min_recent]
        print(f"生成近期概要中================")
        
        if messages_to_summarize:
            # 先构建对话内容
            new_lines = ""
            for msg in messages_to_summarize:
                new_lines += f"{msg.role}: {msg.content}\n"
                
            # 构建完整的提示词
            full_prompt = f"""你是一个对话概要记录员。请基于当前概要和新对话内容,更新生成一段新的对话概要。

要求：
1. 篇幅控制在300字以内,以一段连贯文字呈现
2. 更新原则：
   - 保留当前概要中的关键事实信息
   - 按时间顺序记录新增对话内容
   - 合并相同主题的内容
   - 压缩旧内容但保留关键事件,旧内容占比不超过30%

3. 记录内容范围：
   - 具体发生的事件和行为
   - 明确的约定和承诺
   - 双方使用的称呼
   - 涉及的具体地点、时间、物品
   - 明确表达的情绪状态

4. 使用"USER"和"AI"指代对话双方
5. 删除以下内容:
   - 重复的对话内容
   - 问候等客套内容
   - 没有实质信息的对话
   - 对话内容的推测和解读

当前概要：
{self.summary or '对话开始'}

新的对话内容：
{new_lines}

请直接输出更新后的对话概要："""

            try:
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
                self.chat_history = recent_messages
                
            except Exception as e:
                self._logger.error(f"生成概要时出错: {str(e)}")
                return
                
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