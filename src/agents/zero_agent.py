from typing import Dict, Any, List, Optional, AsyncIterator
from src.agents.base_agent import BaseAgent
from langchain.schema import (
    SystemMessage,
    HumanMessage,
    AIMessage,
    BaseMessage
)
import json
from datetime import datetime, timedelta, timezone
from src.services.db_service import DBService
import uuid

class ZeroAgent(BaseAgent):
    def __init__(self, config: Dict[str, Any], llm=None, memory_llm=None, tools=None):
        """
        初始化函数，用于设置配置和系统提示词。
        
        参数:
        - config: 包含系统配置的字典。
        - llm: 语言模型实例，默认为None。
        - memory: 存储记忆的实例，默认为None。
        - tools: 可用工具的列表，默认为None。
        """
        super().__init__(config, llm, memory_llm, tools)
        # 初始化配置
        self.use_memory_queue = config.get("use_memory_queue", True)  # 是否使用记忆队列
        self.use_combined_query = config.get("use_combined_query", False)  # 是否使用组合查询
        self.memory_queue_limit = config.get("memory_queue_limit", 15)  # 默认队列长度为15
        self.event_queue = []    # 存储历史事件
        self.entity_queue = []   # 存储相关记忆
        self.max_memory_length = 1000  # 每类记忆的最大长度
        self.use_event_summary = config.get("use_event_summary", True)  # 新增配置项
        self.enable_memory_recall = config.get("enable_memory_recall", True)  # 新增
        self._logger.debug(f"[ZeroAgent] 初始化完成，角色ID: {self.role_id}, 名称: {config.get('name')}")
        # 初始化系统提示词
        if config.get("system_prompt"):
            processed_prompt = self._process_template(config["system_prompt"])
            self.messages.append(SystemMessage(content=processed_prompt))
            self.config["system_prompt"] = processed_prompt

    async def load_prompt(self) -> str:
        """加载角色提示词"""
        return self.config.get("system_prompt", "")

    async def update_prompt(self, **kwargs) -> str:
        """更新角色提示词和变量"""
        if kwargs:
            if "system_prompt" in kwargs:
                self.config["system_prompt"] = kwargs["system_prompt"]
            
            if "variables" in kwargs:
                self.variables.update(kwargs["variables"])
        
        # 处理模板
        processed_prompt = self._process_template(self.config["system_prompt"])
        self.config["system_prompt"] = processed_prompt
        
        # 更新系统消息
        for i, msg in enumerate(self.messages):
            if isinstance(msg, SystemMessage):
                self.messages[i] = SystemMessage(content=processed_prompt)
                break

        print(f"[ZeroAgent] Updated prompt: {processed_prompt}")  
        return processed_prompt

    async def think(self, context: Dict[str, Any]) -> List[str]:
        """思考是否需要调用工具"""
        return []  # Zero酱暂时不使用工具

    def _build_scene_info(self) -> str:
        """构建场景信息"""
        # 设置时区为UTC+8
        tz = timezone(timedelta(hours=8))
        now = datetime.now(tz)
        
        # 格式化日期时间
        date_str = now.strftime("%Y年%m月%d日")
        time_str = now.strftime("%H:%M")
        weekday_map = {
            0: "一",
            1: "二", 
            2: "三",
            3: "四",
            4: "五",
            5: "六",
            6: "日"
        }
        weekday = f"星期{weekday_map[now.weekday()]}"
        

        
        scene = f"当前时间：{date_str} {time_str} {weekday}"

            
        return scene

    def _sort_memories(self, memories: List[Dict[str, Any]], time_key: str = 'time') -> List[Dict[str, Any]]:
        """对记忆按实际发生时间排序
        
        Args:
            memories: 记忆列表
            time_key: 时间字段的键名
            
        Returns:
            排序后的记忆列表
        """
        try:
            # 只保留有实际时间的记忆，并按时间排序
            valid_memories = [m for m in memories if m.get(time_key)]
            return sorted(valid_memories, key=lambda x: x[time_key], reverse=False)
        except Exception as e:
            self._logger.error(f"[ZeroAgent] 记忆排序失败: {str(e)}")
            return []

    def _process_memories(self, 
                         new_memories: List[Dict[str, Any]], 
                         queue: List[Dict[str, Any]], 
                         time_key: str = 'updatetime') -> List[Dict[str, Any]]:
        """处理记忆
        
        Args:
            new_memories: 新召回的记忆列表
            queue: 现有的记忆队列
            time_key: 时间字段的键名
            
        Returns:
            处理后的记忆列表
        """
        try:
            if not self.use_memory_queue:
                # 无队列模式：直接使用本次召回的记忆，但仍需排序
                return sorted(new_memories, key=lambda x: x.get(time_key, ''), reverse=False)
            
            # 队列模式：合并新旧记忆并处理
            all_memories = queue + new_memories
            
            # 去重（基于description字段）
            unique_memories = []
            seen_descriptions = set()
            for memory in all_memories:
                desc = memory.get('description', '')
                if desc and desc not in seen_descriptions:
                    seen_descriptions.add(desc)
                    unique_memories.append(memory)
            
            # 如果超出队列长度限制，保留最新的N条
            if len(unique_memories) > self.memory_queue_limit:
                unique_memories = unique_memories[-self.memory_queue_limit:]
            
            # 按时间排序
            return sorted(unique_memories, key=lambda x: x.get(time_key, ''), reverse=False)
            
        except Exception as e:
            self._logger.error(f"[ZeroAgent] 处理记忆失败: {str(e)}")
            # 发生错误时，根据模式返回排序后的新记忆或现有队列
            return sorted(new_memories, key=lambda x: x.get(time_key, ''), reverse=False) if not self.use_memory_queue else queue

    def _format_memory_content(self, content: str) -> str:
        """格式化记忆内容，确保不超过最大长度"""
        if len(content) <= self.max_memory_length:
            return content
        return "..." + content[-self.max_memory_length:]

    async def _build_context(self, input_text: str, remark: str = '') -> Dict[str, Any]:
        """构建上下文信息"""
        self._logger.debug(f"[ZeroAgent] 开始构建上下文，输入文本: {input_text}")
        
        if not self.current_user_id:
            raise ValueError("No user_id set for interaction")
        
        # 获取对话记忆
        self._logger.debug("[ZeroAgent] 正在获取对话记忆...")
        recent_messages = await self.memory.get_recent_messages(
            user_id=self.current_user_id,  # 添加user_id
            limit=20
        )
        
        # 获取实体记忆
        self._logger.debug("[ZeroAgent] 正在获取实体记忆...")
        entity_memories = await self.memory.get_entity_memory(self.current_user_id)  # 添加user_id
        
        # 获取场景信息
        scence_info = self._build_scene_info()
        
        # 获取对话概要
        summary = await self.memory.get_processed_memory(self.current_user_id)  # 添加user_id
        
        # 构建查询文本
        query_text = input_text
        if self.use_combined_query:
            last_assistant_msg = ""
            if recent_messages:
                for msg in reversed(recent_messages):
                    if msg.role == "assistant":
                        last_assistant_msg = msg.content
                        break
            query_text = f"{last_assistant_msg} {input_text}".strip()
        
        self._logger.debug(f"[ZeroAgent] 实体记忆查询文本: {query_text}")
        
        # 修改实体记忆查询部分
        if self.enable_memory_recall:
            self._logger.debug("[ZeroAgent] 正在查询实体记忆...")
            entity_memories = await self.memory.query_entity_memory(query_text)
        else:
            self._logger.debug("[ZeroAgent] 记忆召回已禁用")
            entity_memories = {}
        
        self._logger.debug(f"[ZeroAgent] 获取到实体记忆: {json.dumps(entity_memories, ensure_ascii=False)}")

        # 处理实体记忆
        event_memory = ""
        entity_memory = ""
        
        if entity_memories and isinstance(entity_memories, dict):
            self._logger.debug(f"[ZeroAgent] 开始处理实体记忆... 队列模式: {self.use_memory_queue}")
            
            # 事件记忆处理（根据配置决定是否启用）
            if self.use_event_summary and 'memory_events' in entity_memories:
                processed_events = self._process_memories(
                    entity_memories['memory_events'], 
                    self.event_queue,
                    'updatetime'
                )
                if self.use_memory_queue:
                    self.event_queue = processed_events
                
                self._logger.debug(f"[ZeroAgent] 事件处理完成，数量: {len(processed_events)}")
                
                # 格式化事件记忆（仅在启用时处理）
                event_lines = []
                for event in processed_events:
                    event_time = event['updatetime'].split('T')[0] if event.get('updatetime') else ''
                    description = event.get('description', '')
                    deepinsight = event.get('deepinsight', '')
                    memory_line = f"- {event_time}：{description}"
                    if deepinsight:
                        memory_line += f" ({deepinsight})"
                    event_lines.append(memory_line)
                
                event_memory = "\n".join(event_lines)
                event_memory = self._format_memory_content(event_memory)
            
            # 处理记忆实体
            if 'memory_entities' in entity_memories:
                if entity_memories['memory_entities']:
                    processed_entities = self._process_memories(
                        entity_memories['memory_entities'],
                        self.entity_queue,
                        'updatetime'
                    )
                    if self.use_memory_queue:
                        self.entity_queue = processed_entities
                    
                    self._logger.debug(f"[ZeroAgent] 实体处理完成，数量: {len(processed_entities)}")
                    
                    # 格式化实体记忆
                    entity_lines = []
                    for entity in processed_entities:
                        if description := entity.get('description'):
                            entity_time = entity['updatetime'].split('T')[0] if entity.get('updatetime') else ''
                            entity_lines.append(f"- {entity_time} {description}")
                    
                    entity_memory = "\n".join(entity_lines)
                    entity_memory = self._format_memory_content(entity_memory)
        
        summary = ""
        # event_memory = ""
        # entity_memory = ""
        self._logger.debug(f"[ZeroAgent] 处理后的历史事件: {event_memory}")
        self._logger.debug(f"[ZeroAgent] 处理后的相关记忆: {entity_memory}")
                    
        # 更新提示词

        self._logger.debug("[ZeroAgent] 正在更新系统提示词...")
        sys_prompt = self.config["system_prompt"]
        sys_prompt = sys_prompt.replace("{{chat_summary}}", summary or "无")
        sys_prompt = sys_prompt.replace("{{event_memory}}", event_memory or "无")
        sys_prompt = sys_prompt.replace("{{entity_memory}}", entity_memory or "无")
        sys_prompt = sys_prompt.replace("{{scene}}", scence_info or "无")
        
        # 构建消息列表
        self._logger.debug("[ZeroAgent] 正在构建完整消息列表...")
        messages = [{"role": "system", "content": sys_prompt}]
        for msg in recent_messages:
            messages.append({
                "role": msg.role,
                "content": msg.content
            })
        messages.append({
            "role": "user",
            "content": input_text
        })
        
        # 更新 agent_info
        agent_info = {
            "name": self.llm.__class__.__name__,
            "model_name": getattr(self.llm, "model_name", "unknown"),
            "temperature": getattr(self.llm, "temperature", 0.7),
            "max_tokens": getattr(self.llm, "max_tokens", 4096),
            "use_memory_queue": self.use_memory_queue,
            "memory_queue_limit": self.memory_queue_limit,
            "use_combined_query": self.use_combined_query
        }
        
        context = {
            "messages": messages,
            "summary": summary,
            "query_text": query_text,
            "remark": remark,
            "raw_entity_memory": entity_memories,
            "processed_entity_memory": f"历史事件：\n{event_memory}\n相关记忆：\n{entity_memory}",
            "raw_history": recent_messages,
            "processed_history": messages,
            "prompt": sys_prompt,
            "agent_info": agent_info
        }
        
        self._logger.debug("[ZeroAgent] 上下文构建完成")
        return context

    async def _save_interaction(self, 
                              input_text: str,
                              output_text: str,
                              context: Dict[str, Any]) -> None:
        """保存交互记录"""
        self._logger.debug("[ZeroAgent] 开始保存交互记录...")
        
        await self._ensure_db()
        
        if not self.current_user_id:
            raise ValueError("No user_id set for interaction")
            
        # 获取完整的 LLM 信息
        self._logger.debug("[ZeroAgent] 正在获取LLM信息...")
        agent_info = {
            "name": self.llm.__class__.__name__,
            "model_name": getattr(self.llm, "model_name", "unknown"),
            "temperature": getattr(self.llm, "temperature", 0.7),
            "max_tokens": getattr(self.llm, "max_tokens", 4096),
            "use_memory_queue": self.use_memory_queue,
            "memory_queue_limit": self.memory_queue_limit,
            "use_combined_query": self.use_combined_query
        }
        self._logger.debug(f"[ZeroAgent] LLM信息: {json.dumps(agent_info, ensure_ascii=False)}")
        
        # 序列化原始历史消息
        self._logger.debug("[ZeroAgent] 正在序列化历史消息...")
        raw_history = [
            {
                "role": msg.role,
                "content": msg.content
            } for msg in context["raw_history"]
        ]
        
        # 只获取非系统消息的历史记录
        history_messages = [
            msg for msg in context["processed_history"] 
            if msg["role"] != "system"
        ]
        
        data = {
            "user_id": self.current_user_id,
            "input": input_text,
            "output": output_text,
            "query_text": context.get("query_text", ""),
            "remark": context.get("remark", ""),
            "summary": context["summary"],
            "raw_entity_memory": context["raw_entity_memory"],
            "processed_entity_memory": context["processed_entity_memory"],
            "raw_history": raw_history,
            "processed_history": history_messages,
            "prompt": context["prompt"],
            "agent_info": agent_info,
            "timestamp": datetime.now().isoformat()
        }
        
        # 保存到Redis和MySQL
        self._logger.debug("[ZeroAgent] 正在保存到Redis...")
        await self._db.redis.save_chat_record(
            role_id=self.role_id,
            chat_id=self.chat_id,
            user_id=self.current_user_id,
            data=data
        )
        
        self._logger.debug("[ZeroAgent] 正在保存到MySQL...")
        await self._db.mysql.save_chat_record(
            role_id=self.role_id,
            chat_id=self.chat_id,
            user_id=self.current_user_id,
            data=data
        )
        
        self._logger.debug("[ZeroAgent] 交互记录保存完成")

    async def generate_response(self, input_text: str, user_id: str, remark: str = '') -> str:
        """生成回复"""
        self._logger.debug(f"[ZeroAgent] 开始生成回复，输入文本: {input_text}")
        try:
            self.current_user_id = user_id
            self.chat_id = str(uuid.uuid4())
            

            
            # 构建上下文并生成回复
            context = await self._build_context(input_text, remark)
            response = await self.llm.agenerate(context["messages"])
            
            await self.memory.add_message("user", input_text, user_id)
            await self.memory.add_message("assistant", response, user_id)
            await self._save_interaction(input_text, response, context)
            
            return response
            
        except Exception as e:
            self._logger.error(f"[ZeroAgent] 生成回复时出错: {str(e)}")
            raise

    async def astream_response(self, input_text: str, user_id: str, remark: str = '', config: Optional[Dict[str, Any]] = None) -> AsyncIterator[str]:
        """流式生成回复"""
        try:
            self.current_user_id = user_id
            self.chat_id = str(uuid.uuid4())
            
            # 如果有新的配置，更新 agent 配置
            if config:
                self.use_memory_queue = config.get("use_memory_queue", self.use_memory_queue)
                self.use_combined_query = config.get("use_combined_query", self.use_combined_query)
                self.enable_memory_recall = config.get("enable_memory_recall", self.enable_memory_recall)
                self.memory_queue_limit = config.get("memory_queue_limit", self.memory_queue_limit)
                self.use_event_summary = config.get("use_event_summary", self.use_event_summary)
            
            # 构建上下文
            context = await self._build_context(input_text, remark)
            response = ""
            
            # 流式生成回复
            async for chunk in self.llm.astream(context["messages"]):
                response += chunk
                yield chunk
            
            # 只在流式响应完成后保存一次交互记录
            await self.memory.add_message("user", input_text, user_id)
            await self.memory.add_message("assistant", response, user_id)
            await self._save_interaction(input_text, response, context)
            
        except Exception as e:
            self._logger.error(f"[ZeroAgent] 流式对话出错: {str(e)}")
            raise