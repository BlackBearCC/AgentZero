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

    async def _build_context(self, input_text: str) -> Dict[str, Any]:
        """构建上下文信息
        
        Returns:
            Dict 包含:
            - messages: 消息列表
            - summary: 对话概要
            - entity_memory: 实体记忆
            - history: 历史记录
        """
        self._logger.debug(f"[ZeroAgent] 开始构建上下文，输入文本: {input_text}")
        
        # 获取最新对话概要和历史消息
        self._logger.debug("[ZeroAgent] 正在获取对话记忆...")
        recent_messages = await self.memory.get_recent_messages(limit=20)
        summary = await self.memory.get_summary()
        self._logger.debug(f"[ZeroAgent] 获取到对话概要: {summary}")
        self._logger.debug(f"[ZeroAgent] 获取到历史消息数量: {len(recent_messages)}")
        
        # 查询实体记忆
        self._logger.debug("[ZeroAgent] 正在查询实体记忆...")
        entity_memories = await self.memory.query_entity_memory(input_text)
        self._logger.debug(f"[ZeroAgent] 获取到实体记忆: {json.dumps(entity_memories, ensure_ascii=False)}")

        scence_info  = self._build_scene_info()
        
        # 处理实体记忆
        processed_entity_context = ""
        if entity_memories and isinstance(entity_memories, dict):
            self._logger.debug("[ZeroAgent] 开始处理实体记忆...")
            # 处理记忆事件
            if 'memory_events' in entity_memories:
                processed_entity_context += "历史事件：\n"
                for event in entity_memories['memory_events']:
                    update_time = event.get('updatetime', '')
                    description = event.get('description', '')
                    deepinsight = event.get('deepinsight', '')
                    processed_entity_context += f"- {update_time}：{description}"
                    if deepinsight:
                        processed_entity_context += f" ({deepinsight})"
                    processed_entity_context += "\n"
            
            # 处理记忆实体        
            if 'memory_entities' in entity_memories:
                if entity_memories['memory_entities']:
                    processed_entity_context += "\n相关记忆：\n"
                    for entity in entity_memories['memory_entities']:
                        description = entity.get('description', '')
                        update_time = entity.get('updatetime', '').split('T')[0]
                        if description:
                            processed_entity_context += f"- {update_time} {description}\n"
                else:
                    processed_entity_context += "- 无\n"
        
        self._logger.debug(f"[ZeroAgent] 处理后的实体记忆: {processed_entity_context}")
                    
        # 更新提示词
        self._logger.debug("[ZeroAgent] 正在更新系统提示词...")
        sys_prompt = self.config["system_prompt"]
        sys_prompt = sys_prompt.replace("{{chat_summary}}", summary or "无")
        sys_prompt = sys_prompt.replace("{{entity_memory}}", processed_entity_context or "无")
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
        
        context = {
            "messages": messages,
            "summary": summary,
            "raw_entity_memory": entity_memories,
            "processed_entity_memory": processed_entity_context,
            "raw_history": recent_messages,
            "processed_history": messages,
            "prompt": sys_prompt
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
        
        # 获取 LLM 信息
        self._logger.debug("[ZeroAgent] 正在获取LLM信息...")
        llm_info = {
            "name": self.llm.__class__.__name__,
            "model_name": getattr(self.llm, "model_name", "unknown"),
            "temperature": getattr(self.llm, "temperature", 0.7),
            "max_tokens": getattr(self.llm, "max_tokens", 4096)
        }
        self._logger.debug(f"[ZeroAgent] LLM信息: {json.dumps(llm_info, ensure_ascii=False)}")
        
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
            "input": input_text,
            "output": output_text,
            "summary": context["summary"],
            "raw_entity_memory": context["raw_entity_memory"],
            "processed_entity_memory": context["processed_entity_memory"],
            "raw_history": raw_history,
            "processed_history": history_messages,
            "prompt": context["prompt"],
            "llm_info": llm_info,
            "timestamp": datetime.now().isoformat()
        }
        
        # 保存到Redis和MySQL
        self._logger.debug("[ZeroAgent] 正在保存到Redis...")
        await self._db.redis.save_chat_record(
            role_id=self.role_id,
            chat_id=self.chat_id,
            data=data
        )
        
        self._logger.debug("[ZeroAgent] 正在保存到MySQL...")
        await self._db.mysql.save_chat_record(
            role_id=self.role_id,
            chat_id=self.chat_id,
            data=data
        )
        
        # 更新对话历史
        self._logger.debug("[ZeroAgent] 正在更新对话历史...")
        await self.memory.add_message("user", input_text)
        await self.memory.add_message("assistant", output_text)
        
        self._logger.debug("[ZeroAgent] 交互记录保存完成")

    async def generate_response(self, input_text: str) -> str:
        """生成回复"""
        self._logger.debug(f"[ZeroAgent] 开始生成回复，输入文本: {input_text}")
        try:
            context = await self._build_context(input_text)
            self._logger.debug("[ZeroAgent] 正在调用LLM生成回复...")
            response = await self.llm.agenerate(context["messages"])
            self._logger.debug(f"[ZeroAgent] LLM返回响应: {response}")
            
            await self._save_interaction(input_text, response, context)
            self._logger.debug("[ZeroAgent] 回复生成完成")
            return response
            
        except Exception as e:
            self._logger.error(f"[ZeroAgent] 生成回复时出错: {str(e)}")
            raise
            
    async def astream_response(self, input_text: str) -> AsyncIterator[str]:
        """流式生成回复"""
        try:
            context = await self._build_context(input_text)
            response = ""
            
            async for chunk in self.llm.astream(context["messages"]):
                response += chunk
                yield chunk
                
            await self._save_interaction(input_text, response, context)
            
            # 更新对话历史
            await self.memory.add_message("user", input_text)
            await self.memory.add_message("assistant", response)
                
        except Exception as e:
            self._logger.error(f"ZeroAgent stream error: {str(e)}")
            raise