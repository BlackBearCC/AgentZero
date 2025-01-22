from typing import Dict, Any, List, Optional, AsyncIterator
from src.agents.base_agent import BaseAgent
from langchain.schema import (
    SystemMessage,
    HumanMessage,
    AIMessage,
    BaseMessage
)
import json
from datetime import datetime

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

    async def _build_context(self, input_text: str) -> Dict[str, Any]:
        """构建上下文信息
        
        Returns:
            Dict 包含:
            - messages: 消息列表
            - summary: 对话概要
            - entity_memory: 实体记忆
            - history: 历史记录
        """
        # 获取最新对话概要和历史消息
        recent_messages = await self.memory.get_recent_messages(limit=20)
        summary = await self.memory.get_summary()
        entity_memories = await self.memory.query_entity_memory(input_text)
        
        # 处理实体记忆
        entity_context = ""
        if entity_memories and isinstance(entity_memories, dict):
            # 处理记忆事件
            if 'memory_events' in entity_memories:
                entity_context += "历史事件：\n"
                for event in entity_memories['memory_events']:
                    update_time = event.get('updatetime', '')
                    description = event.get('description', '')
                    deepinsight = event.get('deepinsight', '')
                    entity_context += f"- {update_time}：{description}"
                    if deepinsight:
                        entity_context += f" ({deepinsight})"
                    entity_context += "\n"
            
            # 处理记忆实体        
            if 'memory_entities' in entity_memories:
                if entity_memories['memory_entities']:
                    entity_context += "\n相关记忆：\n"
                    for entity in entity_memories['memory_entities']:
                        description = entity.get('description', '')
                        update_time = entity.get('updatetime', '').split('T')[0]
                        if description:
                            entity_context += f"- {update_time} {description}\n"
                else:
                    entity_context += "- 无\n"
                    
        # 更新提示词
        sys_prompt = self.config["system_prompt"]
        sys_prompt = sys_prompt.replace("{{chat_summary}}", summary or "无")
        sys_prompt = sys_prompt.replace("{{entity_memory}}", entity_context or "无")
        
        # 构建消息列表
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
        
        return {
            "messages": messages,
            "summary": summary,
            "entity_memory": entity_memories,
            "history": messages,
            "prompt": sys_prompt
        }

    async def _save_interaction(self, 
                              input_text: str,
                              output_text: str,
                              context: Dict[str, Any]) -> None:
        """保存交互记录"""
        data = {
            "input": input_text,
            "output": output_text,
            "summary": context["summary"],
            "entity_memory": context["entity_memory"],
            "history": context["history"],
            "prompt": context["prompt"],
            "timestamp": datetime.now().isoformat()
        }
        
        await self.redis.save_chat_record(
            role_id=self.role_id,
            chat_id=self.chat_id,
            data=data
        )
        
        # 更新对话历史
        await self.memory.add_message("user", input_text)
        await self.memory.add_message("assistant", output_text)

    async def generate_response(self, input_text: str) -> str:
        """生成回复"""
        try:
            context = await self._build_context(input_text)
            response = await self.llm.agenerate(context["messages"])
            await self._save_interaction(input_text, response, context)
            return response
            
        except Exception as e:
            self._logger.error(f"Error generating response: {str(e)}")
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