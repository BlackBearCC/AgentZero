from typing import Dict, Any, List, Optional, AsyncIterator
from src.agents.base_agent import BaseAgent
from langchain.schema import (
    SystemMessage,
    HumanMessage,
    AIMessage,
    BaseMessage
)

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

    async def _build_messages(self, input_text: str) -> List[Dict[str, str]]:
        """构建消息列表"""
        recent_messages = await self.memory.get_recent_messages(limit=20)
                        # 获取最新对话概要
        summary = await self.memory.get_summary()
        
        # 更新提示词中的概要
        sys_prompt = self.config["system_prompt"]
        sys_prompt = sys_prompt.replace("{{chat_summary}}", summary or "无")
        messages = [
            {
                "role": "system",
                "content": sys_prompt
            }
        ]
        
        for msg in recent_messages:
            messages.append({
                "role": msg.role,
                "content": msg.content
            })
        
        messages.append({
            "role": "user",
            "content": input_text
        })
        
        # 打印调试信息
        print("\n当前对话上下文:")
        for msg in messages:
            print(f"\n{msg['role']}: {msg['content']}")
        print("\n" + "="*50 + "\n")
        
        return messages

    async def generate_response(self, input_text: str) -> str:
        """生成回复"""
        try:
            messages = await self._build_messages(input_text)
            response = await self.llm.agenerate(messages)
            
            # 更新对话历史
            await self.memory.add_message("user", input_text)
            await self.memory.add_message("assistant", response)
            
            return response
            
        except Exception as e:
            self._logger.error(f"Error generating response: {str(e)}")
            raise
            
    async def astream_response(self, input_text: str) -> AsyncIterator[str]:
        """流式生成回复"""
        try:
            messages = await self._build_messages(input_text)
            
            response = ""
            async for chunk in self.llm.astream(messages):
                response += chunk
                yield chunk
                
            # 更新对话历史
            await self.memory.add_message("user", input_text)
            await self.memory.add_message("assistant", response)
                
        except Exception as e:
            self._logger.error(f"ZeroAgent stream error: {str(e)}")
            raise