from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, AsyncIterator, Tuple
from langchain.schema import (
    SystemMessage,
    HumanMessage,
    AIMessage,
    BaseMessage
)
from langchain.prompts import ChatPromptTemplate
from src.utils.logger import Logger
from src.llm.doubao import DoubaoLLM
import asyncio
import json
from string import Template
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime

@dataclass
class Message:
    role: str
    content: str
    timestamp: datetime = datetime.now()

class Memory:
    def __init__(self, llm=None, max_history: int = 20, min_recent: int = 5):
        self.chat_history: List[Message] = []
        self.summary: str = ""  # 对话概要
        self.max_history = max_history
        self.min_recent = min_recent
        self.llm = llm
        
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
            print(f"完整对话内容: {new_lines}")
            # 构建完整的提示词
            full_prompt = f"""你是一个对话概要分析师。请基于当前概要进行渐进式更新，生成新的对话概要。

要求：
1. 篇幅控制在500字以内,以一段内容形式展示（禁用列表/枚举或其它格式）
2. 渐进式更新原则：
   - 保留当前概要中的关键信息
   - 将新对话内容自然融入现有脉络
   - 对重复出现的主题进行合并和提炼
   - 确保新旧信息的时间顺序连贯
   - 适当压缩旧内容但保留重要节点

3. 按时间顺序记录重要事件和情感变化

4. 使用"USER"和"AI"来指代对话双方

4. 重点记录：
   - 关键话题转换
   - 重要的关系发展节点
   - 具体的行为和承诺
   - 特殊的称呼和关系
   - USER的行为习惯和情感需求
   - 双方的默契和特殊互动
   - 记录有助于AI延续对话的信息

5. 去除重复内容和和无具体信息的客套对话
6. 记录具体明确的事件实体概念相关的内容，忽视空泛或抽象内容
7. 对早期内容进行高度概括但不是删除，保持新内容的细节
8. 只客观记录概要，避免额外描述和修饰

当前概要：
{self.summary or '对话开始'}

新的对话内容：
{new_lines}

请基于以上原则，生成一个连贯的新概要："""

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

                summary = summary.replace("USER", "琦琦").replace("AI", "祁煜")
                self.summary = summary
                      
                # 更新历史记录
                self.chat_history = recent_messages
                print(f"完整结果: {self.summary}")

            except Exception as e:
                print(f"生成概要时出错: {str(e)}")
                return
                
    async def get_full_memory(self) -> Tuple[str, List[Message]]:
        """获取完整上下文（概要 + 最近消息）"""
        return self.summary, self.chat_history
        
    async def get_recent_messages(self, limit: int = 10) -> List[Message]:
        """获取最近的消息"""
        return self.chat_history[-limit:]
        
    async def update_summary(self, new_content: str):
        """更新对话概要"""
        # TODO: 实现对话概要更新逻辑
        pass
        
    async def query_entity_memory(self, query: str) -> List[Dict[str, Any]]:
        """查询实体记忆"""
        # TODO: 实现 RAG 检索逻辑
        pass

class BaseAgent(ABC):
    def __init__(self, 
                 config: Dict[str, Any],
                 llm=None,
                 memory=None,
                 tools=None):
        """
        初始化基础组件

        参数:
        - config: 一个字典，包含配置信息，如名称、角色ID和变量等
        - llm: 语言模型实例，默认为None
        - memory: 存储记忆的实例，默认为None
        - tools: 一个工具列表，默认为空列表
        """
        self.config = config
        self.name = config.get("name", "Assistant")
        self.role_id = config.get("role_id")
        self.variables = config.get("variables", {})
        
        self.llm = llm or DoubaoLLM(
            model_name="ep-20241113173739-b6v4g",
            temperature=0.7,
            max_tokens=4096
        )

        # 将 llm 传给 Memory
        # self.memory = memory 
        self.memory =  Memory(memory)
        self.tools = tools or []
        self._logger = Logger()
        self.messages: List[BaseMessage] = []
        
        # 初始化系统提示词
        if config.get("system_prompt"):
            processed_prompt = self._process_template(config["system_prompt"])
            self.messages.append(SystemMessage(content=processed_prompt))
            self.config["system_prompt"] = processed_prompt
    
    def _process_template(self, template: str) -> str:
        """处理提示词模板，替换变量"""
        try:
            for key, value in self.variables.items():
                template = template.replace(f"{{{{{key}}}}}", value)
            return template
        except Exception as e:
            self._logger.error(f"Error processing template: {str(e)}")
            return template
    
    @abstractmethod
    async def load_prompt(self) -> str:
        """加载角色提示词"""
        pass
        
    @abstractmethod
    async def update_prompt(self, **kwargs) -> str:
        """更新角色提示词"""
        pass
        
    @abstractmethod
    async def think(self, context: Dict[str, Any]) -> List[str]:
        """思考是否需要调用工具"""
        pass
        
    @abstractmethod
    async def generate_response(self, input_text: str) -> str:
        """生成回复"""
        pass
        
    @abstractmethod
    async def astream_response(self, input_text: str) -> AsyncIterator[str]:
        """流式生成回复"""
        pass

    async def _process_memory(self, input_text: str) -> Tuple[str, List[Dict[str, Any]]]:
        """处理记忆并生成上下文"""
        # 获取最近对话历史
        recent_messages = await self.memory.get_recent_messages()
        
        # 获取当前对话概要
        current_summary = self.memory.summary
        
        # 查询相关实体记忆
        relevant_memories = await self.memory.query_entity_memory(input_text)
        
        # 构建上下文
        context = {
            "recent_messages": recent_messages,
            "summary": current_summary,
            "relevant_memories": relevant_memories
        }
        
        return context