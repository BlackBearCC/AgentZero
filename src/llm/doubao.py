from typing import Any, List, Optional, Dict, Iterator
from langchain_core.callbacks.manager import (
    CallbackManagerForLLMRun,
    AsyncCallbackManagerForLLMRun
)
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    ChatMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.outputs import ChatResult, ChatGeneration, ChatGenerationChunk
from langchain_core.messages import AIMessage as AIMessageChunk
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

class DoubaoChatModel(BaseChatModel):
    """豆包 API 的自定义封装"""
    
    client: Any = None
    api_key: str = ""
    url: str = "https://ark.cn-beijing.volces.com/api/v3"
    model: str = "ep-20241113173739-b6v4g"
    temperature: float = 0.7
    max_tokens: int = 4096

    def __init__(
        self,
        api_key: Optional[str] = None,
        url: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        self.api_key = api_key or os.getenv("ARK_API_KEY", "")
        self.url = url or os.getenv("DOUBAO_API_URL", self.url)
        self.model = model or os.getenv("DOUBAO_MODEL_ID", self.model)
        
        if temperature is not None:
            self.temperature = temperature
        if max_tokens is not None:
            self.max_tokens = max_tokens
            
        if not self.api_key:
            raise ValueError("ARK_API_KEY not found")
        self.client = self._create_client()

    def _create_client(self) -> OpenAI:
        """创建 OpenAI 客户端"""
        return OpenAI(
            api_key=self.api_key,
            base_url=self.url
        )

    @property
    def _llm_type(self) -> str:
        """返回 LLM 类型"""
        return "doubao"

    def _convert_messages_to_prompt(self, messages: List[BaseMessage]) -> List[Dict[str, str]]:
        """转换消息格式"""
        formatted_messages = []
        for message in messages:
            if isinstance(message, SystemMessage):
                formatted_messages.append({"role": "system", "content": message.content})
            elif isinstance(message, HumanMessage):
                formatted_messages.append({"role": "user", "content": message.content})
            elif isinstance(message, AIMessage):
                formatted_messages.append({"role": "assistant", "content": message.content})
            else:
                raise ValueError(f"Got unknown message type: {message}")
        return formatted_messages

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """生成回复"""
        try:
            formatted_messages = self._convert_messages_to_prompt(messages)
            if not any(msg["role"] == "system" for msg in formatted_messages):
                formatted_messages.insert(0, {
                    "role": "system",
                    "content": "你是豆包，是由字节跳动开发的 AI 人工智能助手"
                })

            completion = self.client.chat.completions.create(
                model=self.model,
                messages=formatted_messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stream=False
            )
            
            if completion.choices and completion.choices[0].message.content:
                return ChatResult(
                    generations=[
                        ChatGeneration(
                            message=AIMessage(content=completion.choices[0].message.content),
                            generation_info=dict(finish_reason=completion.choices[0].finish_reason)
                        )
                    ]
                )
            else:
                raise ValueError("No content in response")
                
        except Exception as e:
            raise Exception(f"Error calling Doubao API: {str(e)}")

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """异步生成回复"""
        return self._generate(messages, stop, run_manager, **kwargs)

    def stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """流式生成回复"""
        try:
            formatted_messages = self._convert_messages_to_prompt(messages)
            if not any(msg["role"] == "system" for msg in formatted_messages):
                formatted_messages.insert(0, {
                    "role": "system",
                    "content": "你是豆包，是由字节跳动开发的 AI 人工智能助手"
                })

            stream = self.client.chat.completions.create(
                model=self.model,
                messages=formatted_messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stream=True
            )
            
            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield ChatGenerationChunk(
                        message=AIMessageChunk(content=chunk.choices[0].delta.content),
                        generation_info=dict(finish_reason=chunk.choices[0].finish_reason)
                    )
                    
        except Exception as e:
            raise Exception(f"Error calling Doubao API: {str(e)}") 