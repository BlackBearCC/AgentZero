from typing import Any, List, Optional, Dict, AsyncIterator
from langchain_core.callbacks.manager import (
    CallbackManagerForLLMRun,
    AsyncCallbackManagerForLLMRun
)
from langchain_core.language_models.llms import LLM
from langchain_core.outputs import Generation, LLMResult
import aiohttp
import requests
import json
import os
from dotenv import load_dotenv

load_dotenv()

class DeepSeekLLM(LLM):
    """DeepSeek API 的自定义封装"""
    
    api_key: str = os.getenv("DEEPSEEK_API_KEY", "")
    api_base: str = "https://api.deepseek.com/v1"
    model_name: str = "deepseek-chat"
    temperature: float = 0.7
    max_tokens: int = 4096
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if not self.api_key:
            raise ValueError("DEEPSEEK_API_KEY not found in environment variables")

    @property
    def _llm_type(self) -> str:
        """返回 LLM 类型"""
        return "deepseek"
        
    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """返回模型标识参数，用于缓存和追踪"""
        return {
            "model_name": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }

    def _process_messages(self, prompt: str | List[Dict[str, str]]) -> List[Dict[str, str]]:
        """处理输入消息格式
        
        Args:
            prompt: 字符串或消息列表
            
        Returns:
            List[Dict[str, str]]: 标准化的消息列表
        """
        if isinstance(prompt, str):
            return [{"role": "user", "content": prompt}]
        elif isinstance(prompt, (list, tuple)):
            # 验证消息格式
            messages = []
            for msg in prompt:
                if isinstance(msg, dict) and "role" in msg and "content" in msg:
                    if msg["role"] not in ["system", "user", "assistant"]:
                        raise ValueError(f"Invalid role: {msg['role']}")
                    messages.append({
                        "role": msg["role"],
                        "content": str(msg["content"])
                    })
                else:
                    raise ValueError("Invalid message format")
            return messages
        else:
            raise ValueError("Prompt must be string or list of messages")

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """同步调用 DeepSeek API"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }
        
        if stop:
            data["stop"] = stop
            
        response = requests.post(
            f"{self.api_base}/chat/completions",
            headers=headers,
            json=data
        )
        
        if response.status_code != 200:
            raise Exception(f"API call failed: {response.text}")
            
        return response.json()["choices"][0]["message"]["content"]

    async def _acall(
        self,
        prompt: str | List[Dict[str, str]],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """异步调用 DeepSeek API"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        messages = self._process_messages(prompt)
        
        data = {
            "model": self.model_name,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }
        
        if stop:
            data["stop"] = stop
            
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.api_base}/chat/completions",
                headers=headers,
                json=data
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"API call failed: {error_text}")
                    
                result = await response.json()
                return result["choices"][0]["message"]["content"]

    def _generate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> LLMResult:
        """批量生成回复"""
        generations = []
        for prompt in prompts:
            text = self._call(prompt, stop=stop, run_manager=run_manager, **kwargs)
            generations.append([Generation(text=text)])
        return LLMResult(generations=generations)

    async def agenerate(
        self,
        prompts: List[str] | List[List[Dict[str, str]]],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> LLMResult:
        """异步生成回复"""
        generations = []
        for prompt in prompts:
            text = await self._acall(prompt, stop=stop, run_manager=run_manager, **kwargs)
            generations.append([Generation(text=text)])
        return LLMResult(generations=generations)

    async def astream(
        self,
        prompt: str | List[Dict[str, str]],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[Generation]:
        """异步流式生成回复"""
        async for chunk in self._astream(prompt, stop=stop, run_manager=run_manager, **kwargs):
            yield Generation(text=chunk)

    async def _astream(
        self,
        prompt: str | List[Dict[str, str]],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """异步流式调用 DeepSeek API"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        messages = self._process_messages(prompt)
        
        data = {
            "model": self.model_name,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": True
        }
        
        if stop:
            data["stop"] = stop
            
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.api_base}/chat/completions",
                headers=headers,
                json=data
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"API call failed: {error_text}")
                
                async for line in response.content:
                    if line:
                        line = line.decode('utf-8').strip()
                        if line.startswith("data: "):
                            line = line[6:]
                        if line:
                            try:
                                chunk = json.loads(line)
                                if chunk and chunk["choices"] and chunk["choices"][0]["delta"].get("content"):
                                    content = chunk["choices"][0]["delta"]["content"]
                                    yield content
                            except json.JSONDecodeError:
                                continue 