from typing import Any, List, Optional, Dict, AsyncIterator
from langchain_core.callbacks.manager import (
    CallbackManagerForLLMRun,
    AsyncCallbackManagerForLLMRun
)
from langchain_core.language_models.llms import LLM
from langchain_core.outputs import Generation, LLMResult
import aiohttp
import json
import os
from dotenv import load_dotenv

load_dotenv()

class DoubaoLLM(LLM):
    """豆包 API 的自定义封装"""
    
    api_key: str = os.getenv("ARK_API_KEY", "")
    api_base: str = "https://ark.cn-beijing.volces.com/api/v3"
    model_name: str = "ep-20241113173739-b6v4g"
    temperature: float = 0.7
    max_tokens: int = 4096
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if not self.api_key:
            raise ValueError("ARK_API_KEY not found in environment variables")

    @property
    def _llm_type(self) -> str:
        """返回 LLM 类型"""
        return "doubao"
        
    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """返回模型标识参数，用于缓存和追踪"""
        return {
            "model_name": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """同步调用豆包 API"""
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
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """异步调用豆包 API"""
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

    async def _astream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """异步流式调用豆包 API"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": True  # 启用流式输出
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
                
                # 处理流式响应
                async for line in response.content:
                    if line:
                        # 移除 "data: " 前缀并解析 JSON
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

    async def astream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """公开的流式接口"""
        async for chunk in self._astream(prompt, stop, run_manager, **kwargs):
            yield chunk