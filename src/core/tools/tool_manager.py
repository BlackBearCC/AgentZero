from typing import Dict, Any, Callable
from functools import wraps

class ToolManager:
    def __init__(self):
        self.tools: Dict[str, Callable] = {}
    
    def register(self, name: str):
        def decorator(func: Callable):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                return await func(*args, **kwargs)
            self.tools[name] = wrapper
            return wrapper
        return decorator
    
    async def execute(self, tool_name: str, **kwargs) -> Any:
        if tool_name not in self.tools:
            raise ValueError(f"Tool {tool_name} not found")
        return await self.tools[tool_name](**kwargs) 