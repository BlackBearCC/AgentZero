from dataclasses import dataclass
from datetime import datetime

@dataclass
class Message:
    """基础消息数据结构"""
    role: str
    content: str
    timestamp: datetime = datetime.now() 