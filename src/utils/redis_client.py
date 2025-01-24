import json
from typing import Dict, Any, Optional
import redis
from datetime import datetime
import os

class RedisClient:
    def __init__(self):
        """初始化 Redis 客户端"""
        self.redis = redis.Redis(
            host=os.getenv('REDIS_HOST', 'redis'),
            port=int(os.getenv('REDIS_PORT', 6379)),
            decode_responses=True  # 确保返回的数据是解码后的
        )
        
    async def save_chat_record(self, 
                             role_id: str,
                             chat_id: str,
                             data: Dict[str, Any],
                             expire: int = 60 * 60 * 24 * 7):  # 默认保存7天
        """保存聊天记录
        
        Args:
            role_id: 角色ID
            chat_id: 对话ID
            data: 聊天数据
            expire: 过期时间(秒)
        """
        key = f"chat:{role_id}:{chat_id}"
        
        # # 添加调试日志
        # print(f"Saving to Redis - Key: {key}")
        # print(f"Data: {json.dumps(data, ensure_ascii=False, indent=2)}")
        
        try:
            # 处理数据序列化，使用ensure_ascii=False确保中文正常显示
            processed_data = {}
            for k, v in data.items():
                if isinstance(v, (dict, list)):
                    processed_data[k] = json.dumps(v, ensure_ascii=False)
                else:
                    processed_data[k] = str(v)
            
            # 保存到Redis
            self.redis.hset(key, mapping=processed_data)
            self.redis.expire(key, expire)
            
            
        except Exception as e:
            print(f"Redis save error: {str(e)}")
            raise  # 抛出异常以便追踪问题
            
    async def get_chat_record(self, role_id: str, chat_id: str) -> Optional[Dict[str, Any]]:
        """获取聊天记录"""
        key = f"chat:{role_id}:{chat_id}"
        try:
            data = self.redis.hgetall(key)
            if not data:
                return None
                
            # 反序列化JSON字符串
            result = {}
            for k, v in data.items():
                try:
                    result[k] = json.loads(v)
                except json.JSONDecodeError:
                    result[k] = v
            return result
            
        except Exception as e:
            print(f"Redis get error: {str(e)}")
            return None 

    def test_connection(self):
        """测试 Redis 连接"""
        try:
            self.redis.ping()
            print("Redis connection successful!")
            # 写入测试数据
            test_key = "test:connection"
            self.redis.set(test_key, "test_value")
            test_value = self.redis.get(test_key)
            print(f"Test value retrieved: {test_value}")
            self.redis.delete(test_key)
            return True
        except Exception as e:
            print(f"Redis connection error: {str(e)}")
            return False 