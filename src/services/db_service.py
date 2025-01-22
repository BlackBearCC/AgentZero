from typing import Optional
from src.utils.redis_client import RedisClient
from src.utils.mysql_client import MySQLClient

class DBService:
    _instance: Optional['DBService'] = None
    
    def __init__(self):
        self.redis = RedisClient()
        self.mysql = MySQLClient()
        
    @classmethod
    async def get_instance(cls) -> 'DBService':
        if not cls._instance:
            cls._instance = cls()
            # 初始化连接池
            await cls._instance.mysql.init_pool()
        return cls._instance
        
    async def close(self):
        """关闭所有数据库连接"""
        if self.mysql.pool:
            self.mysql.pool.close()
            await self.mysql.pool.wait_closed() 