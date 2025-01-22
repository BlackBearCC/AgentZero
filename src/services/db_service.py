from typing import Optional, AsyncGenerator
from contextlib import asynccontextmanager
from src.utils.redis_client import RedisClient
from src.utils.mysql_client import MySQLClient
from fastapi import Depends

class DBService:
    _instance: Optional['DBService'] = None
    _initialized = False
    
    def __init__(self):
        if not DBService._initialized:
            self.redis = RedisClient()
            self.mysql = MySQLClient()
            DBService._initialized = True
    
    @classmethod
    async def initialize(cls) -> None:
        """初始化数据库连接"""
        if not cls._instance:
            cls._instance = cls()
            await cls._instance.mysql.init_pool()
            if not cls._instance.redis.test_connection():
                raise Exception("Redis connection failed!")
    
    @classmethod
    async def get_instance(cls) -> 'DBService':
        """获取单例实例"""
        if not cls._instance:
            await cls.initialize()
        return cls._instance
    
    @classmethod
    @asynccontextmanager
    async def get_db(cls) -> AsyncGenerator['DBService', None]:
        """获取数据库连接的上下文管理器"""
        if not cls._instance:
            await cls.initialize()
        try:
            yield cls._instance
        except Exception as e:
            print(f"Database error: {str(e)}")
            raise
    
    async def close(self):
        """关闭所有数据库连接"""
        if self.mysql.pool:
            self.mysql.pool.close()
            await self.mysql.pool.wait_closed()

# 依赖注入函数
async def get_db_service() -> DBService:
    async with DBService.get_db() as db:
        yield db 