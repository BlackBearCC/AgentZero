from typing import Optional, AsyncGenerator, List, Dict, Any
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
        """初始化数据库连接和表结构"""
        if not cls._instance:
            cls._instance = cls()
            # 先初始化MySQL连接池
            await cls._instance.mysql.init_pool()
            # 再测试Redis连接
            if not cls._instance.redis.test_connection():
                raise Exception("Redis connection failed!")
            
            # 初始化数据库表结构
            await cls._instance._init_database()
    
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

    async def get_chat_records(self, role_id: str, user_id: str) -> List[Dict[str, Any]]:
        """获取指定用户的聊天记录"""
        return await self.mysql.get_chat_records(role_id=role_id, user_id=user_id)

    async def _init_database(self):
        """初始化或升级数据库表结构"""
        try:
            # 创建或更新聊天记录表
            chat_records_sql = """
            CREATE TABLE IF NOT EXISTS chat_records (
                id BIGINT AUTO_INCREMENT PRIMARY KEY,
                role_id VARCHAR(64) NOT NULL,
                chat_id VARCHAR(64) NOT NULL,
                user_id VARCHAR(64) NOT NULL,
                input TEXT NOT NULL,
                output TEXT NOT NULL,
                query_text TEXT,
                remark TEXT,
                summary TEXT,
                raw_entity_memory JSON,
                processed_entity_memory TEXT,
                raw_history JSON,
                processed_history JSON,
                prompt TEXT,
                agent_info JSON,
                timestamp DATETIME NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                INDEX idx_user_role (user_id, role_id),
                INDEX idx_chat (chat_id)
            ) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
            """
            
            async with self.mysql.pool.acquire() as conn:
                async with conn.cursor() as cur:
                    await cur.execute(chat_records_sql)
                    await conn.commit()
                    
        except Exception as e:
            print(f"数据库初始化失败: {str(e)}")
            raise

# 依赖注入函数
async def get_db_service() -> DBService:
    async with DBService.get_db() as db:
        yield db 