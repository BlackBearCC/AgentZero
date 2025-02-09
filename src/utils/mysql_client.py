import os
from typing import Dict, Any, List, Optional
import aiomysql
import json
from datetime import datetime

from src.utils.logger import Logger


class MySQLClient:
    def __init__(self):
        self.pool = None
        self._logger = Logger()
        
    async def init_pool(self):
        """初始化连接池"""
        try:
            if not self.pool:
                self.pool = await aiomysql.create_pool(
                    host=os.getenv('MYSQL_HOST', 'mysql'),
                    port=int(os.getenv('MYSQL_PORT', 3306)),
                    user=os.getenv('MYSQL_USER', 'root'),
                    password=os.getenv('MYSQL_PASSWORD', 'Jing91101'),
                    db=os.getenv('MYSQL_DATABASE', 'agent_zero'),
                    charset='utf8mb4',
                    autocommit=True
                )
                self._logger.debug("MySQL连接池初始化成功")
        except Exception as e:
            self._logger.error(f"MySQL连接池初始化失败: {str(e)}")
            raise
    
    async def _ensure_pool(self):
        """确保连接池已初始化"""
        if not self.pool:
            await self.init_pool()
    
    async def save_chat_record(
        self,
        role_id: str,
        chat_id: str,
        user_id: str,
        data: Dict[str, Any]
    ) -> bool:
        """保存聊天记录"""
        try:
            # 确保连接池已初始化
            await self._ensure_pool()
            
            sql = """
            INSERT INTO chat_records (
                role_id, chat_id, user_id, input, output,
                query_text, remark, summary, 
                raw_entity_memory, processed_entity_memory,
                raw_history, processed_history, 
                prompt, agent_info, timestamp
            ) VALUES (
                %s, %s, %s, %s, %s, 
                %s, %s, %s, 
                %s, %s,
                %s, %s, 
                %s, %s, %s
            )
            """
            
            async with self.pool.acquire() as conn:
                async with conn.cursor() as cur:
                    await cur.execute(sql, (
                        role_id,
                        chat_id,
                        user_id,
                        data['input'],
                        data['output'],
                        data.get('query_text', ''),
                        data.get('remark', ''),
                        data.get('summary', ''),
                        json.dumps(data.get('raw_entity_memory', {}), ensure_ascii=False),
                        data.get('processed_entity_memory', ''),
                        json.dumps(data.get('raw_history', []), ensure_ascii=False),
                        json.dumps(data.get('processed_history', []), ensure_ascii=False),
                        data.get('prompt', ''),
                        json.dumps(data.get('agent_info', {}), ensure_ascii=False),
                        data.get('timestamp', datetime.now().isoformat())
                    ))
                    await conn.commit()
                    self._logger.debug("MySQL记录保存成功")
                    return True
                    
        except Exception as e:
            self._logger.error(f"MySQL保存记录失败: {str(e)}")
            return False
    
    async def get_chat_records(
        self,
        role_id: str,
        user_id: str,
        chat_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """获取聊天记录"""
        if not self.pool:
            await self.init_pool()
            
        async with self.pool.acquire() as conn:
            async with conn.cursor() as cur:
                try:
                    sql = """
                    SELECT * FROM chat_records 
                    WHERE role_id = %s AND user_id = %s
                    """
                    params = [role_id, user_id]
                    if chat_id:
                        sql += " AND chat_id = %s"
                        params.append(chat_id)
                    
                    await cur.execute(sql, params)
                    records = await cur.fetchall()
                    
                    result = []
                    for record in records:
                        result.append({
                            'id': record[0],
                            'role_id': record[1],
                            'chat_id': record[2],
                            'input': record[3],
                            'output': record[4],
                            'summary': record[5],
                            'raw_entity_memory': json.loads(record[6]) if record[6] else {},
                            'processed_entity_memory': record[7],
                            'raw_history': json.loads(record[8]) if record[8] else [],
                            'processed_history': json.loads(record[9]) if record[9] else [],
                            'prompt': record[10],
                            'agent_info': json.loads(record[11]) if record[11] else {},
                            'created_at': record[12].isoformat()
                        })
                    
                    return result
                    
                except Exception as e:
                    print(f"MySQL get error: {str(e)}")
                    return [] 