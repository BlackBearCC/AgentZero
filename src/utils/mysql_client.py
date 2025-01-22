import os
from typing import Dict, Any, List, Optional
import aiomysql
import json
from datetime import datetime

class MySQLClient:
    def __init__(self):
        self.pool = None
        
    async def init_pool(self):
        """初始化连接池"""
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
    
    async def save_chat_record(self,
                             role_id: str,
                             chat_id: str,
                             data: Dict[str, Any]) -> Optional[int]:
        """保存聊天记录"""
        if not self.pool:
            await self.init_pool()
            
        async with self.pool.acquire() as conn:
            async with conn.cursor() as cur:
                try:
                    # 插入主记录
                    sql = """
                    INSERT INTO chat_records 
                    (role_id, chat_id, input_text, output_text, summary, prompt)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    """
                    await cur.execute(sql, (
                        role_id,
                        chat_id,
                        data['input'],
                        data['output'],
                        data['summary'],
                        data['prompt']
                    ))
                    record_id = cur.lastrowid
                    
                    # 保存实体记忆
                    if data.get('entity_memory'):
                        await cur.execute(
                            """
                            INSERT INTO entity_memories 
                            (chat_record_id, entity_type, entity_data)
                            VALUES (%s, %s, %s)
                            """,
                            (record_id, 'memory', json.dumps(data['entity_memory']))
                        )
                    
                    # 保存对话历史
                    if data.get('history'):
                        await cur.execute(
                            """
                            INSERT INTO chat_histories 
                            (chat_record_id, history_data)
                            VALUES (%s, %s)
                            """,
                            (record_id, json.dumps(data['history']))
                        )
                    
                    return record_id
                    
                except Exception as e:
                    print(f"MySQL save error: {str(e)}")
                    raise
    
    async def get_chat_records(self,
                             role_id: str,
                             chat_id: str) -> List[Dict[str, Any]]:
        """获取聊天记录"""
        if not self.pool:
            await self.init_pool()
            
        async with self.pool.acquire() as conn:
            async with conn.cursor() as cur:
                try:
                    # 获取主记录
                    await cur.execute(
                        """
                        SELECT * FROM chat_records 
                        WHERE role_id = %s AND chat_id = %s
                        ORDER BY created_at DESC
                        """,
                        (role_id, chat_id)
                    )
                    records = await cur.fetchall()
                    
                    result = []
                    for record in records:
                        record_id = record[0]
                        
                        # 获取实体记忆
                        await cur.execute(
                            "SELECT entity_data FROM entity_memories WHERE chat_record_id = %s",
                            (record_id,)
                        )
                        memory = await cur.fetchone()
                        
                        # 获取对话历史
                        await cur.execute(
                            "SELECT history_data FROM chat_histories WHERE chat_record_id = %s",
                            (record_id,)
                        )
                        history = await cur.fetchone()
                        
                        result.append({
                            'id': record[0],
                            'role_id': record[1],
                            'chat_id': record[2],
                            'input': record[3],
                            'output': record[4],
                            'summary': record[5],
                            'prompt': record[6],
                            'created_at': record[7].isoformat(),
                            'entity_memory': json.loads(memory[0]) if memory else None,
                            'history': json.loads(history[0]) if history else None
                        })
                    
                    return result
                    
                except Exception as e:
                    print(f"MySQL get error: {str(e)}")
                    return [] 