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
        """保存聊天记录到单一表"""
        if not self.pool:
            await self.init_pool()
            
        async with self.pool.acquire() as conn:
            async with conn.cursor() as cur:
                try:
                    sql = """
                    INSERT INTO chat_records 
                    (role_id, chat_id, input_text, output_text, summary, 
                     entity_memory, history_messages, prompt, llm_info)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """
                    
                    # 序列化 JSON 字段
                    entity_memory_json = json.dumps(data.get('entity_memory', {}), ensure_ascii=False)
                    history_messages_json = json.dumps(data.get('history_messages', []), ensure_ascii=False)
                    llm_info_json = json.dumps(data.get('llm_info', {}), ensure_ascii=False)
                    
                    await cur.execute(sql, (
                        role_id,
                        chat_id,
                        data['input'],
                        data['output'],
                        data.get('summary', ''),
                        entity_memory_json,
                        history_messages_json,
                        data.get('prompt', ''),
                        llm_info_json
                    ))
                    
                    return cur.lastrowid
                    
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
                        result.append({
                            'id': record[0],
                            'role_id': record[1],
                            'chat_id': record[2],
                            'input': record[3],
                            'output': record[4],
                            'summary': record[5],
                            'entity_memory': json.loads(record[6]) if record[6] else {},
                            'history_messages': json.loads(record[7]) if record[7] else [],
                            'prompt': record[8],
                            'created_at': record[9].isoformat()
                        })
                    
                    return result
                    
                except Exception as e:
                    print(f"MySQL get error: {str(e)}")
                    return [] 