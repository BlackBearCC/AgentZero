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
                    (role_id, chat_id, input_text, output_text, query_text, remark,
                     summary, raw_entity_memory, processed_entity_memory, 
                     raw_history, processed_history, prompt, agent_info)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """
                    
                    # 序列化 JSON 字段
                    raw_entity_memory_json = json.dumps(data.get('raw_entity_memory', {}), ensure_ascii=False)
                    raw_history_json = json.dumps(data.get('raw_history', []), ensure_ascii=False)
                    processed_history_json = json.dumps(data.get('processed_history', []), ensure_ascii=False)
                    agent_info_json = json.dumps(data.get('agent_info', {}), ensure_ascii=False)
                    
                    await cur.execute(sql, (
                        role_id,
                        chat_id,
                        data['input'],
                        data['output'],
                        data.get('query_text', ''),
                        data.get('remark', ''),
                        data.get('summary', ''),
                        raw_entity_memory_json,
                        data.get('processed_entity_memory', ''),
                        raw_history_json,
                        processed_history_json,
                        data.get('prompt', ''),
                        agent_info_json
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