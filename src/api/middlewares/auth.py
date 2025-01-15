from fastapi import Request, HTTPException
from typing import Optional
from jose import JWTError, jwt

async def verify_token(request: Request) -> Optional[dict]:
    token = request.headers.get('Authorization')
    if not token:
        raise HTTPException(status_code=401, detail="Token not found")
    
    try:
        # 从环境变量获取密钥
        payload = jwt.decode(token, "secret_key", algorithms=["HS256"])
        return payload
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token") 