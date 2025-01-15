import pytest
from httpx import AsyncClient
from src.api.main import app

@pytest.mark.asyncio
async def test_create_and_chat_with_agent():
    async with AsyncClient(app=app, base_url="http://localhost:8000") as client:
        # 创建agent
        agent_config = {
            "role_id": "test_assistant",
            "name": "测试助手",
            "description": "用于测试的AI助手",
            "personality": "专业、友善",
            "system_prompt": "你是一个测试助手，用于验证系统功能。",
            "constraints": [],
            "tools": [],
            "memory_config": {
                "type": "conversation",
                "max_history": 10
            }
        }
        
        response = await client.post("/api/v1/agents/create", json=agent_config)
        assert response.status_code == 200
        agent_id = response.json()["agent_id"]
        
        # 测试对话
        chat_request = {
            "message": "你好，请做个自我介绍",
            "context": {}
        }
        
        response = await client.post(
            f"/api/v1/chat/{agent_id}",
            json=chat_request
        )
        assert response.status_code == 200
        assert "content" in response.json() 