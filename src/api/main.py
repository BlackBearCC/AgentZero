from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.api.routes import chat, agent, eval, role
from src.services.agent_service import AgentService

app = FastAPI(title="AgentZero API")

# CORS配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 初始化服务
agent_service = AgentService()

# 注册路由
app.include_router(chat.router, prefix="/api/v1", tags=["chat"])
app.include_router(agent.router, prefix="/api/v1", tags=["agent"])
app.include_router(eval.router, prefix="/api/v1", tags=["evaluation"])
app.include_router(role.router, prefix="/api/v1", tags=["role"])

@app.get("/")
async def health_check():
    """健康检查"""
    return {"status": "healthy", "agents": list(agent_service.agents.keys())} 