from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.api.routes import chat
from src.services.chat_service import ChatService

# Create FastAPI application
app = FastAPI(title="AgentZero API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
chat_service = ChatService()
app.state.chat_service = chat_service

# Register routes
app.include_router(chat.router, prefix="/api/v1/chat", tags=["chat"])

# Health check
@app.get("/")
async def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)