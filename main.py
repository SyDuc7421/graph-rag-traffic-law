from fastapi import FastAPI
from app.api.routes.chat import router as chat_router
from app.core.config import settings

app = FastAPI(
    title=settings.PROJECT_NAME,
    description="API cho Chatbot trả lời câu hỏi về luật giao thông bằng FastAPI, Neo4j GraphRAG và OpenAI.",
    version="2.0.0"
)

# Include routes
app.include_router(chat_router)

@app.get("/")
def read_root():
    return {"health_check": "OK", "version": app.version}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
