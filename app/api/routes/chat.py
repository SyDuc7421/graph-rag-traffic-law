from fastapi import APIRouter, HTTPException
from app.schemas.chat import ChatRequest, ChatResponse
from app.services.chat_service import chat_service
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["chat"])

@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    User endpoint to ask questions about traffic laws.
    Returns analyzed responses from Neo4j through LangChain.
    """
    try:
        if not request.question or not request.question.strip():
            raise HTTPException(status_code=400, detail="Câu hỏi không được để trống")
            
        result = chat_service.ask(request.question)
        
        return ChatResponse(
            question=request.question,
            answer=result.get("answer", "Xin lỗi, tôi không tìm ra câu trả lời cho câu hỏi của bạn."),
            data=result.get("data")
        )
    except Exception as e:
        logger.error(f"Error processing chat request: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal Server Error")
