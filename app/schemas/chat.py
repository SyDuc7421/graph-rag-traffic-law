from pydantic import BaseModel
from typing import Optional, Any

class ChatRequest(BaseModel):
    question: str

class ChatResponse(BaseModel):
    question: str
    answer: str
    data: Optional[Any] = None
