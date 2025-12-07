from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any
import os
from document_processor import UnifiedDocumentProcessor
from langchain_groq import ChatGroq

router = APIRouter()

class ChatRequest(BaseModel):
    query: str
    model: str = "llama-3.3-70b-versatile"

class ChatResponse(BaseModel):
    result: str
    sources: list
    confidence: float

@router.post("/chat", response_model=ChatResponse)
async def process_chat(request: ChatRequest):
    """Process chat queries"""
    try:
        # Initialize LLM
        llm = ChatGroq(
            groq_api_key=os.getenv('GROQ_API_KEY'),
            model_name=request.model
        )
        
        # Get vectorstore from state
        if not hasattr(router, 'vectorstore'):
            raise HTTPException(
                status_code=400, 
                detail="No documents loaded. Please upload documents first."
            )
        
        # Process query
        response = process_chat_query(
            prompt=request.query,
            vectorstore=router.vectorstore,
            llm=llm
        )
        
        return ChatResponse(
            result=response['result'],
            sources=[doc.metadata['source'] for doc in response.get('source_documents', [])],
            confidence=0.95  # You might want to calculate this based on actual confidence
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
