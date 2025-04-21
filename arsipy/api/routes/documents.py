from fastapi import APIRouter, UploadFile, File
from typing import List

router = APIRouter()

@router.post("/documents/upload")
async def upload_documents(files: List[UploadFile] = File(...)):
    """Handle document uploads - reuses existing document processor"""
    try:
        # Reuse existing document processing logic
        pass
    except Exception as e:
        raise HTTPException(status_code=500, str(e))
