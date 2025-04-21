from fastapi import APIRouter, File, UploadFile, HTTPException
from image_analyzer import ImageAnalyzer

router = APIRouter()
analyzer = ImageAnalyzer()

@router.post("/images/analyze")
async def analyze_image(file: UploadFile = File(...)):
    """Process image analysis requests"""
    try:
        contents = await file.read()
        with open("temp_image.jpg", "wb") as temp_file:
            temp_file.write(contents)
            
        result = analyzer.analyze_hybrid("temp_image.jpg")
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
