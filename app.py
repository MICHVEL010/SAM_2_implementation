# app.py (FastAPI Backend)
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
from typing import Tuple
import io
from PIL import Image

app = FastAPI(title="Object Extraction API")

# Enable CORS for Streamlit
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def extract_object(
    image: np.ndarray,
    point: Tuple[int, int],
    margin_percent: float = 0.1,
    iterations: int = 5,
    refine_kernel_size: int = 5
) -> np.ndarray:
    """
    Extracts an object from an image using GrabCut algorithm.
    """
    # Initialize GrabCut models
    mask = np.zeros(image.shape[:2], np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    
    # Calculate rectangle around the point
    h, w = image.shape[:2]
    margin_w = int(w * margin_percent)
    margin_h = int(h * margin_percent)
    rect = (margin_w, margin_h, w - 2*margin_w, h - 2*margin_h)
    
    # Run GrabCut
    cv2.grabCut(image, mask, rect, bgd_model, fgd_model, iterations, cv2.GC_INIT_WITH_RECT)
    
    # Create binary mask
    binary_mask = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    
    # Refine edges
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (refine_kernel_size, refine_kernel_size))
    mask_refined = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
    mask_refined = cv2.GaussianBlur(mask_refined.astype(np.float32), (5, 5), 0)
    
    # Create alpha channel
    alpha = (mask_refined * 255).astype(np.uint8)
    
    # Create RGBA image
    extracted_object = np.dstack([image, alpha])
    
    return extracted_object


@app.post("/extract")
async def extract_object_endpoint(
    file: UploadFile = File(...),
    x: int = Form(...),
    y: int = Form(...)
):
    """
    Extract object from image at given coordinates.
    
    Parameters:
    - file: Image file (jpg, png, etc.)
    - x: X coordinate of object center
    - y: Y coordinate of object center
    
    Returns:
    - PNG image with extracted object (transparent background)
    """
    try:
        # Read uploaded image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Validate coordinates
        h, w = img_rgb.shape[:2]
        if x < 0 or x >= w or y < 0 or y >= h:
            raise HTTPException(
                status_code=400,
                detail=f"Coordinates ({x}, {y}) out of bounds. Valid range: X: 0-{w-1}, Y: 0-{h-1}"
            )
        
        # Extract object
        extracted = extract_object(img_rgb, (x, y))
        
        # Convert RGBA to PNG bytes
        extracted_bgra = cv2.cvtColor(extracted, cv2.COLOR_RGBA2BGRA)
        _, buffer = cv2.imencode('.png', extracted_bgra)
        
        # Return as streaming response
        return StreamingResponse(
            io.BytesIO(buffer.tobytes()),
            media_type="image/png",
            headers={"Content-Disposition": "attachment; filename=extracted_object.png"}
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def root():
    return {
        "message": "Object Extraction API",
        "endpoints": {
            "/extract": "POST - Extract object from image",
            "/docs": "GET - API documentation"
        }
    }


@app.get("/health")
async def health():
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)