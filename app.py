"""
app.py  –  FastAPI Backend
--------------------------
Endpoints
  POST /extract/single   → extract one object by (x, y) click
  POST /extract/multi    → extract multiple objects by list of (x, y) clicks
  POST /extract/preview  → coloured overlay preview (multi-object)
  GET  /health           → health check
  GET  /                 → API info
"""

import io
import json
from typing import List

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from PIL import Image

from segment import (
    apply_mask_to_image,
    overlay_masks_on_image,
    segment_multiple_objects,
    segment_single_object,
)

# ─────────────────────────────────────────────
#  App setup
# ─────────────────────────────────────────────

app = FastAPI(
    title="Object Extraction API",
    description="Single and multi-object extraction using GrabCut segmentation",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────
#  Shared helper
# ─────────────────────────────────────────────

def _read_image(file_bytes: bytes) -> np.ndarray:
    """Decode uploaded bytes → RGB numpy array."""
    nparr = np.frombuffer(file_bytes, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise HTTPException(status_code=400, detail="Could not decode image file.")
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def _validate_coords(x: int, y: int, w: int, h: int) -> None:
    if x < 0 or x >= w or y < 0 or y >= h:
        raise HTTPException(
            status_code=400,
            detail=f"Coordinates ({x}, {y}) out of bounds. "
                   f"Valid range: x 0–{w-1}, y 0–{h-1}.",
        )


def _rgba_to_png_bytes(rgba: np.ndarray) -> bytes:
    """Convert RGBA numpy array → PNG bytes."""
    bgra = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA)
    _, buf = cv2.imencode(".png", bgra)
    return buf.tobytes()


def _rgb_to_png_bytes(rgb: np.ndarray) -> bytes:
    """Convert RGB numpy array → PNG bytes."""
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    _, buf = cv2.imencode(".png", bgr)
    return buf.tobytes()


# ─────────────────────────────────────────────
#  Routes
# ─────────────────────────────────────────────

@app.get("/")
async def root():
    return {
        "message": "Object Extraction API",
        "endpoints": {
            "POST /extract/single":  "Extract one object by click (x, y)",
            "POST /extract/multi":   "Extract multiple objects by JSON list of points",
            "POST /extract/preview": "Coloured overlay preview for multi-object",
            "GET  /health":          "Health check",
            "GET  /docs":            "Swagger UI",
        },
    }


@app.get("/health")
async def health():
    return {"status": "healthy"}


# ── Single object ────────────────────────────

@app.post(
    "/extract/single",
    summary="Extract a single object",
    response_description="PNG image with transparent background",
)
async def extract_single(
    file: UploadFile = File(..., description="Image file (jpg / png)"),
    x: int = Form(..., description="X coordinate of clicked point"),
    y: int = Form(..., description="Y coordinate of clicked point"),
):
    """
    Upload an image and a single (x, y) click coordinate.
    Returns a PNG with the extracted object on a transparent background.
    """
    img = _read_image(await file.read())
    h, w = img.shape[:2]
    _validate_coords(x, y, w, h)

    mask = segment_single_object(img, (x, y))
    rgba = apply_mask_to_image(img, mask)
    png_bytes = _rgba_to_png_bytes(rgba)

    return StreamingResponse(
        io.BytesIO(png_bytes),
        media_type="image/png",
        headers={"Content-Disposition": "attachment; filename=extracted_single.png"},
    )


# ── Multi-object extraction ──────────────────

@app.post(
    "/extract/multi",
    summary="Extract multiple objects",
    response_description="ZIP of PNG images (one per object)",
)
async def extract_multi(
    file: UploadFile = File(..., description="Image file (jpg / png)"),
    points: str = Form(
        ...,
        description='JSON list of [x, y] pairs, e.g. [[100,200],[300,400]]',
    ),
):
    """
    Upload an image and a JSON list of (x, y) click coordinates.
    Returns a ZIP archive containing one PNG per object (transparent background).
    """
    import zipfile

    img = _read_image(await file.read())
    h, w = img.shape[:2]

    # Parse points
    try:
        pts_raw = json.loads(points)
        pts: List = [tuple(p) for p in pts_raw]
    except (json.JSONDecodeError, TypeError, ValueError):
        raise HTTPException(
            status_code=400,
            detail='`points` must be valid JSON, e.g. [[100,200],[300,400]]',
        )

    if not pts:
        raise HTTPException(status_code=400, detail="Provide at least one point.")

    for px, py in pts:
        _validate_coords(int(px), int(py), w, h)

    masks = segment_multiple_objects(img, [(int(px), int(py)) for px, py in pts])

    import cv2 as _cv2
    # Combine all masks into one — union of all object masks
    combined_mask = np.zeros((h, w), dtype=np.float32)
    for mask in masks:
        if mask.shape[:2] != (h, w):
            mask = _cv2.resize(mask, (w, h), interpolation=_cv2.INTER_LINEAR)
        combined_mask = np.maximum(combined_mask, mask)

    # Single RGBA image with all objects visible, background transparent
    rgba = apply_mask_to_image(img, combined_mask)
    png_bytes = _rgba_to_png_bytes(rgba)

    return StreamingResponse(
        io.BytesIO(png_bytes),
        media_type="image/png",
        headers={"Content-Disposition": "attachment; filename=extracted_objects.png"},
    )


# ── Multi-object preview overlay ─────────────

@app.post(
    "/extract/preview",
    summary="Coloured overlay preview",
    response_description="PNG with colour masks overlaid on original image",
)
async def extract_preview(
    file: UploadFile = File(..., description="Image file (jpg / png)"),
    points: str = Form(
        ...,
        description='JSON list of [x, y] pairs, e.g. [[100,200],[300,400]]',
    ),
):
    """
    Same as /extract/multi but returns a single preview image with
    colour-coded masks drawn on top of the original, great for Streamlit.
    """
    img = _read_image(await file.read())
    h, w = img.shape[:2]

    try:
        pts_raw = json.loads(points)
        pts = [(int(p[0]), int(p[1])) for p in pts_raw]
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid `points` JSON.")

    if not pts:
        raise HTTPException(status_code=400, detail="Provide at least one point.")

    for px, py in pts:
        _validate_coords(px, py, w, h)

    masks = segment_multiple_objects(img, pts)
    # Ensure all masks match original image size
    import cv2 as _cv2
    masks = [
        _cv2.resize(m, (w, h), interpolation=_cv2.INTER_LINEAR)
        if m.shape[:2] != (h, w) else m
        for m in masks
    ]
    overlay = overlay_masks_on_image(img, masks)
    png_bytes = _rgb_to_png_bytes(overlay)

    return StreamingResponse(
        io.BytesIO(png_bytes),
        media_type="image/png",
        headers={"Content-Disposition": "inline; filename=preview.png"},
    )


# ─────────────────────────────────────────────
#  Run directly
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True, timeout_keep_alive=300)