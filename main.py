from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import Response
from PIL import Image
import io

app = FastAPI()

@app.post("/image")
async def get_image(
    file: UploadFile = File(...),
    x: int = Form(...),
    y: int = Form(...)
):
    image = Image.open(file.file)
    width, height = image.size

    if x < 0 or y < 0 or x >= width or y >= height:
        raise HTTPException(status_code=400, detail="Coordinates out of bounds")

    pixels = image.load()
    pixels[x, y] = (255, 0, 0)

    img_bytes = io.BytesIO()
    image.save(img_bytes, format="PNG")

    return Response(content=img_bytes.getvalue(), media_type="image/png")