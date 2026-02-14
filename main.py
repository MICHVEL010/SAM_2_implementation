from fastapi import FastAPI, UploadFile, File, Form
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

    pixels = image.load()
    pixels[x, y] = (255, 0, 0) 

    img_bytes = io.BytesIO()
    image.save(img_bytes, format="PNG")

    return Response(content=img_bytes.getvalue(), media_type="image/png")
