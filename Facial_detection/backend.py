from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from deepface import DeepFace
from PIL import Image
import numpy as np
import io

app = FastAPI()
templates = Jinja2Templates(directory="templates")

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/analyze", response_class=HTMLResponse)
def analyze_page(request: Request):
    return templates.TemplateResponse("analyze.html", {"request": request})

@app.post("/analyze/", response_class=HTMLResponse)
async def analyze_image(request: Request, file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        img_np = np.array(image)

        # Analyze with DeepFace
        result = DeepFace.analyze(img_np, actions=['emotion'], enforce_detection=False)

        dominant_emotion = result[0]['dominant_emotion']
        return templates.TemplateResponse("analyze.html", {
            "request": request,
            "result": dominant_emotion
        })
    except Exception as e:
        return templates.TemplateResponse("analyze.html", {
            "request": request,
            "result": f"Error analyzing image: {str(e)}"
        })
