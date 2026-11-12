"""Load the champion once at startup and serve ImageNet-normalized images."""

import io
import os
from contextlib import asynccontextmanager
from typing import Annotated

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

from src.food11.data import CLASSES, resize_image

model = None


@asynccontextmanager
async def lifespan(app):
    global model
    model = None
    if not os.environ.get("SKIP_MODEL"):
        import mlflow

        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
        model = mlflow.pyfunc.load_model("models:/food11@champion")
    yield
    model = None


app = FastAPI(title="Food-11 Prediction API", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: Annotated[UploadFile, File()]):
    if model is None:
        raise HTTPException(503, "Model is unavailable")
    contents = await file.read(10 * 1024 * 1024 + 1)
    if len(contents) > 10 * 1024 * 1024:
        raise HTTPException(413, "Image exceeds 10 MiB")
    try:
        with Image.open(io.BytesIO(contents)) as image:
            image = resize_image(image)
        array = np.asarray(image, dtype=np.float32) / 255.0
    except (
        UnidentifiedImageError,
        OSError,
        ValueError,
        Image.DecompressionBombError,
    ) as exc:
        raise HTTPException(400, "Invalid image file") from exc
    array = (array - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array(
        [0.229, 0.224, 0.225], dtype=np.float32
    )
    batch = array.transpose(2, 0, 1)[None].astype(np.float32)
    logits = np.asarray(model.predict(batch), dtype=float)
    if logits.shape != (1, len(CLASSES)) or not np.isfinite(logits).all():
        raise HTTPException(500, "Model returned invalid logits")
    probabilities = np.exp(logits[0] - logits[0].max())
    probabilities /= probabilities.sum()
    index = int(probabilities.argmax())
    return {"category": CLASSES[index], "confidence": float(probabilities[index])}
