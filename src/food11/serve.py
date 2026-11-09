import io
import os

import mlflow
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image
from torchvision import transforms


MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://127.0.0.1:5000",
)

MODEL_URI = "models:/food11/Staging"

CLASS_NAMES = [
    "Bread",
    "Dairy product",
    "Dessert",
    "Egg",
    "Fried food",
    "Meat",
    "Noodles-Pasta",
    "Rice",
    "Seafood",
    "Soup",
    "Vegetable-Fruit",
]


mlflow.set_tracking_uri(
    MLFLOW_TRACKING_URI
)

print(f"MLflow tracking URI: {MLFLOW_TRACKING_URI}")
print(f"Loading model: {MODEL_URI}")

model = mlflow.pyfunc.load_model(
    MODEL_URI
)

print("Model loaded successfully.")


image_transform = transforms.Compose([
    transforms.Resize((128, 128)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[
            0.485,
            0.456,
            0.406,
        ],
        std=[
            0.229,
            0.224,
            0.225,
        ],
    ),
])


app = FastAPI(
    title="Food-11 Prediction API"
)


@app.get("/health")
def health():
    return {
        "status": "ok"
    }


@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):
    try:
        contents = await file.read()

        image = Image.open(
            io.BytesIO(contents)
        ).convert("RGB")

    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail="Invalid image file",
        ) from exc

    image_tensor = image_transform(
        image
    )

    batch = (
        image_tensor
        .unsqueeze(0)
        .numpy()
    )

    prediction = model.predict(
        batch
    )

    logits = np.asarray(
        prediction
    )

    if logits.ndim == 1:
        logits = logits.reshape(
            1,
            -1
        )

    logits = logits[0]

    logits = logits - np.max(
        logits
    )

    probabilities = np.exp(
        logits
    )

    probabilities = (
        probabilities
        / probabilities.sum()
    )

    predicted_index = int(
        np.argmax(
            probabilities
        )
    )

    confidence = float(
        probabilities[
            predicted_index
        ]
    )

    category = CLASS_NAMES[
        predicted_index
    ]

    return {
        "category": category,
        "confidence": confidence,
    }