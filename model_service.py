import os
import time

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


MODEL_PATH = os.getenv("MODEL_PATH")
model = None
model_error = None

if not MODEL_PATH:
    model_error = "MODEL_PATH environment variable is not set"
else:
    try:
        model = joblib.load(MODEL_PATH)
    except Exception as error:
        model_error = f"Unable to load model: {error}"


app = FastAPI(
    title="Movie Sentiment Model Service",
    version="1.0",
)


class ReviewRequest(BaseModel):
    review: str = Field(min_length=1)


@app.get("/health")
def health():
    if model is None:
        raise HTTPException(
            status_code=503,
            detail=model_error or "Model is not ready",
        )
    return {
        "status": "healthy",
        "model_ready": True,
        "model_path": MODEL_PATH,
    }


@app.post("/predict")
def predict(request: ReviewRequest):
    if model is None:
        raise HTTPException(
            status_code=503,
            detail=model_error or "Model is not ready",
        )

    try:
        start = time.perf_counter()
        prediction = model.predict([request.review])[0]
        latency_ms = round((time.perf_counter() - start) * 1000, 2)
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Model prediction failed: {error}",
        ) from error

    return {
        "sentiment": "positive" if prediction == 1 else "negative",
        "latency_ms": latency_ms,
    }
