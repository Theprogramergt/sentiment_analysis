
import time
import joblib

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(
    title="Movie Sentiment Analysis API",
    description="Compare a model trained on 20% of data with one trained on 100%.",
    version="1.0"
)

# Load both trained models
model_a = joblib.load("models/model_a.joblib")
model_b = joblib.load("models/model_b.joblib")


class ReviewRequest(BaseModel):
    review: str = Field(min_length=1)


def predict(model, review):
    start = time.perf_counter()
    prediction = model.predict([review])[0]
    latency = (time.perf_counter() - start) * 1000

    sentiment = "positive" if prediction == 1 else "negative"

    return {
        "sentiment": sentiment,
        "latency_ms": round(latency, 2)
    }


@app.get("/")
def home():
    return {
        "message": "Movie Sentiment Analysis API is running!"
    }


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/predict/model-a")
def predict_a(request: ReviewRequest):
    return predict(model_a, request.review)


@app.post("/predict/model-b")
def predict_b(request: ReviewRequest):
    return predict(model_b, request.review)


@app.post("/compare")
def compare(request: ReviewRequest):
    result_a = predict(model_a, request.review)
    result_b = predict(model_b, request.review)

    return {
        "model_a": result_a,
        "model_b": result_b,
        "same_prediction": (
            result_a["sentiment"] == result_b["sentiment"]
        )
    }
