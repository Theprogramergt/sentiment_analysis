
import json
import math
import os
import random
import secrets
from pathlib import Path
from threading import Lock
from typing import Optional

import httpx

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi import HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

app = FastAPI(
    title="Movie Sentiment Analysis API",
    description="Compare a model trained on 20% of data with one trained on 100%.",
    version="1.0"
)

MODEL_A_URL = os.getenv("MODEL_A_URL", "http://localhost:8001").rstrip("/")
MODEL_B_URL = os.getenv("MODEL_B_URL", "http://localhost:8002").rstrip("/")
MODEL_REQUEST_TIMEOUT = 10.0

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


class ReviewRequest(BaseModel):
    review: str = Field(min_length=1)


def predict(model_url: str, review: str):
    try:
        with httpx.Client(timeout=MODEL_REQUEST_TIMEOUT) as client:
            response = client.post(
                f"{model_url}/predict",
                json={"review": review},
            )
    except httpx.TimeoutException as error:
        raise HTTPException(
            status_code=504,
            detail=f"Model service timed out: {model_url}",
        ) from error
    except httpx.RequestError as error:
        raise HTTPException(
            status_code=503,
            detail=f"Unable to connect to model service: {model_url}",
        ) from error

    if response.status_code >= 400:
        try:
            detail = response.json().get("detail")
        except (ValueError, AttributeError):
            detail = None
        raise HTTPException(
            status_code=502,
            detail=detail or f"Model service returned HTTP {response.status_code}",
        )

    try:
        payload = response.json()
        sentiment = payload["sentiment"]
        latency_ms = payload["latency_ms"]
        if sentiment not in {"positive", "negative"}:
            raise ValueError("Invalid sentiment")
        if not isinstance(latency_ms, (int, float)):
            raise ValueError("Invalid latency")
    except (ValueError, KeyError, TypeError) as error:
        raise HTTPException(
            status_code=502,
            detail=f"Invalid response from model service: {model_url}",
        ) from error

    return {
        "sentiment": sentiment,
        "latency_ms": round(float(latency_ms), 2),
    }


HELDOUT_DATA_PATH = Path("data/heldout_imdb.jsonl")
AB_TEST_SEED = 2026
ab_test_lock = Lock()
ab_test_results = None
ab_test_run_count = 0
ab_test_results_by_seed = {}


class ExperimentSeedRequest(BaseModel):
    seed: Optional[int] = Field(default=None, ge=0)


def load_heldout_reviews():
    reviews = []
    if HELDOUT_DATA_PATH.exists():
        seen_texts = set()
        with HELDOUT_DATA_PATH.open("r", encoding="utf-8") as dataset_file:
            for line in dataset_file:
                if not line.strip():
                    continue
                row = json.loads(line)
                if "text" in row and "label" in row:
                    text = str(row["text"])
                    if text not in seen_texts:
                        seen_texts.add(text)
                        reviews.append({
                            "text": text,
                            "label": int(row["label"]),
                        })
        return reviews

    try:
        from datasets import load_dataset
        dataset = load_dataset("stanfordnlp/imdb", split="test")
        indices = list(range(len(dataset)))
        random.Random(AB_TEST_SEED).shuffle(indices)
        seen_texts = set()
        for index in indices:
            text = str(dataset[index]["text"])
            if text in seen_texts:
                continue
            seen_texts.add(text)
            reviews.append({
                "text": text,
                "label": int(dataset[index]["label"]),
            })
            if len(reviews) == 1000:
                break
    except Exception:
        return []
    return reviews


heldout_reviews = load_heldout_reviews()


def pooled_two_proportion_z_test(model_a_correct, model_a_total,
                                 model_b_correct, model_b_total):
    if model_a_total <= 0 or model_b_total <= 0:
        raise ValueError("Both models require at least one evaluation sample")

    proportion_a = model_a_correct / model_a_total
    proportion_b = model_b_correct / model_b_total
    pooled = (
        (model_a_correct + model_b_correct) /
        (model_a_total + model_b_total)
    )
    standard_error = math.sqrt(
        pooled * (1 - pooled) * (
            1 / model_a_total + 1 / model_b_total
        )
    )
    if standard_error == 0:
        z_statistic = 0.0
    else:
        z_statistic = (proportion_b - proportion_a) / standard_error
    p_value = math.erfc(abs(z_statistic) / math.sqrt(2))

    if abs(z_statistic) >= 1.96:
        if z_statistic > 0:
            conclusion = "Model B is statistically better."
            recommendation = "Recommend Model B."
        else:
            conclusion = "Model A is statistically better."
            recommendation = "Recommend Model A."
    else:
        conclusion = "No statistically significant difference."
        recommendation = (
            "Keep the current model until more evidence is available."
        )

    return {
        "z_statistic": round(z_statistic, 4),
        "significance_threshold": 1.96,
        "p_value": round(p_value, 6),
        "conclusion": conclusion,
        "recommendation": recommendation,
        "test": "Pooled two-proportion Z-test",
        "assumptions": (
            "Independent evaluation samples, binary outcomes, "
            "and sufficiently large sample sizes."
        ),
    }


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"title": "Movie Sentiment AI"},
    )


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/predict/model-a")
def predict_a(request: ReviewRequest):
    return predict(MODEL_A_URL, request.review)


@app.post("/predict/model-b")
def predict_b(request: ReviewRequest):
    return predict(MODEL_B_URL, request.review)


@app.post("/compare")
def compare(request: ReviewRequest):
    result_a = predict(MODEL_A_URL, request.review)
    result_b = predict(MODEL_B_URL, request.review)

    return {
        "model_a": result_a,
        "model_b": result_b,
        "same_prediction": (
            result_a["sentiment"] == result_b["sentiment"]
        )
    }


@app.get("/ab-test/dataset")
def ab_test_dataset():
    if not heldout_reviews:
        raise HTTPException(
            status_code=503,
            detail="The held-out IMDb evaluation dataset is unavailable.",
        )
    return {
        "available": True,
        "review_count": len(heldout_reviews),
        "source": "IMDb test split",
    }


@app.post("/ab-test/start")
def start_ab_test(request: Optional[ExperimentSeedRequest] = None):
    global ab_test_results, ab_test_run_count

    if not heldout_reviews:
        raise HTTPException(
            status_code=503,
            detail="The held-out IMDb evaluation dataset is unavailable.",
        )

    with ab_test_lock:
        ab_test_run_count += 1
        experiment_seed = (
            request.seed
            if request is not None and request.seed is not None
            else secrets.randbits(63)
        )
        if experiment_seed in ab_test_results_by_seed:
            ab_test_results = dict(ab_test_results_by_seed[experiment_seed])
            ab_test_results["experiment_id"] = ab_test_run_count
            ab_test_results["seed"] = experiment_seed
            return ab_test_results

        indices = list(range(len(heldout_reviews)))
        random.Random(experiment_seed).shuffle(indices)

        model_a_indices = indices[::2]
        model_b_indices = indices[1::2]
        model_results = {
            "model_a": {"correct": 0, "latencies": []},
            "model_b": {"correct": 0, "latencies": []},
        }
        sample_assignments = []

        for index in model_a_indices:
            review = heldout_reviews[index]
            result = predict(MODEL_A_URL, review["text"])
            predicted_label = 1 if result["sentiment"] == "positive" else 0
            if predicted_label == review["label"]:
                model_results["model_a"]["correct"] += 1
            model_results["model_a"]["latencies"].append(result["latency_ms"])
            if len([sample for sample in sample_assignments
                    if sample["assigned_model"] == "A"]) < 4:
                sample_assignments.append({
                    "review_id": index,
                    "preview": review["text"][:160],
                    "assigned_model": "A",
                    "actual_sentiment": (
                        "positive" if review["label"] == 1 else "negative"
                    ),
                })

        for index in model_b_indices:
            review = heldout_reviews[index]
            result = predict(MODEL_B_URL, review["text"])
            predicted_label = 1 if result["sentiment"] == "positive" else 0
            if predicted_label == review["label"]:
                model_results["model_b"]["correct"] += 1
            model_results["model_b"]["latencies"].append(result["latency_ms"])
            if len([sample for sample in sample_assignments
                    if sample["assigned_model"] == "B"]) < 4:
                sample_assignments.append({
                    "review_id": index,
                    "preview": review["text"][:160],
                    "assigned_model": "B",
                    "actual_sentiment": (
                        "positive" if review["label"] == 1 else "negative"
                    ),
                })

        def summarize(name):
            data = model_results[name]
            total = len(data["latencies"])
            correct = data["correct"]
            latencies = data["latencies"]
            total_latency = sum(latencies)
            return {
                "processed": total,
                "correct": correct,
                "accuracy": round(correct / total, 6) if total else 0.0,
                "accuracy_percent": round(
                    (correct / total * 100) if total else 0.0,
                    2,
                ),
                "average_inference_ms": round(
                    total_latency / total if total else 0.0,
                    2,
                ),
                "total_inference_ms": round(total_latency, 2),
            }

        model_a_summary = summarize("model_a")
        model_b_summary = summarize("model_b")
        significance = pooled_two_proportion_z_test(
            model_a_summary["correct"],
            model_a_summary["processed"],
            model_b_summary["correct"],
            model_b_summary["processed"],
        )

        ab_test_results = {
            "status": "completed",
            "experiment_id": ab_test_run_count,
            "seed": experiment_seed,
            "review_count": len(heldout_reviews),
            "model_a": model_a_summary,
            "model_b": model_b_summary,
            "significance": significance,
            "traffic": {
                "total": len(heldout_reviews),
                "model_a": {
                    "count": model_a_summary["processed"],
                    "percentage": round(
                        model_a_summary["processed"] /
                        len(heldout_reviews) * 100,
                        1,
                    ),
                },
                "model_b": {
                    "count": model_b_summary["processed"],
                    "percentage": round(
                        model_b_summary["processed"] /
                        len(heldout_reviews) * 100,
                        1,
                    ),
                },
            },
            "sample_assignments": sample_assignments,
        }
        ab_test_results_by_seed[experiment_seed] = dict(ab_test_results)
        return ab_test_results


@app.get("/ab-test/results")
def get_ab_test_results():
    if ab_test_results is None:
        return {
            "status": "not_started",
            "seed": AB_TEST_SEED,
            "review_count": len(heldout_reviews),
        }
    return ab_test_results
