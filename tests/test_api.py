
from fastapi.testclient import TestClient
from app import app
import httpx
import pytest
import app as app_module

client = TestClient(app)


class MockModelClient:
    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def post(self, url, json):
        request = httpx.Request("POST", url)
        if "8001" in url:
            return httpx.Response(
                200,
                json={"sentiment": "positive", "latency_ms": 2.25},
                request=request,
            )
        if "8002" in url:
            return httpx.Response(
                200,
                json={"sentiment": "negative", "latency_ms": 3.50},
                request=request,
            )
        return httpx.Response(
            502,
            json={"detail": "Unexpected model URL"},
            request=request,
        )


@pytest.fixture(autouse=True)
def mock_model_services(monkeypatch):
    monkeypatch.setattr(app_module.httpx, "Client", MockModelClient)


def test_home():
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_compare():
    response = client.post(
        "/compare",
        json={"review": "This movie was amazing and wonderful!"}
    )
    assert response.status_code == 200

    result = response.json()
    assert result["model_a"]["sentiment"] in ["positive", "negative"]
    assert result["model_b"]["sentiment"] in ["positive", "negative"]
    assert "same_prediction" in result


def test_ab_test_dataset_is_available():
    response = client.get("/ab-test/dataset")
    assert response.status_code == 200
    result = response.json()
    assert result["available"] is True
    assert result["review_count"] > 0


def test_ab_test_start_and_results():
    start_response = client.post("/ab-test/start")
    assert start_response.status_code == 200
    result = start_response.json()

    assert result["status"] == "completed"
    assert result["model_a"]["processed"] == result["model_b"]["processed"]
    assert result["model_a"]["processed"] + result["model_b"]["processed"] == result["review_count"]
    assert result["model_a"]["processed"] > 0
    assert result["model_b"]["processed"] > 0
    assert abs(
        result["model_a"]["accuracy"]
        - result["model_a"]["correct"] / result["model_a"]["processed"]
    ) < 0.000001
    assert abs(
        result["model_b"]["accuracy"]
        - result["model_b"]["correct"] / result["model_b"]["processed"]
    ) < 0.000001
    assert result["significance"]["significance_threshold"] == 1.96
    assert result["significance"]["conclusion"] in {
        "Model B is statistically better.",
        "Model A is statistically better.",
        "No statistically significant difference.",
    }
    assert result["traffic"]["model_a"]["count"] == result["model_a"]["processed"]
    assert result["traffic"]["model_b"]["count"] == result["model_b"]["processed"]
    assert (
        result["traffic"]["model_a"]["count"]
        + result["traffic"]["model_b"]["count"]
        == result["traffic"]["total"]
        == result["review_count"]
    )
    assert abs(
        result["traffic"]["model_a"]["percentage"]
        - result["traffic"]["model_a"]["count"] / result["review_count"] * 100
    ) < 0.11
    assert abs(
        result["traffic"]["model_b"]["percentage"]
        - result["traffic"]["model_b"]["count"] / result["review_count"] * 100
    ) < 0.11
    assert 0 < len(result["sample_assignments"]) <= 8
    assert all(
        sample["assigned_model"] in {"A", "B"}
        and sample["actual_sentiment"] in {"positive", "negative"}
        and sample["preview"]
        for sample in result["sample_assignments"]
    )

    results_response = client.get("/ab-test/results")
    assert results_response.status_code == 200
    assert results_response.json()["status"] == "completed"


def test_z_test_conclusion_uses_threshold():
    from app import pooled_two_proportion_z_test

    no_difference = pooled_two_proportion_z_test(50, 100, 50, 100)
    assert no_difference["conclusion"] == "No statistically significant difference."

    model_b_wins = pooled_two_proportion_z_test(50, 100, 90, 100)
    assert model_b_wins["conclusion"] == "Model B is statistically better."


def test_heldout_reviews_are_unique_and_assignments_are_disjoint():
    import random

    from app import AB_TEST_SEED, heldout_reviews

    texts = [review["text"] for review in heldout_reviews]
    assert len(texts) == 1000
    assert len(set(texts)) == 1000
    assert all(review["label"] in {0, 1} for review in heldout_reviews)

    indices = list(range(len(heldout_reviews)))
    random.Random(AB_TEST_SEED).shuffle(indices)
    model_a_indices = set(indices[::2])
    model_b_indices = set(indices[1::2])

    assert len(model_a_indices) == 500
    assert len(model_b_indices) == 500
    assert model_a_indices.isdisjoint(model_b_indices)
    assert model_a_indices | model_b_indices == set(range(1000))


def test_repeated_starts_create_new_experiment_samples():
    first = client.post("/ab-test/start").json()
    second = client.post("/ab-test/start").json()

    assert second["experiment_id"] > first["experiment_id"]
    assert second["seed"] != first["seed"]
    assert len(second["sample_assignments"]) == 8

    first_ids = {sample["review_id"] for sample in first["sample_assignments"]}
    second_ids = {sample["review_id"] for sample in second["sample_assignments"]}
    assert first_ids != second_ids


def test_same_seed_reproduces_assignments_and_different_seed_changes_them():
    import random

    first_seed_run = client.post(
        "/ab-test/start",
        json={"seed": 424242},
    ).json()
    repeated_seed_run = client.post(
        "/ab-test/start",
        json={"seed": 424242},
    ).json()
    different_seed_run = client.post(
        "/ab-test/start",
        json={"seed": 424243},
    ).json()

    assert first_seed_run["seed"] == 424242
    assert repeated_seed_run["seed"] == 424242
    assert first_seed_run["model_a"] == repeated_seed_run["model_a"]
    assert first_seed_run["model_b"] == repeated_seed_run["model_b"]
    assert first_seed_run["traffic"] == repeated_seed_run["traffic"]
    assert (
        first_seed_run["sample_assignments"]
        == repeated_seed_run["sample_assignments"]
    )

    first_indices = list(range(first_seed_run["review_count"]))
    random.Random(424242).shuffle(first_indices)
    different_indices = list(range(different_seed_run["review_count"]))
    random.Random(424243).shuffle(different_indices)
    assert first_indices != different_indices
    assert set(first_indices[::2]) != set(different_indices[::2])


def test_model_service_connection_error(monkeypatch):
    class ConnectionErrorClient(MockModelClient):
        def post(self, url, json):
            raise httpx.ConnectError(
                "Service unavailable",
                request=httpx.Request("POST", url),
            )

    monkeypatch.setattr(app_module.httpx, "Client", ConnectionErrorClient)
    response = client.post(
        "/predict/model-a",
        json={"review": "This movie was excellent"},
    )
    assert response.status_code == 503
    assert "Unable to connect" in response.json()["detail"]


def test_model_service_invalid_response(monkeypatch):
    class InvalidResponseClient(MockModelClient):
        def post(self, url, json):
            return httpx.Response(
                200,
                json={"sentiment": "unknown", "latency_ms": "fast"},
                request=httpx.Request("POST", url),
            )

    monkeypatch.setattr(app_module.httpx, "Client", InvalidResponseClient)
    response = client.post(
        "/predict/model-b",
        json={"review": "This movie was disappointing"},
    )
    assert response.status_code == 502
    assert "Invalid response" in response.json()["detail"]
