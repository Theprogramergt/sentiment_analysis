
from fastapi.testclient import TestClient
from app import app

client = TestClient(app)


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
