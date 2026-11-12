
from app import home, health, compare
from app import ReviewRequest


def test_home():
    result = home()
    assert "running" in result["message"]


def test_health():
    result = health()
    assert result["status"] == "healthy"


def test_compare():
    request = ReviewRequest(
        review="This movie was amazing and wonderful!"
    )

    result = compare(request)

    assert result["model_a"]["sentiment"] in ["positive", "negative"]
    assert result["model_b"]["sentiment"] in ["positive", "negative"]
    assert "same_prediction" in result
