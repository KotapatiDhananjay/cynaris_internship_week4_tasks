
"""Automated tests for the W4D4 FastAPI model serving endpoint."""

from fastapi.testclient import TestClient
from W4D4_Model_API import app
from sklearn.datasets import load_breast_cancer

client = TestClient(app)


def test_home_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["model_loaded"] is True


def test_valid_prediction():
    data = load_breast_cancer()
    response = client.post(
        "/predict",
        json={"features": data.data[0].tolist()},
    )

    assert response.status_code == 200
    result = response.json()
    assert result["prediction"] in [0, 1]
    assert result["diagnosis"] in ["malignant", "benign"]
    assert 0 <= result["probability_malignant"] <= 1
    assert 0 <= result["probability_benign"] <= 1
    assert abs(
        result["probability_malignant"]
        + result["probability_benign"] - 1
    ) < 1e-6


def test_invalid_feature_count():
    response = client.post("/predict", json={"features": [1.0, 2.0]})
    assert response.status_code == 422


def test_api_documentation():
    response = client.get("/openapi.json")
    assert response.status_code == 200
    assert "/predict" in response.json()["paths"]


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
