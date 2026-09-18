"""Unit tests for FastAPI REST API endpoints."""

import pytest
from fastapi.testclient import TestClient
from src.api import app

client = TestClient(app)

SAMPLE_CUSTOMER = {
    "tenure": 3,
    "MonthlyCharges": 85.50,
    "TotalCharges": 256.50,
    "Contract": "Month-to-month",
    "InternetService": "Fiber optic",
    "PaymentMethod": "Electronic check",
}


def test_root_endpoint():
    """Verify GET / returns 200 and basic API info."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "health_check" in data


def test_health_endpoint():
    """Verify GET /health returns status and model availability."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ["healthy", "model_unavailable"]
    assert "model_loaded" in data


def test_predict_endpoint():
    """Verify POST /predict returns valid churn classification and probabilities."""
    response = client.post("/predict", json=SAMPLE_CUSTOMER)
    assert response.status_code == 200
    data = response.json()
    assert data["prediction"] in [0, 1]
    assert data["label"] in ["Churn", "No Churn"]
    assert 0.0 <= data["churn_probability"] <= 1.0
    assert data["risk_level"] in ["Low", "Medium", "High"]


def test_predict_endpoint_missing_total_charges():
    """Verify POST /predict handles customer with omitted TotalCharges."""
    customer_no_tc = {
        "tenure": 0,
        "MonthlyCharges": 45.0,
        "Contract": "One year",
        "InternetService": "DSL",
        "PaymentMethod": "Mailed check",
    }
    response = client.post("/predict", json=customer_no_tc)
    assert response.status_code == 200
    data = response.json()
    assert data["prediction"] in [0, 1]


def test_predict_batch_dict_format():
    """Verify POST /predict/batch with {'customers': [...]} format."""
    response = client.post("/predict/batch", json={"customers": [SAMPLE_CUSTOMER]})
    assert response.status_code == 200
    data = response.json()
    assert data["total_customers"] == 1
    assert len(data["predictions"]) == 1


def test_predict_batch_list_format():
    """Verify POST /predict/batch with direct list [...] format."""
    response = client.post("/predict/batch", json=[SAMPLE_CUSTOMER, SAMPLE_CUSTOMER])
    assert response.status_code == 200
    data = response.json()
    assert data["total_customers"] == 2
    assert len(data["predictions"]) == 2


def test_predict_batch_empty_fails():
    """Verify POST /predict/batch rejects empty payload with 400."""
    response = client.post("/predict/batch", json=[])
    assert response.status_code == 400
