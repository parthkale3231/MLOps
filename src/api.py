"""FastAPI REST API for Customer Churn Prediction.

Provides:
- GET  /         : Welcome message and API status
- GET  /health   : Health check with model loading status
- POST /predict  : Single customer churn inference
- POST /predict/batch : Batch customer churn inference
"""

import sys
from pathlib import Path
from typing import List, Optional
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Ensure root directory is on Python path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.config import ALL_FEATURES, MODEL_PATH

# ---------------------------------------------------------------------------
# FastAPI Application & Middleware
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Customer Churn Prediction API",
    description="Production REST API for real-time customer churn probability scoring and risk assessment.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Model Loader with Error Handling
# ---------------------------------------------------------------------------
_model = None


def get_model():
    """Lazily loads and caches the trained model pipeline."""
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model file not found at '{MODEL_PATH}'. Run 'python -m src.train' to train and save the model."
            )
        _model = joblib.load(MODEL_PATH)
    return _model


# Attempt eager load on startup, but allow server to boot even if model is not yet trained
try:
    _model = get_model()
except Exception:
    _model = None


# ---------------------------------------------------------------------------
# Pydantic Schemas
# ---------------------------------------------------------------------------
class Customer(BaseModel):
    """Schema for individual customer profile data."""

    tenure: int = Field(
        ...,
        ge=0,
        description="Number of months the customer has stayed with the company.",
        examples=[3],
    )
    MonthlyCharges: float = Field(
        ...,
        ge=0.0,
        description="The monthly fee billed to the customer.",
        examples=[85.50],
    )
    TotalCharges: float = Field(
        ...,
        ge=0.0,
        description="The total amount billed to the customer across tenure.",
        examples=[256.50],
    )
    Contract: str = Field(
        ...,
        description="Contract term ('Month-to-month', 'One year', 'Two year').",
        examples=["Month-to-month"],
    )
    InternetService: str = Field(
        ...,
        description="Internet service provider ('DSL', 'Fiber optic', 'No').",
        examples=["Fiber optic"],
    )
    PaymentMethod: str = Field(
        ...,
        description="Payment method ('Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)').",
        examples=["Electronic check"],
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "tenure": 3,
                    "MonthlyCharges": 85.50,
                    "TotalCharges": 256.50,
                    "Contract": "Month-to-month",
                    "InternetService": "Fiber optic",
                    "PaymentMethod": "Electronic check",
                }
            ]
        }
    }


class PredictionResponse(BaseModel):
    """Schema for prediction output."""

    prediction: int = Field(..., description="Binary churn classification (1: Churn, 0: No Churn).")
    label: str = Field(..., description="Human-readable label ('Churn' or 'No Churn').")
    churn_probability: float = Field(..., description="Probability of customer churning (0.0 to 1.0).")
    risk_level: str = Field(..., description="Risk tier: 'High' (>0.6), 'Medium' (0.3-0.6), or 'Low' (<0.3).")


class BatchCustomerInput(BaseModel):
    """Schema for batch inference requests."""

    customers: List[Customer]


class BatchPredictionResponse(BaseModel):
    """Schema for batch inference results."""

    total_customers: int
    predictions: List[PredictionResponse]


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------
@app.get("/", tags=["General"])
def root():
    """Root endpoint providing API information."""
    return {
        "message": "Customer Churn Prediction API is running",
        "docs_url": "/docs",
        "health_check": "/health",
        "model_loaded": _model is not None,
    }


@app.get("/health", tags=["Health"])
def health():
    """Health check endpoint indicating service and model readiness."""
    is_loaded = _model is not None or MODEL_PATH.exists()
    return {
        "status": "healthy",
        "model_loaded": is_loaded,
        "model_path": str(MODEL_PATH),
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
def predict(customer: Customer):
    """Predict churn probability and classification for a single customer."""
    try:
        model = get_model()
    except FileNotFoundError as fnf_err:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(fnf_err),
        )

    try:
        # Prepare DataFrame conforming to expected features
        data = pd.DataFrame([customer.model_dump()])
        prediction = int(model.predict(data)[0])
        churn_prob = float(model.predict_proba(data)[0][1])

        # Assign risk tier
        if churn_prob >= 0.6:
            risk_level = "High"
        elif churn_prob >= 0.3:
            risk_level = "Medium"
        else:
            risk_level = "Low"

        return PredictionResponse(
            prediction=prediction,
            label="Churn" if prediction == 1 else "No Churn",
            churn_probability=round(churn_prob, 4),
            risk_level=risk_level,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Inference error: {str(exc)}",
        )


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Inference"])
def predict_batch(batch: BatchCustomerInput):
    """Predict churn probability and classification for multiple customers."""
    try:
        model = get_model()
    except FileNotFoundError as fnf_err:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(fnf_err),
        )

    if not batch.customers:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The batch customer list is empty.",
        )

    try:
        data = pd.DataFrame([c.model_dump() for c in batch.customers])
        predictions = model.predict(data)
        probabilities = model.predict_proba(data)

        results = []
        for pred, probs in zip(predictions, probabilities):
            pred_int = int(pred)
            churn_p = float(probs[1])
            risk = "High" if churn_p >= 0.6 else ("Medium" if churn_p >= 0.3 else "Low")
            results.append(
                PredictionResponse(
                    prediction=pred_int,
                    label="Churn" if pred_int == 1 else "No Churn",
                    churn_probability=round(churn_p, 4),
                    risk_level=risk,
                )
            )

        return BatchPredictionResponse(
            total_customers=len(results),
            predictions=results,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Batch inference error: {str(exc)}",
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.api:app", host="0.0.0.0", port=8000, reload=True)