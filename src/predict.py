"""Prediction and inference module for Customer Churn Prediction.

Executes sequential workflow:
Load saved model -> Accept customer data -> Predict -> Return probability
"""

import sys
from pathlib import Path
from typing import Any, Dict, Union
import joblib
import pandas as pd

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.config import MODEL_PATH
else:
    from .config import MODEL_PATH


def load_model(model_path: Union[str, Path] = None):
    """Loads the serialized scikit-learn model pipeline."""
    path = Path(model_path) if model_path is not None else MODEL_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Trained model not found at {path}. "
            "Please run 'python -m src.train' first."
        )
    return joblib.load(path)


def predict(
    customer_data: Union[Dict[str, Any], pd.DataFrame],
    model=None,
    model_path: Union[str, Path] = None,
) -> Dict[str, Any]:
    """Runs inference on input customer data and returns prediction and probabilities."""
    # 1. Load saved model if not provided
    if model is None:
        model = load_model(model_path)

    # 2. Accept customer data
    if isinstance(customer_data, dict):
        df_input = pd.DataFrame([customer_data])
    elif isinstance(customer_data, pd.DataFrame):
        df_input = customer_data.copy()
    else:
        raise TypeError("customer_data must be a dict or a pandas DataFrame.")

    # 3. Predict class
    pred_class = int(model.predict(df_input)[0])

    # 4. Return probability
    probabilities = model.predict_proba(df_input)[0]
    prob_no_churn = float(probabilities[0])
    prob_churn = float(probabilities[1])

    return {
        "prediction": pred_class,
        "label": "Churn" if pred_class == 1 else "No Churn",
        "probability_churn": prob_churn,
        "probability_no_churn": prob_no_churn,
    }


def main():
    print("=" * 60)
    print("CUSTOMER CHURN INFERENCE PIPELINE")
    print("=" * 60)

    # 1. Load model
    print(f"Loading model from: {MODEL_PATH}")
    model = load_model()

    # 2. Accept new customer data
    new_customer = {
        "tenure": 3,
        "MonthlyCharges": 85.5,
        "TotalCharges": 256.5,
        "Contract": "Month-to-month",
        "InternetService": "Fiber optic",
        "PaymentMethod": "Electronic check",
    }
    print(f"\nNew Customer Data:\n{new_customer}")

    # 3 & 4. Predict and return probability
    result = predict(new_customer, model=model)

    print("\n--- Prediction Result ---")
    print(f"Prediction         : {result['label']} (Class {result['prediction']})")
    print(f"Probability Churn  : {result['probability_churn'] * 100:.2f}%")
    print(f"Probability NoChurn: {result['probability_no_churn'] * 100:.2f}%")
    print("=" * 60)


if __name__ == "__main__":
    main()
