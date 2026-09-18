"""Model evaluation module and quality gate for Customer Churn Prediction.

Loads the serialized model and dataset, computes classification metrics
(Accuracy, Precision, Recall, F1), and enforces a quality gate.
"""

import os
import sys
from pathlib import Path
from typing import Dict, Optional, Union
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# Support both `python -m src.evaluate` and `python src/evaluate.py`
if __package__ is None or __package__ == "":
    BASE_DIR = Path(__file__).resolve().parent.parent
    if str(BASE_DIR) not in sys.path:
        sys.path.insert(0, str(BASE_DIR))
    from src.config import ALL_FEATURES, DATA_PATH, MODEL_PATH, RANDOM_STATE, TARGET, TEST_SIZE
    from src.data_preprocessing import clean_data, load_data, split_data
else:
    from .config import ALL_FEATURES, DATA_PATH, MODEL_PATH, RANDOM_STATE, TARGET, TEST_SIZE
    from .data_preprocessing import clean_data, load_data, split_data


def evaluate(
    data_path: Optional[Union[str, Path]] = None,
    model_path: Optional[Union[str, Path]] = None,
    use_test_split: bool = True,
    min_f1: Optional[float] = None,
) -> Dict[str, float]:
    """Evaluates the trained model on customer churn data and verifies quality gates.

    Args:
        data_path: Path to dataset CSV (defaults to config.DATA_PATH).
        model_path: Path to serialized model joblib file (defaults to config.MODEL_PATH).
        use_test_split: If True, evaluates on holdout test partition matching training.
        min_f1: Minimum F1 score required to pass quality gate (defaults to 0.55 or env var MIN_F1).

    Returns:
        Dictionary containing calculated metrics (accuracy, precision, recall, f1).

    Raises:
        FileNotFoundError: If dataset or model file does not exist.
        RuntimeError: If model fails the quality gate threshold.
    """
    resolved_model_path = Path(model_path) if model_path is not None else MODEL_PATH
    if not resolved_model_path.exists():
        raise FileNotFoundError(
            f"Trained model not found at '{resolved_model_path}'. "
            "Please run 'python -m src.train' first."
        )

    # 1. Load and clean data
    raw_df = load_data(data_path if data_path is not None else DATA_PATH)
    cleaned_df = clean_data(raw_df)

    X = cleaned_df[ALL_FEATURES]
    y = cleaned_df[TARGET]

    # 2. Select evaluation partition
    if use_test_split:
        _, X_eval, _, y_eval = split_data(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    else:
        X_eval, y_eval = X, y

    # 3. Load model and run inference
    model = joblib.load(resolved_model_path)
    predictions = model.predict(X_eval)

    # 4. Calculate metrics
    accuracy = float(accuracy_score(y_eval, predictions))
    precision = float(precision_score(y_eval, predictions, zero_division=0))
    recall = float(recall_score(y_eval, predictions, zero_division=0))
    f1 = float(f1_score(y_eval, predictions, zero_division=0))

    print(f"Evaluation Dataset : {len(X_eval)} samples ({'Holdout Test Split' if use_test_split else 'Full Dataset'})")
    print(f"Accuracy           : {accuracy:.4f}")
    print(f"Precision          : {precision:.4f}")
    print(f"Recall             : {recall:.4f}")
    print(f"F1                 : {f1:.4f}")

    # 5. Model quality gate
    # Default threshold 0.55 (matches production baseline ~0.5857), configurable via MIN_F1 env var
    if min_f1 is None:
        min_f1 = float(os.getenv("MIN_F1", "0.55"))

    if f1 < min_f1:
        raise RuntimeError(
            f"Model quality gate failed. F1={f1:.4f} is below minimum required threshold {min_f1:.4f}"
        )

    print(f"Model quality gate passed (F1={f1:.4f} >= threshold={min_f1:.4f}).")

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


if __name__ == "__main__":
    evaluate()