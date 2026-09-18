"""Data preprocessing module for Customer Churn Prediction.

Contains:
- Feature definitions
- Data loading
- Data cleaning & missing value imputation
- ColumnTransformer preprocessing definition
- Train/test splitting
"""

import sys
from pathlib import Path
from typing import Tuple, Union
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.config import (
        ALL_FEATURES,
        CATEGORICAL_FEATURES,
        DATA_PATH,
        FEATURES,
        NUMERIC_FEATURES,
        RANDOM_STATE,
        TARGET,
        TEST_SIZE,
    )
else:
    from .config import (
        ALL_FEATURES,
        CATEGORICAL_FEATURES,
        DATA_PATH,
        FEATURES,
        NUMERIC_FEATURES,
        RANDOM_STATE,
        TARGET,
        TEST_SIZE,
    )


# ---------------------------------------------------------------------------
# Feature Definitions
# ---------------------------------------------------------------------------
NUMERIC_COLS = NUMERIC_FEATURES
CATEGORICAL_COLS = CATEGORICAL_FEATURES
ALL_COLS = ALL_FEATURES
TARGET_COL = TARGET


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------
def load_data(filepath: Union[str, Path] = None) -> pd.DataFrame:
    """Loads dataset from CSV file into a pandas DataFrame."""
    path = Path(filepath) if filepath is not None else DATA_PATH
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}")

    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"Dataset at {path} is empty.")
    return df


# ---------------------------------------------------------------------------
# Data Cleaning
# ---------------------------------------------------------------------------
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Cleans raw dataset by handling TotalCharges whitespace/nulls and encoding Churn."""
    cleaned = df.copy()

    # Convert TotalCharges to numeric (handles whitespace strings ' ' and invalid values)
    if "TotalCharges" in cleaned.columns:
        cleaned["TotalCharges"] = pd.to_numeric(cleaned["TotalCharges"], errors="coerce")
        median_total = cleaned["TotalCharges"].median()
        # In case median is NaN (e.g. all values NaN in a synthetic test), fallback to 0.0
        fill_val = median_total if pd.notna(median_total) else 0.0
        cleaned["TotalCharges"] = cleaned["TotalCharges"].fillna(fill_val)

    # Encode Churn to binary integers (Yes -> 1, No -> 0)
    if "Churn" in cleaned.columns:
        if cleaned["Churn"].dtype == object or isinstance(cleaned["Churn"].iloc[0], str):
            cleaned["Churn"] = cleaned["Churn"].map({"Yes": 1, "No": 0})
            if cleaned["Churn"].isna().any():
                cleaned["Churn"] = cleaned["Churn"].fillna(0).astype(int)
        cleaned["Churn"] = cleaned["Churn"].astype(int)

    return cleaned


# ---------------------------------------------------------------------------
# Preprocessor Definition
# ---------------------------------------------------------------------------
def get_preprocessor() -> ColumnTransformer:
    """Creates and returns the scikit-learn ColumnTransformer."""
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_COLS),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLS),
        ]
    )


# ---------------------------------------------------------------------------
# Data Splitting
# ---------------------------------------------------------------------------
def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Splits dataset into stratified train and test partitions."""
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
