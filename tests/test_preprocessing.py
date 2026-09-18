"""Unit tests for data preprocessing and cleaning functions."""

import numpy as np
import pandas as pd
import pytest
from src.data_preprocessing import (
    CATEGORICAL_COLS,
    NUMERIC_COLS,
    clean_data,
    get_preprocessor,
    load_data,
    split_data,
)


def test_load_data():
    """Test 1: Verify data loading returns a non-empty DataFrame with required columns."""
    df = load_data()
    assert isinstance(df, pd.DataFrame), "Loaded object should be a pandas DataFrame"
    assert not df.empty, "Loaded DataFrame should not be empty"
    for col in NUMERIC_COLS + CATEGORICAL_COLS + ["Churn"]:
        assert col in df.columns, f"Expected column '{col}' missing from loaded dataset"


def test_churn_conversion():
    """Test 2: Verify Churn 'Yes'/'No' string labels are mapped to 1 and 0."""
    sample_df = pd.DataFrame({
        "tenure": [1, 24, 12],
        "MonthlyCharges": [29.85, 56.95, 42.30],
        "TotalCharges": ["29.85", "1889.5", "500.0"],
        "Contract": ["Month-to-month", "One year", "Two year"],
        "InternetService": ["DSL", "Fiber optic", "No"],
        "PaymentMethod": ["Electronic check", "Mailed check", "Bank transfer (automatic)"],
        "Churn": ["Yes", "No", "Yes"],
    })

    cleaned = clean_data(sample_df)

    assert cleaned["Churn"].dtype in [np.int64, np.int32, int], "Churn should be integer type"
    assert list(cleaned["Churn"]) == [1, 0, 1], "Churn 'Yes'/'No' should be mapped to [1, 0, 1]"
    assert cleaned["Churn"].isna().sum() == 0, "There should be no missing values in Churn"


def test_missing_total_charges_handling():
    """Test 3: Verify whitespace and missing TotalCharges are converted to float and imputed."""
    sample_df = pd.DataFrame({
        "tenure": [0, 5, 0],
        "MonthlyCharges": [20.0, 50.0, 70.0],
        "TotalCharges": [" ", "250.0", np.nan],
        "Contract": ["Month-to-month", "One year", "Month-to-month"],
        "InternetService": ["DSL", "Fiber optic", "DSL"],
        "PaymentMethod": ["Electronic check", "Mailed check", "Electronic check"],
        "Churn": ["No", "No", "Yes"],
    })

    cleaned = clean_data(sample_df)

    assert pd.api.types.is_numeric_dtype(cleaned["TotalCharges"]), "TotalCharges must be numeric"
    assert cleaned["TotalCharges"].isna().sum() == 0, "No NaN should remain in TotalCharges"
    assert (cleaned["TotalCharges"] >= 0).all(), "TotalCharges should be non-negative"
    # Position 1 had valid 250.0, so median was 250.0 and imputed to position 0 and 2
    assert cleaned["TotalCharges"].iloc[1] == 250.0


def test_preprocessor_transformation():
    """Test 4: Verify ColumnTransformer transforms features without crashing."""
    sample_df = pd.DataFrame({
        "tenure": [1, 12, 36],
        "MonthlyCharges": [30.0, 70.0, 90.0],
        "TotalCharges": [30.0, 840.0, 3240.0],
        "Contract": ["Month-to-month", "One year", "Two year"],
        "InternetService": ["DSL", "Fiber optic", "No"],
        "PaymentMethod": ["Electronic check", "Mailed check", "Bank transfer (automatic)"],
    })

    preprocessor = get_preprocessor()
    transformed = preprocessor.fit_transform(sample_df)

    assert transformed is not None
    assert transformed.shape[0] == 3
    # 3 numeric + one-hot encoded categories (> 3)
    assert transformed.shape[1] > 3


def test_split_data():
    """Test 5: Verify split_data creates properly stratified train and test partitions."""
    X = pd.DataFrame({
        "tenure": list(range(100)),
        "MonthlyCharges": [50.0] * 100,
        "TotalCharges": [500.0] * 100,
        "Contract": ["Month-to-month"] * 100,
        "InternetService": ["DSL"] * 100,
        "PaymentMethod": ["Electronic check"] * 100,
    })
    y = pd.Series([0] * 70 + [1] * 30)

    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2, random_state=42)

    assert len(X_train) == 80
    assert len(X_test) == 20
    assert sum(y_test == 1) == 6  # 30% of 20 = 6
