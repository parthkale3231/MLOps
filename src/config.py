"""Central configuration file for the Customer Churn ML project."""

from pathlib import Path

# Project directory paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"

# Ensure directories exist
MODELS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Primary and fallback data paths
DATA_PATH = DATA_DIR / "customer_churn.csv"
if not DATA_PATH.exists():
    fallback_path = DATA_DIR / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
    if fallback_path.exists():
        DATA_PATH = fallback_path

MODEL_PATH = MODELS_DIR / "churn_model.joblib"
LOG_PATH = LOGS_DIR / "training.log"

# Feature definitions
NUMERIC_FEATURES = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
]

CATEGORICAL_FEATURES = [
    "Contract",
    "InternetService",
    "PaymentMethod",
]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
ALL_FEATURES = FEATURES
TARGET = "Churn"

# Model hyperparameters & training parameters
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
SCORING = "f1"

import os

# MLflow Configuration
DEFAULT_SQLITE_URI = f"sqlite:///{(BASE_DIR / 'mlflow.db').as_posix()}"
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", DEFAULT_SQLITE_URI)
MLFLOW_EXPERIMENT_NAME = "customer-churn-prediction"
REGISTERED_MODEL_NAME = "customer-churn-model"

# Experiment configurations for Task 6
EXPERIMENT_RUNS = [
    {"run_name": "Run_1_RF_100_depth_5", "n_estimators": 100, "max_depth": 5},
    {"run_name": "Run_2_RF_100_depth_10", "n_estimators": 100, "max_depth": 10},
    {"run_name": "Run_3_RF_200_depth_5", "n_estimators": 200, "max_depth": 5},
    {"run_name": "Run_4_RF_200_depth_10", "n_estimators": 200, "max_depth": 10},
    {"run_name": "Run_5_RF_300_depth_10", "n_estimators": 300, "max_depth": 10},
]
