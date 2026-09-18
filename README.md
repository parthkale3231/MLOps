# Customer Churn Prediction ML Project

An end-to-end production-ready machine learning system designed to predict customer churn, identify key attrition risk drivers, and support proactive retention strategies.

## Project Structure

```text
customer-churn-ml/
│
├── data/
│   └── customer_churn.csv        # Primary Telco customer dataset
│
├── models/
│   └── churn_model.joblib        # Serialized trained scikit-learn pipeline
│
├── notebooks/
│   └── exploration.ipynb         # EDA, feature engineering, and experimentation
│
├── src/
│   ├── __init__.py               # Package initializer
│   ├── config.py                 # Central configuration and hyperparameters
│   ├── data_preprocessing.py     # Data loading, cleaning, and ColumnTransformer
│   ├── train.py                  # Sequential training workflow with logging
│   └── predict.py                # Inference script for customer churn probability
│
├── tests/
│   ├── __init__.py               # Test package initializer
│   └── test_preprocessing.py     # Unit tests for loading, cleaning, and preprocessing
│
├── logs/
│   └── training.log              # Persistent training logs and evaluation metrics
│
├── .gitignore                    # Version control ignore rules
├── requirements.txt              # Project dependencies
└── README.md                     # Project documentation
```

---

## Getting Started

### 1. Environment Setup

```bash
# Create and activate virtual environment
python -m venv .venv

# Windows (PowerShell):
.\.venv\Scripts\Activate.ps1
# Linux/macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## Execution Guide

### 1. Train the Model
Runs the sequential training workflow (`Load -> Clean -> Split -> Train -> Evaluate -> Save`) and generates `logs/training.log`:

```bash
python -m src.train
```

### 2. Run Inference
Loads `models/churn_model.joblib`, inputs customer profile data, and computes churn probability:

```bash
python -m src.predict
```

### 3. Run Unit Tests
Executes automated test suite testing data loading, missing value imputation, churn encoding, and preprocessing:

```bash
pytest
```
