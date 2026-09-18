import joblib
import pandas as pd

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from src.config import MODEL_PATH


def evaluate():

    data = pd.read_csv("data/customer_churn.csv")

    X = data.drop("Churn", axis=1)
    y = data["Churn"]

    model = joblib.load(MODEL_PATH)

    predictions = model.predict(X)

    accuracy = accuracy_score(y, predictions)
    precision = precision_score(y, predictions)
    recall = recall_score(y, predictions)
    f1 = f1_score(y, predictions)

    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1:", f1)

    # Model quality gate
    MIN_F1 = 0.70

    if f1 < MIN_F1:
        raise RuntimeError(
            f"Model quality gate failed. F1={f1:.3f}"
        )

    print("Model quality gate passed.")


if __name__ == "__main__":
    evaluate()