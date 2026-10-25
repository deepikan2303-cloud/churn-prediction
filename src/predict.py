"""Predict churn for new customers with the saved model.

Usage (from the project root, with the venv active):
    python src/predict.py                       # demo: scores 5 customers from data/telco_churn.csv
    python src/predict.py path\\to\\customers.csv  # scores every row, writes <name>_predictions.csv

The input CSV needs the same columns as the Telco dataset (customerID and Churn are optional).
"""
import os
import sys

import joblib
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# The pickled pipeline references src.features.add_features
sys.path.insert(0, PROJECT_ROOT)

MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "churn_model.pkl")
THRESHOLD = 0.5


def predict(customers, model=None):
    """Return churn probability and Yes/No prediction for each customer row."""
    model = model or joblib.load(MODEL_PATH)
    proba = model.predict_proba(customers)[:, 1]
    result = pd.DataFrame(index=customers.index)
    if "customerID" in customers:
        result["customerID"] = customers["customerID"]
    result["churn_probability"] = proba.round(3)
    result["churn_prediction"] = ["Yes" if p >= THRESHOLD else "No" for p in proba]
    return result


def main():
    if len(sys.argv) > 1:
        input_path = sys.argv[1]
        customers = pd.read_csv(input_path)
        result = predict(customers)
        output_path = os.path.splitext(input_path)[0] + "_predictions.csv"
        result.to_csv(output_path, index=False)
        print(result.head(10).to_string(index=False))
        print(f"\nSaved {len(result)} predictions to {output_path}")
    else:
        customers = pd.read_csv(os.path.join(PROJECT_ROOT, "data", "telco_churn.csv")).head(5)
        result = predict(customers)
        result["actual_churn"] = customers["Churn"]
        print(result.to_string(index=False))


if __name__ == "__main__":
    main()
