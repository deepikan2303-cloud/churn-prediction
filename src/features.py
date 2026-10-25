"""Cleaning and feature engineering shared by the notebook and predict.py.

The trained pipeline calls `add_features` as its first step, so raw customer
rows (same columns as the Telco CSV) can be passed straight to the model.
"""
import numpy as np
import pandas as pd

# Add-on services that are answered "Yes" / "No" (or "No internet/phone service")
SERVICE_COLS = [
    "PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
]

NUMERIC_FEATURES = [
    "SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges",
    "avg_monthly_spend", "num_services",
]

CATEGORICAL_FEATURES = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod", "tenure_group",
]


def clean(df):
    """Fix data types and drop columns that are not features."""
    df = df.drop(columns=["customerID", "Churn"], errors="ignore").copy()
    # TotalCharges is text with blanks for brand-new customers (tenure 0):
    # they have not been billed yet, so 0 is the correct value.
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)
    return df


def add_features(df):
    """Clean the raw columns and add engineered features."""
    df = clean(df)
    df["tenure_group"] = pd.cut(
        df["tenure"],
        bins=[-1, 12, 24, 48, np.inf],
        labels=["0-12", "13-24", "25-48", "49+"],
    ).astype(str)
    # Average spend per month so far; new customers fall back to the current bill
    df["avg_monthly_spend"] = np.where(
        df["tenure"] > 0,
        df["TotalCharges"] / df["tenure"].clip(lower=1),
        df["MonthlyCharges"],
    )
    df["num_services"] = (df[SERVICE_COLS] == "Yes").sum(axis=1) + (
        df["InternetService"] != "No"
    ).astype(int)
    return df
