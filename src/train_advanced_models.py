from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DATA = BASE_DIR / "data" / "raw" / "creditpilot_dataset.csv"
MODEL_PATH = BASE_DIR / "models" / "advanced_bundle.pkl"


def build_preprocessor(df: pd.DataFrame) -> ColumnTransformer:
    categorical = ["employment_type", "loan_type"]
    numeric = [col for col in df.columns if col not in categorical]
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
        ]
    )


def train_bundle() -> dict:
    df = pd.read_csv(RAW_DATA)

    approval_features = [
        "age",
        "employment_type",
        "employment_length_years",
        "annual_income",
        "income_stability_score",
        "loan_amount",
        "loan_term_months",
        "interest_rate",
        "loan_type",
        "emi",
        "total_existing_loans",
        "total_debt",
        "debt_to_income_ratio",
        "credit_score",
        "credit_utilization_ratio",
        "emi_to_income_ratio",
        "loan_to_income_ratio",
        "num_credit_inquiries",
        "late_payment_count",
        "default_history",
        "repayment_history_score",
        "credit_history_length_years",
        "job_stability_score",
        "spending_to_income_ratio",
    ]
    X_cls = df[approval_features]
    y_cls = df["loan_status"]
    X_train, X_test, y_train, y_test = train_test_split(
        X_cls, y_cls, test_size=0.2, random_state=42, stratify=y_cls
    )
    cls_pipe = Pipeline(
        steps=[
            ("prep", build_preprocessor(X_cls)),
            ("model", RandomForestClassifier(n_estimators=240, max_depth=10, min_samples_leaf=4, random_state=42)),
        ]
    )
    cls_pipe.fit(X_train, y_train)
    cls_pred = cls_pipe.predict(X_test)
    cls_prob = cls_pipe.predict_proba(X_test)[:, 1]

    interest_features = [
        "age",
        "employment_type",
        "employment_length_years",
        "annual_income",
        "income_stability_score",
        "loan_amount",
        "loan_term_months",
        "loan_type",
        "emi",
        "total_existing_loans",
        "total_debt",
        "debt_to_income_ratio",
        "credit_score",
        "credit_utilization_ratio",
        "emi_to_income_ratio",
        "loan_to_income_ratio",
        "num_credit_inquiries",
        "late_payment_count",
        "default_history",
        "repayment_history_score",
        "credit_history_length_years",
        "job_stability_score",
        "spending_to_income_ratio",
        "loan_status",
    ]
    X_reg = df[interest_features]
    y_reg = df["interest_rate"]
    Xr_train, Xr_test, yr_train, yr_test = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)
    reg_pipe = Pipeline(
        steps=[
            ("prep", build_preprocessor(X_reg)),
            ("model", RandomForestRegressor(n_estimators=220, max_depth=12, min_samples_leaf=3, random_state=42)),
        ]
    )
    reg_pipe.fit(Xr_train, yr_train)
    reg_pred = reg_pipe.predict(Xr_test)

    bundle = {
        "approval_model": cls_pipe,
        "interest_model": reg_pipe,
        "metrics": {
            "approval": {
                "accuracy": float(accuracy_score(y_test, cls_pred)),
                "roc_auc": float(roc_auc_score(y_test, cls_prob)),
            },
            "interest": {
                "mae": float(mean_absolute_error(yr_test, reg_pred)),
                "r2": float(r2_score(yr_test, reg_pred)),
            },
        },
        "feature_columns": {
            "approval": approval_features,
            "interest": interest_features,
        },
    }
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, MODEL_PATH)
    return bundle


if __name__ == "__main__":
    trained = train_bundle()
    print("Saved:", MODEL_PATH)
    print("Approval metrics:", trained["metrics"]["approval"])
    print("Interest metrics:", trained["metrics"]["interest"])
