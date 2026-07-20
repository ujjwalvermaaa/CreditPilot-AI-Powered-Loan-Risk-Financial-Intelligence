from pathlib import Path
import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
import xgboost as xgb


BASE = Path(__file__).resolve().parent.parent
RAW_PATH = BASE / "data" / "raw" / "creditpilot_dataset.csv"
PROC_PATH = BASE / "data" / "processed" / "final_dataset.csv"
ASSETS = BASE / "paper_assets"
ASSETS.mkdir(exist_ok=True)


def generate_assets() -> dict:
    raw = pd.read_csv(RAW_PATH)
    proc = pd.read_csv(PROC_PATH)
    bundle = joblib.load(BASE / "models" / "advanced_bundle.pkl")

    summary = {
        "raw_shape": list(raw.shape),
        "processed_shape": list(proc.shape),
        "default_rate": float(raw["loan_status"].mean()),
        "approval_rate": float(1 - raw["loan_status"].mean()),
        "loan_type_counts": raw["loan_type"].value_counts().to_dict(),
        "employment_counts": raw["employment_type"].value_counts().to_dict(),
        "credit_score_mean": float(raw["credit_score"].mean()),
        "interest_rate_mean": float(raw["interest_rate"].mean()),
        "loan_amount_mean": float(raw["loan_amount"].mean()),
        "dti_mean": float(raw["debt_to_income_ratio"].mean()),
        "util_mean": float(raw["credit_utilization_ratio"].mean()),
        "advanced_bundle_metrics": bundle["metrics"],
    }

    X = proc.drop(columns=["loan_status"])
    y = proc["loan_status"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "XGBoost": xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42),
    }

    perf_rows = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else pred
        perf_rows.append(
            {
                "Model": name,
                "Accuracy": accuracy_score(y_test, pred),
                "Precision": precision_score(y_test, pred),
                "Recall": recall_score(y_test, pred),
                "F1": f1_score(y_test, pred),
                "ROC_AUC": roc_auc_score(y_test, prob),
            }
        )

        if name == "XGBoost":
            cm = confusion_matrix(y_test, pred)
            plt.figure(figsize=(5, 4))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
            plt.title("Confusion Matrix - XGBoost")
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            plt.tight_layout()
            plt.savefig(ASSETS / "figure_confusion_matrix_xgb.png", dpi=220)
            plt.close()

            importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False).head(12)
            plt.figure(figsize=(8, 5))
            sns.barplot(x=importances.values, y=importances.index, hue=importances.index, dodge=False, palette="viridis", legend=False)
            plt.title("Top 12 Feature Importances - XGBoost")
            plt.xlabel("Importance")
            plt.ylabel("Feature")
            plt.tight_layout()
            plt.savefig(ASSETS / "figure_feature_importance.png", dpi=220)
            plt.close()

    perf_df = pd.DataFrame(perf_rows)
    perf_df.to_csv(ASSETS / "table_model_performance.csv", index=False)
    summary["benchmark_metrics"] = perf_rows

    plt.figure(figsize=(7, 4.5))
    sns.countplot(data=raw, x="loan_type", hue="loan_status", palette="Set2")
    plt.title("Loan Status Distribution by Loan Type")
    plt.xlabel("Loan Type")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(ASSETS / "figure_loan_type_status.png", dpi=220)
    plt.close()

    plt.figure(figsize=(7, 4.5))
    sns.histplot(data=raw, x="credit_score", hue="loan_status", bins=35, kde=True, palette="Set2")
    plt.title("Credit Score Distribution by Loan Status")
    plt.xlabel("Credit Score")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(ASSETS / "figure_credit_score_distribution.png", dpi=220)
    plt.close()

    corr_cols = [
        "annual_income",
        "loan_amount",
        "emi",
        "total_debt",
        "debt_to_income_ratio",
        "credit_score",
        "credit_utilization_ratio",
        "emi_to_income_ratio",
        "interest_rate",
        "repayment_history_score",
    ]
    plt.figure(figsize=(8, 6))
    sns.heatmap(raw[corr_cols].corr(numeric_only=True), cmap="coolwarm", annot=True, fmt=".2f")
    plt.title("Correlation Heatmap of Key Financial Variables")
    plt.tight_layout()
    plt.savefig(ASSETS / "figure_correlation_heatmap.png", dpi=220)
    plt.close()

    scenario_rows = []
    for income_drop, rate_up, util_up, label in [(10, 1, 0.05, "Mild"), (25, 3, 0.15, "Moderate"), (45, 6, 0.30, "Severe")]:
        stressed_income = raw["annual_income"] * (1 - income_drop / 100.0)
        avg_emi_ratio = (
            raw["emi"] * (1 + rate_up / 100.0) / (stressed_income / 12.0)
        ).replace([np.inf, -np.inf], np.nan).dropna().mean()
        avg_util = np.minimum(1.0, raw["credit_utilization_ratio"] + util_up).mean()
        scenario_rows.append(
            {
                "Scenario": label,
                "Income Drop (%)": income_drop,
                "Rate Increase (%)": rate_up,
                "Avg EMI-to-Income": float(avg_emi_ratio),
                "Avg Credit Utilization": float(avg_util),
            }
        )
    scenario_df = pd.DataFrame(scenario_rows)
    scenario_df.to_csv(ASSETS / "table_stress_scenarios.csv", index=False)
    summary["stress_scenarios"] = scenario_rows

    raw.describe(include="all").transpose().to_csv(ASSETS / "table_dataset_describe.csv")
    raw.groupby("loan_type")["loan_status"].agg(["count", "mean"]).to_csv(ASSETS / "table_loan_type_default_rate.csv")
    raw.groupby("employment_type")["loan_status"].agg(["count", "mean"]).to_csv(ASSETS / "table_employment_default_rate.csv")

    (ASSETS / "paper_metrics.json").write_text(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    print(json.dumps(generate_assets(), indent=2))
