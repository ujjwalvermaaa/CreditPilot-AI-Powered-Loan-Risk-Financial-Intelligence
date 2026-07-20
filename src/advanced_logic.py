import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from logic import (
    calculate_emi,
    calculate_fsi,
    check_affordability,
    final_decision,
    get_risk,
    loan_health_score,
    prepare_features,
    stress_impact,
    suggest_improvements,
)


BASE_DIR = Path(__file__).resolve().parent.parent
HISTORY_DIR = BASE_DIR / "data" / "app_history"
HISTORY_FILE = HISTORY_DIR / "profile_history.json"

LOAN_CONFIG: Dict[str, Dict[str, Any]] = {
    "personal": {"label": "Personal", "terms": [12, 24, 36, 48, 60], "cap_multiple": 0.45},
    "auto": {"label": "Car", "terms": [24, 36, 48, 60, 84], "cap_multiple": 0.6},
    "education": {"label": "Education", "terms": [24, 36, 48, 60, 84, 120], "cap_multiple": 0.5},
    "home": {"label": "Home", "terms": [120, 180, 240, 300], "cap_multiple": 0.9},
}


def format_currency(value: float) -> str:
    return f"Rs. {value:,.0f}"


def to_model_payload(payload: Dict[str, Any], loan_type: Optional[str] = None) -> Dict[str, Any]:
    data = dict(payload)
    if loan_type is not None:
        data["loan_type"] = loan_type
    data["emi"] = calculate_emi(data["loan_amount"], data["interest_rate"], int(data["loan_term_months"]))
    data = prepare_features(data)
    return data


def build_payload_from_form(
    age: int,
    employment_type: str,
    employment_length_years: int,
    annual_income: float,
    income_stability_score: float,
    loan_amount: float,
    loan_term_months: int,
    interest_rate: float,
    loan_type: str,
    total_existing_loans: int,
    total_debt: float,
    credit_score: int,
    credit_utilization_ratio: float,
    num_credit_inquiries: int,
    late_payment_count: int,
    default_history: int,
    repayment_history_score: float,
    credit_history_length_years: int,
    job_stability_score: float,
    spending_to_income_ratio: float,
) -> Dict[str, Any]:
    return {
        "age": int(age),
        "employment_type": employment_type,
        "employment_length_years": int(employment_length_years),
        "annual_income": float(annual_income),
        "income_stability_score": float(income_stability_score),
        "loan_amount": float(loan_amount),
        "loan_term_months": int(loan_term_months),
        "interest_rate": float(interest_rate),
        "loan_type": loan_type,
        "total_existing_loans": int(total_existing_loans),
        "total_debt": float(total_debt),
        "credit_score": int(credit_score),
        "credit_utilization_ratio": float(credit_utilization_ratio),
        "num_credit_inquiries": int(num_credit_inquiries),
        "late_payment_count": int(late_payment_count),
        "default_history": int(default_history),
        "repayment_history_score": float(repayment_history_score),
        "credit_history_length_years": int(credit_history_length_years),
        "job_stability_score": float(job_stability_score),
        "spending_to_income_ratio": float(spending_to_income_ratio),
    }


def build_feature_frame(payload: Dict[str, Any]) -> pd.DataFrame:
    data = to_model_payload(payload)
    cols = [
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
        "credit_risk_score",
    ]
    return pd.DataFrame([{col: data[col] for col in cols}])


def _bundle_metrics(bundle: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if isinstance(bundle, dict):
        return bundle.get("metrics", {})
    return {}


def predict_approval_probability(bundle: Optional[Dict[str, Any]], payload: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
    info: Dict[str, Any] = {"mode": "logic_fallback"}
    model_payload = to_model_payload(payload)
    if isinstance(bundle, dict) and bundle.get("approval_model") is not None:
        features = pd.DataFrame([model_payload])
        try:
            prob = float(bundle["approval_model"].predict_proba(features)[0, 1])
            info["mode"] = "advanced_bundle"
            info["model_metrics"] = _bundle_metrics(bundle).get("approval")
            return prob, info
        except Exception as exc:
            info["bundle_error"] = str(exc)
    risk = (model_payload["credit_risk_score"] - 50) / 40.0
    return float(1 / (1 + np.exp(-risk))), info


def estimate_interest_rate(bundle: Optional[Dict[str, Any]], payload: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
    info: Dict[str, Any] = {"mode": "heuristic_rate"}
    model_payload = to_model_payload(payload)
    if isinstance(bundle, dict) and bundle.get("interest_model") is not None:
        features = pd.DataFrame([model_payload])
        try:
            rate = float(bundle["interest_model"].predict(features)[0])
            rate = float(np.clip(rate, 6.5, 18.5))
            info["mode"] = "advanced_bundle"
            info["model_metrics"] = _bundle_metrics(bundle).get("interest")
            return rate, info
        except Exception as exc:
            info["bundle_error"] = str(exc)

    base_rate = 8.0
    score_penalty = max(0.0, (700 - model_payload["credit_score"]) / 55.0)
    dti_penalty = max(0.0, (model_payload["debt_to_income_ratio"] - 0.35) * 7.5)
    util_penalty = max(0.0, (model_payload["credit_utilization_ratio"] - 0.3) * 6.0)
    stability_discount = max(0.0, (model_payload["income_stability_score"] - 60) / 40.0)
    loan_type_penalty = {"personal": 2.2, "auto": 1.6, "education": 1.3, "home": 0.8}
    rate = base_rate + score_penalty + dti_penalty + util_penalty + loan_type_penalty.get(model_payload["loan_type"], 1.8) - stability_discount
    return float(np.clip(rate, 6.5, 18.5)), info


def evaluate_application(bundle: Optional[Dict[str, Any]], payload: Dict[str, Any]) -> Dict[str, Any]:
    prob, prob_info = predict_approval_probability(bundle, payload)
    logic_payload = to_model_payload(payload)
    fsi = calculate_fsi(logic_payload)
    risk = get_risk(prob, fsi)
    decision = final_decision(prob, fsi)
    health = loan_health_score(prob, fsi)
    emi = logic_payload["emi"]
    affordability = check_affordability(emi, logic_payload["annual_income"] / 12)
    rejection = rejection_breakdown(logic_payload, prob, fsi)
    suggestions = combined_suggestions(logic_payload, prob, fsi)
    early_warning = default_early_warning(logic_payload, prob, fsi)
    return {
        "payload": logic_payload,
        "probability": prob,
        "probability_info": prob_info,
        "fsi": fsi,
        "risk": risk,
        "decision": decision,
        "health_score": health,
        "affordability": affordability,
        "emi": emi,
        "rejection_breakdown": rejection,
        "suggestions": suggestions,
        "early_warning": early_warning,
    }


def rejection_breakdown(data: Dict[str, Any], prob: float, fsi: float) -> List[Dict[str, Any]]:
    raw_scores = {
        "Low credit score": max(0.0, (680 - data["credit_score"]) / 180),
        "High debt-to-income": max(0.0, (data["debt_to_income_ratio"] - 0.35) / 0.35),
        "High EMI burden": max(0.0, (data["emi_to_income_ratio"] - 0.35) / 0.35),
        "High credit utilization": max(0.0, (data["credit_utilization_ratio"] - 0.45) / 0.4),
        "Recent repayment stress": max(0.0, (70 - data["repayment_history_score"]) / 50),
        "Default probability pressure": max(0.0, (prob - 0.45) / 0.55),
        "Low financial stability": max(0.0, (65 - fsi) / 35),
    }
    total = sum(raw_scores.values())
    if total <= 0:
        return [{"reason": "Profile looks stable", "share": 100.0, "detail": "No major rejection drivers detected."}]
    rows = []
    for reason, score in raw_scores.items():
        if score <= 0:
            continue
        rows.append(
            {
                "reason": reason,
                "share": round(score / total * 100, 1),
                "detail": reason_detail(reason, data, prob, fsi),
            }
        )
    rows.sort(key=lambda item: item["share"], reverse=True)
    return rows


def reason_detail(reason: str, data: Dict[str, Any], prob: float, fsi: float) -> str:
    mapping = {
        "Low credit score": f"Credit score is {int(data['credit_score'])}, which weakens pricing and approval confidence.",
        "High debt-to-income": f"Debt-to-income ratio is {data['debt_to_income_ratio']:.2f}; lenders usually prefer a lower burden.",
        "High EMI burden": f"EMI-to-income ratio is {data['emi_to_income_ratio']:.2f}, showing monthly pressure.",
        "High credit utilization": f"Credit utilization is {data['credit_utilization_ratio']:.2f}; lower usage improves risk profile.",
        "Recent repayment stress": f"Repayment history score is {data['repayment_history_score']:.1f}, indicating uneven discipline.",
        "Default probability pressure": f"Predicted default probability is {prob:.1%}.",
        "Low financial stability": f"Financial Stability Index is {fsi:.1f} on a 0-100 scale.",
    }
    return mapping.get(reason, reason)


def combined_suggestions(data: Dict[str, Any], prob: float, fsi: float) -> List[str]:
    suggestions = list(suggest_improvements(data))
    if data["credit_utilization_ratio"] > 0.5:
        suggestions.append("Bring credit utilization below 30% to improve approval strength.")
    if data["debt_to_income_ratio"] > 0.45:
        suggestions.append("Reduce debt before applying or stretch tenure to ease debt pressure.")
    if data["emi_to_income_ratio"] > 0.4:
        suggestions.append("Lower loan amount or add a co-applicant to keep EMI manageable.")
    if prob > 0.55:
        suggestions.append("Apply after 3-6 months of stronger repayment behavior to improve probability.")
    if fsi < 60:
        suggestions.append("Increase emergency buffer and income stability for a better financial health score.")
    seen = set()
    unique = []
    for item in suggestions:
        if item not in seen:
            unique.append(item)
            seen.add(item)
    return unique


def credit_utilization_analyzer(utilization: float) -> Dict[str, str]:
    if utilization < 0.3:
        level = "Healthy"
        impact = "Low risk impact"
        suggestion = "Current utilization is well controlled."
    elif utilization < 0.6:
        level = "Watch"
        impact = "Moderate risk impact"
        suggestion = "Try to reduce balances before the next billing cycle."
    else:
        level = "High"
        impact = "Strong negative impact"
        suggestion = "High utilization is lifting risk. Pay down revolving debt quickly."
    return {"level": level, "impact": impact, "suggestion": suggestion}


def default_early_warning(data: Dict[str, Any], prob: float, fsi: float) -> Dict[str, str]:
    stress_index = (
        prob * 0.45
        + min(1.0, data["emi_to_income_ratio"]) * 0.25
        + min(1.0, data["debt_to_income_ratio"]) * 0.2
        + data["credit_utilization_ratio"] * 0.1
    )
    if stress_index > 0.7 or fsi < 50:
        level = "High"
        note = "Profile shows elevated future default risk under normal conditions."
    elif stress_index > 0.48 or fsi < 65:
        level = "Moderate"
        note = "Profile is workable but sensitive to income or rate shocks."
    else:
        level = "Low"
        note = "Profile appears resilient with manageable payment stress."
    return {"level": level, "note": note}


def generate_probability_curve(bundle: Optional[Dict[str, Any]], payload: Dict[str, Any]) -> pd.DataFrame:
    base = dict(payload)
    rows = []
    for credit_score in range(500, 851, 25):
        sim = dict(base)
        sim["credit_score"] = credit_score
        prob, _ = predict_approval_probability(bundle, sim)
        fsi = calculate_fsi(to_model_payload(sim))
        rows.append({"credit_score": credit_score, "approval_chance": round((1 - prob) * 100, 2), "fsi": fsi})
    return pd.DataFrame(rows)


def variable_impact_table(bundle: Optional[Dict[str, Any]], payload: Dict[str, Any]) -> pd.DataFrame:
    base = evaluate_application(bundle, payload)
    candidates = [
        ("Credit Score +50", {"credit_score": min(850, payload["credit_score"] + 50)}),
        ("Income +20%", {"annual_income": payload["annual_income"] * 1.2}),
        ("Loan Amount -15%", {"loan_amount": payload["loan_amount"] * 0.85}),
        ("Utilization -15%", {"credit_utilization_ratio": max(0.0, payload["credit_utilization_ratio"] - 0.15)}),
        ("Tenure +24m", {"loan_term_months": payload["loan_term_months"] + 24}),
    ]
    rows = []
    for label, patch in candidates:
        sim = dict(payload)
        sim.update(patch)
        result = evaluate_application(bundle, sim)
        rows.append(
            {
                "scenario": label,
                "approval_change": round((1 - result["probability"] - (1 - base["probability"])) * 100, 2),
                "fsi_change": round(result["fsi"] - base["fsi"], 2),
                "emi_change": round(result["emi"] - base["emi"], 2),
            }
        )
    return pd.DataFrame(rows)


def safe_loan_limit(payload: Dict[str, Any]) -> float:
    monthly_income = payload["annual_income"] / 12
    max_emi = monthly_income * 0.35
    term = max(int(payload["loan_term_months"]), 12)
    rate = max(float(payload["interest_rate"]), 6.5)
    monthly_rate = rate / (12 * 100)
    if monthly_rate == 0:
        base_limit = max_emi * term
    else:
        base_limit = max_emi * (((1 + monthly_rate) ** term - 1) / (monthly_rate * (1 + monthly_rate) ** term))
    profile_factor = (
        payload["credit_score"] / 850 * 0.35
        + (1 - min(1.0, payload["credit_utilization_ratio"])) * 0.2
        + min(1.0, payload["income_stability_score"] / 100) * 0.25
        + (1 - min(1.0, payload["spending_to_income_ratio"])) * 0.2
    )
    return max(50000.0, base_limit * (0.8 + profile_factor))


def recommend_loans(bundle: Optional[Dict[str, Any]], payload: Dict[str, Any]) -> pd.DataFrame:
    rows = []
    safe_limit = safe_loan_limit(payload)
    for loan_type, cfg in LOAN_CONFIG.items():
        type_cap = safe_limit * cfg["cap_multiple"]
        candidate_amounts = sorted(
            {
                round(min(payload["loan_amount"], type_cap), -3),
                round(min(payload["loan_amount"] * 0.85, type_cap), -3),
                round(min(payload["loan_amount"] * 0.7, type_cap), -3),
            }
        )
        candidate_amounts = [amount for amount in candidate_amounts if amount >= 50000]
        for term in cfg["terms"]:
            for amount in candidate_amounts:
                sim = dict(payload)
                sim["loan_type"] = loan_type
                sim["loan_term_months"] = term
                sim["loan_amount"] = float(amount)
                sim["interest_rate"], _ = estimate_interest_rate(bundle, sim)
                result = evaluate_application(bundle, sim)
                approval = 1 - result["probability"]
                utility = approval * 0.55 + (result["health_score"] / 100) * 0.25
                utility -= min(1.0, result["payload"]["emi_to_income_ratio"]) * 0.2
                rows.append(
                    {
                        "loan_type": cfg["label"],
                        "loan_type_key": loan_type,
                        "loan_amount": sim["loan_amount"],
                        "loan_term_months": term,
                        "interest_rate": sim["interest_rate"],
                        "emi": result["emi"],
                        "approval_chance": round(approval * 100, 2),
                        "risk": result["risk"],
                        "decision": result["decision"],
                        "health_score": result["health_score"],
                        "utility_score": round(utility * 100, 2),
                    }
                )
    rec_df = pd.DataFrame(rows)
    rec_df = rec_df.sort_values(["utility_score", "approval_chance", "health_score"], ascending=False)
    return rec_df.head(12).reset_index(drop=True)


def coapplicant_optimization(bundle: Optional[Dict[str, Any]], payload: Dict[str, Any], co_income: float, co_credit_score: int, debt_share_ratio: float) -> Dict[str, Any]:
    base = evaluate_application(bundle, payload)
    sim = dict(payload)
    sim["annual_income"] = payload["annual_income"] + max(0.0, co_income)
    sim["credit_score"] = int(round((payload["credit_score"] * 0.65) + (co_credit_score * 0.35)))
    sim["total_debt"] = max(0.0, payload["total_debt"] * (1 - debt_share_ratio))
    sim["income_stability_score"] = min(100.0, payload["income_stability_score"] + 8)
    sim["job_stability_score"] = min(100.0, payload["job_stability_score"] + 5)
    upgraded = evaluate_application(bundle, sim)
    delta_approval = round(((1 - upgraded["probability"]) - (1 - base["probability"])) * 100, 2)
    return {
        "base": base,
        "with_coapplicant": upgraded,
        "approval_gain": delta_approval,
        "should_add": delta_approval > 5 or upgraded["decision"] != base["decision"],
    }


def compare_loan_scenarios(bundle: Optional[Dict[str, Any]], scenarios: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for item in scenarios:
        label = item["label"]
        payload = dict(item["payload"])
        result = evaluate_application(bundle, payload)
        schedule = amortization_schedule(
            payload["loan_amount"], payload["interest_rate"], int(payload["loan_term_months"]), extra_payment=0.0
        )
        rows.append(
            {
                "scenario": label,
                "loan_type": payload["loan_type"].title(),
                "loan_amount": payload["loan_amount"],
                "tenure_months": payload["loan_term_months"],
                "rate": payload["interest_rate"],
                "emi": result["emi"],
                "approval_chance": round((1 - result["probability"]) * 100, 2),
                "total_interest": round(float(schedule["interest"].sum()), 2),
                "decision": result["decision"],
            }
        )
    return pd.DataFrame(rows)


def amortization_schedule(loan_amount: float, annual_rate: float, months: int, extra_payment: float = 0.0) -> pd.DataFrame:
    r = annual_rate / (12 * 100.0)
    emi = calculate_emi(loan_amount, annual_rate, months)
    balance = float(loan_amount)
    rows = []
    month = 1
    while month <= months and balance > 0:
        interest = balance * r
        principal = min(balance, max(0.0, emi - interest) + extra_payment)
        balance = max(0.0, balance - principal)
        rows.append({"month": month, "interest": interest, "principal": principal, "balance": balance})
        month += 1
        if balance <= 0:
            break
    return pd.DataFrame(rows)


def prepayment_summary(loan_amount: float, annual_rate: float, months: int, extra_payment: float) -> Dict[str, float]:
    base = amortization_schedule(loan_amount, annual_rate, months, extra_payment=0.0)
    new = amortization_schedule(loan_amount, annual_rate, months, extra_payment=extra_payment)
    base_interest = float(base["interest"].sum())
    new_interest = float(new["interest"].sum())
    base_months = int(base["month"].max()) if not base.empty else 0
    new_months = int(new["month"].max()) if not new.empty else 0
    return {
        "interest_saved": round(base_interest - new_interest, 2),
        "months_saved": max(0, base_months - new_months),
        "base_interest": round(base_interest, 2),
        "new_interest": round(new_interest, 2),
    }


def financial_goal_plan(_bundle: Optional[Dict[str, Any]], payload: Dict[str, Any], goal_type: str, down_payment: float, years_to_goal: int) -> Dict[str, Any]:
    horizon_factor = max(1.0, years_to_goal / 2)
    safe_limit = safe_loan_limit(payload) * min(1.2, 0.85 + 0.08 * horizon_factor)
    goal_caps = {"Buy House": 1.15, "Buy Car": 0.65, "Higher Education": 0.55, "Business Expansion": 0.7}
    target_loan = safe_limit * goal_caps.get(goal_type, 0.7)
    target_loan = max(50000.0, target_loan - down_payment * 0.2)
    monthly_income = payload["annual_income"] / 12
    comfortable_emi = monthly_income * 0.32
    return {
        "goal_type": goal_type,
        "safe_loan_amount": round(target_loan, 2),
        "comfortable_emi": round(comfortable_emi, 2),
        "years_to_goal": years_to_goal,
        "down_payment": down_payment,
        "note": f"You can target around {format_currency(target_loan)} safely for {goal_type.lower()}.",
    }


def applicant_benchmark(payload: Dict[str, Any], raw_df: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        ("Credit Score", payload["credit_score"], float(raw_df["credit_score"].median()), "Higher is better"),
        ("Annual Income", payload["annual_income"], float(raw_df["annual_income"].median()), "Higher is better"),
        ("Debt To Income", payload["total_debt"] / max(payload["annual_income"], 1.0), float(raw_df["debt_to_income_ratio"].median()), "Lower is better"),
        ("Credit Utilization", payload["credit_utilization_ratio"], float(raw_df["credit_utilization_ratio"].median()), "Lower is better"),
        ("Repayment Score", payload["repayment_history_score"], float(raw_df["repayment_history_score"].median()), "Higher is better"),
    ]
    rows = []
    for label, applicant_value, portfolio_value, rule in metrics:
        if rule == "Higher is better":
            status = "Stronger" if applicant_value >= portfolio_value else "Needs improvement"
        else:
            status = "Stronger" if applicant_value <= portfolio_value else "Needs improvement"
        rows.append(
            {
                "metric": label,
                "applicant_value": round(float(applicant_value), 2),
                "portfolio_median": round(float(portfolio_value), 2),
                "status": status,
            }
        )
    return pd.DataFrame(rows)


def risk_heatmap_frame(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work["credit_band"] = pd.cut(work["credit_score"], bins=[300, 600, 680, 750, 850], labels=["300-600", "600-680", "680-750", "750-850"], include_lowest=True)
    work["dti_band"] = pd.cut(work["debt_to_income_ratio"], bins=[0, 0.5, 1, 2, 10], labels=["0-0.5", "0.5-1", "1-2", "2+"], include_lowest=True)
    pivot = work.pivot_table(index="dti_band", columns="credit_band", values="loan_status", aggfunc="mean", observed=False)
    return pivot.fillna(0.0)


def approval_trend_frame(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work["approval_flag"] = 1 - work["loan_status"]
    return work.groupby("loan_type", as_index=False).agg(
        approval_rate=("approval_flag", "mean"),
        avg_interest_rate=("interest_rate", "mean"),
        avg_credit_score=("credit_score", "mean"),
    )


def portfolio_summary(df: pd.DataFrame) -> Dict[str, Any]:
    work = df.copy()
    work["approval_flag"] = 1 - work["loan_status"]
    work["risk_flag"] = np.where(
        (work["credit_score"] < 650)
        | (work["credit_utilization_ratio"] > 0.6)
        | (work["debt_to_income_ratio"] > 0.7),
        "High",
        np.where((work["credit_score"] < 720) | (work["debt_to_income_ratio"] > 0.45), "Medium", "Low"),
    )
    total_approved = int(work["approval_flag"].sum())
    high_risk_share = float((work["risk_flag"] == "High").mean())
    return {
        "total_approved": total_approved,
        "portfolio_size": int(work.shape[0]),
        "approval_rate": float(work["approval_flag"].mean()),
        "high_risk_share": high_risk_share,
        "warning": "Warning: high-risk mix is elevated." if high_risk_share > 0.3 else "Portfolio mix is within a healthy range.",
    }


def save_profile_event(event: Dict[str, Any]) -> None:
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    history = load_profile_history()
    history.append(event)
    HISTORY_FILE.write_text(json.dumps(history, indent=2))


def load_profile_history() -> List[Dict[str, Any]]:
    if not HISTORY_FILE.exists():
        return []
    try:
        return json.loads(HISTORY_FILE.read_text())
    except Exception:
        return []


def build_history_frame() -> pd.DataFrame:
    history = load_profile_history()
    if not history:
        return pd.DataFrame(columns=["timestamp", "name", "loan_type", "loan_amount", "approval_chance", "decision", "risk"])
    rows = []
    for item in history:
        rows.append(
            {
                "timestamp": item.get("timestamp"),
                "name": item.get("name", "Applicant"),
                "loan_type": item.get("loan_type", "").title(),
                "loan_amount": item.get("loan_amount"),
                "approval_chance": item.get("approval_chance"),
                "decision": item.get("decision"),
                "risk": item.get("risk"),
            }
        )
    return pd.DataFrame(rows)


def assistant_action_plan(payload: Dict[str, Any], bundle: Optional[Dict[str, Any]] = None) -> List[str]:
    result = evaluate_application(bundle, payload)
    actions = list(result["suggestions"][:4])
    if payload["credit_score"] < 700:
        actions.append("Focus on improving credit score before asking for a larger ticket size.")
    if payload["credit_utilization_ratio"] > 0.5:
        actions.append("Pay down revolving credit first because utilization is hurting approval quality.")
    if result["affordability"] != "Affordable":
        actions.append("Reduce EMI pressure by extending tenure or lowering requested amount.")
    if result["decision"] != "APPROVED":
        actions.append("Reapply with a stronger profile snapshot after the next repayment cycle.")
    seen = set()
    unique = []
    for item in actions:
        if item not in seen:
            unique.append(item)
            seen.add(item)
    return unique[:5]


def assistant_response(choice: str, payload: Optional[Dict[str, Any]] = None, bundle: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    if payload is None:
        return {
            "title": "Assistant Ready",
            "answer": "Open Smart Application, enter a borrower profile, and I will generate personalized guidance.",
            "bullets": [
                "I can explain eligibility.",
                "I can suggest better loan structures.",
                "I can show EMI and prepayment guidance.",
            ],
            "highlights": [],
        }

    result = evaluate_application(bundle, payload)
    approval_chance = (1 - result["probability"]) * 100
    recs = recommend_loans(bundle, payload)
    top = recs.iloc[0]

    if choice == "Check EMI":
        return {
            "title": "EMI Guidance",
            "answer": f"Estimated EMI is {format_currency(result['emi'])} per month and the profile is currently marked as {result['affordability'].lower()}.",
            "bullets": [
                f"Current tenure is {payload['loan_term_months']} months.",
                f"Monthly burden ratio is {result['payload']['emi_to_income_ratio']:.2f}.",
                "Use the EMI and Prepayment tab to test lower tenure or faster closure.",
            ],
            "highlights": [
                ("EMI", format_currency(result["emi"])),
                ("Affordability", result["affordability"]),
            ],
        }
    if choice == "Loan eligibility help":
        return {
            "title": "Eligibility Guidance",
            "answer": f"Current decision trend is {result['decision']} with {approval_chance:.1f}% approval chance and {result['risk']} classification.",
            "bullets": [
                f"FSI is {result['fsi']:.1f}, which reflects financial stability.",
                f"Early warning level is {result['early_warning']['level']}.",
                result["rejection_breakdown"][0]["detail"],
            ],
            "highlights": [
                ("Approval", f"{approval_chance:.1f}%"),
                ("FSI", f"{result['fsi']:.1f}"),
                ("Decision", result["decision"]),
            ],
        }
    if choice == "Improve approval chances":
        return {
            "title": "Approval Improvement Plan",
            "answer": "These are the strongest next steps to improve approval quality for the current profile.",
            "bullets": assistant_action_plan(payload, bundle),
            "highlights": [
                ("Top Risk", result["rejection_breakdown"][0]["reason"]),
                ("Health Score", f"{result['health_score']:.1f}"),
            ],
        }
    if choice == "Best loan type":
        return {
            "title": "Best Loan Match",
            "answer": f"Best current fit is a {top['loan_type']} loan with {int(top['loan_term_months'])} months tenure and {format_currency(float(top['loan_amount']))} amount.",
            "bullets": [
                f"Estimated approval chance is {float(top['approval_chance']):.1f}%.",
                f"Expected EMI is {format_currency(float(top['emi']))}.",
                f"Recommended decision outlook is {top['decision']}.",
            ],
            "highlights": [
                ("Loan Type", top["loan_type"]),
                ("Tenure", f"{int(top['loan_term_months'])}m"),
                ("Amount", format_currency(float(top["loan_amount"]))),
            ],
        }
    if choice == "Can I lower my EMI":
        lower_amount = max(50000.0, payload["loan_amount"] * 0.85)
        lower_tenure = payload["loan_term_months"] + 24
        new_emi = calculate_emi(lower_amount, payload["interest_rate"], lower_tenure)
        return {
            "title": "EMI Reduction Strategy",
            "answer": f"If you reduce the loan amount to {format_currency(lower_amount)} and extend tenure to {lower_tenure} months, EMI can move near {format_currency(new_emi)}.",
            "bullets": [
                "Lower amount has the fastest EMI impact.",
                "Longer tenure reduces EMI but may increase total interest.",
                "Use prepayment later to recover some interest cost.",
            ],
            "highlights": [
                ("Current EMI", format_currency(result["emi"])),
                ("Target EMI", format_currency(new_emi)),
            ],
        }
    if choice == "Should I add co-applicant":
        return {
            "title": "Co-Applicant Advice",
            "answer": "A co-applicant is most useful when income support and risk balancing are both needed.",
            "bullets": [
                "Add a co-applicant if EMI burden feels high relative to income.",
                "It helps more when the co-applicant has stronger credit than the primary borrower.",
                "Use Recommendation Lab to test income and credit score improvements live.",
            ],
            "highlights": [
                ("Current EMI Burden", f"{result['payload']['emi_to_income_ratio']:.2f}"),
                ("Current Decision", result["decision"]),
            ],
        }
    if choice == "Rate negotiation tips":
        est_rate, _ = estimate_interest_rate(bundle, payload)
        return {
            "title": "Rate Negotiation Tips",
            "answer": f"Your smart estimated rate is around {est_rate:.2f}%, which you can use as a benchmark while negotiating.",
            "bullets": [
                "Negotiate after improving credit utilization and repayment behavior.",
                "A stronger co-applicant can help justify better pricing.",
                "Lower requested amount and stronger down payment also support a better offer.",
            ],
            "highlights": [
                ("Estimated Rate", f"{est_rate:.2f}%"),
                ("Credit Score", f"{payload['credit_score']}"),
            ],
        }

    return {
        "title": "Assistant Ready",
        "answer": "Choose an action and I will guide you using the current borrower profile.",
        "bullets": [],
        "highlights": [],
    }


def stress_delta_summary(base_prob: float, stress_prob: float) -> Dict[str, Any]:
    return {
        "base_approval": round((1 - base_prob) * 100, 2),
        "stress_approval": round((1 - stress_prob) * 100, 2),
        "impact": stress_impact(base_prob, stress_prob),
    }
