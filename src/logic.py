import numpy as np

# -----------------------------
# 1. EMI CALCULATOR
# -----------------------------
def calculate_emi(loan_amount, interest_rate, tenure):
    monthly_rate = interest_rate / (12 * 100)

    emi = (loan_amount * monthly_rate * (1 + monthly_rate) ** tenure) / \
          ((1 + monthly_rate) ** tenure - 1)

    return round(emi, 2)


# -----------------------------
# 2. FEATURE ENGINEERING
# -----------------------------
def prepare_features(data):
    annual_income = data["annual_income"]
    loan_amount = data["loan_amount"]
    emi = data["emi"]
    total_debt = data["total_debt"]

    data["emi_to_income_ratio"] = emi / (annual_income / 12)
    data["loan_to_income_ratio"] = loan_amount / annual_income
    data["debt_to_income_ratio"] = total_debt / annual_income

    data["credit_risk_score"] = (
        (850 - data["credit_score"]) * 0.4 +
        data["debt_to_income_ratio"] * 100 * 0.3 +
        data["credit_utilization_ratio"] * 100 * 0.3
    )

    return data


# -----------------------------
# 3. FINANCIAL STABILITY INDEX
# -----------------------------
def calculate_fsi(data):
    credit_norm = (data["credit_score"] - 300) / (850 - 300)

    dti_score = 1 - data["debt_to_income_ratio"]
    emi_score = 1 - data["emi_to_income_ratio"]
    util_score = 1 - data["credit_utilization_ratio"]

    fsi = (
        0.25 * (data["income_stability_score"] / 100) +
        0.20 * credit_norm +
        0.20 * dti_score +
        0.15 * emi_score +
        0.20 * util_score
    ) * 100

    return round(fsi, 2)


# -----------------------------
# 4. RISK CATEGORY
# -----------------------------
def get_risk(prob, fsi):
    if prob > 0.7 or fsi < 50:
        return "High Risk"
    elif prob > 0.4 or fsi < 65:
        return "Medium Risk"
    else:
        return "Low Risk"


# -----------------------------
# 5. FINAL DECISION
# -----------------------------
def final_decision(prob, fsi):
    if prob < 0.4 and fsi > 70:
        return "APPROVED"
    elif prob < 0.6 and fsi > 55:
        return "CONDITIONAL"
    else:
        return "REJECTED"


# -----------------------------
# 6. AFFORDABILITY CHECK
# -----------------------------
def check_affordability(emi, monthly_income):
    ratio = emi / monthly_income

    if ratio < 0.3:
        return "Affordable"
    elif ratio < 0.5:
        return "Moderate Risk"
    else:
        return "Not Affordable"


# -----------------------------
# 7. EXPLAIN DECISION (XAI)
# -----------------------------
def explain_decision(data, prob, fsi):
    reasons = []

    if data["credit_score"] < 650:
        reasons.append("Low credit score")

    if data["debt_to_income_ratio"] > 0.4:
        reasons.append("High debt-to-income ratio")

    if data["emi_to_income_ratio"] > 0.4:
        reasons.append("High EMI burden")

    if data["credit_utilization_ratio"] > 0.6:
        reasons.append("High credit utilization")

    if prob > 0.6:
        reasons.append("High default probability")

    if fsi < 60:
        reasons.append("Low financial stability")

    return reasons


# -----------------------------
# 8. IMPROVEMENT SUGGESTIONS
# -----------------------------
def suggest_improvements(data):
    suggestions = []

    if data["credit_score"] < 700:
        suggestions.append("Improve credit score with timely payments")

    if data["debt_to_income_ratio"] > 0.4:
        suggestions.append("Reduce existing debt obligations")

    if data["emi_to_income_ratio"] > 0.4:
        suggestions.append("Consider reducing loan amount")

    if data["credit_utilization_ratio"] > 0.6:
        suggestions.append("Lower credit utilization")

    if data["income_stability_score"] < 60:
        suggestions.append("Increase income stability")

    return suggestions


# -----------------------------
# 9. LOAN HEALTH SCORE
# -----------------------------
def loan_health_score(prob, fsi):
    score = (1 - prob) * 50 + (fsi / 100) * 50
    return round(score, 2)


# -----------------------------
# 10. STRESS IMPACT ANALYSIS
# -----------------------------
def stress_impact(base_prob, stress_prob):
    change = stress_prob - base_prob

    if change > 0.3:
        return "Severely Impacted"
    elif change > 0.1:
        return "Moderately Impacted"
    else:
        return "Stable"