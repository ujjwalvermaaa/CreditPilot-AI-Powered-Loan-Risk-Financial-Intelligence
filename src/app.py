import base64
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

try:
    from joblib import load as joblib_load
except Exception:
    joblib_load = None

from advanced_logic import (
    amortization_schedule,
    applicant_benchmark,
    assistant_response,
    approval_trend_frame,
    build_history_frame,
    build_payload_from_form,
    coapplicant_optimization,
    compare_loan_scenarios,
    credit_utilization_analyzer,
    estimate_interest_rate,
    evaluate_application,
    financial_goal_plan,
    format_currency,
    generate_probability_curve,
    load_profile_history,
    portfolio_summary,
    prepayment_summary,
    recommend_loans,
    risk_heatmap_frame,
    save_profile_event,
    stress_delta_summary,
    variable_impact_table,
)
from logic import calculate_emi


st.set_page_config(page_title="CreditPilot", page_icon="CP", layout="wide")

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DATA = BASE_DIR / "data" / "raw" / "creditpilot_dataset.csv"
PROCESSED_DATA = BASE_DIR / "data" / "processed" / "final_dataset.csv"
ADVANCED_BUNDLE = BASE_DIR / "models" / "advanced_bundle.pkl"
FINAL_MODEL = BASE_DIR / "models" / "final_model.pkl"
SCALER_MODEL = BASE_DIR / "models" / "scaler.pkl"


def _inject_style() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Sora:wght@600;700;800&display=swap');
        :root{
            --accent:#60a5fa;
            --accent-2:#22d3ee;
            --accent-3:#a78bfa;
            --bg:#07111f;
            --panel:rgba(12,20,37,0.68);
            --card:rgba(16,24,39,0.56);
            --text:#edf4ff;
            --muted:#b7c6d9;
            --stroke:rgba(255,255,255,0.14);
            --shadow:0 16px 40px rgba(2, 8, 23, 0.42);
        }
        [data-testid="stSidebar"] {display:none;}
        html, body, .stApp {background: var(--bg); color: var(--text); font-family: Inter, sans-serif;}
        .block-container {padding-top: 3.6rem; padding-bottom: 2rem; max-width: 1380px;}
        h1, h2, h3, h4, h5, h6 {font-family: Sora, Inter, sans-serif; color:#ffffff !important;}
        .cp-hero {
            padding: 28px;
            border-radius: 22px;
            background: linear-gradient(135deg, rgba(59,130,246,0.18), rgba(34,211,238,0.12), rgba(167,139,250,0.16));
            border: 1px solid rgba(255,255,255,0.18);
            box-shadow: var(--shadow);
            backdrop-filter: blur(18px);
            margin-bottom: 18px;
        }
        .cp-title {font-size: 34px; font-weight: 800; color:#fff; margin:0;}
        .cp-subtitle {font-size: 14px; color:#dbeafe; margin-top:8px;}
        .cp-pill {
            display:inline-block;
            padding:7px 12px;
            border-radius:999px;
            margin-top:12px;
            margin-right:8px;
            font-size:12px;
            font-weight:800;
            color:#08111d;
            background:linear-gradient(135deg, #67e8f9, #60a5fa, #c084fc);
        }
        .cp-card, .cp-metric {
            border-radius: 18px;
            padding: 16px 18px;
            background: linear-gradient(180deg, rgba(15,23,42,0.62), rgba(2,6,23,0.52));
            border: 1px solid rgba(255,255,255,0.14);
            box-shadow: var(--shadow);
            backdrop-filter: blur(14px);
        }
        .cp-metric-label {font-size:12px; color:#a5b4c7; font-weight:700;}
        .cp-metric-value {font-size:24px; font-weight:800; margin-top:4px;}
        .cp-metric-note {font-size:12px; color:#d7e5f8; margin-top:4px;}
        .cp-section {font-size: 20px; font-weight:800; color:#fff; margin: 10px 0 10px;}
        .cp-note {color:#d8e3f4; font-size:13px;}
        .stButton > button {
            background: linear-gradient(135deg, #67e8f9, #60a5fa);
            color: #08111d;
            border: 0;
            border-radius: 12px;
            font-weight: 800;
            box-shadow: 0 10px 24px rgba(96,165,250,0.28);
        }
        .stButton > button:hover {filter:brightness(1.05);}
        [data-testid="stTabs"] [role="tablist"] {
            gap: 8px;
            background: rgba(9,16,30,0.55);
            padding: 8px;
            border-radius: 16px;
            border: 1px solid rgba(255,255,255,0.12);
        }
        [data-testid="stTabs"] [role="tab"] {
            border-radius: 12px;
            color: #eef6ff;
            font-weight: 800;
            padding: 10px 16px;
            background: rgba(15,23,42,0.62);
            border: 1px solid rgba(255,255,255,0.10);
        }
        [data-testid="stTabs"] [aria-selected="true"] {
            background: linear-gradient(135deg, rgba(96,165,250,0.30), rgba(34,211,238,0.18));
            border: 1px solid rgba(255,255,255,0.24);
        }
        [data-testid="stPlotlyChart"] {
            border-radius: 18px;
            padding: 10px;
            background: rgba(12,20,37,0.52);
            border: 1px solid rgba(255,255,255,0.14);
            box-shadow: var(--shadow);
        }
        [data-testid="stTextInput"] > div,
        [data-testid="stNumberInput"] > div,
        [data-testid="stSelectbox"] > div,
        [data-testid="stSlider"] > div,
        [data-testid="stTextArea"] > div {
            background: rgba(12,20,37,0.55) !important;
            border-radius: 14px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    try:
        bg_path = Path(__file__).resolve().parent / "bg.jpeg"
        if bg_path.exists():
            b64 = base64.b64encode(bg_path.read_bytes()).decode()
            st.markdown(
                f"""
                <style>
                .stApp {{
                    background-image: linear-gradient(rgba(3,8,18,.72), rgba(3,8,18,.82)), url("data:image/jpeg;base64,{b64}");
                    background-size: cover;
                    background-attachment: fixed;
                    background-position: center;
                }}
                [data-testid="stHeader"], [data-testid="stToolbar"] {{
                    background: transparent !important;
                }}
                </style>
                """,
                unsafe_allow_html=True,
            )
    except Exception:
        pass


def header() -> None:
    st.markdown(
        """
        <div class="cp-hero">
            <div class="cp-title">CreditPilot - Advanced AI Loan Intelligence Suite</div>
            <div class="cp-subtitle">Advisory engine, personalized pricing, prepayment lab, admin dashboards, history tracking, and a smart credit assistant.</div>
            <div class="cp-pill">Recommendation Engine</div>
            <div class="cp-pill">Interest Estimator</div>
            <div class="cp-pill">Risk Intelligence</div>
            <div class="cp-pill">Portfolio Dashboard</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def kpi_row(items):
    cols = st.columns(len(items))
    for idx, (label, value, note) in enumerate(items):
        with cols[idx]:
            st.markdown(
                f"""
                <div class="cp-metric">
                    <div class="cp-metric-label">{label}</div>
                    <div class="cp-metric-value">{value}</div>
                    <div class="cp-metric-note">{note}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def card(title: str, body: str) -> None:
    st.markdown(
        f"""
        <div class="cp-card">
            <div class="cp-section" style="margin-top:0;">{title}</div>
            <div class="cp-note">{body}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def themed(fig: go.Figure) -> go.Figure:
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e5efff"),
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig


@st.cache_data(show_spinner=False)
def load_raw_data() -> Optional[pd.DataFrame]:
    if RAW_DATA.exists():
        return pd.read_csv(RAW_DATA)
    return None


@st.cache_data(show_spinner=False)
def load_processed_data() -> Optional[pd.DataFrame]:
    if PROCESSED_DATA.exists():
        return pd.read_csv(PROCESSED_DATA)
    return None


@st.cache_resource(show_spinner=False)
def load_advanced_bundle() -> Optional[Dict[str, Any]]:
    if ADVANCED_BUNDLE.exists() and joblib_load is not None:
        return joblib_load(ADVANCED_BUNDLE)
    return None


def set_sample_state(row: pd.Series) -> None:
    st.session_state["app_name"] = f"Sample User {int(row.get('user_id', 0))}"
    mapping = {
        "main_age": int(row["age"]),
        "main_employment_type": row["employment_type"],
        "main_employment_length": int(row["employment_length_years"]),
        "main_annual_income": float(row["annual_income"]),
        "main_income_stability": float(row["income_stability_score"]),
        "main_loan_amount": float(row["loan_amount"]),
        "main_loan_term": int(row["loan_term_months"]),
        "main_interest_rate": float(row["interest_rate"]),
        "main_loan_type": row["loan_type"],
        "main_existing_loans": int(row["total_existing_loans"]),
        "main_total_debt": float(row["total_debt"]),
        "main_credit_score": int(round(row["credit_score"])),
        "main_utilization": float(row["credit_utilization_ratio"]),
        "main_inquiries": int(row["num_credit_inquiries"]),
        "main_late_payments": int(row["late_payment_count"]),
        "main_default_history": int(row["default_history"]),
        "main_repayment_score": float(row["repayment_history_score"]),
        "main_credit_history": int(row["credit_history_length_years"]),
        "main_job_stability": float(row["job_stability_score"]),
        "main_spending_ratio": float(row["spending_to_income_ratio"]),
    }
    for key, value in mapping.items():
        st.session_state[key] = value


def applicant_form(raw_df: Optional[pd.DataFrame]) -> Dict[str, Any]:
    st.markdown('<div class="cp-section">Applicant Studio</div>', unsafe_allow_html=True)

    sample_col, button_col, _ = st.columns([1.2, 0.6, 2.2])
    with sample_col:
        sample_index = st.number_input(
            "Load dataset sample",
            min_value=1,
            max_value=int(raw_df.shape[0]) if isinstance(raw_df, pd.DataFrame) else 1,
            value=1,
            step=1,
        )
    with button_col:
        st.write("")
        st.write("")
        if raw_df is not None and st.button("Use Sample"):
            set_sample_state(raw_df.iloc[int(sample_index) - 1])
            st.rerun()

    applicant_name = st.text_input("Applicant Name", value=st.session_state.get("app_name", "Ujjwal Applicant"), key="app_name")
    c1, c2, c3 = st.columns(3)
    with c1:
        age = st.number_input("Age", 18, 80, value=st.session_state.get("main_age", 32), key="main_age")
        employment_type = st.selectbox(
            "Employment Type",
            ["salaried", "self-employed", "unemployed"],
            index=["salaried", "self-employed", "unemployed"].index(st.session_state.get("main_employment_type", "salaried")),
            key="main_employment_type",
        )
        employment_length_years = st.number_input(
            "Employment Length (years)", 0, 40, value=st.session_state.get("main_employment_length", 6), key="main_employment_length"
        )
        annual_income = st.number_input(
            "Annual Income", 50000, 10000000, value=int(st.session_state.get("main_annual_income", 900000)), step=10000, key="main_annual_income"
        )
        income_stability_score = st.slider(
            "Income Stability Score", 0.0, 100.0, value=float(st.session_state.get("main_income_stability", 72.0)), step=0.5, key="main_income_stability"
        )
        loan_amount = st.number_input(
            "Loan Amount", 50000, 10000000, value=int(st.session_state.get("main_loan_amount", 700000)), step=10000, key="main_loan_amount"
        )
        loan_type = st.selectbox(
            "Loan Type",
            ["personal", "auto", "education", "home"],
            index=["personal", "auto", "education", "home"].index(st.session_state.get("main_loan_type", "personal")),
            key="main_loan_type",
        )
    with c2:
        loan_term_months = st.selectbox(
            "Loan Term (months)",
            [12, 24, 36, 48, 60, 84, 120, 180, 240, 300],
            index=[12, 24, 36, 48, 60, 84, 120, 180, 240, 300].index(int(st.session_state.get("main_loan_term", 48))),
            key="main_loan_term",
        )
        interest_rate = st.slider(
            "Current Interest Rate (%)", 5.0, 20.0, value=float(st.session_state.get("main_interest_rate", 10.8)), step=0.1, key="main_interest_rate"
        )
        total_existing_loans = st.number_input(
            "Total Existing Loans", 0, 20, value=st.session_state.get("main_existing_loans", 1), key="main_existing_loans"
        )
        total_debt = st.number_input(
            "Total Debt", 0, 10000000, value=int(st.session_state.get("main_total_debt", 180000)), step=10000, key="main_total_debt"
        )
        credit_score = st.slider("Credit Score", 300, 850, value=st.session_state.get("main_credit_score", 720), key="main_credit_score")
        credit_utilization_ratio = st.slider(
            "Credit Utilization Ratio", 0.0, 1.0, value=float(st.session_state.get("main_utilization", 0.32)), step=0.01, key="main_utilization"
        )
        num_credit_inquiries = st.number_input("Credit Inquiries", 0, 20, value=st.session_state.get("main_inquiries", 2), key="main_inquiries")
    with c3:
        late_payment_count = st.number_input("Late Payments", 0, 30, value=st.session_state.get("main_late_payments", 0), key="main_late_payments")
        default_history = st.selectbox("Past Default", [0, 1], index=[0, 1].index(st.session_state.get("main_default_history", 0)), key="main_default_history")
        repayment_history_score = st.slider(
            "Repayment History Score", 0.0, 100.0, value=float(st.session_state.get("main_repayment_score", 78.0)), step=0.5, key="main_repayment_score"
        )
        credit_history_length_years = st.number_input(
            "Credit History Length (years)", 0, 40, value=st.session_state.get("main_credit_history", 7), key="main_credit_history"
        )
        job_stability_score = st.slider(
            "Job Stability Score", 0.0, 100.0, value=float(st.session_state.get("main_job_stability", 74.0)), step=0.5, key="main_job_stability"
        )
        spending_to_income_ratio = st.slider(
            "Spending To Income Ratio", 0.0, 1.0, value=float(st.session_state.get("main_spending_ratio", 0.42)), step=0.01, key="main_spending_ratio"
        )

    payload = build_payload_from_form(
        age=age,
        employment_type=employment_type,
        employment_length_years=employment_length_years,
        annual_income=annual_income,
        income_stability_score=income_stability_score,
        loan_amount=loan_amount,
        loan_term_months=loan_term_months,
        interest_rate=interest_rate,
        loan_type=loan_type,
        total_existing_loans=total_existing_loans,
        total_debt=total_debt,
        credit_score=credit_score,
        credit_utilization_ratio=credit_utilization_ratio,
        num_credit_inquiries=num_credit_inquiries,
        late_payment_count=late_payment_count,
        default_history=default_history,
        repayment_history_score=repayment_history_score,
        credit_history_length_years=credit_history_length_years,
        job_stability_score=job_stability_score,
        spending_to_income_ratio=spending_to_income_ratio,
    )
    st.session_state["current_payload"] = payload
    st.session_state["current_applicant_name"] = applicant_name
    return {"name": applicant_name, "payload": payload}


def overview_page(raw_df: Optional[pd.DataFrame], _processed_df: Optional[pd.DataFrame], bundle: Optional[Dict[str, Any]]) -> None:
    header()
    st.markdown('<div class="cp-section">Overview</div>', unsafe_allow_html=True)
    metrics = bundle.get("metrics", {}) if isinstance(bundle, dict) else {}
    total_records = int(raw_df.shape[0]) if raw_df is not None else 0
    kpi_row(
        [
            ("Records", f"{total_records:,}" if total_records else "-", "Raw dataset rows"),
            ("Advanced Bundle", "Ready" if bundle is not None else "Missing", "Approval + rate models"),
            ("Approval Accuracy", f"{metrics.get('approval', {}).get('accuracy', 0):.3f}" if metrics else "-", "Retrained pipeline"),
            ("Rate MAE", f"{metrics.get('interest', {}).get('mae', 0):.3f}" if metrics else "-", "Interest estimator error"),
        ]
    )
    c1, c2 = st.columns([1.15, 1])
    with c1:
        card(
            "What Is New",
            "CreditPilot now behaves like an advisory platform. It recommends loan structures, estimates personalized rates, explains rejections, tests co-applicant impact, tracks profiles, and exposes admin-grade portfolio analytics.",
        )
        st.markdown('<div class="cp-section">Advanced Feature Map</div>', unsafe_allow_html=True)
        feature_df = pd.DataFrame(
            {
                "Feature": [
                    "Loan recommendation engine",
                    "Smart interest estimation",
                    "Prepayment and foreclosure analyzer",
                    "Co-applicant optimizer",
                    "Approval probability graph",
                    "Financial goal planner",
                    "Risk heatmap and trends",
                    "Scenario comparison and history tracker",
                ],
                "Status": ["Live"] * 8,
            }
        )
        st.dataframe(feature_df, width="stretch", hide_index=True)
    with c2:
        artifact_df = pd.DataFrame(
            {
                "Artifact": ["Raw Data", "Processed Data", "Final Model", "Scaler", "Advanced Bundle"],
                "Available": [
                    RAW_DATA.exists(),
                    PROCESSED_DATA.exists(),
                    FINAL_MODEL.exists(),
                    SCALER_MODEL.exists(),
                    ADVANCED_BUNDLE.exists(),
                ],
            }
        )
        st.markdown('<div class="cp-section">Artifacts</div>', unsafe_allow_html=True)
        st.dataframe(artifact_df, width="stretch", hide_index=True)
        if raw_df is not None:
            business_snapshot = pd.DataFrame(
                {
                    "Metric": [
                        "Average credit score",
                        "Average interest rate",
                        "Average loan amount",
                        "Lowest-risk segment",
                    ],
                    "Value": [
                        f"{raw_df['credit_score'].mean():.0f}",
                        f"{raw_df['interest_rate'].mean():.2f}%",
                        format_currency(float(raw_df["loan_amount"].mean())),
                        raw_df.groupby("loan_type")["loan_status"].mean().sort_values().index[0].title(),
                    ],
                }
            )
            st.markdown('<div class="cp-section">Portfolio Business Snapshot</div>', unsafe_allow_html=True)
            st.dataframe(business_snapshot, width="stretch", hide_index=True)


def smart_application_page(raw_df: Optional[pd.DataFrame], bundle: Optional[Dict[str, Any]]) -> None:
    header()
    form = applicant_form(raw_df)
    payload = form["payload"]
    name = form["name"]
    result = evaluate_application(bundle, payload)
    rate_offer, rate_info = estimate_interest_rate(bundle, payload)
    util = credit_utilization_analyzer(payload["credit_utilization_ratio"])

    st.markdown('<div class="cp-section">Decision Engine</div>', unsafe_allow_html=True)
    kpi_row(
        [
            ("Approval Chance", f"{(1 - result['probability']):.1%}", result["probability_info"].get("mode", "prediction")),
            ("Decision", result["decision"], result["risk"]),
            ("FSI", f"{result['fsi']:.1f}", "Financial stability index"),
            ("EMI", format_currency(result["emi"]), result["affordability"]),
            ("Health Score", f"{result['health_score']:.1f}", result["early_warning"]["level"] + " warning"),
        ]
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        card("Personalized Rate Offer", f"Estimated smart rate: <b>{rate_offer:.2f}%</b><br>Inference: {rate_info.get('mode', 'heuristic')}")
    with c2:
        card("Credit Utilization Analyzer", f"Status: <b>{util['level']}</b><br>{util['impact']}<br>{util['suggestion']}")
    with c3:
        card("Default Early Warning", f"Level: <b>{result['early_warning']['level']}</b><br>{result['early_warning']['note']}")

    curve_df = generate_probability_curve(bundle, payload)
    impact_df = variable_impact_table(bundle, payload)
    g1, g2 = st.columns(2)
    with g1:
        fig_curve = px.line(curve_df, x="credit_score", y="approval_chance", title="Approval Chance vs Credit Score")
        fig_curve.add_vline(x=payload["credit_score"], line_dash="dash", line_color="#67e8f9")
        themed(fig_curve)
        st.plotly_chart(fig_curve, width="stretch")
    with g2:
        fig_impact = px.bar(impact_df, x="scenario", y="approval_change", color="approval_change", title="Impact Of Key Variables")
        themed(fig_impact)
        st.plotly_chart(fig_impact, width="stretch")

    rejection_df = pd.DataFrame(result["rejection_breakdown"])
    if not rejection_df.empty:
        top_reason = rejection_df.iloc[0]
        st.markdown('<div class="cp-section">Top Decision Driver</div>', unsafe_allow_html=True)
        card(top_reason["reason"], f"{top_reason['share']:.1f}% contribution<br>{top_reason['detail']}")
        with st.expander("View decision driver details"):
            st.dataframe(rejection_df, width="stretch", hide_index=True)

    if raw_df is not None:
        st.markdown('<div class="cp-section">Borrower Benchmark</div>', unsafe_allow_html=True)
        benchmark_df = applicant_benchmark(payload, raw_df)
        b1, b2 = st.columns([1.05, 1])
        with b1:
            st.dataframe(benchmark_df, width="stretch", hide_index=True)
        with b2:
            fig_bench = px.bar(
                benchmark_df,
                x="metric",
                y="applicant_value",
                color="status",
                title="Applicant vs Portfolio Benchmark",
                hover_data=["portfolio_median"],
            )
            themed(fig_bench)
            st.plotly_chart(fig_bench, width="stretch")

    st.markdown('<div class="cp-section">Financial Health Suggestions</div>', unsafe_allow_html=True)
    for item in result["suggestions"][:6]:
        st.markdown(f"- {item}")

    save_col, _ = st.columns([0.45, 2.55])
    with save_col:
        if st.button("Save Profile Snapshot"):
            save_profile_event(
                {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "name": name,
                    "loan_type": payload["loan_type"],
                    "loan_amount": payload["loan_amount"],
                    "approval_chance": round((1 - result["probability"]) * 100, 2),
                    "decision": result["decision"],
                    "risk": result["risk"],
                }
            )
            st.success("Profile snapshot saved.")


def recommendation_page(bundle: Optional[Dict[str, Any]]) -> None:
    header()
    payload = st.session_state.get("current_payload")
    if not payload:
        st.info("Open Smart Application first and enter an applicant profile.")
        return

    rec_df = recommend_loans(bundle, payload)
    top = rec_df.iloc[0]
    st.markdown('<div class="cp-section">Loan Recommendation Engine</div>', unsafe_allow_html=True)
    kpi_row(
        [
            ("Best Loan Type", top["loan_type"], "Highest utility fit"),
            ("Optimal Tenure", f"{int(top['loan_term_months'])} months", "Recommended structure"),
            ("Optimal Amount", format_currency(float(top["loan_amount"])), "Safer offer size"),
            ("Expected Approval", f"{float(top['approval_chance']):.1f}%", top["decision"]),
        ]
    )

    c1, c2 = st.columns([1.25, 1])
    with c1:
        st.dataframe(
            rec_df[["loan_type", "loan_amount", "loan_term_months", "interest_rate", "emi", "approval_chance", "decision"]],
            width="stretch",
            hide_index=True,
        )
    with c2:
        fig = px.bar(rec_df.head(8), x="loan_type", y="approval_chance", color="utility_score", title="Top Recommended Structures")
        themed(fig)
        st.plotly_chart(fig, width="stretch")

    st.markdown('<div class="cp-section">Co-Applicant Optimization</div>', unsafe_allow_html=True)
    a1, a2, a3 = st.columns(3)
    with a1:
        co_income = st.number_input("Co-applicant income", 0, 5000000, 250000, step=10000)
    with a2:
        co_score = st.slider("Co-applicant credit score", 300, 850, 730)
    with a3:
        debt_share = st.slider("Debt share reduction", 0.0, 0.6, 0.2, 0.05)
    co_result = coapplicant_optimization(bundle, payload, co_income, co_score, debt_share)
    kpi_row(
        [
            ("Base Approval", f"{(1 - co_result['base']['probability']):.1%}", co_result["base"]["decision"]),
            ("With Co-applicant", f"{(1 - co_result['with_coapplicant']['probability']):.1%}", co_result["with_coapplicant"]["decision"]),
            ("Approval Gain", f"{co_result['approval_gain']:.1f} pts", "Improvement"),
            ("Recommendation", "Add Co-applicant" if co_result["should_add"] else "Optional", "Optimization view"),
        ]
    )

    st.markdown('<div class="cp-section">Multi-Loan Scenario Comparison</div>', unsafe_allow_html=True)
    s1, s2 = st.columns(2)
    with s1:
        type_a = st.selectbox("Loan A Type", ["personal", "auto", "education", "home"], index=0)
        amount_a = st.number_input("Loan A Amount", 50000, 5000000, int(payload["loan_amount"]), step=10000)
        term_a = st.selectbox("Loan A Tenure", [12, 24, 36, 48, 60, 84, 120, 180, 240, 300], index=3)
    with s2:
        type_b = st.selectbox("Loan B Type", ["personal", "auto", "education", "home"], index=1)
        amount_b = st.number_input("Loan B Amount", 50000, 5000000, int(payload["loan_amount"] * 0.9), step=10000)
        term_b = st.selectbox("Loan B Tenure", [12, 24, 36, 48, 60, 84, 120, 180, 240, 300], index=4)

    scenario_a = dict(payload)
    scenario_a["loan_type"] = type_a
    scenario_a["loan_amount"] = float(amount_a)
    scenario_a["loan_term_months"] = int(term_a)
    scenario_a["interest_rate"], _ = estimate_interest_rate(bundle, scenario_a)

    scenario_b = dict(payload)
    scenario_b["loan_type"] = type_b
    scenario_b["loan_amount"] = float(amount_b)
    scenario_b["loan_term_months"] = int(term_b)
    scenario_b["interest_rate"], _ = estimate_interest_rate(bundle, scenario_b)

    comparison = compare_loan_scenarios(
        bundle,
        [
            {"label": "Loan A", "payload": scenario_a},
            {"label": "Loan B", "payload": scenario_b},
        ],
    )
    st.dataframe(comparison, width="stretch", hide_index=True)


def emi_page() -> None:
    header()
    st.markdown('<div class="cp-section">EMI Calculator And Prepayment Planner</div>', unsafe_allow_html=True)
    c1, c2 = st.columns([1, 1])
    with c1:
        loan_amount = st.number_input("Planner Loan Amount", 50000.0, 10000000.0, 800000.0, step=10000.0)
        annual_rate = st.slider("Planner Interest Rate (%)", 5.0, 25.0, 10.5, 0.1)
        tenure_years = st.slider("Planner Tenure (years)", 1, 30, 5)
        extra_payment = st.number_input("Extra Monthly Payment", 0.0, 200000.0, 5000.0, step=500.0)
    months = tenure_years * 12
    emi = calculate_emi(loan_amount, annual_rate, months)
    schedule = amortization_schedule(loan_amount, annual_rate, months, extra_payment=extra_payment)
    summary = prepayment_summary(loan_amount, annual_rate, months, extra_payment)
    with c2:
        kpi_row(
            [
                ("Live EMI", format_currency(emi), "Updates instantly"),
                ("Interest Saved", format_currency(summary["interest_saved"]), "With prepayment"),
                ("Tenure Reduced", f"{summary['months_saved']} months", "Foreclosure benefit"),
                ("Total Interest", format_currency(summary["new_interest"]), "After extra payment"),
            ]
        )

    g1, g2 = st.columns(2)
    with g1:
        fig_balance = px.line(schedule, x="month", y="balance", title="Remaining Balance Trend")
        themed(fig_balance)
        st.plotly_chart(fig_balance, width="stretch")
    with g2:
        fig_mix = px.area(schedule, x="month", y=["principal", "interest"], title="Principal vs Interest Mix")
        themed(fig_mix)
        st.plotly_chart(fig_mix, width="stretch")
    with st.expander("View Full Amortization Schedule"):
        st.dataframe(schedule, width="stretch", hide_index=True)


def fsi_page(bundle: Optional[Dict[str, Any]]) -> None:
    header()
    payload = st.session_state.get("current_payload")
    if not payload:
        st.info("Open Smart Application first and enter an applicant profile.")
        return

    result = evaluate_application(bundle, payload)
    model_payload = result["payload"]
    component_df = pd.DataFrame(
        [
            {"component": "Income Stability", "score": round(model_payload["income_stability_score"], 2)},
            {"component": "Credit Strength", "score": round(((model_payload["credit_score"] - 300) / (850 - 300)) * 100, 2)},
            {"component": "Debt Headroom", "score": round(max(0.0, 1 - model_payload["debt_to_income_ratio"]) * 100, 2)},
            {"component": "EMI Headroom", "score": round(max(0.0, 1 - model_payload["emi_to_income_ratio"]) * 100, 2)},
            {"component": "Utilization Headroom", "score": round(max(0.0, 1 - model_payload["credit_utilization_ratio"]) * 100, 2)},
        ]
    )

    st.markdown('<div class="cp-section">Financial Stability Index</div>', unsafe_allow_html=True)
    kpi_row(
        [
            ("FSI", f"{result['fsi']:.1f}", "Composite resilience score"),
            ("Risk Class", result["risk"], "Current borrower posture"),
            ("Affordability", result["affordability"], "Monthly capacity view"),
            ("Health Score", f"{result['health_score']:.1f}", "Combined profile health"),
        ]
    )

    g1, g2 = st.columns(2)
    with g1:
        fig_gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=result["fsi"],
                title={"text": "FSI Gauge"},
                number={"suffix": "/100"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#67e8f9"},
                    "steps": [
                        {"range": [0, 50], "color": "rgba(239,68,68,0.35)"},
                        {"range": [50, 70], "color": "rgba(250,204,21,0.30)"},
                        {"range": [70, 100], "color": "rgba(74,222,128,0.30)"},
                    ],
                },
            )
        )
        themed(fig_gauge)
        st.plotly_chart(fig_gauge, width="stretch")
    with g2:
        fig_components = px.bar(component_df, x="component", y="score", color="score", title="FSI Component Breakdown")
        fig_components.update_layout(xaxis_title="", yaxis_title="Score")
        themed(fig_components)
        st.plotly_chart(fig_components, width="stretch")

    fig_radar = go.Figure()
    fig_radar.add_trace(
        go.Scatterpolar(
            r=component_df["score"].tolist(),
            theta=component_df["component"].tolist(),
            fill="toself",
            name="FSI mix",
            line=dict(color="#60a5fa"),
        )
    )
    fig_radar.update_layout(title="Financial Stability Radar", polar=dict(radialaxis=dict(range=[0, 100])))
    themed(fig_radar)
    st.plotly_chart(fig_radar, width="stretch")

    st.markdown('<div class="cp-section">Stability Notes</div>', unsafe_allow_html=True)
    notes_df = pd.DataFrame(
        [
            {"metric": "Credit score", "value": model_payload["credit_score"], "note": "Higher values improve FSI."},
            {"metric": "Debt-to-income", "value": round(model_payload["debt_to_income_ratio"], 2), "note": "Lower leverage improves resilience."},
            {"metric": "EMI-to-income", "value": round(model_payload["emi_to_income_ratio"], 2), "note": "High monthly pressure reduces stability."},
            {"metric": "Credit utilization", "value": round(model_payload["credit_utilization_ratio"], 2), "note": "Lower revolving usage is healthier."},
        ]
    )
    st.dataframe(notes_df, width="stretch", hide_index=True)


def stress_lab_page(bundle: Optional[Dict[str, Any]]) -> None:
    header()
    payload = st.session_state.get("current_payload")
    if not payload:
        st.info("Open Smart Application first and enter an applicant profile.")
        return

    base_result = evaluate_application(bundle, payload)
    st.markdown('<div class="cp-section">Stress Lab</div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        income_shock = st.slider("Income Shock (%)", -50, 20, -10, 5)
    with c2:
        rate_shock = st.slider("Rate Shock (%)", 0.0, 6.0, 1.0, 0.1)
    with c3:
        debt_shock = st.slider("Debt Shock (%)", 0, 100, 20, 5)
    with c4:
        util_shock = st.slider("Utilization Shock", 0.0, 0.5, 0.1, 0.01)

    stressed_payload = dict(payload)
    stressed_payload["annual_income"] = max(50000.0, payload["annual_income"] * (1 + income_shock / 100))
    stressed_payload["interest_rate"] = min(25.0, payload["interest_rate"] + rate_shock)
    stressed_payload["total_debt"] = max(0.0, payload["total_debt"] * (1 + debt_shock / 100))
    stressed_payload["credit_utilization_ratio"] = min(1.0, payload["credit_utilization_ratio"] + util_shock)
    stressed_result = evaluate_application(bundle, stressed_payload)
    summary = stress_delta_summary(base_result["probability"], stressed_result["probability"])

    kpi_row(
        [
            ("Base Approval", f"{summary['base_approval']:.1f}%", base_result["decision"]),
            ("Stress Approval", f"{summary['stress_approval']:.1f}%", stressed_result["decision"]),
            ("Stress Impact", summary["impact"], "Scenario effect"),
            ("FSI Delta", f"{stressed_result['fsi'] - base_result['fsi']:+.1f}", "Resilience change"),
            ("EMI Delta", format_currency(stressed_result["emi"] - base_result["emi"]), "Monthly payment shift"),
        ]
    )

    compare_df = pd.DataFrame(
        [
            {"metric": "Approval chance", "base": summary["base_approval"], "stress": summary["stress_approval"]},
            {"metric": "FSI", "base": base_result["fsi"], "stress": stressed_result["fsi"]},
            {"metric": "EMI", "base": base_result["emi"], "stress": stressed_result["emi"]},
            {"metric": "Health score", "base": base_result["health_score"], "stress": stressed_result["health_score"]},
        ]
    )
    compare_long = compare_df.melt(id_vars="metric", var_name="scenario", value_name="value")

    g1, g2 = st.columns(2)
    with g1:
        fig_compare = px.bar(compare_long, x="metric", y="value", color="scenario", barmode="group", title="Base vs Stress Comparison")
        themed(fig_compare)
        st.plotly_chart(fig_compare, width="stretch")
    with g2:
        shock_rows = []
        for shock in range(0, 41, 5):
            sim = dict(payload)
            sim["annual_income"] = max(50000.0, payload["annual_income"] * (1 - shock / 100))
            sim_result = evaluate_application(bundle, sim)
            shock_rows.append(
                {
                    "income_drop_pct": shock,
                    "approval_chance": round((1 - sim_result["probability"]) * 100, 2),
                    "fsi": sim_result["fsi"],
                }
            )
        shock_df = pd.DataFrame(shock_rows)
        fig_curve = px.line(shock_df, x="income_drop_pct", y=["approval_chance", "fsi"], markers=True, title="Sensitivity To Income Stress")
        themed(fig_curve)
        st.plotly_chart(fig_curve, width="stretch")

    details_df = pd.DataFrame(
        [
            {"field": "Annual income", "base": format_currency(payload["annual_income"]), "stress": format_currency(stressed_payload["annual_income"])},
            {"field": "Interest rate", "base": f"{payload['interest_rate']:.2f}%", "stress": f"{stressed_payload['interest_rate']:.2f}%"},
            {"field": "Total debt", "base": format_currency(payload["total_debt"]), "stress": format_currency(stressed_payload["total_debt"])},
            {"field": "Credit utilization", "base": f"{payload['credit_utilization_ratio']:.2f}", "stress": f"{stressed_payload['credit_utilization_ratio']:.2f}"},
        ]
    )
    st.dataframe(details_df, width="stretch", hide_index=True)


def goal_planner_page(bundle: Optional[Dict[str, Any]]) -> None:
    header()
    payload = st.session_state.get("current_payload")
    if not payload:
        st.info("Open Smart Application first and enter an applicant profile.")
        return
    st.markdown('<div class="cp-section">Financial Goal Planner</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        goal_type = st.selectbox("Goal Type", ["Buy House", "Buy Car", "Higher Education", "Business Expansion"])
    with c2:
        years_to_goal = st.slider("Years To Goal", 1, 15, 4)
    with c3:
        down_payment = st.number_input("Planned Down Payment", 0, 5000000, 250000, step=10000)
    plan = financial_goal_plan(bundle, payload, goal_type, down_payment, years_to_goal)

    loan_type_map = {
        "Buy House": "home",
        "Buy Car": "auto",
        "Higher Education": "education",
        "Business Expansion": "personal",
    }
    goal_payload = dict(payload)
    goal_payload["loan_type"] = loan_type_map[goal_type]
    goal_payload["loan_amount"] = float(plan["safe_loan_amount"])
    goal_payload["loan_term_months"] = 240 if goal_type == "Buy House" else (60 if goal_type == "Buy Car" else 84)
    goal_payload["interest_rate"], _ = estimate_interest_rate(bundle, goal_payload)
    goal_eval = evaluate_application(bundle, goal_payload)

    kpi_row(
        [
            ("Safe Loan Amount", format_currency(plan["safe_loan_amount"]), goal_type),
            ("Comfortable EMI", format_currency(plan["comfortable_emi"]), "Safe monthly zone"),
            ("Projected Approval", f"{(1 - goal_eval['probability']):.1%}", goal_eval["decision"]),
            ("Recommended Rate", f"{goal_payload['interest_rate']:.2f}%", "Estimated offer"),
        ]
    )
    card("Planner Insight", plan["note"])


def admin_dashboard_page(raw_df: Optional[pd.DataFrame]) -> None:
    header()
    if raw_df is None:
        st.error("Raw dataset not found.")
        return
    st.markdown('<div class="cp-section">Admin Dashboards</div>', unsafe_allow_html=True)
    portfolio = portfolio_summary(raw_df)
    kpi_row(
        [
            ("Portfolio Size", f"{portfolio['portfolio_size']:,}", "All applications"),
            ("Approved Loans", f"{portfolio['total_approved']:,}", "Estimated from dataset"),
            ("Approval Rate", f"{portfolio['approval_rate']:.1%}", "Portfolio conversion"),
            ("High Risk Share", f"{portfolio['high_risk_share']:.1%}", portfolio["warning"]),
        ]
    )

    heatmap_df = risk_heatmap_frame(raw_df)
    trend_df = approval_trend_frame(raw_df)

    raw_work = raw_df.copy()
    raw_work["risk_cluster"] = raw_work["credit_score"].apply(lambda x: "High Risk" if x < 650 else ("Medium Risk" if x < 720 else "Low Risk"))

    g1, g2 = st.columns(2)
    with g1:
        fig_heat = px.imshow(
            heatmap_df,
            text_auto=".0%",
            color_continuous_scale="RdYlBu_r",
            title="Risk Heatmap: Default Rate by DTI and Credit Band",
        )
        themed(fig_heat)
        st.plotly_chart(fig_heat, width="stretch")
    with g2:
        fig_cluster = px.scatter(
            raw_work.sample(min(1200, len(raw_work)), random_state=42),
            x="credit_score",
            y="debt_to_income_ratio",
            color="risk_cluster",
            size="loan_amount",
            title="High-Risk Cluster View",
            hover_data=["loan_type", "interest_rate"],
        )
        themed(fig_cluster)
        st.plotly_chart(fig_cluster, width="stretch")

    fig_trend = px.bar(trend_df, x="loan_type", y="approval_rate", color="avg_interest_rate", title="Approval Trend by Loan Type")
    themed(fig_trend)
    st.plotly_chart(fig_trend, width="stretch")


def profile_tracker_page() -> None:
    header()
    st.markdown('<div class="cp-section">User Profile And Loan History Tracker</div>', unsafe_allow_html=True)
    history = load_profile_history()
    history_df = build_history_frame()
    if not history:
        st.info("No saved profile snapshots yet. Save one from Smart Application.")
        return
    kpi_row(
        [
            ("Saved Profiles", f"{len(history):,}", "Snapshots stored locally"),
            ("Latest Applicant", history[-1].get("name", "-"), "Most recent"),
            ("Latest Decision", history[-1].get("decision", "-"), history[-1].get("risk", "-")),
            ("Latest Approval", f"{history[-1].get('approval_chance', 0):.1f}%", "Chance at save time"),
        ]
    )
    st.dataframe(history_df, width="stretch", hide_index=True)
    if len(history_df) > 1:
        fig_hist = px.line(history_df, x="timestamp", y="approval_chance", color="name", markers=True, title="Approval Trend Over Time")
        themed(fig_hist)
        st.plotly_chart(fig_hist, width="stretch")


def assistant_page(bundle: Optional[Dict[str, Any]]) -> None:
    header()
    payload = st.session_state.get("current_payload")
    st.markdown('<div class="cp-section">Smart Credit Assistant</div>', unsafe_allow_html=True)
    c1, c2 = st.columns([1.05, 1.45])
    with c1:
        choice = st.radio(
            "Choose a guidance action",
            [
                "Loan eligibility help",
                "Improve approval chances",
                "Best loan type",
                "Check EMI",
                "Can I lower my EMI",
                "Should I add co-applicant",
                "Rate negotiation tips",
            ],
        )
        tone = st.selectbox("Assistant mode", ["Executive summary", "Detailed advisor"], index=0)
        if payload is None:
            st.info("Load a borrower profile in Smart Application to unlock personalized assistant guidance.")
    response = assistant_response(choice, payload, bundle)
    with c2:
        card(response["title"], response["answer"])
        if response["highlights"]:
            kpi_row([(label, value, "Assistant insight") for label, value in response["highlights"]])
        st.markdown('<div class="cp-section">Action Points</div>', unsafe_allow_html=True)
        if response["bullets"]:
            bullets = response["bullets"] if tone == "Detailed advisor" else response["bullets"][:3]
            for item in bullets:
                st.markdown(f"- {item}")
        else:
            st.markdown("- Add a borrower profile to get specific recommendations.")


def visualisations_page(raw_df: Optional[pd.DataFrame], bundle: Optional[Dict[str, Any]]) -> None:
    header()
    if raw_df is None:
        st.error("Raw dataset not found.")
        return

    st.markdown('<div class="cp-section">Visualisations</div>', unsafe_allow_html=True)
    work = raw_df.copy()
    work = work[work["loan_type"] != "home"]
    work["approval_flag"] = 1 - work["loan_status"]
    work["decision"] = work["approval_flag"].map({1: "Approved", 0: "Rejected"})
    work["risk_cluster"] = work["credit_score"].apply(lambda x: "High Risk" if x < 650 else ("Medium Risk" if x < 720 else "Low Risk"))

    g1, g2 = st.columns(2)
    with g1:
        approval_by_type = work.groupby("loan_type", as_index=False)["approval_flag"].mean()
        fig_approval = px.bar(approval_by_type, x="loan_type", y="approval_flag", color="approval_flag", title="Approval Rate By Loan Type")
        fig_approval.update_layout(yaxis_tickformat=".0%")
        themed(fig_approval)
        st.plotly_chart(fig_approval, width="stretch")
    with g2:
        fig_score = px.histogram(
            work,
            x="credit_score",
            color="decision",
            nbins=30,
            barmode="overlay",
            title="Credit Score Distribution By Decision",
        )
        themed(fig_score)
        st.plotly_chart(fig_score, width="stretch")

    g3, g4 = st.columns(2)
    with g3:
        scatter_df = work.sample(min(1200, len(work)), random_state=42)
        fig_scatter = px.scatter(
            scatter_df,
            x="credit_score",
            y="interest_rate",
            color="loan_type",
            size="loan_amount",
            title="Credit Score vs Interest Rate",
            hover_data=["annual_income", "decision"],
        )
        themed(fig_scatter)
        st.plotly_chart(fig_scatter, width="stretch")
    with g4:
        fig_income = px.scatter(
            scatter_df,
            x="annual_income",
            y="loan_amount",
            color="risk_cluster",
            title="Income vs Loan Amount",
            hover_data=["loan_type", "credit_score"],
        )
        themed(fig_income)
        st.plotly_chart(fig_income, width="stretch")

    g5, g6 = st.columns(2)
    with g5:
        fig_box = px.box(work, x="employment_type", y="debt_to_income_ratio", color="employment_type", title="Debt-To-Income By Employment Type")
        themed(fig_box)
        st.plotly_chart(fig_box, width="stretch")
    with g6:
        heatmap_source = work.pivot_table(
            index="employment_type",
            columns="loan_type",
            values="approval_flag",
            aggfunc="mean",
            observed=False,
        ).fillna(0.0)
        fig_heat = px.imshow(
            heatmap_source,
            text_auto=".0%",
            color_continuous_scale="Blues",
            title="Approval Heatmap: Employment vs Loan Type",
        )
        themed(fig_heat)
        st.plotly_chart(fig_heat, width="stretch")

    g7, g8 = st.columns(2)
    with g7:
        fig_pie = px.pie(work, names="employment_type", hole=0.55, title="Employment Mix")
        themed(fig_pie)
        st.plotly_chart(fig_pie, width="stretch")
    with g8:
        corr_cols = [
            "annual_income",
            "loan_amount",
            "interest_rate",
            "credit_score",
            "debt_to_income_ratio",
            "credit_utilization_ratio",
            "repayment_history_score",
        ]
        corr_df = work[corr_cols].corr().round(2)
        fig_corr = px.imshow(corr_df, text_auto=True, color_continuous_scale="RdBu_r", title="Correlation Matrix")
        themed(fig_corr)
        st.plotly_chart(fig_corr, width="stretch")

    payload = st.session_state.get("current_payload")
    if payload:
        result = evaluate_application(bundle, payload)
        rejection_df = pd.DataFrame(result["rejection_breakdown"])
        applicant_curve = generate_probability_curve(bundle, payload)
        v1, v2 = st.columns(2)
        with v1:
            fig_curve = px.line(applicant_curve, x="credit_score", y="approval_chance", title="Applicant Approval Curve")
            fig_curve.add_vline(x=payload["credit_score"], line_dash="dash", line_color="#67e8f9")
            themed(fig_curve)
            st.plotly_chart(fig_curve, width="stretch")
        with v2:
            fig_rej = px.bar(rejection_df, x="reason", y="share", text="share", color="share", title="Applicant Decision Driver Breakdown")
            fig_rej.update_traces(texttemplate="%{text:.1f}%")
            themed(fig_rej)
            st.plotly_chart(fig_rej, width="stretch")


def main() -> None:
    _inject_style()
    raw_df = load_raw_data()
    processed_df = load_processed_data()
    bundle = load_advanced_bundle()

    tabs = st.tabs(
        [
            "Overview",
            "Smart Application",
            "Recommendation Lab",
            "Stress Lab",
            "FSI",
            "EMI Calculator",
            "Visualisations",
            "Goal Planner",
            "Admin Dashboards",
            "Profile Tracker",
            "Assistant",
        ]
    )

    with tabs[0]:
        overview_page(raw_df, processed_df, bundle)
    with tabs[1]:
        smart_application_page(raw_df, bundle)
    with tabs[2]:
        recommendation_page(bundle)
    with tabs[3]:
        stress_lab_page(bundle)
    with tabs[4]:
        fsi_page(bundle)
    with tabs[5]:
        emi_page()
    with tabs[6]:
        visualisations_page(raw_df, bundle)
    with tabs[7]:
        goal_planner_page(bundle)
    with tabs[8]:
        admin_dashboard_page(raw_df)
    with tabs[9]:
        profile_tracker_page()
    with tabs[10]:
        assistant_page(bundle)


if __name__ == "__main__":
    main()
