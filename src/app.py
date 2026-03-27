import os
import pickle
import json
import base64
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import xgboost as xgb
try:
    from joblib import load as joblib_load
except Exception:
    joblib_load = None

from logic import (
    calculate_emi,
    prepare_features,
    calculate_fsi,
    get_risk,
    final_decision,
    check_affordability,
    explain_decision,
    suggest_improvements,
    loan_health_score,
    stress_impact,
)


st.set_page_config(page_title="CreditPilot", page_icon="💳", layout="wide")


def _inject_style():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Sora:wght@600;700;800&display=swap');
        :root{
            --accent:#3b82f6;
            --bg:#0b1220;
            --panel:rgba(15,23,42,0.55);
            --text:#E6EDF7;
            --muted:#93a3b8;
            --radius:14px;
            --pad:14px;
            --font-body:'Inter',system-ui,-apple-system,Segoe UI,Roboto,Ubuntu,'Helvetica Neue',Arial,'Noto Sans';
            --font-display:'Sora','Inter',system-ui,-apple-system,Segoe UI,Roboto,Ubuntu,'Helvetica Neue',Arial,'Noto Sans';
        }
        [data-testid="stSidebar"] {display: none;}
        html, body {background: var(--bg); color: #ffffff; font-family: var(--font-body);}
        .block-container {padding-top: 4.2rem; padding-bottom: 2rem; max-width: 1320px; margin: 0 auto; padding-left: 24px; padding-right: 24px;}
        .block-container h1 {color:#f8fafc; font-size:2.6rem; line-height:1.15; letter-spacing:.3px; text-shadow:0 3px 14px rgba(2,6,23,.7); margin:.2rem 0 .1rem; font-family:var(--font-display);}
        .block-container h1, .block-container h2, .block-container h3, .block-container h4, .block-container h5, .block-container h6 {
            color:#ffffff !important; font-weight:900; text-shadow:0 2px 10px rgba(0,0,0,.68); letter-spacing:.2px; font-family:var(--font-display);
        }
        [data-testid="stCaptionContainer"], .block-container .caption { color:#ffffff !important; font-size:1.05rem !important; text-shadow:0 2px 12px rgba(0,0,0,.65); opacity:1 !important; font-family:var(--font-body); font-weight:700; }
        .block-container p, .block-container li, .block-container span, .block-container a, .block-container small, .block-container strong, .block-container em, .block-container label {
            color:#ffffff !important; text-shadow:0 1px 6px rgba(0,0,0,.68); font-weight:700;
        }
        .block-container b, .block-container strong { font-weight:900 !important; }
        .glass {
            background: rgba(17, 24, 39, 0.45);
            border: 1px solid rgba(255, 255, 255, 0.08);
            box-shadow: 0 8px 28px rgba(2, 6, 23, 0.45);
            backdrop-filter: blur(12px) saturate(120%);
            -webkit-backdrop-filter: blur(12px) saturate(120%);
            border-radius: 16px;
        }
        .creditpilot-header.glass {
            padding: 26px 28px;
            color: #ffffff;
            margin-bottom: 18px;
            background: linear-gradient(180deg, rgba(15,23,42,0.35), rgba(2,6,23,0.35));
            border: 1px solid rgba(255,255,255,0.22);
            box-shadow: 0 8px 28px rgba(2, 6, 23, 0.45);
            backdrop-filter: blur(14px) saturate(120%);
        }
        .creditpilot-title {font-size: 34px; font-weight: 900; margin: 0; line-height: 1.15; letter-spacing: .2px; font-family: var(--font-display); text-shadow: 0 2px 8px rgba(0,0,0,.55);}
        .creditpilot-subtitle {font-size: 14px; opacity: 1; margin: 6px 0 0; color: #e5e7eb; text-shadow: 0 2px 6px rgba(0,0,0,.5);}
        .pill {
            display: inline-block; padding: 6px 12px; margin-right: 8px; margin-top: 10px;
            border-radius: 999px;
            background: linear-gradient(135deg,#22d3ee 0%, #3b82f6 55%, #7c3aed 100%);
            color: #0b1220; font-size: 12px; font-weight: 800;
            box-shadow: inset 0 0 0 1px rgba(255,255,255,0.35);
        }
        .card {
            border-radius: 16px; padding: 16px;
            background: linear-gradient(180deg, rgba(15,23,42,0.35), rgba(2,6,23,0.35));
            border: 1px solid rgba(255,255,255,0.22);
            box-shadow: 0 8px 28px rgba(2, 6, 23, 0.45);
            backdrop-filter: blur(14px) saturate(120%);
            color: #ffffff;
        }
        .card * { color:#ffffff !important; text-shadow:0 1px 6px rgba(0,0,0,.6); font-weight:700; }
        .metric-card {
            border-radius: 16px; padding: 16px;
            background: linear-gradient(180deg, rgba(15,23,42,0.35), rgba(2,6,23,0.35));
            border: 1px solid rgba(255,255,255,0.22);
            box-shadow: 0 8px 28px rgba(2, 6, 23, 0.45);
            backdrop-filter: blur(14px) saturate(120%);
            color: #ffffff;
        }
        .metric-card * { color:#ffffff !important; text-shadow:0 1px 6px rgba(0,0,0,.6); font-weight:700; }
        .metric-card .footer-note { color:#e5e7eb !important; font-weight:600 !important; }
        .section-title {font-weight: 900; font-size: 18px; margin: 4px 0 10px; color: #ffffff; text-shadow: 0 1px 6px rgba(0,0,0,.5);}
        .nav-tabs {
            background: linear-gradient(180deg, rgba(15,23,42,0.35), rgba(2,6,23,0.35));
            border: 1px solid rgba(255,255,255,0.22);
            box-shadow: 0 6px 20px rgba(2,6,23,0.35);
            backdrop-filter: blur(14px) saturate(120%);
            border-radius: 14px;
            padding: 8px;
            position: sticky; top: 0; z-index: 60;
            margin-bottom: 8px;
        }
        .nav-tabs [role="tablist"] {
            gap: 8px;
            padding: 0;
            border: 0;
            background: transparent;
        }
        .nav-tabs button[kind="secondary"] {
            border-radius: 12px;
            color: #ffffff !important;
            background: rgba(15,23,42,0.7);
            border: 1px solid rgba(255,255,255,0.28);
            text-shadow: 0 1px 6px rgba(0,0,0,.7);
            font-weight: 800;
        }
        .nav-tabs button[kind="secondary"][aria-selected="true"]{
            background: rgba(15,23,42,0.95);
            color: #ffffff !important; font-weight: 900;
            border: 1px solid rgba(255,255,255,0.7);
            box-shadow: 0 -3px 0 var(--accent) inset, 0 6px 16px rgba(2,6,23,.45);
        }
        .footer-note {font-size: 12px; color: #93a3b8;}
        .status-dot {display:inline-block;width:10px;height:10px;border-radius:999px;margin-right:8px;}
        .status-ok {background:#10b981;}
        .status-bad {background:#ef4444;}
        .heading {font-size:22px;font-weight:900;color:#ffffff !important;margin:4px 0 8px; text-shadow:0 2px 10px rgba(0,0,0,.68);}
        .lead {font-size:14px;color:#C7D2FE;}
        label, .st-emotion-cache-18ni7ap p { color: #ffffff !important; text-shadow: 0 1px 4px rgba(0,0,0,.4); }
        .icon-badge {display:inline-flex;align-items:center;gap:8px;font-weight:800;color:#E6EDF7;}
        .icon-badge .dot {width:8px;height:8px;border-radius:999px;background:#22d3ee;display:inline-block;}
        .stButton > button {
            background: var(--accent);
            color: #0b1220; font-weight: 900; border: 0; border-radius: 12px; padding: 8px 14px;
            box-shadow: 0 6px 18px rgba(59,130,246,0.35);
        }
        .stButton > button:hover { filter: brightness(1.06); transform: translateY(-1px); box-shadow: 0 10px 22px rgba(59,130,246,0.30);}
        [data-testid="stPlotlyChart"] {
            position: relative;
            border-radius: 16px;
            padding: 12px;
            background: rgba(17,24,39,0.55) !important;
            border: 1px solid rgba(255,255,255,0.22) !important;
            box-shadow: 0 8px 28px rgba(2, 6, 23, 0.45) !important;
            min-height: 260px;
        }
        [data-testid="stPlotlyChart"] > div { position: relative; z-index: 1; }
        [data-testid="stPlotlyChart"]::before {
            content: "";
            position: absolute; inset: 0; z-index: 0;
            border-radius: 16px;
            background: rgba(17,24,39,0.50);
            border: 1px solid rgba(255,255,255,0.22);
            box-shadow: 0 8px 28px rgba(2, 6, 23, 0.45);
            backdrop-filter: blur(12px) saturate(120%);
            will-change: opacity;
            opacity: 1 !important;
            transition: none !important;
            pointer-events: none;
        }
        .flow-steps {
            display: flex; gap: 12px; align-items: stretch; overflow-x: auto; padding: 8px;
            scrollbar-width: thin;
        }
        .flow-steps::-webkit-scrollbar {height: 8px;}
        .flow-steps::-webkit-scrollbar-thumb {background: rgba(148,163,184,0.35); border-radius: 999px;}
        .step-card {
            min-width: 200px; border-radius: 14px; padding: 14px 16px; position: relative;
            border: 1px solid rgba(255,255,255,0.22);
            background: rgba(17,24,39,0.50);
            box-shadow: 0 8px 20px rgba(2, 6, 23, 0.35);
            backdrop-filter: blur(10px);
        }
        .step-card * { color: #ffffff !important; text-shadow: 0 1px 4px rgba(0,0,0,.45) !important; }
        .step-title {font-weight: 900; font-size: 14px; letter-spacing: .2px;}
        .step-desc {font-size: 12px; opacity: .85; margin-top: 6px; font-weight: 700; color: #cbd5e1 !important;}
        .step-index {
            position: absolute; top: 10px; right: 12px; font-weight: 900; font-size: 12px;
            color: #0b1220 !important; background: #e2e8f0; border-radius: 999px; padding: 2px 8px;
        }
        .step-card:hover { border-color: rgba(255,255,255,0.45); box-shadow: 0 10px 24px rgba(2,6,23,0.45); }
        .step-arrow {
            display: flex; align-items: center; justify-content: center; padding: 0 6px;
            font-weight: 900; color: #cbd5e1;
        }
        [data-testid="stTextInput"]>div,
        [data-testid="stTextArea"]>div,
        [data-testid="stNumberInput"]>div,
        [data-testid="stSelectbox"]>div,
        [data-testid="stMultiSelect"]>div {
            background: rgba(15,23,42,0.55) !important;
            border: 1px solid rgba(255,255,255,0.18);
            border-radius: var(--radius);
            box-shadow: inset 0 1px 0 rgba(255,255,255,0.06), 0 6px 20px rgba(2,6,23,0.25);
            backdrop-filter: blur(8px) saturate(1.05);
        }
        [data-testid="stExpander"] > details {
            background: rgba(17,24,39,0.45);
            border: 1px solid rgba(255,255,255,0.06);
            border-radius: 12px; color: #E6EDF7;
        }
        ::-webkit-scrollbar {height: 10px; width: 10px;}
        ::-webkit-scrollbar-thumb {background: rgba(148,163,184,0.35); border-radius: 999px;}
        </style>
        """,
        unsafe_allow_html=True,
    )
    try:
        base_dir = Path(__file__).resolve().parent
        for fname in ["bg.jpeg", "bg.jpg", "bg.png"]:
            bg_path = base_dir / fname
            if bg_path.exists():
                ext = bg_path.suffix.lower()
                mime = "image/jpeg" if ext in [".jpeg", ".jpg"] else "image/png"
                b64 = base64.b64encode(bg_path.read_bytes()).decode()
                st.markdown(
                    f"""
                    <style>
                    html, body, .stApp, [data-testid="stAppViewContainer"] {{
                        background-image: url('data:{mime};base64,{b64}') !important;
                        background-size: cover !important;
                        background-repeat: no-repeat !important;
                        background-position: center !important;
                        background-attachment: fixed !important;
                        background-color: #0b1220 !important;
                    }}
                    [data-testid="stHeader"], [data-testid="stToolbar"] {{
                        background: transparent !important;
                    }}
                    </style>
                    """,
                    unsafe_allow_html=True,
                )
                break
    except Exception:
        pass


BASE_DIR = Path(__file__).resolve().parent.parent


@st.cache_data(show_spinner=False)
def load_raw_data() -> Optional[pd.DataFrame]:
    p = BASE_DIR / "data" / "raw" / "creditpilot_dataset.csv"
    if p.exists():
        return pd.read_csv(p)
    return None


@st.cache_data(show_spinner=False)
def load_processed_data() -> Optional[pd.DataFrame]:
    p = BASE_DIR / "data" / "processed" / "final_dataset.csv"
    if p.exists():
        return pd.read_csv(p)
    return None


@st.cache_resource(show_spinner=False)
def load_artifacts() -> Tuple[Optional[Any], Optional[Any], Optional[List[str]], Dict[str, Optional[str]]]:
    scaler_p = BASE_DIR / "models" / "scaler.pkl"
    model_p = BASE_DIR / "models" / "final_model.pkl"
    scaler = None
    model = None
    feature_names = None
    errors: Dict[str, Optional[str]] = {"scaler": None, "model": None}
    if scaler_p.exists():
        try:
            with open(scaler_p, "rb") as f:
                scaler = pickle.load(f)
        except Exception as e:
            try:
                if joblib_load is not None:
                    scaler = joblib_load(scaler_p)
                else:
                    raise e
            except Exception as e2:
                errors["scaler"] = str(e2)
                scaler = None
    else:
        errors["scaler"] = "File not found"
    if model_p.exists():
        try:
            with open(model_p, "rb") as f:
                model = pickle.load(f)
        except Exception as e:
            try:
                if joblib_load is not None:
                    model = joblib_load(model_p)
                else:
                    raise e
            except Exception as e2:
                errors["model"] = str(e2)
                model = None
    else:
        errors["model"] = "File not found"
    try:
        if hasattr(model, "feature_names_in_"):
            feature_names = list(model.feature_names_in_)
    except Exception:
        pass
    return scaler, model, feature_names, errors


def header():
    st.markdown(
        """
        <div class="creditpilot-header glass">
            <div class="creditpilot-title">CreditPilot — AI Powered Loan Risk & Financial Intelligence</div>
            <div class="creditpilot-subtitle">Predictions • FSI • Risk & Decision • Explainability • Advisor • Stress Simulations • Visual Analytics</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def kpi_row(cols: List[Tuple[str, str, str]]):
    c = st.columns(len(cols))
    for i, (label, value, helptext) in enumerate(cols):
        with c[i]:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div style="font-size:12px;color:#475569;font-weight:600;">{label}</div>
                    <div style="font-size:24px;font-weight:800;margin-top:4px;">{value}</div>
                    <div class="footer-note">{helptext}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

def _theme_fig(fig):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#E6EDF7"),
        margin=dict(l=24, r=24, t=56, b=32),
    )
    return fig


def encode_category(value: str, mapping: Dict[str, int]) -> int:
    return mapping.get(value, 0)


def build_feature_frame_raw(user_input: Dict[str, Any]) -> pd.DataFrame:
    d = dict(user_input)
    d["emi"] = calculate_emi(d["loan_amount"], d["interest_rate"], d["loan_term_months"])
    d["emi_to_income_ratio"] = d["emi"] / (d["annual_income"] / 12)
    d["loan_to_income_ratio"] = d["loan_amount"] / d["annual_income"]
    d["debt_to_income_ratio"] = d["total_debt"] / d["annual_income"]
    d["credit_risk_score"] = (
        (850 - d["credit_score"]) * 0.4
        + d["debt_to_income_ratio"] * 100 * 0.3
        + d["credit_utilization_ratio"] * 100 * 0.3
    )
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
    return pd.DataFrame([{k: d[k] for k in cols}])


def predict_with_model(row_processed: Optional[pd.Series], row_raw: Optional[pd.Series], user_df_raw: Optional[pd.DataFrame]) -> Tuple[float, Dict[str, Any]]:
    scaler, model, model_features, _ = load_artifacts()
    info = {}
    if model is not None:
        try:
            if row_processed is not None:
                X = row_processed.drop(labels=["loan_status"], errors="ignore").to_frame().T
                y_prob = None
                if hasattr(model, "predict_proba"):
                    y_prob = model.predict_proba(X.values)[:, 1][0]
                else:
                    pred = model.predict(X.values)
                    y_prob = float(np.clip(pred, 0, 1))
                info["mode"] = "dataset_processed_inference"
                return float(y_prob), info
        except Exception as e:
            info["processed_error"] = str(e)
        try:
            if user_df_raw is not None:
                X = user_df_raw.copy()
                X_values = X.values
                if scaler is not None and hasattr(scaler, "transform"):
                    X_values = scaler.transform(X_values)
                y_prob = None
                if hasattr(model, "predict_proba"):
                    y_prob = model.predict_proba(X_values)[:, 1][0]
                else:
                    pred = model.predict(X_values)
                    y_prob = float(np.clip(pred, 0, 1))
                info["mode"] = "manual_form_inference"
                return float(y_prob), info
        except Exception as e:
            info["manual_error"] = str(e)
    if row_raw is not None:
        tmp = {
            "annual_income": float(row_raw.get("annual_income", 0)),
            "loan_amount": float(row_raw.get("loan_amount", 0)),
            "emi": float(row_raw.get("emi", 0)),
            "total_debt": float(row_raw.get("total_debt", 0)),
            "credit_score": float(row_raw.get("credit_score", 650)),
            "credit_utilization_ratio": float(row_raw.get("credit_utilization_ratio", 0.3)),
            "income_stability_score": float(row_raw.get("income_stability_score", 60)),
        }
        tmp = prepare_features(tmp)
        risk = (tmp["credit_risk_score"] - 50) / 40.0
        prob = float(1 / (1 + np.exp(-risk)))
        info["mode"] = "logic_fallback_from_dataset"
        return prob, info
    if user_df_raw is not None:
        d = user_df_raw.iloc[0].to_dict()
        tmp = prepare_features(d)
        risk = (tmp["credit_risk_score"] - 50) / 40.0
        prob = float(1 / (1 + np.exp(-risk)))
        info["mode"] = "logic_fallback_from_form"
        return prob, info
    info["mode"] = "default_zero"
    return 0.0, info


def overview_page(raw_df: Optional[pd.DataFrame], processed_df: Optional[pd.DataFrame]):
    header()
    total_rows = int(raw_df.shape[0]) if isinstance(raw_df, pd.DataFrame) else (int(processed_df.shape[0]) if isinstance(processed_df, pd.DataFrame) else 0)
    scaler, mdl, _, load_errs = load_artifacts()
    model_ok = mdl is not None
    scaler_ok = scaler is not None
    st.markdown('<div class="heading">Overview</div>', unsafe_allow_html=True)
    kpi_row(
        [
            ("Records", f"{total_rows:,}" if total_rows else "—", "Rows available"),
            ("Model", "Loaded" if model_ok else "Unavailable", "XGBoost/Classifier"),
            ("FSI Engine", "Active", "0–100 Financial Health"),
            ("Stress Lab", "Active", "Shock income/interest"),
        ]
    )
    col1, col2 = st.columns([1.35, 1])
    with col1:
        st.markdown('<div class="section-title">About CreditPilot</div>', unsafe_allow_html=True)
        st.markdown('<div class="card">AI‑powered loan risk and financial intelligence engine that predicts default probability, evaluates stability (FSI), classifies risk and decision, explains outcomes, provides advice, and simulates stress.</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-title" style="margin-top:14px;">Capabilities</div>', unsafe_allow_html=True)
        st.markdown('<div class="card"><ul style="margin:0;padding-left:18px;"><li>Default probability via trained model</li><li>Financial Stability Index (FSI)</li><li>Risk tiers and final decision</li><li>Explainability and improvement tips</li><li>Stress testing for resilience</li><li>Interactive Plotly analytics</li></ul></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="section-title">Artifacts</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            mtext = "Loaded" if model_ok else f"Unavailable{(' — ' + load_errs.get('model','')) if load_errs.get('model') else ''}"
            stext = "Loaded" if scaler_ok else f"Unavailable{(' — ' + load_errs.get('scaler','')) if load_errs.get('scaler') else ''}"
            st.markdown(f'<div class="card"><span class="status-dot {"status-ok" if model_ok else "status-bad"}"></span><strong>Model</strong><br>{mtext}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="card" style="margin-top:10px;"><span class="status-dot {"status-ok" if scaler_ok else "status-bad"}"></span><strong>Scaler</strong><br>{stext}</div>', unsafe_allow_html=True)
        with c2:
            raw_ok = raw_df is not None
            proc_ok = processed_df is not None
            st.markdown(f'<div class="card"><span class="status-dot {"status-ok" if raw_ok else "status-bad"}"></span><strong>Raw Data</strong><br>{"Yes" if raw_ok else "No"}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="card" style="margin-top:10px;"><span class="status-dot {"status-ok" if proc_ok else "status-bad"}"></span><strong>Processed Data</strong><br>{"Yes" if proc_ok else "No"}</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title" style="margin-top:16px;">Feature Deck</div>', unsafe_allow_html=True)
    f1, f2, f3 = st.columns([1, 1, 1])
    with f1:
        st.markdown('<div class="card"><div class="icon-badge"><span class="dot"></span>Prediction</div><div style="margin-top:6px;color:#93a3b8;">Model probability with engineered features and scaled inputs.</div></div>', unsafe_allow_html=True)
    with f2:
        st.markdown('<div class="card"><div class="icon-badge"><span class="dot" style="background:#34d399;"></span>Risk & Decision</div><div style="margin-top:6px;color:#93a3b8;">Risk tiers from probability + FSI; Approval engine for outcomes.</div></div>', unsafe_allow_html=True)
    with f3:
        st.markdown('<div class="card"><div class="icon-badge"><span class="dot" style="background:#a78bfa;"></span>Advisor & Stress</div><div style="margin-top:6px;color:#93a3b8;">EMI tools, affordability checks, suggestions, and stress tests.</div></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title" style="margin-top:16px;">Workflow</div>', unsafe_allow_html=True)
    steps = [
        ("Data", "Raw dataset ingestion"),
        ("Clean & Encode", "Quality fixes + encoding"),
        ("Feature Eng.", "Ratios and derived signals"),
        ("Scale/Model", "Normalize + train/infer"),
        ("App", "UX layers and tools"),
        ("Predictions", "Outputs and insights"),
    ]
    html = '<div class="flow-steps">'
    for i, (title, desc) in enumerate(steps):
        html += f'<div class="step-card"><div class="step-index">{i+1}</div><div class="step-title">{title}</div><div class="step-desc">{desc}</div></div>'
        if i < len(steps) - 1:
            html += '<div class="step-arrow">→</div>'
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)
    with st.expander("Data Flow (Sankey)"):
        labels = [s[0] for s in steps]
        idx = {lab: i for i, lab in enumerate(labels)}
        sources = [idx["Data"], idx["Clean & Encode"], idx["Feature Eng."], idx["Scale/Model"], idx["App"]]
        targets = [idx["Clean & Encode"], idx["Feature Eng."], idx["Scale/Model"], idx["App"], idx["Predictions"]]
        values = [1, 1, 1, 1, 1]
        node_colors = [
            "rgba(6,182,212,0.92)",   # cyan
            "rgba(99,102,241,0.92)",  # indigo
            "rgba(236,72,153,0.92)",  # pink
            "rgba(34,197,94,0.92)",   # green
            "rgba(245,158,11,0.92)",  # amber
            "rgba(250,204,21,0.92)",  # yellow
        ]
        link_colors = [
            "rgba(6,182,212,0.35)",
            "rgba(99,102,241,0.35)",
            "rgba(236,72,153,0.35)",
            "rgba(34,197,94,0.35)",
            "rgba(245,158,11,0.35)",
        ]
        x_pos = [0.02, 0.21, 0.40, 0.59, 0.78, 0.96]
        y_pos = [0.5] * len(labels)
        fig_flow = go.Figure(
            data=[
                go.Sankey(
                    arrangement="snap",
                    valueformat=".0f",
                    valuesuffix=" flow",
                    node=dict(
                        pad=22,
                        thickness=22,
                        line=dict(color="rgba(255,255,255,0.60)", width=2),
                        label=labels,
                        color=node_colors,
                        x=x_pos,
                        y=y_pos,
                        hovertemplate="%{label}<extra></extra>",
                    ),
                    link=dict(
                        source=sources,
                        target=targets,
                        value=values,
                        color=link_colors,
                        hoverlabel=dict(bgcolor="rgba(17,24,39,0.85)"),
                        hovertemplate="%{source.label} → %{target.label}<br>Flow: %{value}<extra></extra>",
                    ),
                )
            ]
        )
        fig_flow.update_layout(
            title="CreditPilot Data Flow",
            hoverlabel=dict(font_size=12, font_family="Inter"),
            font=dict(family="Sora, Inter", size=12, color="#E6EDF7"),
        )
        _theme_fig(fig_flow)
        st.plotly_chart(fig_flow, use_container_width=True)


def predict_page(raw_df: Optional[pd.DataFrame], processed_df: Optional[pd.DataFrame]):
    header()
    st.markdown('<div class="heading">Predict & Explain</div>', unsafe_allow_html=True)
    tab1, tab2 = st.tabs(["Pick Sample", "Fill Form"])
    with tab1:
        if raw_df is None and processed_df is None:
            st.error("Dataset not found. Place files in data/raw or data/processed.")
        else:
            max_id = raw_df.shape[0] if raw_df is not None else processed_df.shape[0]
            idx = st.number_input("Select record index", min_value=1, max_value=int(max_id), value=1, step=1, key="pick_idx")
            raw_row = None
            proc_row = None
            if raw_df is not None and 0 < idx <= raw_df.shape[0]:
                raw_row = raw_df.iloc[idx - 1]
            if processed_df is not None and 0 < idx <= processed_df.shape[0]:
                proc_row = processed_df.iloc[idx - 1]
            st.write("Raw Snapshot")
            if raw_row is not None:
                st.dataframe(pd.DataFrame(raw_row).T, width='stretch')
            prob, info = predict_with_model(proc_row, raw_row, None)
            d = {}
            if raw_row is not None:
                d = {
                    "annual_income": float(raw_row.get("annual_income", 0)),
                    "loan_amount": float(raw_row.get("loan_amount", 0)),
                    "emi": float(raw_row.get("emi", 0)),
                    "total_debt": float(raw_row.get("total_debt", 0)),
                    "credit_score": float(raw_row.get("credit_score", 650)),
                    "credit_utilization_ratio": float(raw_row.get("credit_utilization_ratio", 0.3)),
                    "income_stability_score": float(raw_row.get("income_stability_score", 60)),
                    "debt_to_income_ratio": float(raw_row.get("debt_to_income_ratio", 0)),
                    "emi_to_income_ratio": float(raw_row.get("emi_to_income_ratio", raw_row.get("emi", 0) / max(raw_row.get("annual_income", 1) / 12, 1))),
                }
            else:
                d = {
                    "annual_income": 0,
                    "loan_amount": 0,
                    "emi": 0,
                    "total_debt": 0,
                    "credit_score": 650,
                    "credit_utilization_ratio": 0.3,
                    "income_stability_score": 60,
                    "debt_to_income_ratio": 0,
                    "emi_to_income_ratio": 0,
                }
            d = prepare_features(d)
            fsi = calculate_fsi(d)
            risk_cat = get_risk(prob, fsi)
            decision = final_decision(prob, fsi)
            kpi_row(
                [
                    ("Default Probability", f"{prob:.2%}", info.get("mode", "—")),
                    ("FSI", f"{fsi:.0f}", "0–100"),
                    ("Risk", risk_cat, "Low/Medium/High"),
                    ("Decision", decision, "Final"),
                ]
            )
            clr = "#10b981" if risk_cat == "Low Risk" else ("#f59e0b" if risk_cat == "Medium Risk" else "#ef4444")
            st.markdown(f'<div class="card" style="display:inline-block;padding:10px 14px;"><span style="display:inline-block;width:10px;height:10px;border-radius:999px;background:{clr};margin-right:8px;"></span><strong style="color:#E6EDF7;">{risk_cat}</strong></div>', unsafe_allow_html=True)
            reasons = explain_decision(d, prob, fsi)
            sugg = suggest_improvements(d)
            with st.expander("Reasons"):
                if reasons:
                    st.write("\n".join([f"- {r}" for r in reasons]))
                else:
                    st.write("No critical risk factors.")
            with st.expander("Suggestions"):
                if sugg:
                    st.write("\n".join([f"- {s}" for s in sugg]))
                else:
                    st.write("Healthy profile.")
    with tab2:
        col1, col2, col3 = st.columns(3)
        with col1:
            age = st.number_input("Age", 18, 80, 35, key="form_age")
            employment_type = st.selectbox("Employment Type", ["salaried", "self-employed", "unemployed"], key="form_emp_type")
            employment_length_years = st.number_input("Employment Length (years)", 0, 40, 5, key="form_emp_len")
            annual_income = st.number_input("Annual Income", 50000, 3000000, 600000, step=5000, key="form_income")
            income_stability_score = st.slider("Income Stability Score", 0, 100, 60, key="form_income_stability")
            loan_amount = st.number_input("Loan Amount", 50000, 3000000, 500000, step=5000, key="form_loan_amount")
        with col2:
            loan_term_months = st.selectbox("Loan Term (months)", [12, 24, 36, 48, 60, 84, 120, 180, 240, 300], key="form_term")
            interest_rate = st.slider("Interest Rate (%)", 5.0, 20.0, 11.0, 0.1, key="form_interest_rate")
            loan_type = st.selectbox("Loan Type", ["personal", "auto", "education", "home"], key="form_loan_type")
            total_existing_loans = st.number_input("Total Existing Loans", 0, 20, 1, key="form_existing_loans")
            total_debt = st.number_input("Total Debt", 0, 5000000, 200000, step=5000, key="form_total_debt")
            credit_score = st.slider("Credit Score", 300, 850, 700, key="form_credit_score")
        with col3:
            credit_utilization_ratio = st.slider("Credit Utilization Ratio", 0.0, 1.0, 0.3, 0.01, key="form_util")
            num_credit_inquiries = st.number_input("Credit Inquiries (24m)", 0, 20, 2, key="form_inquiries")
            late_payment_count = st.number_input("Late Payments (24m)", 0, 30, 0, key="form_late")
            default_history = st.selectbox("Past Default", [0, 1], key="form_default_hist")
            repayment_history_score = st.slider("Repayment History Score", 0.0, 100.0, 70.0, 0.1, key="form_repay")
            credit_history_length_years = st.number_input("Credit History Length (years)", 0, 40, 5, key="form_ch_len")
            job_stability_score = st.slider("Job Stability Score", 0.0, 100.0, 70.0, 0.1, key="form_job_stability")
            spending_to_income_ratio = st.slider("Spending to Income Ratio", 0.0, 1.0, 0.4, 0.01, key="form_spend")
        if st.button("Predict"):
            cat_map_emp = {"salaried": 0, "self-employed": 1, "unemployed": -1}
            cat_map_loan = {"personal": 0, "auto": 1, "education": 2, "home": 3}
            payload = {
                "age": age,
                "employment_type": encode_category(employment_type, cat_map_emp),
                "employment_length_years": employment_length_years,
                "annual_income": float(annual_income),
                "income_stability_score": float(income_stability_score),
                "loan_amount": float(loan_amount),
                "loan_term_months": int(loan_term_months),
                "interest_rate": float(interest_rate),
                "loan_type": encode_category(loan_type, cat_map_loan),
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
            user_df = build_feature_frame_raw(payload)
            prob, info = predict_with_model(None, None, user_df)
            feat_for_logic = prepare_features(user_df.iloc[0].to_dict())
            fsi = calculate_fsi(feat_for_logic)
            risk_cat = get_risk(prob, fsi)
            decision = final_decision(prob, fsi)
            kpi_row(
                [
                    ("Default Probability", f"{prob:.2%}", info.get("mode", "—")),
                    ("FSI", f"{fsi:.0f}", "0–100"),
                    ("Risk", risk_cat, "Low/Medium/High"),
                    ("Decision", decision, "Final"),
                ]
            )
            emi_val = calculate_emi(payload["loan_amount"], payload["interest_rate"], payload["loan_term_months"])
            aff = check_affordability(emi_val, payload["annual_income"] / 12)
            st.info(f"EMI: {emi_val:.2f} • Affordability: {aff}")
            reasons = explain_decision(feat_for_logic, prob, fsi)
            sugg = suggest_improvements(feat_for_logic)
            with st.expander("Reasons"):
                if reasons:
                    st.write("\n".join([f"- {r}" for r in reasons]))
                else:
                    st.write("No critical risk factors.")
            with st.expander("Suggestions"):
                if sugg:
                    st.write("\n".join([f"- {s}" for s in sugg]))
                else:
                    st.write("Healthy profile.")


def advisor_page():
    header()
    st.markdown('<div class="heading">Advisor</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-title">Quick Affordability</div>', unsafe_allow_html=True)
        emi_in = st.number_input("Monthly EMI", 1000.0, 500000.0, 20000.0, step=500.0, key="aff_emi")
        mi = st.number_input("Monthly Income", 5000.0, 500000.0, 60000.0, step=1000.0, key="aff_mi")
        aff = check_affordability(emi_in, mi)
        st.markdown(f'<div class="card">Affordability: <strong>{aff}</strong></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="section-title">Loan Health</div>', unsafe_allow_html=True)
        prob_in = st.slider("Default Probability", 0.0, 1.0, 0.35, 0.01, key="advisor_prob")
        fsi_in = st.slider("FSI", 0.0, 100.0, 70.0, 1.0, key="advisor_fsi")
        st.markdown(f'<div class="card">Loan Health Score: <strong>{loan_health_score(prob_in, fsi_in):.1f}</strong></div>', unsafe_allow_html=True)


def amortization_schedule(loan_amount: float, annual_rate: float, months: int, extra_payment: float = 0.0) -> pd.DataFrame:
    r = annual_rate / (12 * 100.0)
    emi = calculate_emi(loan_amount, annual_rate, months)
    bal = float(loan_amount)
    rows = []
    m = 1
    while m <= months and bal > 0:
        interest = bal * r
        principal = min(bal, emi - interest + extra_payment)
        if principal < 0:
            principal = 0
        bal = max(0.0, bal - principal)
        rows.append({"month": m, "interest": interest, "principal": principal, "balance": bal})
        m += 1
        if bal <= 0:
            break
    df = pd.DataFrame(rows)
    return df


def emi_page():
    header()
    st.markdown('<div class="heading">EMI Calculator</div>', unsafe_allow_html=True)
    with st.container():
        c1, c2 = st.columns([1, 1])
        with c1:
            la = st.number_input("Loan Amount", 50000.0, 10000000.0, 500000.0, step=10000.0, key="emi2_la")
            ir = st.slider("Interest Rate (%)", 5.0, 30.0, 11.0, 0.1, key="emi2_ir")
            tm_years = st.slider("Tenure (years)", 1, 30, 5, 1, key="emi2_years")
            extra = st.number_input("Extra Monthly Payment (optional)", 0.0, 100000.0, 0.0, step=500.0, key="emi2_extra")
            tm = tm_years * 12
        with c2:
            emi_val = calculate_emi(la, ir, tm)
            sch = amortization_schedule(la, ir, tm, extra_payment=extra)
            tot_interest = float(sch["interest"].sum())
            tot_principal = float(sch["principal"].sum())
            months_actual = int(sch["month"].max()) if not sch.empty else 0
            kpi_row(
                [
                    ("EMI", f"{emi_val:,.0f}", "Monthly payment"),
                    ("Total Interest", f"{tot_interest:,.0f}", "Over loan life"),
                    ("Payoff Months", f"{months_actual}", "With extra payment"),
                    ("Total Paid", f"{(tot_interest+tot_principal):,.0f}", "Principal + interest"),
                ]
            )
    st.markdown("")
    colA, colB = st.columns([1, 1])
    with colA:
        fig = px.line(sch, x="month", y="balance", title="Remaining Balance")
        _theme_fig(fig)
        st.plotly_chart(fig, use_container_width=True)
    with colB:
        pie_df = pd.DataFrame({"type": ["Principal", "Interest"], "amount": [tot_principal, tot_interest]})
        fig2 = px.pie(pie_df, names="type", values="amount", title="Principal vs Interest", hole=0.4,
                      color_discrete_sequence=px.colors.sequential.Blues_r)
        _theme_fig(fig2)
        st.plotly_chart(fig2, use_container_width=True)
    with st.expander("Amortization Schedule"):
        st.dataframe(sch, width='stretch')
    prob_in = st.slider("Assumed Default Probability", 0.0, 1.0, 0.35, 0.01, key="emi_prob")
    fsi_in = st.slider("FSI", 0.0, 100.0, 70.0, 1.0, key="emi_fsi")
    st.success(f"Loan Health Score: {loan_health_score(prob_in, fsi_in):.1f}")


def stress_page(raw_df: Optional[pd.DataFrame], processed_df: Optional[pd.DataFrame]):
    header()
    st.markdown('<div class="heading">Stress Simulation</div>', unsafe_allow_html=True)
    if raw_df is None and processed_df is None:
        st.error("Dataset not found.")
        return
    max_id = raw_df.shape[0] if raw_df is not None else processed_df.shape[0]
    idx = st.number_input("Select record index", min_value=1, max_value=int(max_id), value=1, step=1, key="stress_idx")
    raw_row = raw_df.iloc[idx - 1] if raw_df is not None and 0 < idx <= raw_df.shape[0] else None
    proc_row = processed_df.iloc[idx - 1] if processed_df is not None and 0 < idx <= processed_df.shape[0] else None
    base_prob, info = predict_with_model(proc_row, raw_row, None)
    base_dict = {}
    if raw_row is not None:
        base_dict = {
            "annual_income": float(raw_row.get("annual_income", 0)),
            "loan_amount": float(raw_row.get("loan_amount", 0)),
            "emi": float(raw_row.get("emi", 0)),
            "total_debt": float(raw_row.get("total_debt", 0)),
            "credit_score": float(raw_row.get("credit_score", 650)),
            "credit_utilization_ratio": float(raw_row.get("credit_utilization_ratio", 0.3)),
            "income_stability_score": float(raw_row.get("income_stability_score", 60)),
            "debt_to_income_ratio": float(raw_row.get("debt_to_income_ratio", 0)),
            "emi_to_income_ratio": float(raw_row.get("emi_to_income_ratio", raw_row.get("emi", 0) / max(raw_row.get("annual_income", 1) / 12, 1))),
        }
    else:
        st.error("Raw record unavailable for stress inputs.")
        return
    st.markdown('<div class="section-title">Profile Snapshot</div>', unsafe_allow_html=True)
    kpi_row(
        [
            ("Credit Score", f"{int(base_dict['credit_score'])}", ""),
            ("DTI", f"{base_dict['debt_to_income_ratio']:.2f}", "Debt / Income"),
            ("Utilization", f"{base_dict['credit_utilization_ratio']:.2f}", "Credit use"),
            ("EMI / Income", f"{base_dict['emi_to_income_ratio']:.2f}", "Monthly"),
        ]
    )
    p1, p2, p3 = st.columns(3)
    with p1:
        if st.button("Mild Shock", key="preset_mild"):
            st.session_state["stress_income_drop"] = 10
            st.session_state["stress_rate_up"] = 1
            st.session_state["stress_util_up"] = 0.05
            st.rerun()
    with p2:
        if st.button("Moderate Shock", key="preset_moderate"):
            st.session_state["stress_income_drop"] = 25
            st.session_state["stress_rate_up"] = 3
            st.session_state["stress_util_up"] = 0.15
            st.rerun()
    with p3:
        if st.button("Severe Shock", key="preset_severe"):
            st.session_state["stress_income_drop"] = 45
            st.session_state["stress_rate_up"] = 6
            st.session_state["stress_util_up"] = 0.3
            st.rerun()
    c1, c2, c3 = st.columns(3)
    with c1:
        income_drop = st.slider("Income Drop (%)", 0, 60, 20, 1, key="stress_income_drop")
    with c2:
        interest_up = st.slider("Interest Increase (%)", 0, 10, 2, 1, key="stress_rate_up")
    with c3:
        util_up = st.slider("Credit Utilization +", 0.0, 0.5, 0.1, 0.01, key="stress_util_up")
    stressed = dict(base_dict)
    stressed["annual_income"] = max(1.0, base_dict["annual_income"] * (1 - income_drop / 100))
    stressed["credit_utilization_ratio"] = min(1.0, base_dict["credit_utilization_ratio"] + util_up)
    emi_base = base_dict.get("emi", 0.0)
    emi_stress = emi_base * (1 + interest_up / 100)
    stressed["emi"] = emi_stress
    stressed = prepare_features(stressed)
    fsi_base = calculate_fsi(base_dict if "debt_to_income_ratio" in base_dict else prepare_features(base_dict))
    fsi_stress = calculate_fsi(stressed)
    risk_cat_base = get_risk(base_prob, fsi_base)
    prob_stress = min(1.0, max(0.0, base_prob + (interest_up / 100) * 0.2 + (income_drop / 100) * 0.3 + util_up * 0.25))
    risk_cat_stress = get_risk(prob_stress, fsi_stress)
    clr_base = "#10b981" if risk_cat_base == "Low Risk" else ("#f59e0b" if risk_cat_base == "Medium Risk" else "#ef4444")
    clr_stress = "#10b981" if risk_cat_stress == "Low Risk" else ("#f59e0b" if risk_cat_stress == "Medium Risk" else "#ef4444")
    kpi_row([("Base Prob", f"{base_prob:.2%}", risk_cat_base),
             ("Stress Prob", f"{prob_stress:.2%}", risk_cat_stress),
             ("FSI Base", f"{fsi_base:.0f}", "0–100"),
             ("FSI Stress", f"{fsi_stress:.0f}", stress_impact(base_prob, prob_stress))])
    st.markdown(
        f'<div class="card" style="display:flex;gap:12px;align-items:center;max-width:fit-content;margin-top:6px;"><span class="pill" style="background:{clr_base};color:#0b1220;">Base: {risk_cat_base}</span><span class="pill" style="background:{clr_stress};color:#0b1220;">Stress: {risk_cat_stress}</span></div>',
        unsafe_allow_html=True,
    )
    comp = pd.DataFrame(
        {"metric": ["Probability", "Probability", "FSI", "FSI"], "scenario": ["Base", "Stress", "Base", "Stress"], "value": [base_prob * 100, prob_stress * 100, fsi_base, fsi_stress]}
    )
    fig_bar = px.bar(comp, x="metric", y="value", color="scenario", barmode="group", color_discrete_sequence=px.colors.qualitative.Set2, title="Base vs Stress Comparison")
    _theme_fig(fig_bar)
    st.plotly_chart(fig_bar, use_container_width=True)
    g = go.Figure()
    g.add_trace(go.Indicator(mode="gauge+number", value=fsi_base, title={"text": "FSI Base"}, gauge={"axis": {"range": [0, 100]}}))
    g.add_trace(go.Indicator(mode="gauge+number", value=fsi_stress, title={"text": "FSI Stress"}, gauge={"axis": {"range": [0, 100]}}, domain={"x": [0.52, 1], "y": [0, 1]}))
    g.update_layout(height=260, margin=dict(l=24, r=24, t=24, b=24))
    st.plotly_chart(g, use_container_width=True)


def viz_page(raw_df: Optional[pd.DataFrame], processed_df: Optional[pd.DataFrame]):
    header()
    st.markdown('<div class="heading">Visual Analytics</div>', unsafe_allow_html=True)
    df = raw_df
    if df is None:
        st.error("Raw dataset not found.")
        return
    colA, colB = st.columns([1, 1])
    with colA:
        fig = px.histogram(df, x="credit_score", nbins=40, color="loan_status", barmode="overlay", color_discrete_sequence=px.colors.qualitative.Set2)
        _theme_fig(fig)
        fig.update_layout(title="Credit Score Distribution")
        st.plotly_chart(fig, use_container_width=True)
    with colB:
        fig = px.scatter(
            df,
            x="annual_income",
            y="loan_amount",
            color="loan_status",
            size="emi",
            hover_data=["employment_type", "interest_rate"],
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        _theme_fig(fig)
        fig.update_layout(title="Income vs Loan Amount")
        st.plotly_chart(fig, use_container_width=True)
    colC, colD = st.columns([1, 1])
    with colC:
        fig = px.box(df, x="loan_status", y="debt_to_income_ratio", color="loan_status", color_discrete_sequence=px.colors.qualitative.Set2)
        _theme_fig(fig)
        fig.update_layout(title="Debt-to-Income by Status")
        st.plotly_chart(fig, use_container_width=True)
    with colD:
        fig = px.violin(df, x="employment_type", y="emi_to_income_ratio", color="employment_type", box=True, color_discrete_sequence=px.colors.qualitative.Set2)
        _theme_fig(fig)
        fig.update_layout(title="EMI Burden by Employment Type")
        st.plotly_chart(fig, use_container_width=True)
    num_cols = [
        "annual_income",
        "loan_amount",
        "emi",
        "total_debt",
        "debt_to_income_ratio",
        "credit_score",
        "credit_utilization_ratio",
        "emi_to_income_ratio",
        "loan_to_income_ratio",
        "interest_rate",
        "repayment_history_score",
        "spending_to_income_ratio",
    ]
    corr_df = df[num_cols].corr(numeric_only=True)
    fig = px.imshow(corr_df, text_auto=True, color_continuous_scale="RdBu", origin="lower")
    _theme_fig(fig)
    fig.update_layout(title="Correlation Heatmap")
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('<div class="section-title">Segmented Default Rate</div>', unsafe_allow_html=True)
    seg = df.groupby("employment_type")["loan_status"].mean().reset_index(name="default_rate")
    fig = px.bar(seg, x="employment_type", y="default_rate", text="default_rate", color="employment_type", color_discrete_sequence=px.colors.qualitative.Set2)
    _theme_fig(fig)
    fig.update_traces(texttemplate="%{text:.1%}", textposition="outside")
    fig.update_layout(yaxis_tickformat=".0%")
    st.plotly_chart(fig, width='stretch')


def main():
    _inject_style()
    raw_df = load_raw_data()
    processed_df = load_processed_data()
    pages = ["Overview", "Predict", "EMI Calculator", "Advisor", "Stress Lab", "Visualizations"]
    st.markdown('<div class="nav-tabs">', unsafe_allow_html=True)
    tabs = st.tabs(pages)
    st.markdown("</div>", unsafe_allow_html=True)
    with tabs[0]:
        overview_page(raw_df, processed_df)
    with tabs[1]:
        predict_page(raw_df, processed_df)
    with tabs[2]:
        emi_page()
    with tabs[3]:
        advisor_page()
    with tabs[4]:
        stress_page(raw_df, processed_df)
    with tabs[5]:
        viz_page(raw_df, processed_df)


if __name__ == "__main__":
    main()
