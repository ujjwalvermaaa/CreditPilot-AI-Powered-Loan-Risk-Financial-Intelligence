# CreditPilot — Advanced AI Loan Intelligence Suite

CreditPilot is an AI-powered loan risk assessment and financial intelligence platform. It goes beyond a simple credit score — it gives borrowers a detailed breakdown of their loan eligibility, a custom Financial Stability Index (FSI), a Stress Lab to simulate economic shocks, an AI financial coach for personalized advice, and a portfolio dashboard for lenders. Built entirely in Python with Streamlit.

---

## The Problem It Solves

Traditional credit systems are black boxes. When a loan is rejected, applicants get vague reasons and no guidance on how to improve. CreditPilot flips this — every decision is explained, every variable is quantified, and every user gets a roadmap to approval.

---

## Key Features

| Feature | Description |
|---|---|
| **Risk Classification** | XGBoost model classifies applicants as Approved / Conditional / Rejected with >90% accuracy |
| **Financial Stability Index (FSI)** | A 0–100 composite score built from 5 financial health pillars |
| **Interest Rate Estimation** | Personalized interest rate predicted by a Gradient Boosting Regressor |
| **Stress Lab** | Simulate income drops, rate hikes, or debt increases and see real-time impact on approval |
| **AI Financial Coach** | Conversational assistant that explains rejections and gives actionable advice in plain language |
| **Recommendation Lab** | Shows exactly how much debt to pay off / income to increase to flip a rejection to approval |
| **History Tracking** | Local profile snapshots let users track their financial progress over time |
| **Admin Dashboard** | Portfolio-level analytics: approval trends, risk clusters, model performance metrics |

---

## Tech Stack

| Layer | Technology |
|---|---|
| UI | Streamlit |
| ML — Classification | XGBoost |
| ML — Regression | Gradient Boosting Regressor (scikit-learn) |
| Explainability | SHAP-inspired feature importance breakdowns |
| Data | Pandas, NumPy |
| Visualization | Plotly (interactive gauges, charts), Seaborn, Matplotlib |
| AI Coach | OpenAI GPT-4 API (optional; falls back to local deterministic responses) |
| Model Serialization | joblib |

---

## Project Structure

```
CreditPilot/
├── src/
│   ├── app.py                    # Main Streamlit multi-page entry point
│   ├── logic.py                  # Core financial formulas: EMI, DTI, FSI calculation
│   ├── advanced_logic.py         # Stress Lab, co-applicant optimization, goal planning
│   ├── train_advanced_models.py  # Model training script (XGBoost + GBR)
│   └── generate_paper_assets.py  # Generates diagrams and report assets
├── models/
│   ├── xgb_classifier.pkl        # Trained XGBoost classification model
│   └── gbr_rate_estimator.pkl    # Trained interest rate estimator
├── data/
│   └── loan_dataset.csv          # 10,000-record synthetic retail loan dataset
├── notebooks/
│   └── exploratory_analysis.ipynb
├── paper_assets/                 # Diagrams, charts for research paper
├── requirements.txt
└── CreditPilot_Project_Report.txt   # Full academic project report
```

---

## Prerequisites

- **Python 3.9, 3.10, or 3.11**
- **4 GB+ RAM** recommended
- **OpenAI API key** (optional — only needed for the AI Coach feature)

---

## Setup

```bash
# 1. Clone / download the project
cd CreditPilot

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate       # macOS/Linux
.venv\Scripts\activate          # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Training the Models

Before running the app, you need to generate the model `.pkl` files:

```bash
cd src
python train_advanced_models.py
```

This will:
1. Load and preprocess `data/loan_dataset.csv`
2. Train the XGBoost classifier and GBR interest estimator
3. Save `models/xgb_classifier.pkl` and `models/gbr_rate_estimator.pkl`

---

## Running the App

```bash
# From the project root
streamlit run src/app.py
```

Open **http://localhost:8501** in your browser.

---

## How to Use

### For Loan Applicants

1. Go to the **Studio** tab
2. Enter your financial details: income, employment type, loan amount, existing debt, credit score, credit history, credit utilization, etc.
3. Click **Analyze My Profile**
4. Review your:
   - **Approval Status** (Approved / Conditional / Rejected)
   - **Estimated Interest Rate**
   - **Financial Stability Index (FSI)** with breakdown across 5 pillars
   - **Top rejection drivers** — specific variables holding you back
5. Use the **Stress Lab** to simulate scenarios: "What if my income drops 20%?" or "What if rates rise to 12%?"
6. Check the **Recommendation Lab** for exact steps to reach Approved status

### For Portfolio Managers

1. Go to the **Admin Dashboard** tab
2. View aggregate approval rates, risk cluster heatmaps, and model performance metrics (Accuracy, Precision, Recall, AUC)

---

## Financial Stability Index (FSI)

The FSI is a 0–100 composite score built from 5 weighted pillars:

| Pillar | Weight | What it measures |
|---|---|---|
| Income Stability | 25% | Employment type and tenure |
| Credit History | 20% | Credit score and length of history |
| Debt Management | 20% | DTI ratio and EMI-to-income ratio |
| Repayment Discipline | 15% | History of late payments and defaults |
| Credit Utilization | 20% | Current balance vs. credit limits |

---

## Model Performance

Evaluated against Logistic Regression, Random Forest, and SVM baselines:

| Metric | XGBoost (CreditPilot) |
|---|---|
| Accuracy | 90.3% |
| Precision | 0.93 |
| Recall | 0.90 |
| ROC-AUC | 0.967 |

---

## Environment Variables (Optional)

For AI Coach with GPT-4, create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_openai_api_key_here
```

Without this, the AI Coach uses built-in deterministic responses.

---

## Troubleshooting

| Issue | Fix |
|---|---|
| `FileNotFoundError: models/xgb_classifier.pkl` | Run `python src/train_advanced_models.py` first |
| Streamlit shows blank page | Ensure you're running from the project root: `streamlit run src/app.py` |
| Slow inference | This is normal on first run — models are loaded into memory once and cached |
| OpenAI API errors | Check your API key in `.env` or the app will fall back to local responses automatically |
