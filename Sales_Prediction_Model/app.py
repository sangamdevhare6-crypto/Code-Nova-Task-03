"""
Sales Prediction Model — Streamlit Dashboard
=============================================
Three tabs:
  1. Predict Sales      – live single-record prediction
  2. Model Evaluation   – metrics, charts, feature importance
  3. Predictions        – full test-set results with CSV download

Run:
    streamlit run app.py
"""

import os
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Sales Prediction Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.main { background: linear-gradient(135deg, #0f0c29, #302b63, #24243e); }

.metric-card {
    background: rgba(255,255,255,0.07);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 14px;
    padding: 22px 20px;
    text-align: center;
    backdrop-filter: blur(10px);
    transition: transform .2s;
}
.metric-card:hover { transform: translateY(-3px); }
.metric-card .label { font-size: 0.78rem; color: #a0aec0; letter-spacing: 1px; text-transform: uppercase; }
.metric-card .value { font-size: 2rem; font-weight: 700; color: #63b3ed; margin-top: 4px; }

.predict-box {
    background: linear-gradient(135deg, rgba(99,179,237,.15), rgba(154,107,254,.15));
    border: 1px solid rgba(99,179,237,.35);
    border-radius: 16px;
    padding: 28px;
    text-align: center;
    margin-top: 20px;
}
.predict-box .result-label { font-size: 1rem; color: #a0aec0; }
.predict-box .result-value { font-size: 3rem; font-weight: 700;
    background: linear-gradient(90deg, #63b3ed, #9a6bfe);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; }

.section-header {
    font-size: 1.15rem; font-weight: 600; color: #e2e8f0;
    border-left: 4px solid #63b3ed; padding-left: 10px;
    margin: 24px 0 14px;
}
.info-badge {
    display: inline-block;
    background: rgba(99,179,237,.18);
    color: #63b3ed;
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 0.82rem;
    font-weight: 600;
    margin-bottom: 8px;
}
</style>
""", unsafe_allow_html=True)

# ── Paths ────────────────────────────────────────────────────────────────────
OUTPUT_DIR      = "output"
MODEL_PATH      = os.path.join(OUTPUT_DIR, "best_sales_model.joblib")
SUMMARY_JSON    = os.path.join(OUTPUT_DIR, "summary.json")
COMPARISON_CSV  = os.path.join(OUTPUT_DIR, "model_comparison.csv")
PREDICTIONS_CSV = os.path.join(OUTPUT_DIR, "predictions.csv")
GRAPHS_DIR      = os.path.join(OUTPUT_DIR, "graphs")

STORES   = ["Store_A", "Store_B", "Store_C", "Store_D"]
PRODUCTS = ["Electronics", "Clothing", "Groceries", "Furniture", "Sports"]

# ── Load artefacts (cached) ──────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)

@st.cache_data(show_spinner=False)
def load_summary():
    if not os.path.exists(SUMMARY_JSON):
        return None
    with open(SUMMARY_JSON) as f:
        return json.load(f)

@st.cache_data(show_spinner=False)
def load_comparison():
    if not os.path.exists(COMPARISON_CSV):
        return None
    return pd.read_csv(COMPARISON_CSV)

@st.cache_data(show_spinner=False)
def load_predictions():
    if not os.path.exists(PREDICTIONS_CSV):
        return None
    return pd.read_csv(PREDICTIONS_CSV)

def artefacts_missing():
    return not os.path.exists(MODEL_PATH)

# ── Helper: build a single-row input dataframe ───────────────────────────────
def build_input_row(date, store, product, price, ad_spend, promotion, holiday):
    return pd.DataFrame([{
        "Store":             store,
        "Product":           product,
        "Price":             float(price),
        "Advertising_Spend": float(ad_spend),
        "Promotion":         int(promotion),
        "Holiday":           int(holiday),
        "Year":              date.year,
        "Month":             date.month,
        "DayOfWeek":         date.weekday(),
        "DayOfYear":         date.timetuple().tm_yday,
    }])

# ── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; padding: 30px 0 10px;">
  <h1 style="font-size:2.6rem; font-weight:700;
      background:linear-gradient(90deg,#63b3ed,#9a6bfe);
      -webkit-background-clip:text; -webkit-text-fill-color:transparent;">
    📈 Sales Prediction Dashboard
  </h1>
  <p style="color:#a0aec0; font-size:1rem; margin-top:-6px;">
    Machine Learning · Regression · Synthetic Data Demo
  </p>
</div>
""", unsafe_allow_html=True)

# ── Missing-artefacts warning ────────────────────────────────────────────────
if artefacts_missing():
    st.error(
        "**Model artefacts not found.**\n\n"
        "Please run the training pipeline first:\n"
        "```\npython main.py\n```\n"
        "This will generate the model, graphs, and CSV files in the `output/` folder."
    )
    st.stop()

model   = load_model()
summary = load_summary()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🔮 Predict Sales", "📊 Model Evaluation", "📋 Predictions"])

# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — Predict Sales
# ════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-header">Enter Prediction Inputs</div>',
                unsafe_allow_html=True)

    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        pred_date  = st.date_input("📅 Prediction Date", value=pd.Timestamp("2024-01-15"))
        store      = st.selectbox("🏪 Store", STORES)
        product    = st.selectbox("📦 Product", PRODUCTS)

    with col_right:
        price      = st.number_input("💲 Product Price ($)", min_value=1.0,
                                     max_value=10000.0, value=150.0, step=5.0)
        ad_spend   = st.number_input("📣 Advertising Spend ($)", min_value=0.0,
                                     max_value=50000.0, value=500.0, step=50.0)
        promotion  = st.toggle("🎁 Promotion Active", value=False)
        holiday    = st.toggle("🏖️ Weekend / Holiday", value=False)

    st.markdown("---")

    if st.button("🔮 Predict Sales", type="primary", width='stretch'):
        if price <= 0:
            st.warning("Please enter a valid price greater than $0.")
        else:
            row   = build_input_row(pred_date, store, product,
                                    price, ad_spend, int(promotion), int(holiday))
            pred  = model.predict(row)[0]
            pred  = max(0.0, pred)   # clamp negatives

            st.markdown(f"""
            <div class="predict-box">
              <div class="result-label">Estimated Sales</div>
              <div class="result-value">${pred:,.2f}</div>
              <div style="color:#718096; font-size:0.8rem; margin-top:10px;">
                ⚠️ This is a model estimate, not a guarantee of actual sales.
                Predictions are based on synthetic training data only.
              </div>
            </div>
            """, unsafe_allow_html=True)

            # Breakdown card
            st.markdown('<div class="section-header">Input Summary</div>',
                        unsafe_allow_html=True)
            summary_df = pd.DataFrame({
                "Field":  ["Date", "Store", "Product", "Price",
                           "Ad Spend", "Promotion", "Holiday"],
                "Value":  [str(pred_date), store, product,
                           f"${price:,.2f}", f"${ad_spend:,.2f}",
                           "Yes" if promotion else "No",
                           "Yes" if holiday else "No"],
            })
            st.dataframe(summary_df, width='stretch', hide_index=True)


# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — Model Evaluation
# ════════════════════════════════════════════════════════════════════════════
with tab2:
    if summary is None:
        st.error("summary.json not found. Run `python main.py` first.")
        st.stop()

    best_name = summary["best_model"]
    metrics   = summary["metrics"]

    st.markdown(f"""
    <div style="text-align:center; margin-bottom:20px;">
      <span class="info-badge">🏆 Best Model</span>
      <h2 style="color:#e2e8f0; margin:4px 0;">{best_name}</h2>
      <p style="color:#718096; font-size:0.85rem;">
        Selected by <strong>lowest RMSE</strong> on the held-out test set (chronological 80/20 split)
      </p>
    </div>
    """, unsafe_allow_html=True)

    # Metric cards
    mc1, mc2, mc3 = st.columns(3)
    with mc1:
        st.markdown(f"""<div class="metric-card">
          <div class="label">R² Score</div>
          <div class="value">{metrics['R2']:.4f}</div>
        </div>""", unsafe_allow_html=True)
    with mc2:
        st.markdown(f"""<div class="metric-card">
          <div class="label">MAE ($)</div>
          <div class="value">{metrics['MAE']:,.2f}</div>
        </div>""", unsafe_allow_html=True)
    with mc3:
        st.markdown(f"""<div class="metric-card">
          <div class="label">RMSE ($)</div>
          <div class="value">{metrics['RMSE']:,.2f}</div>
        </div>""", unsafe_allow_html=True)

    # Metric explanations
    with st.expander("ℹ️ What do these metrics mean?"):
        st.markdown("""
| Metric | Meaning | Better when |
|--------|---------|-------------|
| **R²** | Proportion of variance explained (0–1) | Closer to 1 |
| **MAE** | Average absolute prediction error in $ | Lower |
| **RMSE** | Root-mean-square error; penalises large errors | Lower |
        """)

    st.markdown("---")

    # Model comparison table
    comp_df = load_comparison()
    if comp_df is not None:
        st.markdown('<div class="section-header">All Models Comparison</div>',
                    unsafe_allow_html=True)
        styled = comp_df.copy()
        styled["R2"]   = styled["R2"].map(lambda x: f"{x:.4f}")
        styled["MAE"]  = styled["MAE"].map(lambda x: f"${x:,.2f}")
        styled["RMSE"] = styled["RMSE"].map(lambda x: f"${x:,.2f}")
        st.dataframe(styled, width='stretch', hide_index=True)

    st.markdown("---")

    # Charts
    g1, g2 = st.columns(2)
    avp_path = os.path.join(GRAPHS_DIR, "actual_vs_predicted.png")
    cmp_path = os.path.join(GRAPHS_DIR, "model_comparison.png")
    trend_path = os.path.join(GRAPHS_DIR, "sales_trend.png")
    fimp_path  = os.path.join(GRAPHS_DIR, "feature_importance.png")

    with g1:
        st.markdown('<div class="section-header">Actual vs Predicted</div>',
                    unsafe_allow_html=True)
        if os.path.exists(avp_path):
            st.image(avp_path, width='stretch')
        else:
            st.info("Chart not found — run `python main.py`.")

    with g2:
        st.markdown('<div class="section-header">Model Comparison Chart</div>',
                    unsafe_allow_html=True)
        if os.path.exists(cmp_path):
            st.image(cmp_path, width='stretch')
        else:
            st.info("Chart not found — run `python main.py`.")

    st.markdown('<div class="section-header">Historical Sales Trend</div>',
                unsafe_allow_html=True)
    if os.path.exists(trend_path):
        st.image(trend_path, width='stretch')
    else:
        st.info("Chart not found — run `python main.py`.")

    if os.path.exists(fimp_path):
        st.markdown('<div class="section-header">Feature Importance</div>',
                    unsafe_allow_html=True)
        st.image(fimp_path, width='stretch')
    else:
        st.info("Feature importance chart not available for this model type.")


# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — Predictions
# ════════════════════════════════════════════════════════════════════════════
with tab3:
    pred_df = load_predictions()
    if pred_df is None:
        st.error("predictions.csv not found. Run `python main.py` first.")
        st.stop()

    st.markdown('<div class="section-header">Test-Set Predictions</div>',
                unsafe_allow_html=True)
    st.caption(f"Showing {len(pred_df):,} records from the held-out test set (last 20% chronologically).")

    # Search / filter
    search = st.text_input("🔍 Filter by Store or Product", "")
    display_df = pred_df.copy()
    if search:
        mask = (
            display_df["Store"].astype(str).str.contains(search, case=False, na=False)
            | display_df["Product"].astype(str).str.contains(search, case=False, na=False)
        )
        display_df = display_df[mask]

    st.dataframe(display_df, width='stretch', height=420)

    # Download button
    csv_bytes = pred_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download Predictions as CSV",
        data=csv_bytes,
        file_name="sales_predictions.csv",
        mime="text/csv",
        width='stretch',
    )

    # Summary stats
    st.markdown("---")
    st.markdown('<div class="section-header">Prediction Statistics</div>',
                unsafe_allow_html=True)
    sc1, sc2, sc3, sc4 = st.columns(4)
    sc1.metric("Records",      f"{len(pred_df):,}")
    sc2.metric("Avg Actual",   f"${pred_df['Actual_Sales'].mean():,.2f}")
    sc3.metric("Avg Predicted",f"${pred_df['Predicted_Sales'].mean():,.2f}")
    err = (pred_df["Actual_Sales"] - pred_df["Predicted_Sales"]).abs().mean()
    sc4.metric("Mean Abs Error", f"${err:,.2f}")
