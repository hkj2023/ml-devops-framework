import streamlit as st
import pandas as pd
# from streamlit_autorefresh import st_autorefresh
import json
import time
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go

# ============================================
# PAGE CONFIG
# ============================================
st.set_page_config(
    page_title="ML DevOps Framework Dashboard",
    page_icon="🚀",
    layout="wide"
)
# ============================================
# AUTO REFRESH (DEVOPS STYLE MONITORING)
# ============================================

# REFRESH_INTERVAL = 5  # seconds

st.sidebar.title("⚙️ Dashboard Controls")
# ===================================
# auto_refresh = st.sidebar.toggle("Enable Auto Refresh", value=True)

#if auto_refresh:
 #   st.sidebar.info(f"Refreshing every {REFRESH_INTERVAL}s")
  #  time.sleep(REFRESH_INTERVAL)
   # st.rerun()

# Refresh every 5 seconds (5000 ms)
#st_autorefresh(interval=5000, limit=None, key="ml_refresh")
# ============================================
# PATHS
# ============================================
BASE_DIR = Path(__file__).resolve().parent
OUTPUTS = BASE_DIR / "outputs"

METRICS_FILE = OUTPUTS / "metrics.json"
RISK_FILE = OUTPUTS / "risk_prediction.json"
ANOMALY_SUMMARY_FILE = OUTPUTS / "anomaly_summary.json"
ANOMALY_RESULTS_FILE = OUTPUTS / "anomaly_results.csv"
CLUSTER_SUMMARY_FILE = OUTPUTS / "cluster_summary.json"
CLUSTER_RESULTS_FILE = OUTPUTS / "cluster_results.csv"
TEST_RESULTS_FILE = BASE_DIR / "results.json"

# ============================================
# HELPERS
# ============================================
def load_json(path):
    if path.exists():
        with open(path, "r") as f:
            return json.load(f)
    return {}

def load_csv(path):
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()

# Load files
metrics = load_json(METRICS_FILE)
risk = load_json(RISK_FILE)
anomaly_summary = load_json(ANOMALY_SUMMARY_FILE)
cluster_summary = load_json(CLUSTER_SUMMARY_FILE)
test_results = load_json(TEST_RESULTS_FILE)

anomaly_df = load_csv(ANOMALY_RESULTS_FILE)
cluster_df = load_csv(CLUSTER_RESULTS_FILE)

# ============================================
# HEADER
# ============================================
st.title("🚀 ML-Enabled DevOps Framework Dashboard")
st.caption("Monitoring ML model, risk prediction, anomaly detection, clustering & CI/CD tests")

st.divider()

# ============================================
# TOP KPI ROW
# ============================================
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Accuracy", f"{metrics.get('accuracy', 0):.3f}" if metrics else "N/A")

with col2:
    # FIX: use 'auroc' instead of 'roc_auc'
    st.metric("ROC AUC", f"{metrics.get('auroc', 0):.3f}" if metrics else "N/A")

with col3:
    prob = risk.get("defect_probability", 0)
    st.metric("Defect Probability", f"{prob:.2%}" if risk else "N/A")

with col4:
    st.metric("Risk Level", risk.get("risk_level", "N/A"))

st.divider()

# ============================================
# MODEL PERFORMANCE
# ============================================
st.subheader("📈 Model Performance")

if metrics:
    perf_df = pd.DataFrame({
        "Metric": ["Accuracy", "Precision", "Recall", "F1 Score", "ROC AUC"],
        "Value": [
            metrics.get("accuracy", 0),
            metrics.get("precision", 0),
            metrics.get("recall", 0),
            metrics.get("f1_score", 0),
            metrics.get("auroc", 0)
        ]
    })

    fig = px.bar(perf_df, x="Metric", y="Value", title="Evaluation Metrics")
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No metrics.json found.")

# ============================================
# RISK PREDICTION
# ============================================
st.subheader("⚠️ Risk Prediction")

if risk:
    c1, c2 = st.columns([1, 2])

    with c1:
        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            title={"text": "Defect Probability (%)"},
            gauge={"axis": {"range": [0, 100]}}
        ))
        st.plotly_chart(gauge, use_container_width=True)

    with c2:
        st.json(risk)
else:
    st.info("No risk_prediction.json found.")

# ============================================
# ANOMALY DETECTION
# ============================================
st.subheader("🕵️ Anomaly Detection")

if anomaly_summary:
    a1, a2 = st.columns(2)

    with a1:
        st.metric("Anomalies Detected", anomaly_summary.get("anomalies_detected", 0))
        st.metric("Anomaly Rate", f"{anomaly_summary.get('anomaly_rate', 0):.2%}")

    with a2:
        pie_df = pd.DataFrame({
            "Type": ["Normal", "Anomaly"],
            "Count": [
                anomaly_summary.get("total_samples", 0) - anomaly_summary.get("anomalies_detected", 0),
                anomaly_summary.get("anomalies_detected", 0)
            ]
        })

        fig = px.pie(pie_df, names="Type", values="Count", title="Anomaly Distribution")
        st.plotly_chart(fig, use_container_width=True)

    if not anomaly_df.empty:
        st.markdown("### Sample Anomalies")
        st.dataframe(anomaly_df.head(20), use_container_width=True)
else:
    st.info("No anomaly outputs found.")

# ============================================
# CLUSTERING INSIGHTS
# ============================================
st.subheader("🧩 Clustering Insights")

if cluster_summary:
    dist = cluster_summary.get("cluster_distribution", {})

    if dist:
        cluster_df_plot = pd.DataFrame({
            "Cluster": list(dist.keys()),
            "Count": list(dist.values())
        })

        fig = px.bar(cluster_df_plot, x="Cluster", y="Count", title="Cluster Distribution")
        st.plotly_chart(fig, use_container_width=True)

    if not cluster_df.empty:
        st.markdown("### Clustered Data Sample")
        st.dataframe(cluster_df.head(20), use_container_width=True)
else:
    st.info("No clustering outputs found.")

# ============================================
# TEST RESULTS
# ============================================
st.subheader("✅ Adaptive Test Results")

if test_results:
    summary = test_results.get("summary", {})

    t1, t2, t3 = st.columns(3)
    t1.metric("Passed", summary.get("passed", 0))
    t2.metric("Failed", summary.get("failed", 0))
    t3.metric("Total", summary.get("total", 0))

    st.json(test_results)
else:
    st.info("No results.json found. Run pytest with --json-report.")

st.divider()
st.success("Dashboard loaded successfully 🚀")