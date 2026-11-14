from __future__ import annotations

import os
from typing import Any

import pandas as pd
import requests
import streamlit as st

FEATURES = [
    "fPeak", "pLow", "pMid", "pHigh", "pKurtosis", "qMean", "qVar",
    "qSkewness", "qKurtosis", "qPeak2Peak", "qCrest", "qRMS", "qMAD", "qCSRange",
]
API_DEFAULT = os.getenv("API_BASE_URL", "http://localhost:8000").rstrip("/")

st.set_page_config(
    page_title="PumpSense | Predictive Maintenance",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .stApp { background: #0b1220; color: #e5edf8; }
    [data-testid="stHeader"] { background: rgba(11,18,32,0.85); }
    [data-testid="stSidebar"] { background: #101b2d; }
    .hero { padding: 1.15rem 1.35rem; border: 1px solid #263751; border-radius: 16px;
            background: linear-gradient(120deg,#14243b,#101a2b); margin-bottom: 1rem; }
    .hero h1 { margin: 0; color: #f4f8ff; font-size: 2rem; }
    .hero p { color: #a9bbd2; margin: .35rem 0 0 0; }
    div[data-testid="stMetric"] { background: #111f33; border: 1px solid #263751;
        padding: 14px 16px; border-radius: 12px; }
    div[data-testid="stMetricLabel"] { color: #a9bbd2; }
    div[data-testid="stMetricValue"] { color: #f4f8ff; }
    .section-note { color: #9fb1c8; font-size: .88rem; }
    .small-muted { color: #9fb1c8; font-size: .8rem; }
</style>
<div class="hero">
  <h1>⚙️ PumpSense</h1>
  <p>Microservices-based predictive maintenance · Synthetic triplex-pump experiment</p>
</div>
""", unsafe_allow_html=True)


def api_get(base: str, path: str, params: dict | None = None) -> Any:
    response = requests.get(f"{base}{path}", params=params, timeout=12)
    response.raise_for_status()
    return response.json()


def api_post(base: str, path: str, payload: dict | None = None) -> Any:
    response = requests.post(f"{base}{path}", json=payload, timeout=120)
    response.raise_for_status()
    return response.json()


def show_error(label: str, exc: Exception) -> None:
    st.error(f"{label}: {exc}")


with st.sidebar:
    st.markdown("### System connection")
    api_base = st.text_input("API Gateway URL", value=API_DEFAULT).rstrip("/")
    st.caption("The API Gateway must be running on port 8000.")
    refresh = st.button("↻ Refresh data", use_container_width=True)
    st.divider()
    st.markdown("**Services**")
    st.markdown("- API Gateway · `:8000`\n- Data Handler · `:8001`\n- ML Operate · `:8002`\n- ML Maintenance · `:8003`\n- MLflow UI · `:5002`")
    st.divider()
    st.caption("Research-aligned prototype. Pump signals are generated synthetically in Python, not MATLAB Simscape.")

if refresh:
    st.rerun()

# Load the data independently so one unavailable endpoint does not blank the entire dashboard.
health = None
stats = None
readings: list[dict] = []
health_error = stats_error = readings_error = None
try:
    health = api_get(api_base, "/health")
except Exception as exc:
    health_error = exc
try:
    stats = api_get(api_base, "/stats")
except Exception as exc:
    stats_error = exc
try:
    readings_payload = api_get(api_base, "/readings", {"limit": 20000})
    readings = readings_payload.get("records", [])
except Exception as exc:
    readings_error = exc

if health_error:
    st.error(f"Cannot reach the API Gateway at {api_base}. Start your Docker services and try again. Details: {health_error}")
    st.code("docker compose ps", language="powershell")
else:
    st.success("API Gateway connected")

# Summary metrics
m1, m2, m3, m4 = st.columns(4)
if stats:
    m1.metric("Sensor measurements", f"{stats.get('total_readings', len(readings)):,}")
    m2.metric("Labelled measurements", f"{stats.get('labelled_readings', 0):,}")
else:
    m1.metric("Sensor measurements", f"{len(readings):,}")
    m2.metric("Labelled measurements", "—")

service_count = 0
if isinstance(health, dict):
    service_count = sum(1 for value in health.values() if isinstance(value, dict) and value.get("status") == "ok")
m3.metric("Healthy backend services", f"{service_count}/3" if health else "—")
m4.metric("Engineered features", "14")

if stats_error:
    st.warning(f"Statistics endpoint unavailable: {stats_error}")
if readings_error:
    st.warning(f"Could not load readings: {readings_error}")

# Tabs
tab_overview, tab_predict, tab_drift, tab_maintenance = st.tabs([
    "Overview", "Live prediction", "Drift monitoring", "Model maintenance"
])

with tab_overview:
    st.subheader("Experiment overview")
    st.markdown('<p class="section-note">Explore the sensor measurements ingested by the Data Handler. The charts are based on the records currently stored in your local experiment database.</p>', unsafe_allow_html=True)
    if readings:
        df = pd.DataFrame(readings)
        if "scenario" in df.columns:
            counts = df["scenario"].fillna("unlabelled").value_counts().rename_axis("Scenario").reset_index(name="Measurements")
            left, right = st.columns([1, 1.5])
            with left:
                st.markdown("**Measurements by scenario**")
                st.bar_chart(counts.set_index("Scenario"))
            with right:
                st.markdown("**Sensor feature trends**")
                chart_features = [f for f in ["qMean", "qRMS", "qVar", "pLow", "pMid", "pHigh"] if f in df.columns]
                if chart_features:
                    plot_df = df[chart_features].reset_index(drop=True)
                    st.line_chart(plot_df, height=260)
        if "severity" in df.columns:
            st.markdown("**Progressive leak severity**")
            leak_df = df[df.get("scenario", pd.Series(index=df.index, dtype=str)).eq("leak")]
            if not leak_df.empty:
                severity_cols = [c for c in ["severity", "qMean", "qRMS"] if c in leak_df.columns]
                st.line_chart(leak_df[severity_cols].reset_index(drop=True), height=220)
        st.markdown("**Latest stored measurements**")
        display_cols = [c for c in ["id", "scenario", "severity", "elapsed_seconds", "failure"] + FEATURES if c in df.columns]
        st.dataframe(df[display_cols].tail(12).iloc[::-1], use_container_width=True, hide_index=True)
    else:
        st.info("No measurements were returned. Generate and ingest the pump experiment, then refresh this page.")
        st.code("python scripts/generate_pump_experiment.py\npython scripts/ingest_pump_csv.py", language="powershell")

with tab_predict:
    st.subheader("Single-measurement failure prediction")
    st.markdown('<p class="section-note">The form sends one 14-feature measurement to the ML Operate service through the API Gateway.</p>', unsafe_allow_html=True)
    default_values = {}
    if readings:
        latest = readings[-1]
        default_values = {f: float(latest.get(f, 0.0) or 0.0) for f in FEATURES}
    with st.form("prediction_form"):
        st.caption("Feature values are prefilled from the latest stored measurement when available. You can edit them.")
        cols = st.columns(3)
        values = {}
        for index, feature in enumerate(FEATURES):
            with cols[index % 3]:
                values[feature] = st.number_input(feature, value=default_values.get(feature, 0.0), format="%.6f", key=f"pred_{feature}")
        submitted = st.form_submit_button("Run failure prediction", type="primary", use_container_width=True)
    if submitted:
        try:
            result = api_post(api_base, "/predict", values)
            prediction = int(result.get("failure_prediction", 0))
            probability = float(result.get("failure_probability", 0.0))
            a, b, c = st.columns(3)
            a.metric("Prediction", "FAILURE" if prediction == 1 else "NORMAL")
            b.metric("Failure probability", f"{probability:.2%}")
            c.metric("Model", result.get("model", "—"))
            if prediction == 1:
                st.error("The model classified this measurement as a potential failure.")
            else:
                st.success("The model classified this measurement as normal.")
            st.json(result)
        except Exception as exc:
            show_error("Prediction request failed", exc)

with tab_drift:
    st.subheader("Input feature drift")
    st.markdown('<p class="section-note">The KS two-sample test compares reference and recent windows. A small p-value indicates evidence that a feature distribution changed.</p>', unsafe_allow_html=True)
    d1, d2, d3 = st.columns([1, 1, 1])
    with d1:
        reference_size = st.number_input("Reference window", min_value=10, max_value=5000, value=100, step=10)
    with d2:
        current_size = st.number_input("Current window", min_value=10, max_value=5000, value=100, step=10)
    with d3:
        st.write("")
        st.write("")
        run_drift = st.button("Run input drift test", type="primary", use_container_width=True)
    if run_drift:
        try:
            result = api_get(api_base, "/drift/input", {"reference_size": int(reference_size), "window_size": int(current_size)})
            st.session_state["last_input_drift"] = result
        except Exception as exc:
            show_error("Input drift request failed", exc)
    drift_result = st.session_state.get("last_input_drift")
    if drift_result:
        affected = drift_result.get("affected_features", [])
        if drift_result.get("detected"):
            st.warning(f"Distribution drift detected in {len(affected)} of {len(drift_result.get('features', {}))} features: {', '.join(affected)}")
        else:
            st.success("No statistically significant input feature drift detected in the selected windows.")
        feature_results = drift_result.get("features", {})
        if feature_results:
            drift_df = pd.DataFrame([{"Feature": name, "KS statistic": item.get("ks_statistic"), "p-value": item.get("p_value"), "Drift": item.get("drift")} for name, item in feature_results.items()])
            st.dataframe(drift_df, use_container_width=True, hide_index=True)
        st.caption(f"Method: {drift_result.get('method', 'KS two-sample test')} · Significance level α = {drift_result.get('alpha', 0.05)}")
    else:
        st.info("Run the input drift test to compare the selected reference and current windows.")

    st.divider()
    st.subheader("Model performance evaluation")
    st.markdown('<p class="section-note">This endpoint evaluates the production model on its registered held-out test records. It is not continuous live concept-drift monitoring.</p>', unsafe_allow_html=True)
    if st.button("Evaluate production model", use_container_width=True):
        try:
            result = api_get(api_base, "/drift/output")
            st.session_state["last_output_eval"] = result
        except Exception as exc:
            show_error("Model evaluation request failed", exc)
    output_result = st.session_state.get("last_output_eval")
    if output_result:
        x1, x2, x3, x4 = st.columns(4)
        x1.metric("Held-out F1", f"{float(output_result.get('current_f1', 0)):.4f}")
        x2.metric("Accuracy", f"{float(output_result.get('accuracy', 0)):.4f}")
        x3.metric("Precision", f"{float(output_result.get('precision', 0)):.4f}" if output_result.get("precision") is not None else "—")
        x4.metric("Recall", f"{float(output_result.get('recall', 0)):.4f}" if output_result.get("recall") is not None else "—")
        st.write(f"**Model version:** `{output_result.get('model_version', 'not reported')}`")
        st.write(f"**Evaluation type:** `{output_result.get('evaluation_type', 'not reported')}`")
        if "confusion_matrix" in output_result:
            cm = output_result["confusion_matrix"]
            st.markdown("**Confusion matrix** · rows = actual [normal, failure], columns = predicted [normal, failure]")
            st.dataframe(pd.DataFrame(cm, index=["Actual normal", "Actual failure"], columns=["Predicted normal", "Predicted failure"]), use_container_width=True)
        if output_result.get("note"):
            st.info(output_result["note"])
        st.json(output_result)
    else:
        st.info("Run the evaluation to display the latest saved held-out test-set results.")

with tab_maintenance:
    st.subheader("Model retraining and versioning")
    st.markdown('<p class="section-note">Retraining calls the ML Maintenance service. Review the returned metrics and then inspect the logged run in MLflow.</p>', unsafe_allow_html=True)
    st.warning("Retraining can change the active production model. Run it only when the labelled dataset contains both normal and failure examples.")
    if st.button("Train and register a new model", type="primary"):
        try:
            with st.spinner("Training Random Forest and logging the MLflow run…"):
                result = api_post(api_base, "/retrain")
            st.session_state["last_retrain"] = result
            st.success("Retraining request completed.")
        except Exception as exc:
            show_error("Retraining failed", exc)
    retrain_result = st.session_state.get("last_retrain")
    if retrain_result:
        r1, r2, r3 = st.columns(3)
        r1.metric("Test F1", f"{float(retrain_result.get('test_f1', 0)):.4f}")
        r2.metric("Test accuracy", f"{float(retrain_result.get('test_accuracy', 0)):.4f}")
        r3.metric("Training samples", str(retrain_result.get("training_samples", "—")))
        st.write(f"**Model version:** `{retrain_result.get('version', '—')}`")
        st.write(f"**MLflow run ID:** `{retrain_result.get('mlflow_run_id', '—')}`")
        st.json(retrain_result)
    st.markdown("**MLflow experiment tracker**")
    st.link_button("Open MLflow UI", "http://localhost:5002", use_container_width=True)

st.divider()
st.markdown('<p class="small-muted">PumpSense is a demonstration dashboard for a research-aligned prototype. Synthetic Python pump signals are used; metrics shown are experiment-specific and should not be interpreted as real-world industrial validation.</p>', unsafe_allow_html=True)
