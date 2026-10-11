from __future__ import annotations

import io
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from ml_pipeline import (
    REQUIRED_COLUMNS, clean_dataset, build_supervised_dataset,
    train_and_evaluate, predict_latest, cost_summary, data_quality_summary,
)

st.set_page_config(
    page_title="Cloud Resource Forecasting",
    page_icon="☁️",
    layout="wide",
    initial_sidebar_state="expanded",
)

@st.cache_resource(show_spinner=False)
def get_model_results(dataframe: pd.DataFrame):
    # Cache model training for the current uploaded dataset across Streamlit reruns/tabs.
    return train_and_evaluate(dataframe)


st.markdown("""
<style>
.stApp { background: #0b1220; color: #e7eefb; }
[data-testid="stSidebar"] { background: #101a2b; }
[data-testid="stMetric"] {
    background: linear-gradient(135deg, #14233a, #101a2b);
    border: 1px solid #263852; padding: 16px; border-radius: 14px;
}
div[data-testid="stMetricValue"] { color: #61d7ff; }
h1, h2, h3 { color: #f3f7ff; }
.small-note { color: #9fb0c8; font-size: 0.9rem; }
</style>
""", unsafe_allow_html=True)

st.title("☁️ Cloud Resource Forecasting")
st.caption("CPU forecasting • Resource analytics • Cost optimization")
st.markdown(
    "Analyze server telemetry, forecast CPU utilization five minutes ahead, "
    "and compare an illustrative CPU-based compute-tier policy with a fixed tier."
)

with st.sidebar:
    st.header("Data source")
    uploaded = st.file_uploader("Upload cloud CPU dataset (CSV)", type=["csv"])
    st.markdown("---")
    st.subheader("Project scope")
    st.write("**Forecast horizon:** 5 minutes")
    st.write("**Model candidates:** Linear Regression, Ridge")
    st.write("**Evaluation:** chronological train / validation / test")
    st.markdown("---")
    st.caption("This dashboard analyzes uploaded data. It does not connect to or monitor live cloud servers.")

if uploaded is None:
    st.info("Upload your cloud CPU dataset to start. The CSV should contain the following core columns:")
    st.code(", ".join(REQUIRED_COLUMNS))
    st.markdown("""
    **Expected data shape**
    - One row per server measurement and timestamp (duplicates are aggregated).
    - `timestamp` must be parseable as a date/time.
    - `server_id` identifies the server.
    - `cpu_utilization` should be a percentage from 0 to 100.
    - Optional resource columns include memory, storage, network traffic, active users,
      request rate, temperature, and workload level.
    """)
    st.warning("No dataset is bundled with this project. Upload your real CSV; no real-world metrics are fabricated.")
    st.stop()

try:
    raw = pd.read_csv(uploaded)
except Exception as exc:
    st.error(f"Could not read this CSV: {exc}")
    st.stop()

try:
    df, cleaning_notes = clean_dataset(raw)
except ValueError as exc:
    st.error(str(exc))
    st.stop()

quality = data_quality_summary(raw, df)
st.sidebar.success(f"Loaded {len(df):,} cleaned rows")

overview_tab, explorer_tab, forecast_tab, model_tab, cost_tab = st.tabs([
    "Overview", "Data Explorer", "CPU Forecast", "Model Evaluation", "Cost Optimization"
])

with overview_tab:
    st.subheader("Infrastructure overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Cleaned records", f"{len(df):,}")
    c2.metric("Servers", f"{df['server_id'].nunique():,}")
    c3.metric("Mean CPU", f"{df['cpu_utilization'].mean():.1f}%")
    memory = df["memory_utilization"].mean() if "memory_utilization" in df else np.nan
    c4.metric("Mean memory", f"{memory:.1f}%" if pd.notna(memory) else "Not available")

    c1, c2, c3, c4 = st.columns(4)
    storage = df["storage_utilization"].mean() if "storage_utilization" in df else np.nan
    network = df["network_traffic"].mean() if "network_traffic" in df else np.nan
    c1.metric("Mean storage", f"{storage:.1f}%" if pd.notna(storage) else "Not available")
    c2.metric("Mean network traffic", f"{network:.1f}" if pd.notna(network) else "Not available")
    c3.metric("Time range start", df["timestamp"].min().strftime("%Y-%m-%d %H:%M"))
    c4.metric("Time range end", df["timestamp"].max().strftime("%Y-%m-%d %H:%M"))

    left, right = st.columns(2)
    with left:
        server_trend = df.groupby("timestamp", as_index=False)["cpu_utilization"].mean()
        fig = px.line(server_trend, x="timestamp", y="cpu_utilization",
                      title="Average CPU utilization over time",
                      labels={"timestamp": "Timestamp", "cpu_utilization": "CPU (%)"})
        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)
    with right:
        server_cpu = df.groupby("server_id", as_index=False)["cpu_utilization"].mean()
        fig = px.bar(server_cpu, x="server_id", y="cpu_utilization",
                     title="Mean CPU utilization by server",
                     labels={"server_id": "Server", "cpu_utilization": "CPU (%)"})
        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

    if "workload_level" in df:
        counts = df["workload_level"].value_counts(dropna=False).rename_axis("workload_level").reset_index(name="count")
        fig = px.bar(counts, x="workload_level", y="count", title="Workload-level distribution")
        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Data quality")
    st.dataframe(pd.DataFrame({"Check": list(quality.keys()), "Result": list(quality.values())}),
                 use_container_width=True, hide_index=True)
    with st.expander("Cleaning actions"):
        for note in cleaning_notes:
            st.write(f"• {note}")

with explorer_tab:
    st.subheader("Dataset explorer")
    st.write("Preview of the cleaned dataset")
    st.dataframe(df.head(100), use_container_width=True)
    st.caption(f"Showing up to 100 rows of {len(df):,} cleaned records.")
    left, right = st.columns(2)
    with left:
        st.markdown("**Descriptive statistics**")
        st.dataframe(df.select_dtypes(include=np.number).describe().T, use_container_width=True)
    with right:
        missing = df.isna().sum().sort_values(ascending=False).rename_axis("column").reset_index(name="missing values")
        fig = px.bar(missing, x="column", y="missing values", title="Missing values by column")
        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    if numeric_cols:
        chosen = st.selectbox("Select a numeric field for distribution", numeric_cols)
        fig = px.histogram(df, x=chosen, nbins=35, title=f"Distribution of {chosen}")
        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)
    if len(numeric_cols) >= 2:
        corr = df[numeric_cols].corr(numeric_only=True)
        fig = px.imshow(corr, text_auto=".2f", aspect="auto", color_continuous_scale="RdBu_r",
                        zmin=-1, zmax=1, title="Numeric feature correlation")
        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

with forecast_tab:
    st.subheader("Five-minute CPU forecast")
    try:
        with st.spinner("Training and evaluating models on the uploaded dataset..."):
            result = get_model_results(df)
        servers = sorted(df["server_id"].astype(str).unique().tolist())
        selected_server = st.selectbox("Select server", servers)
        forecast = predict_latest(df, result, selected_server)
        a, b, c = st.columns(3)
        a.metric("Latest observed CPU", f"{forecast['current_cpu']:.2f}%")
        b.metric("Predicted CPU (+5 min)", f"{forecast['prediction']:.2f}%")
        c.metric("Forecast time", forecast["forecast_time"].strftime("%Y-%m-%d %H:%M"))
        st.caption(f"Selected model: {result['best_model_name']}. This is an offline forecast from uploaded telemetry, not a live server reading.")
        recent = df[df["server_id"].astype(str) == str(selected_server)].tail(100)
        fig = px.line(recent, x="timestamp", y="cpu_utilization",
                      title=f"Recent CPU observations — {selected_server}",
                      labels={"timestamp": "Timestamp", "cpu_utilization": "CPU (%)"})
        fig.add_hline(y=forecast["prediction"], line_dash="dash",
                      annotation_text=f"5-minute forecast: {forecast['prediction']:.1f}%")
        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)
    except ValueError as exc:
        st.error(str(exc))
    except Exception as exc:
        st.error(f"Forecast could not be calculated. Check that the dataset contains enough valid time points. Details: {exc}")

with model_tab:
    st.subheader("Model evaluation")
    try:
        result = get_model_results(df)
        st.write("Models are selected using validation MAE. The test set is reserved for final evaluation.")
        st.dataframe(result["comparison"], use_container_width=True, hide_index=True)
        st.markdown("**Final test metrics**")
        m1, m2, m3 = st.columns(3)
        m1.metric("MAE", f"{result['test_metrics']['MAE']:.3f}")
        m2.metric("RMSE", f"{result['test_metrics']['RMSE']:.3f}")
        m3.metric("R²", f"{result['test_metrics']['R2']:.3f}")
        pred_df = result["test_predictions"].copy()
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=pred_df["actual"].head(150), mode="lines", name="Actual CPU"))
        fig.add_trace(go.Scatter(y=pred_df["predicted"].head(150), mode="lines", name="Predicted CPU"))
        fig.update_layout(title="Actual vs predicted CPU — test set (first 150 observations)",
                          xaxis_title="Test observation", yaxis_title="CPU (%)",
                          template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)
        scatter = px.scatter(pred_df, x="actual", y="predicted", title="Actual vs predicted CPU",
                             labels={"actual": "Actual CPU (%)", "predicted": "Predicted CPU (%)"})
        scatter.add_shape(type="line", x0=0, y0=0, x1=100, y1=100, line=dict(dash="dash"))
        scatter.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                              plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(scatter, use_container_width=True)
        residual = pred_df.assign(residual=pred_df["actual"] - pred_df["predicted"])
        fig = px.scatter(residual, x="predicted", y="residual", title="Residual analysis",
                         labels={"predicted": "Predicted CPU (%)", "residual": "Residual (actual - predicted)"})
        fig.add_hline(y=0, line_dash="dash")
        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)
    except ValueError as exc:
        st.error(str(exc))
    except Exception as exc:
        st.error(f"Model evaluation could not be completed: {exc}")

with cost_tab:
    st.subheader("Illustrative compute cost optimization")
    st.warning("These are illustrative USD prices and a simplified CPU-threshold policy, not live prices or actual cloud-provider billing.")
    rates = {}
    cols = st.columns(3)
    defaults = [0.020, 0.040, 0.080]
    for col, tier, default in zip(cols, ["Small", "Medium", "Large"], defaults):
        with col:
            rates[tier] = st.number_input(f"{tier} tier (USD/hour)", min_value=0.0, value=default, step=0.005, format="%.3f")
    c1, c2 = st.columns(2)
    with c1:
        small_max = st.slider("Small tier maximum CPU (%)", 10, 60, 35)
    with c2:
        medium_max = st.slider("Medium tier maximum CPU (%)", max(20, small_max + 1), 90, max(70, small_max + 1))
    try:
        result = get_model_results(df)
        pred = np.clip(result["test_predictions"]["predicted"].to_numpy(), 0, 100)
        summary = cost_summary(pred, rates, small_max, medium_max)
        x1, x2, x3 = st.columns(3)
        x1.metric("Fixed Large-tier test-period cost", f"${summary['fixed_cost']:.4f}")
        x2.metric("CPU-based tier estimate", f"${summary['scaled_cost']:.4f}")
        x3.metric("Estimated savings", f"${summary['savings']:.4f} ({summary['savings_pct']:.1f}%)")
        compare = pd.DataFrame({
            "Policy": ["Fixed Large tier", "CPU-based tier policy"],
            "Estimated cost (USD)": [summary["fixed_cost"], summary["scaled_cost"]]
        })
        fig = px.bar(compare, x="Policy", y="Estimated cost (USD)", title="Fixed vs CPU-based estimated compute cost",
                     text_auto=".4f")
        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)
        st.write("**Tier selection distribution**")
        st.dataframe(pd.DataFrame({"Tier": list(summary["tier_counts"].keys()),
                                   "Intervals": list(summary["tier_counts"].values())}),
                     use_container_width=True, hide_index=True)
        st.caption(f"Monthly projection assumes {summary['hours_per_month']} hours/month and scales the test-period average. It is an illustrative projection, not a billing forecast.")
        st.metric("Projected monthly savings", f"${summary['monthly_savings']:.2f}")
    except ValueError as exc:
        st.error(str(exc))
    except Exception as exc:
        st.error(f"Cost estimate could not be calculated: {exc}")

st.markdown("---")
st.caption("Cloud Resource Forecasting & Cost Optimization • Academic project demonstration • Uploaded data only")
