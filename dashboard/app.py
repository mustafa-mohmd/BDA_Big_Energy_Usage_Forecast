"""
Streamlit Dashboard - Energy Consumption Forecasting Using Big Data Technologies

Launch: streamlit run dashboard/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dashboard.components.charts import (
    consumption_over_time,
    dow_profile,
    forecast_chart,
    hourly_profile,
    model_comparison_chart,
    monthly_profile,
    weekday_weekend_chart,
)
from src.config import get_config
from src.utils import load_json

st.set_page_config(
    page_title="Energy Consumption Forecasting",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

config = get_config()


@st.cache_data
def load_hourly_data() -> pd.DataFrame | None:
    """Load processed hourly data."""
    path = config.PROCESSED_DATA_PATH / "hourly"
    if not path.exists():
        return None
    try:
        df = pd.read_parquet(path)
        df[config.TIMESTAMP_COL] = pd.to_datetime(df[config.TIMESTAMP_COL])
        df["hour"] = df[config.TIMESTAMP_COL].dt.hour
        return df
    except Exception as e:
        st.error(f"Error loading hourly data: {e}")
        return None


@st.cache_data
def load_model_comparison() -> pd.DataFrame | None:
    """Load model comparison table."""
    path = config.METRICS_PATH / "model_comparison.csv"
    if path.exists():
        return pd.read_csv(path)
    results_path = config.METRICS_PATH / "model_results.json"
    if results_path.exists():
        results = load_json(results_path)
        rows = []
        for name, metrics in results["validation_metrics"].items():
            rows.append({
                "Model": name.replace("_", " ").title(),
                "MAE": round(metrics["MAE"], 4),
                "RMSE": round(metrics["RMSE"], 4),
                "MAPE": round(metrics["MAPE"], 4),
                "R²": round(metrics["R2"], 4),
            })
        return pd.DataFrame(rows)
    return None


@st.cache_data
def load_forecast() -> pd.DataFrame | None:
    """Load forecast predictions."""
    path = config.PREDICTIONS_PATH / "forecast_24h.csv"
    if path.exists():
        return pd.read_csv(path)
    return None


@st.cache_data
def load_insights() -> dict | None:
    """Load EDA insights."""
    path = config.METRICS_PATH / "eda_insights.json"
    if path.exists():
        return load_json(path)
    return None


@st.cache_data
def load_preprocessing_stats() -> dict | None:
    """Load preprocessing statistics."""
    path = config.METRICS_PATH / "preprocessing_stats.json"
    if path.exists():
        return load_json(path)
    return None


def page_home():
    st.title("⚡ Energy Consumption Forecasting")
    st.subheader("Using Big Data Technologies")

    st.markdown("""
    This project demonstrates an end-to-end **Big Data pipeline** for forecasting
    household electricity consumption using **Hadoop HDFS**, **Apache Spark/PySpark**,
    and **Machine Learning**.

    ### Problem Statement
    Accurate energy consumption forecasting helps utilities optimize generation,
    reduce costs, and support smart grid initiatives.

    ### Pipeline Overview
    """)

    st.code("""
RAW DATA → INGESTION → HDFS → PySpark → CLEANING → FEATURE ENGINEERING
    → EDA → MODEL TRAINING → EVALUATION → BEST MODEL → 24-HOUR FORECAST → DASHBOARD
    """, language="text")

    st.markdown("""
    ### Technology Stack
    | Layer | Technology |
    |-------|-----------|
    | Storage | Hadoop HDFS |
    | Processing | Apache Spark / PySpark |
    | ML | Spark MLlib (Linear Regression, Random Forest, GBT) |
    | Visualization | Plotly, Matplotlib |
    | Dashboard | Streamlit |
    | Language | Python 3.10+ |

    ### Dataset
    UCI Individual Household Electric Power Consumption (~2M minute-level records)
    """)


def page_dataset():
    st.header("📊 Dataset Overview")

    df = load_hourly_data()
    stats = load_preprocessing_stats()

    if df is None:
        st.warning(
            "Processed data not found. Run the pipeline first:\n\n"
            "`python scripts/run_pipeline.py`"
        )
        raw_exists = config.RAW_FILE.exists()
        st.info(f"Raw dataset present: {'Yes' if raw_exists else 'No'}")
        return

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Hourly Records", f"{len(df):,}")
    with col2:
        st.metric("Start Date", str(df[config.TIMESTAMP_COL].min())[:10])
    with col3:
        st.metric("End Date", str(df[config.TIMESTAMP_COL].max())[:10])
    with col4:
        st.metric("Columns", len(df.columns))

    st.subheader("Columns")
    st.write(list(df.columns))

    st.subheader("Descriptive Statistics")
    st.dataframe(df.describe(), use_container_width=True)

    if stats:
        st.subheader("Preprocessing Info")
        st.json(stats)


def page_energy_analysis():
    st.header("📈 Energy Analysis")

    df = load_hourly_data()
    if df is None:
        st.warning("No data available. Run the pipeline first.")
        return

    target = config.HOURLY_TARGET
    ts_col = config.TIMESTAMP_COL

    st.plotly_chart(consumption_over_time(df, ts_col, target), use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(hourly_profile(df, target), use_container_width=True)
    with col2:
        st.plotly_chart(dow_profile(df, target), use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        st.plotly_chart(monthly_profile(df, ts_col, target), use_container_width=True)
    with col4:
        st.plotly_chart(weekday_weekend_chart(df, ts_col, target), use_container_width=True)


def page_model_performance():
    st.header("🎯 Model Performance")

    comparison = load_model_comparison()
    if comparison is None:
        st.warning("Model results not found. Run training first.")
        return

    st.subheader("Model Comparison Table")
    st.dataframe(comparison, use_container_width=True, hide_index=True)

    st.plotly_chart(model_comparison_chart(comparison), use_container_width=True)

    results_path = config.METRICS_PATH / "model_results.json"
    if results_path.exists():
        results = load_json(results_path)
        st.subheader("Best Model")
        st.success(f"**{results['best_model'].replace('_', ' ').title()}** selected by validation RMSE")

        st.subheader("Test Set Metrics (Unseen Data)")
        test = results.get("test_metrics", {})
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("MAE", f"{test.get('MAE', 'N/A'):.4f}")
        c2.metric("RMSE", f"{test.get('RMSE', 'N/A'):.4f}")
        c3.metric("MAPE", f"{test.get('MAPE', 'N/A'):.2f}%")
        c4.metric("R²", f"{test.get('R2', 'N/A'):.4f}")


def page_forecast():
    st.header("🔮 24-Hour Forecast")

    forecast_df = load_forecast()
    if forecast_df is None:
        st.warning("Forecast not generated. Run the full pipeline first.")
        return

    meta_path = config.PREDICTIONS_PATH / "forecast_metadata.json"
    if meta_path.exists():
        meta = load_json(meta_path)
        st.info(
            f"Model: **{meta.get('model', 'N/A')}** | "
            f"Start: **{meta.get('forecast_start', 'N/A')}** | "
            f"Method: {meta.get('method', 'N/A')}"
        )

    st.plotly_chart(forecast_chart(forecast_df), use_container_width=True)

    st.subheader("Prediction Table")
    st.dataframe(forecast_df, use_container_width=True, hide_index=True)

    avg_pred = forecast_df["predicted_energy_consumption"].mean()
    max_pred = forecast_df["predicted_energy_consumption"].max()
    min_pred = forecast_df["predicted_energy_consumption"].min()
    c1, c2, c3 = st.columns(3)
    c1.metric("Average Predicted", f"{avg_pred:.3f} kWh")
    c2.metric("Peak Predicted", f"{max_pred:.3f} kWh")
    c3.metric("Minimum Predicted", f"{min_pred:.3f} kWh")


def page_insights():
    st.header("💡 Insights")

    insights = load_insights()
    if insights is None:
        st.warning("Insights not available. Run EDA step first.")
        return

    c1, c2 = st.columns(2)
    with c1:
        st.metric("Peak Consumption Hour", f"{insights.get('peak_consumption_hour', 'N/A')}:00")
        st.metric("Minimum Consumption Hour", f"{insights.get('minimum_consumption_hour', 'N/A')}:00")
        st.metric("Average Consumption", f"{insights.get('average_consumption_kwh', 'N/A')} kWh")
    with c2:
        st.metric("Weekday Average", f"{insights.get('weekday_avg_kwh', 'N/A')} kWh")
        st.metric("Weekend Average", f"{insights.get('weekend_avg_kwh', 'N/A')} kWh")
        diff = insights.get("weekday_weekend_diff_kwh", 0)
        st.metric("Weekday − Weekend", f"{diff:+.4f} kWh")

    st.subheader("Key Findings")
    peak = insights.get("peak_consumption_hour", 0)
    minimum = insights.get("minimum_consumption_hour", 0)
    st.markdown(f"""
    - **Peak demand** occurs around **{peak}:00**, likely corresponding to evening appliance usage.
    - **Lowest consumption** is around **{minimum}:00**, during early morning hours.
    - **Weekday consumption** is {"higher" if diff > 0 else "lower"} than weekends by **{abs(diff):.4f} kWh** on average.
    - The dataset spans from **{insights.get('date_range', {}).get('start', 'N/A')[:10]}**
      to **{insights.get('date_range', {}).get('end', 'N/A')[:10]}**.
    """)

    figures_path = config.FIGURES_PATH
    if figures_path.exists():
        pngs = sorted(figures_path.glob("*.png"))
        if pngs:
            st.subheader("EDA Figures")
            cols = st.columns(2)
            for i, png in enumerate(pngs[:6]):
                cols[i % 2].image(str(png), caption=png.stem.replace("_", " ").title())


# Sidebar navigation
PAGES = {
    "Home": page_home,
    "Dataset Overview": page_dataset,
    "Energy Analysis": page_energy_analysis,
    "Model Performance": page_model_performance,
    "Forecast": page_forecast,
    "Insights": page_insights,
}

with st.sidebar:
    st.title("Navigation")
    selection = st.radio("Go to", list(PAGES.keys()))
    st.markdown("---")
    st.markdown("**Energy Forecasting**")
    st.markdown("Big Data Final Year Project")
    st.markdown("HDFS + Spark + ML")

PAGES[selection]()
