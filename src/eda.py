"""
Exploratory Data Analysis - generates figures saved to results/figures/.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pyspark.sql import SparkSession

from src.config import Config, get_config
from src.utils import logger, save_json


def run_eda(spark: SparkSession, config: Config | None = None) -> dict:
    """Generate EDA figures and summary statistics."""
    config = config or get_config()
    logger.info("Running exploratory data analysis")

    hourly_path = config.processed_hourly_path
    if not Path(hourly_path.replace("file:///", "")).exists() and not config.USE_HDFS:
        # Try direct path
        local = config.PROCESSED_DATA_PATH / "hourly"
        if not local.exists():
            raise FileNotFoundError("Processed hourly data not found. Run preprocessing first.")
        hourly_path = str(local)

    df = spark.read.parquet(hourly_path).toPandas()
    ts_col = config.TIMESTAMP_COL
    target = config.HOURLY_TARGET
    df[ts_col] = pd.to_datetime(df[ts_col])

    figures_path = config.FIGURES_PATH
    figures_path.mkdir(parents=True, exist_ok=True)

    # 1. Overall consumption trend (weekly resample for readability)
    fig, ax = plt.subplots(figsize=(14, 5))
    weekly = df.set_index(ts_col)[target].resample("W").mean()
    weekly.plot(ax=ax, color="#2563eb")
    ax.set_title("Weekly Average Energy Consumption Over Time")
    ax.set_xlabel("Date")
    ax.set_ylabel("Energy Consumption (kWh)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(figures_path / "01_consumption_trend.png", dpi=150)
    plt.close(fig)

    # 2. Hourly profile
    df["hour"] = df[ts_col].dt.hour
    hourly_profile = df.groupby("hour")[target].mean()
    fig, ax = plt.subplots(figsize=(10, 5))
    hourly_profile.plot(kind="bar", ax=ax, color="#059669")
    ax.set_title("Average Consumption by Hour of Day")
    ax.set_xlabel("Hour")
    ax.set_ylabel("Avg Consumption (kWh)")
    fig.tight_layout()
    fig.savefig(figures_path / "02_hourly_profile.png", dpi=150)
    plt.close(fig)

    # 3. Day of week profile
    df["dow"] = df[ts_col].dt.day_name()
    dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    dow_profile = df.groupby("dow")[target].mean().reindex(dow_order)
    fig, ax = plt.subplots(figsize=(10, 5))
    dow_profile.plot(kind="bar", ax=ax, color="#7c3aed")
    ax.set_title("Average Consumption by Day of Week")
    ax.set_xlabel("Day")
    ax.set_ylabel("Avg Consumption (kWh)")
    fig.tight_layout()
    fig.savefig(figures_path / "03_dow_profile.png", dpi=150)
    plt.close(fig)

    # 4. Monthly profile
    df["month"] = df[ts_col].dt.month
    monthly = df.groupby("month")[target].mean()
    fig, ax = plt.subplots(figsize=(10, 5))
    monthly.plot(kind="bar", ax=ax, color="#dc2626")
    ax.set_title("Average Consumption by Month")
    ax.set_xlabel("Month")
    ax.set_ylabel("Avg Consumption (kWh)")
    fig.tight_layout()
    fig.savefig(figures_path / "04_monthly_profile.png", dpi=150)
    plt.close(fig)

    # 5. Weekday vs weekend
    df["is_weekend"] = df[ts_col].dt.dayofweek >= 5
    wk = df.groupby("is_weekend")[target].mean()
    fig, ax = plt.subplots(figsize=(6, 5))
    wk.index = ["Weekday", "Weekend"]
    wk.plot(kind="bar", ax=ax, color=["#0891b2", "#ea580c"])
    ax.set_title("Weekday vs Weekend Consumption")
    ax.set_ylabel("Avg Consumption (kWh)")
    fig.tight_layout()
    fig.savefig(figures_path / "05_weekday_weekend.png", dpi=150)
    plt.close(fig)

    # 6. Distribution
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(df[target], bins=50, kde=True, ax=ax, color="#4f46e5")
    ax.set_title("Distribution of Hourly Energy Consumption")
    ax.set_xlabel("Consumption (kWh)")
    fig.tight_layout()
    fig.savefig(figures_path / "06_distribution.png", dpi=150)
    plt.close(fig)

    # 7. Correlation heatmap
    numeric_cols = [target, "avg_reactive_power", "avg_voltage", "avg_intensity",
                    "sub_metering_1_kwh", "sub_metering_2_kwh", "sub_metering_3_kwh"]
    available = [c for c in numeric_cols if c in df.columns]
    if len(available) > 1:
        corr = df[available].corr()
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(corr, annot=True, cmap="coolwarm", center=0, ax=ax, fmt=".2f")
        ax.set_title("Feature Correlation Matrix")
        fig.tight_layout()
        fig.savefig(figures_path / "07_correlation.png", dpi=150)
        plt.close(fig)

    # Compute insights
    peak_hour = int(hourly_profile.idxmax())
    min_hour = int(hourly_profile.idxmin())
    avg_consumption = float(df[target].mean())
    weekday_avg = float(df[~df["is_weekend"]][target].mean())
    weekend_avg = float(df[df["is_weekend"]][target].mean())

    insights = {
        "peak_consumption_hour": peak_hour,
        "minimum_consumption_hour": min_hour,
        "average_consumption_kwh": round(avg_consumption, 4),
        "weekday_avg_kwh": round(weekday_avg, 4),
        "weekend_avg_kwh": round(weekend_avg, 4),
        "weekday_weekend_diff_kwh": round(weekday_avg - weekend_avg, 4),
        "date_range": {
            "start": str(df[ts_col].min()),
            "end": str(df[ts_col].max()),
        },
        "total_hourly_records": len(df),
    }

    save_json(insights, config.METRICS_PATH / "eda_insights.json")
    logger.info("EDA complete. Figures saved to %s", figures_path)

    return insights
