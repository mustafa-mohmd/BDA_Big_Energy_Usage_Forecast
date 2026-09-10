"""Plotly chart components for the dashboard."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def consumption_over_time(df: pd.DataFrame, ts_col: str, target: str) -> go.Figure:
    """Line chart of energy consumption over time."""
    sample = df.sort_values(ts_col)
    if len(sample) > 5000:
        sample = sample.iloc[::max(1, len(sample) // 5000)]

    fig = px.line(
        sample, x=ts_col, y=target,
        title="Energy Consumption Over Time",
        labels={target: "Consumption (kWh)", ts_col: "Date"},
    )
    fig.update_layout(height=400, template="plotly_white")
    return fig


def hourly_profile(df: pd.DataFrame, target: str) -> go.Figure:
    """Bar chart of average consumption by hour."""
    hourly = df.groupby("hour")[target].mean().reset_index()
    fig = px.bar(
        hourly, x="hour", y=target,
        title="Average Consumption by Hour of Day",
        labels={target: "Avg Consumption (kWh)", "hour": "Hour"},
        color_discrete_sequence=["#059669"],
    )
    fig.update_layout(height=350, template="plotly_white")
    return fig


def dow_profile(df: pd.DataFrame, target: str) -> go.Figure:
    """Bar chart by day of week."""
    dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    if "dow" not in df.columns:
        df = df.copy()
        df["dow"] = pd.to_datetime(df.iloc[:, 0]).dt.day_name()
    profile = df.groupby("dow")[target].mean().reindex(dow_order).reset_index()
    fig = px.bar(
        profile, x="dow", y=target,
        title="Average Consumption by Day of Week",
        labels={target: "Avg Consumption (kWh)", "dow": "Day"},
        color_discrete_sequence=["#7c3aed"],
    )
    fig.update_layout(height=350, template="plotly_white")
    return fig


def monthly_profile(df: pd.DataFrame, ts_col: str, target: str) -> go.Figure:
    """Bar chart by month."""
    df = df.copy()
    df["month"] = pd.to_datetime(df[ts_col]).dt.month
    monthly = df.groupby("month")[target].mean().reset_index()
    fig = px.bar(
        monthly, x="month", y=target,
        title="Average Consumption by Month",
        labels={target: "Avg Consumption (kWh)", "month": "Month"},
        color_discrete_sequence=["#dc2626"],
    )
    fig.update_layout(height=350, template="plotly_white")
    return fig


def weekday_weekend_chart(df: pd.DataFrame, ts_col: str, target: str) -> go.Figure:
    """Compare weekday vs weekend consumption."""
    df = df.copy()
    df["period"] = pd.to_datetime(df[ts_col]).dt.dayofweek.apply(
        lambda x: "Weekend" if x >= 5 else "Weekday"
    )
    grouped = df.groupby("period")[target].mean().reset_index()
    fig = px.bar(
        grouped, x="period", y=target,
        title="Weekday vs Weekend Consumption",
        labels={target: "Avg Consumption (kWh)", "period": ""},
        color="period",
        color_discrete_map={"Weekday": "#0891b2", "Weekend": "#ea580c"},
    )
    fig.update_layout(height=350, template="plotly_white", showlegend=False)
    return fig


def model_comparison_chart(comparison_df: pd.DataFrame) -> go.Figure:
    """Grouped bar chart comparing model metrics."""
    metrics = ["MAE", "RMSE", "MAPE"]
    fig = go.Figure()
    colors = ["#2563eb", "#059669", "#dc2626"]
    for i, metric in enumerate(metrics):
        if metric in comparison_df.columns:
            fig.add_trace(go.Bar(
                name=metric,
                x=comparison_df["Model"],
                y=comparison_df[metric],
                marker_color=colors[i],
            ))
    fig.update_layout(
        title="Model Performance Comparison (Validation Set)",
        barmode="group",
        height=400,
        template="plotly_white",
        yaxis_title="Error Value",
    )
    return fig


def forecast_chart(forecast_df: pd.DataFrame) -> go.Figure:
    """Line chart of 24-hour forecast."""
    forecast_df = forecast_df.copy()
    forecast_df["timestamp"] = pd.to_datetime(forecast_df["timestamp"])
    fig = px.line(
        forecast_df, x="timestamp", y="predicted_energy_consumption",
        title="24-Hour Energy Consumption Forecast",
        labels={
            "predicted_energy_consumption": "Predicted Consumption (kWh)",
            "timestamp": "Time",
        },
        markers=True,
    )
    fig.update_layout(height=400, template="plotly_white")
    fig.update_traces(line_color="#2563eb")
    return fig
