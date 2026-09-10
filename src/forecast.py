"""
24-hour recursive forecasting using the best trained model.

Uses only past actual values and previously predicted values for lag features.
Does NOT use future actual consumption values.
"""

from __future__ import annotations

from datetime import timedelta

import pandas as pd
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from src.config import Config, get_config
from src.utils import load_json, logger, save_json


def load_best_model(spark: SparkSession, config: Config | None = None):
    """Load the best model identified during training."""
    config = config or get_config()
    best_info = load_json(config.METRICS_PATH / "best_model.json")
    model_name = best_info["best_model"]
    model_path = str(config.MODEL_PATH / model_name)

    from pyspark.ml.regression import (
        GBTRegressionModel,
        LinearRegressionModel,
        RandomForestRegressionModel,
    )

    loaders = {
        config.MODEL_LINEAR_REGRESSION: LinearRegressionModel,
        config.MODEL_RANDOM_FOREST: RandomForestRegressionModel,
        config.MODEL_GBT: GBTRegressionModel,
    }

    if model_name not in loaders:
        raise ValueError(f"Unknown model: {model_name}")

    logger.info("Loading best model: %s from %s", model_name, model_path)
    return loaders[model_name].load(model_path), model_name


def generate_forecast(
    spark: SparkSession,
    config: Config | None = None,
    forecast_start: str | None = None,
    horizon: int | None = None,
) -> pd.DataFrame:
    """
    Generate recursive 24-hour forecast.

    For each future hour:
    1. Build feature vector from known history + prior predictions
    2. Predict consumption
    3. Append prediction to history for next step's lag features
    """
    config = config or get_config()
    horizon = horizon or config.FORECAST_HORIZON

    model, model_name = load_best_model(spark, config)

    # Load historical hourly data for context
    hourly_df = spark.read.parquet(config.processed_hourly_path)
    hourly_pd = hourly_df.orderBy(config.TIMESTAMP_COL).toPandas()
    hourly_pd[config.TIMESTAMP_COL] = pd.to_datetime(hourly_pd[config.TIMESTAMP_COL])

    if forecast_start:
        start_ts = pd.Timestamp(forecast_start)
    else:
        # Default: start from last available timestamp + 1 hour
        start_ts = hourly_pd[config.TIMESTAMP_COL].max() + timedelta(hours=1)

    # Build history array for lag computation
    history = hourly_pd[config.HOURLY_TARGET].tolist()
    history_ts = hourly_pd[config.TIMESTAMP_COL].tolist()

    predictions = []
    assembler = VectorAssembler(
        inputCols=config.FEATURE_COLUMNS,
        outputCol="features",
    )

    for step in range(horizon):
        ts = start_ts + timedelta(hours=step)

        # Compute features from history (no future actuals)
        hour = ts.hour
        day = ts.day
        dow = ts.dayofweek + 1  # Spark dayofweek: 1=Sun
        dom = ts.day
        month = ts.month
        year = ts.year
        is_weekend = 1 if ts.dayofweek >= 5 else 0

        n = len(history)
        lag_1 = history[-1] if n >= 1 else None
        lag_24 = history[-24] if n >= 24 else None
        lag_168 = history[-168] if n >= 168 else None
        rolling_mean_24 = (
            sum(history[-24:]) / min(24, len(history[-24:]))
            if n >= 1 else None
        )

        if any(v is None for v in [lag_1, lag_24, lag_168, rolling_mean_24]):
            logger.warning("Insufficient history at step %d, stopping forecast", step)
            break

        feature_row = {
            "hour": float(hour),
            "day": float(day),
            "day_of_week": float(dow),
            "day_of_month": float(dom),
            "month": float(month),
            "year": float(year),
            "is_weekend": float(is_weekend),
            "lag_1": float(lag_1),
            "lag_24": float(lag_24),
            "lag_168": float(lag_168),
            "rolling_mean_24": float(rolling_mean_24),
            config.HOURLY_TARGET: float(lag_1),  # placeholder label
        }

        row_df = spark.createDataFrame([feature_row])
        assembled = assembler.transform(row_df)
        pred_row = model.transform(assembled).select("prediction").collect()[0]
        pred_value = float(pred_row["prediction"])
        pred_value = max(0.0, pred_value)  # consumption cannot be negative

        predictions.append({
            "timestamp": ts.isoformat(),
            "predicted_energy_consumption": pred_value,
        })

        # Append prediction to history for next recursive step
        history.append(pred_value)
        history_ts.append(ts)

    forecast_df = pd.DataFrame(predictions)
    logger.info("Generated %d-hour forecast starting %s", len(forecast_df), start_ts)

    # Save predictions
    csv_path = config.PREDICTIONS_PATH / "forecast_24h.csv"
    forecast_df.to_csv(csv_path, index=False)

    save_json(
        {
            "model": model_name,
            "forecast_start": start_ts.isoformat(),
            "horizon": len(forecast_df),
            "method": "recursive (one-step-ahead with predicted lags)",
        },
        config.PREDICTIONS_PATH / "forecast_metadata.json",
    )

    return forecast_df


def run_forecast(spark: SparkSession, config: Config | None = None) -> pd.DataFrame:
    """Execute forecasting pipeline."""
    config = config or get_config()
    logger.info("Starting 24-hour forecast generation")
    return generate_forecast(spark, config)
