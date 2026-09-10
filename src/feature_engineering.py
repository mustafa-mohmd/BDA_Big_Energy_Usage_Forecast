"""
Time-series feature engineering using PySpark window functions.

Lag features (hourly data):
  lag_1   = previous hour
  lag_24  = same hour yesterday
  lag_168 = same hour last week (7 * 24)

rolling_mean_24 = mean of previous 24 hours (excluding current)

Rows without valid lag values are removed to prevent leakage and NaN features.
"""

from __future__ import annotations

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

from src.config import Config, get_config
from src.utils import logger, save_json


def add_time_features(df: DataFrame, config: Config | None = None) -> DataFrame:
    """Extract calendar and cyclical time features."""
    config = config or get_config()
    logger.info("Generating time features")

    ts = config.TIMESTAMP_COL
    df = (
        df.withColumn("hour", F.hour(F.col(ts)))
        .withColumn("day", F.dayofmonth(F.col(ts)))
        .withColumn("day_of_week", F.dayofweek(F.col(ts)))
        .withColumn("day_of_month", F.dayofmonth(F.col(ts)))
        .withColumn("month", F.month(F.col(ts)))
        .withColumn("year", F.year(F.col(ts)))
        .withColumn(
            "is_weekend",
            F.when(F.dayofweek(F.col(ts)).isin(1, 7), 1).otherwise(0),
        )
    )
    return df


def add_lag_features(df: DataFrame, config: Config | None = None) -> DataFrame:
    """
    Add lag and rolling features using chronological window functions.

    Uses only past data (no future leakage).
    """
    config = config or get_config()
    logger.info("Generating lag and rolling features")

    target = config.HOURLY_TARGET
    w = Window.orderBy(config.TIMESTAMP_COL)

    df = (
        df.withColumn("lag_1", F.lag(target, 1).over(w))
        .withColumn("lag_24", F.lag(target, 24).over(w))
        .withColumn("lag_168", F.lag(target, 168).over(w))
    )

    # Rolling mean of previous 24 hours (exclude current row)
    w_rolling = w.rowsBetween(-24, -1)
    df = df.withColumn("rolling_mean_24", F.avg(target).over(w_rolling))

    return df


def remove_incomplete_rows(df: DataFrame, config: Config | None = None) -> DataFrame:
    """Remove rows that cannot have valid lag/rolling features."""
    config = config or get_config()
    initial = df.count()

    df = df.dropna(subset=config.FEATURE_COLUMNS + [config.HOURLY_TARGET])
    final = df.count()
    logger.info(
        "Removed %d rows without valid lag features (%d -> %d)",
        initial - final, initial, final,
    )
    return df


def save_features(df: DataFrame, config: Config | None = None) -> str:
    """Write feature dataset to Parquet."""
    config = config or get_config()
    path = config.features_path
    logger.info("Saving features to %s", path)
    df.write.mode("overwrite").parquet(path)
    return path


def run_feature_engineering(
    spark: SparkSession,
    config: Config | None = None,
    hourly_df: DataFrame | None = None,
) -> DataFrame:
    """Execute full feature engineering pipeline."""
    config = config or get_config()

    if hourly_df is None:
        path = config.processed_hourly_path
        logger.info("Loading hourly data from %s", path)
        hourly_df = spark.read.parquet(path)

    df = add_time_features(hourly_df, config)
    df = add_lag_features(df, config)
    df = remove_incomplete_rows(df, config)
    save_features(df, config)

    stats = {
        "feature_records": df.count(),
        "features": config.FEATURE_COLUMNS,
        "target": config.HOURLY_TARGET,
        "lag_definitions": {
            "lag_1": "previous hour consumption",
            "lag_24": "same hour previous day",
            "lag_168": "same hour previous week",
            "rolling_mean_24": "mean of previous 24 hours",
        },
    }
    save_json(stats, config.METRICS_PATH / "feature_stats.json")

    return df


def chronological_split(
    df: DataFrame, config: Config | None = None
) -> tuple[DataFrame, DataFrame, DataFrame]:
    """
    Split data chronologically (NOT random shuffle).

    Time-series requires preserving temporal order:
    70% train, 15% validation, 15% test.
    """
    config = config or get_config()
    logger.info("Performing chronological train/val/test split")

    w = Window.orderBy(config.TIMESTAMP_COL)
    df = df.withColumn("_row_num", F.row_number().over(w))

    total = df.count()
    train_end = int(total * config.TRAIN_RATIO)
    val_end = int(total * (config.TRAIN_RATIO + config.VAL_RATIO))

    train_df = df.filter(F.col("_row_num") <= train_end).drop("_row_num")
    val_df = df.filter(
        (F.col("_row_num") > train_end) & (F.col("_row_num") <= val_end)
    ).drop("_row_num")
    test_df = df.filter(F.col("_row_num") > val_end).drop("_row_num")

    logger.info(
        "Split sizes - train: %d, val: %d, test: %d",
        train_df.count(), val_df.count(), test_df.count(),
    )
    return train_df, val_df, test_df
