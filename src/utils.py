"""
Shared utilities: Spark session, logging, metrics, and helpers.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from pyspark.sql import SparkSession

from src.config import Config, get_config


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """Configure and return the project logger."""
    logger = logging.getLogger("energy_forecasting")
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter("%(levelname)s - %(message)s")
        )
        logger.addHandler(handler)
        logger.setLevel(level)
    return logger


logger = setup_logging()


def create_spark_session(config: Optional[Config] = None) -> SparkSession:
    """
    Create a SparkSession with project defaults.

    Falls back to local mode when HDFS is unavailable.
    """
    config = config or get_config()
    logger.info("Connecting to Spark (master=%s)", config.SPARK_MASTER)

    builder = (
        SparkSession.builder
        .appName(config.SPARK_APP_NAME)
        .master(config.SPARK_MASTER)
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.driver.memory", "4g")
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.sql.adaptive.enabled", "true")
        # The dataset uses legacy day/month timestamp strings such as
        # "24/6/2008 04:58:00". Spark 3+ defaults to the new parser and
        # rejects these values unless compatibility mode is enabled.
        .config("spark.sql.legacy.timeParserPolicy", "LEGACY")
    )

    if config.USE_HDFS:
        builder = builder.config(
            "spark.hadoop.fs.defaultFS", "hdfs://localhost:9000"
        )

    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    logger.info("Spark session created successfully")
    return spark


def safe_mape(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-8) -> float:
    """
    Compute Mean Absolute Percentage Error safely.

    MAPE becomes unstable when actual values are zero or near-zero.
    We mask those values and document the limitation.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.abs(y_true) > epsilon
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute regression metrics: MAE, RMSE, MAPE, R²."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

    mape = safe_mape(y_true, y_pred)

    return {"MAE": mae, "RMSE": rmse, "MAPE": mape, "R2": r2}


def save_json(data: Any, path: Path) -> None:
    """Save data as formatted JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)
    logger.info("Saved JSON to %s", path)


def load_json(path: Path) -> Any:
    """Load JSON from file."""
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def check_dataset_exists(config: Optional[Config] = None) -> bool:
    """Check if the raw dataset file exists locally."""
    config = config or get_config()
    return config.RAW_FILE.exists()
