"""
Central configuration for the Energy Consumption Forecasting project.

Supports Windows, Linux, and WSL environments with optional HDFS or local fallback.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List


def _project_root() -> Path:
    """Return the project root directory."""
    return Path(__file__).resolve().parent.parent


@dataclass
class Config:
    """Project-wide configuration."""

    # Paths
    PROJECT_ROOT: Path = field(default_factory=_project_root)
    DATA_PATH: Path = field(default_factory=lambda: _project_root() / "data" / "raw")
    PROCESSED_DATA_PATH: Path = field(
        default_factory=lambda: _project_root() / "data" / "processed"
    )
    MODEL_PATH: Path = field(default_factory=lambda: _project_root() / "models")
    RESULTS_PATH: Path = field(default_factory=lambda: _project_root() / "results")
    FIGURES_PATH: Path = field(
        default_factory=lambda: _project_root() / "results" / "figures"
    )
    METRICS_PATH: Path = field(
        default_factory=lambda: _project_root() / "results" / "metrics"
    )
    PREDICTIONS_PATH: Path = field(
        default_factory=lambda: _project_root() / "results" / "predictions"
    )

    # Dataset
    RAW_FILENAME: str = "household_power_consumption.txt"
    RAW_FILE: Path = field(init=False)
    DATASET_URL: str = (
        "https://archive.ics.uci.edu/static/public/235/"
        "individual+household+electric+power+consumption.zip"
    )

    # HDFS paths (logical Hadoop directories)
    HDFS_BASE: str = "/energy_forecasting"
    HDFS_RAW_PATH: str = "/energy_forecasting/raw"
    HDFS_PROCESSED_PATH: str = "/energy_forecasting/processed"
    HDFS_FEATURES_PATH: str = "/energy_forecasting/features"
    HDFS_RESULTS_PATH: str = "/energy_forecasting/results"

    # Spark
    SPARK_APP_NAME: str = "EnergyConsumptionForecasting"
    SPARK_MASTER: str = "local[*]"
    USE_HDFS: bool = False  # Set True when Hadoop cluster is available

    # Local fallback paths (file:// URIs when HDFS unavailable)
    LOCAL_RAW_URI: str = field(init=False)
    LOCAL_PROCESSED_URI: str = field(init=False)
    LOCAL_FEATURES_URI: str = field(init=False)

    # Preprocessing
    TARGET_COLUMN: str = "Global_active_power"
    HOURLY_TARGET: str = "hourly_energy_consumption"
    TIMESTAMP_COL: str = "timestamp"
    MISSING_VALUE: str = "?"
    DATE_FORMAT: str = "dd/MM/yyyy HH:mm:ss"

    # Feature columns
    FEATURE_COLUMNS: List[str] = field(default_factory=lambda: [
        "hour", "day", "day_of_week", "day_of_month", "month", "year",
        "is_weekend", "lag_1", "lag_24", "lag_168", "rolling_mean_24",
    ])

    # Train/validation/test split (chronological)
    TRAIN_RATIO: float = 0.70
    VAL_RATIO: float = 0.15
    TEST_RATIO: float = 0.15

    # Forecasting
    FORECAST_HORIZON: int = 24

    # Model names
    MODEL_LINEAR_REGRESSION: str = "linear_regression"
    MODEL_RANDOM_FOREST: str = "random_forest"
    MODEL_GBT: str = "gradient_boosting"

    # Random seed for reproducibility
    RANDOM_SEED: int = 42

    def __post_init__(self) -> None:
        self.RAW_FILE = self.DATA_PATH / self.RAW_FILENAME
        self.LOCAL_RAW_URI = self.RAW_FILE.as_uri()
        self.LOCAL_PROCESSED_URI = (self.PROCESSED_DATA_PATH / "hourly").as_uri()
        self.LOCAL_FEATURES_URI = (self.PROCESSED_DATA_PATH / "features").as_uri()

        # Allow environment overrides
        if os.getenv("USE_HDFS", "").lower() in ("1", "true", "yes"):
            self.USE_HDFS = True
        if os.getenv("SPARK_MASTER"):
            self.SPARK_MASTER = os.getenv("SPARK_MASTER", self.SPARK_MASTER)

        # Ensure directories exist
        for path in [
            self.DATA_PATH,
            self.PROCESSED_DATA_PATH,
            self.MODEL_PATH,
            self.RESULTS_PATH,
            self.FIGURES_PATH,
            self.METRICS_PATH,
            self.PREDICTIONS_PATH,
        ]:
            path.mkdir(parents=True, exist_ok=True)

    @property
    def raw_data_path(self) -> str:
        """Return HDFS or local URI for raw data."""
        if self.USE_HDFS:
            return f"hdfs://localhost:9000{self.HDFS_RAW_PATH}/{self.RAW_FILENAME}"
        return str(self.RAW_FILE)

    @property
    def processed_hourly_path(self) -> str:
        """Return HDFS or local path for hourly aggregated data."""
        if self.USE_HDFS:
            return f"hdfs://localhost:9000{self.HDFS_PROCESSED_PATH}/hourly"
        return str(self.PROCESSED_DATA_PATH / "hourly")

    @property
    def features_path(self) -> str:
        """Return HDFS or local path for feature-engineered data."""
        if self.USE_HDFS:
            return f"hdfs://localhost:9000{self.HDFS_FEATURES_PATH}/features"
        return str(self.PROCESSED_DATA_PATH / "features")

    @property
    def hdfs_enabled(self) -> bool:
        return self.USE_HDFS


def get_config() -> Config:
    """Return a fresh configuration instance."""
    return Config()
