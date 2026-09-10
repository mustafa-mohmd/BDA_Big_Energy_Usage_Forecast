"""Tests for configuration module."""

from pathlib import Path

from src.config import Config, get_config


def test_config_creation():
    config = get_config()
    assert config.PROJECT_ROOT.exists()
    assert config.RAW_FILENAME == "household_power_consumption.txt"


def test_config_paths():
    config = Config()
    assert config.DATA_PATH == config.PROJECT_ROOT / "data" / "raw"
    assert config.PROCESSED_DATA_PATH == config.PROJECT_ROOT / "data" / "processed"
    assert config.MODEL_PATH == config.PROJECT_ROOT / "models"


def test_train_val_test_ratios():
    config = get_config()
    total = config.TRAIN_RATIO + config.VAL_RATIO + config.TEST_RATIO
    assert abs(total - 1.0) < 0.001


def test_feature_columns():
    config = get_config()
    assert "lag_1" in config.FEATURE_COLUMNS
    assert "lag_24" in config.FEATURE_COLUMNS
    assert "lag_168" in config.FEATURE_COLUMNS
    assert "rolling_mean_24" in config.FEATURE_COLUMNS


def test_hdfs_paths():
    config = get_config()
    assert config.HDFS_RAW_PATH.startswith("/energy_forecasting")
    assert "raw" in config.HDFS_RAW_PATH
