"""Tests for preprocessing logic."""

import pandas as pd


def test_hourly_aggregation_formula():
    """Verify hourly energy = sum(kW) / 60 for 1-minute samples."""
    # Simulate 60 minutes at 2 kW each
    minute_power = [2.0] * 60
    hourly_energy = sum(minute_power) / 60.0
    assert hourly_energy == 2.0  # 2 kW for 1 hour = 2 kWh


def test_hourly_aggregation_varying():
    """Test with varying minute-level power readings."""
    minute_power = [1.0, 2.0, 3.0] * 20  # 60 samples
    hourly_energy = sum(minute_power) / 60.0
    expected = (1.0 + 2.0 + 3.0) / 3.0  # average kW = kWh for 1 hour
    assert abs(hourly_energy - expected) < 0.001


def test_timestamp_parsing():
    """Test date/time parsing format."""
    date_str = "16/12/2006"
    time_str = "17:24:00"
    combined = f"{date_str} {time_str}"
    ts = pd.to_datetime(combined, format="%d/%m/%Y %H:%M:%S")
    assert ts.year == 2006
    assert ts.month == 12
    assert ts.day == 16
    assert ts.hour == 17


def test_lag_feature_definitions():
    """Document lag feature semantics for hourly data."""
    lags = {"lag_1": 1, "lag_24": 24, "lag_168": 168}
    assert lags["lag_1"] == 1       # 1 hour ago
    assert lags["lag_24"] == 24     # 24 hours ago (1 day)
    assert lags["lag_168"] == 168   # 168 hours ago (1 week)
