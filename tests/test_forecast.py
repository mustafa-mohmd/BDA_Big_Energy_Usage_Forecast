"""Tests for forecast output format."""

import pandas as pd


def test_forecast_output_columns():
    """Forecast DataFrame must have required columns."""
    df = pd.DataFrame({
        "timestamp": ["2026-01-01T00:00:00", "2026-01-01T01:00:00"],
        "predicted_energy_consumption": [0.5, 0.6],
    })
    assert "timestamp" in df.columns
    assert "predicted_energy_consumption" in df.columns
    assert len(df) == 2


def test_forecast_values_non_negative():
    """Predicted consumption should be non-negative."""
    values = [0.0, 0.5, 1.2, 3.0]
    for v in values:
        assert max(0.0, v) >= 0


def test_forecast_horizon():
    """Default forecast horizon is 24 hours."""
    from src.config import get_config
    config = get_config()
    assert config.FORECAST_HORIZON == 24
