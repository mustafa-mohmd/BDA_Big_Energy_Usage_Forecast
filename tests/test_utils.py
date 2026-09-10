"""Tests for utility functions."""

import numpy as np
import pytest

from src.utils import compute_metrics, safe_mape


def test_compute_metrics_perfect_prediction():
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    metrics = compute_metrics(y, y)
    assert metrics["MAE"] == 0.0
    assert metrics["RMSE"] == 0.0
    assert metrics["R2"] == 1.0


def test_compute_metrics_with_error():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.5, 2.5, 2.5])
    metrics = compute_metrics(y_true, y_pred)
    assert metrics["MAE"] > 0
    assert metrics["RMSE"] > 0
    assert metrics["R2"] < 1.0


def test_safe_mape_normal():
    y_true = np.array([10.0, 20.0, 30.0])
    y_pred = np.array([11.0, 19.0, 31.0])
    mape = safe_mape(y_true, y_pred)
    assert 0 < mape < 20


def test_safe_mape_with_zeros():
    y_true = np.array([0.0, 0.0, 10.0])
    y_pred = np.array([1.0, 2.0, 11.0])
    mape = safe_mape(y_true, y_pred)
    assert not np.isnan(mape)


def test_safe_mape_all_zeros():
    y_true = np.array([0.0, 0.0])
    y_pred = np.array([1.0, 2.0])
    mape = safe_mape(y_true, y_pred)
    assert np.isnan(mape)
