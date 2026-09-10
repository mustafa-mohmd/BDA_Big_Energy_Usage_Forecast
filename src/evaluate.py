"""
Model evaluation: metrics computation and comparison table generation.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import Config, get_config
from src.utils import load_json, logger, save_json


def build_comparison_table(config: Config | None = None) -> pd.DataFrame:
    """Build model comparison table from saved validation metrics."""
    config = config or get_config()
    results_path = config.METRICS_PATH / "model_results.json"

    if not results_path.exists():
        raise FileNotFoundError(
            "Model results not found. Run training first: python scripts/run_pipeline.py"
        )

    results = load_json(results_path)
    val_metrics = results["validation_metrics"]

    rows = []
    for model_name, metrics in val_metrics.items():
        rows.append({
            "Model": model_name.replace("_", " ").title(),
            "MAE": round(metrics["MAE"], 4),
            "RMSE": round(metrics["RMSE"], 4),
            "MAPE": round(metrics["MAPE"], 4) if metrics["MAPE"] == metrics["MAPE"] else "N/A",
            "R²": round(metrics["R2"], 4),
        })

    df = pd.DataFrame(rows)
    logger.info("Model comparison table:\n%s", df.to_string(index=False))

    csv_path = config.METRICS_PATH / "model_comparison.csv"
    df.to_csv(csv_path, index=False)
    logger.info("Saved comparison table to %s", csv_path)

    return df


def run_evaluation(config: Config | None = None) -> dict:
    """Run evaluation and generate comparison artifacts."""
    config = config or get_config()
    logger.info("Evaluating models")

    comparison_df = build_comparison_table(config)
    results = load_json(config.METRICS_PATH / "model_results.json")

    summary = {
        "best_model": results["best_model"],
        "validation_metrics": results["validation_metrics"],
        "test_metrics": results["test_metrics"],
        "comparison_table": comparison_df.to_dict(orient="records"),
    }

    save_json(summary, config.METRICS_PATH / "evaluation_summary.json")
    return summary
