"""
Model training using Spark MLlib: Linear Regression and Random Forest.
"""

from __future__ import annotations

from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import GBTRegressor, LinearRegression, RandomForestRegressor
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.sql import DataFrame, SparkSession

from src.config import Config, get_config
from src.feature_engineering import chronological_split, run_feature_engineering
from src.utils import compute_metrics, logger, save_json


def prepare_ml_data(df: DataFrame, config: Config | None = None) -> DataFrame:
    """Assemble feature vector for MLlib."""
    config = config or get_config()
    assembler = VectorAssembler(
        inputCols=config.FEATURE_COLUMNS,
        outputCol="features",
        handleInvalid="skip",
    )
    return assembler.transform(df).select(
        "features", config.HOURLY_TARGET, config.TIMESTAMP_COL
    )


def train_linear_regression(
    train_df: DataFrame, config: Config | None = None
):
    """Train Spark MLlib Linear Regression."""
    config = config or get_config()
    logger.info("Training Linear Regression")

    lr = LinearRegression(
        featuresCol="features",
        labelCol=config.HOURLY_TARGET,
        maxIter=100,
        regParam=0.1,
        elasticNetParam=0.0,
    )
    model = lr.fit(train_df)
    logger.info("Linear Regression training complete")
    return model


def train_random_forest(
    train_df: DataFrame, config: Config | None = None
):
    """Train Spark MLlib Random Forest Regressor."""
    config = config or get_config()
    logger.info("Training Random Forest")

    rf = RandomForestRegressor(
        featuresCol="features",
        labelCol=config.HOURLY_TARGET,
        numTrees=50,
        maxDepth=10,
        seed=config.RANDOM_SEED,
    )
    model = rf.fit(train_df)
    logger.info("Random Forest training complete")
    return model


def train_gbt(
    train_df: DataFrame, config: Config | None = None
):
    """Train Spark MLlib Gradient Boosted Trees (optional third model)."""
    config = config or get_config()
    logger.info("Training Gradient Boosting")

    gbt = GBTRegressor(
        featuresCol="features",
        labelCol=config.HOURLY_TARGET,
        maxIter=50,
        maxDepth=5,
        seed=config.RANDOM_SEED,
    )
    model = gbt.fit(train_df)
    logger.info("Gradient Boosting training complete")
    return model


def predict_and_collect(
    model, df: DataFrame, config: Config | None = None
) -> tuple[list[float], list[float]]:
    """Generate predictions and collect actual vs predicted values."""
    config = config or get_config()
    predictions = model.transform(df)
    rows = predictions.select(
        config.HOURLY_TARGET, "prediction"
    ).collect()
    y_true = [float(r[config.HOURLY_TARGET]) for r in rows]
    y_pred = [float(r["prediction"]) for r in rows]
    return y_true, y_pred


def save_model(model, name: str, config: Config | None = None) -> str:
    """Save trained Spark ML model."""
    config = config or get_config()
    path = str(config.MODEL_PATH / name)
    model.write().overwrite().save(path)
    logger.info("Saved model to %s", path)
    return path


def run_training(spark: SparkSession, config: Config | None = None) -> dict:
    """Execute full model training and validation pipeline."""
    config = config or get_config()

    features_df = run_feature_engineering(spark, config)
    train_df, val_df, test_df = chronological_split(features_df, config)

    train_ml = prepare_ml_data(train_df, config)
    val_ml = prepare_ml_data(val_df, config)
    test_ml = prepare_ml_data(test_df, config)

    models = {
        config.MODEL_LINEAR_REGRESSION: train_linear_regression(train_ml, config),
        config.MODEL_RANDOM_FOREST: train_random_forest(train_ml, config),
        config.MODEL_GBT: train_gbt(train_ml, config),
    }

    val_metrics = {}
    for name, model in models.items():
        save_model(model, name, config)
        y_true, y_pred = predict_and_collect(model, val_ml, config)
        val_metrics[name] = compute_metrics(
            __import__("numpy").array(y_true),
            __import__("numpy").array(y_pred),
        )
        logger.info("Validation metrics for %s: %s", name, val_metrics[name])

    # Select best model by validation RMSE
    best_name = min(val_metrics, key=lambda k: val_metrics[k]["RMSE"])
    logger.info("Best model (validation RMSE): %s", best_name)

    best_model = models[best_name]
    y_true_test, y_pred_test = predict_and_collect(best_model, test_ml, config)
    test_metrics = compute_metrics(
        __import__("numpy").array(y_true_test),
        __import__("numpy").array(y_pred_test),
    )

    results = {
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
        "best_model": best_name,
        "model_info": {
            config.MODEL_LINEAR_REGRESSION: {
                "algorithm": "Linear Regression (Spark MLlib)",
                "hyperparameters": {"maxIter": 100, "regParam": 0.1},
                "advantages": "Fast, interpretable, good baseline",
                "limitations": "Assumes linear relationships",
            },
            config.MODEL_RANDOM_FOREST: {
                "algorithm": "Random Forest Regressor (Spark MLlib)",
                "hyperparameters": {"numTrees": 50, "maxDepth": 10},
                "advantages": "Handles non-linearity, robust to outliers",
                "limitations": "Less interpretable, slower training",
            },
            config.MODEL_GBT: {
                "algorithm": "Gradient Boosted Trees (Spark MLlib)",
                "hyperparameters": {"maxIter": 50, "maxDepth": 5},
                "advantages": "Strong predictive performance",
                "limitations": "Risk of overfitting, longer training",
            },
        },
    }

    save_json(results, config.METRICS_PATH / "model_results.json")
    save_json(
        {"best_model": best_name, "validation_rmse": val_metrics[best_name]["RMSE"]},
        config.METRICS_PATH / "best_model.json",
    )

    return results
