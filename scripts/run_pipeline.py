#!/usr/bin/env python3
"""
Run the complete Energy Consumption Forecasting pipeline.

Usage:
    python scripts/run_pipeline.py [--skip-download] [--use-hdfs]

Steps:
    1. Download dataset (if needed)
    2. Preprocessing (clean + hourly aggregation)
    3. Feature engineering
    4. EDA
    5. Model training
    6. Evaluation
    7. 24-hour forecast
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import get_config
from src.eda import run_eda
from src.evaluate import run_evaluation
from src.forecast import run_forecast
from src.preprocessing import run_preprocessing
from src.train_models import run_training
from src.utils import check_dataset_exists, create_spark_session, logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Energy Forecasting Pipeline")
    parser.add_argument("--skip-download", action="store_true", help="Skip dataset download")
    parser.add_argument("--use-hdfs", action="store_true", help="Use HDFS instead of local storage")
    parser.add_argument("--skip-training", action="store_true", help="Skip model training")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = get_config()

    if args.use_hdfs:
        config.USE_HDFS = True
        logger.info("HDFS mode enabled")

    # Step 1: Download dataset
    if not args.skip_download and not check_dataset_exists(config):
        logger.info("Downloading dataset...")
        from scripts.download_dataset import main as download_main
        if download_main() != 0:
            logger.error("Dataset download failed. See manual instructions above.")
            return 1

    if not check_dataset_exists(config) and not config.USE_HDFS:
        logger.error("Dataset not available. Run: python scripts/download_dataset.py")
        return 1

    spark = create_spark_session(config)

    try:
        # Step 2: Preprocessing
        logger.info("=" * 50)
        logger.info("STEP 2: Preprocessing")
        run_preprocessing(spark, config)

        # Step 3-4: Feature engineering happens inside training, but EDA uses hourly data
        logger.info("=" * 50)
        logger.info("STEP 4: Exploratory Data Analysis")
        run_eda(spark, config)

        if not args.skip_training:
            # Step 5: Training (includes feature engineering)
            logger.info("=" * 50)
            logger.info("STEP 5: Model Training")
            run_training(spark, config)

            # Step 6: Evaluation
            logger.info("=" * 50)
            logger.info("STEP 6: Model Evaluation")
            run_evaluation(config)

            # Step 7: Forecast
            logger.info("=" * 50)
            logger.info("STEP 7: 24-Hour Forecast")
            run_forecast(spark, config)

        logger.info("=" * 50)
        logger.info("Pipeline completed successfully!")
        logger.info("Launch dashboard: streamlit run dashboard/app.py")
        return 0

    except Exception as e:
        logger.error("Pipeline failed: %s", e, exc_info=True)
        return 1
    finally:
        spark.stop()


if __name__ == "__main__":
    sys.exit(main())
