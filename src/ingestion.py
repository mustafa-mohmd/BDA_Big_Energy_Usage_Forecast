"""
Data ingestion: load raw UCI dataset into Spark from local filesystem or HDFS.
"""

from __future__ import annotations

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.types import (
    DoubleType,
    StringType,
    StructField,
    StructType,
)

from src.config import Config, get_config
from src.utils import logger


RAW_SCHEMA = StructType([
    StructField("Date", StringType(), True),
    StructField("Time", StringType(), True),
    StructField("Global_active_power", DoubleType(), True),
    StructField("Global_reactive_power", DoubleType(), True),
    StructField("Voltage", DoubleType(), True),
    StructField("Global_intensity", DoubleType(), True),
    StructField("Sub_metering_1", DoubleType(), True),
    StructField("Sub_metering_2", DoubleType(), True),
    StructField("Sub_metering_3", DoubleType(), True),
])


def read_raw_data(spark: SparkSession, config: Config | None = None) -> DataFrame:
    """
    Read the raw household power consumption dataset.

    Reads from HDFS when USE_HDFS=True, otherwise from local data/raw/.
    """
    config = config or get_config()

    if config.USE_HDFS:
        path = f"hdfs://localhost:9000{config.HDFS_RAW_PATH}/{config.RAW_FILENAME}"
        logger.info("Reading data from HDFS: %s", path)
    else:
        if not config.RAW_FILE.exists():
            raise FileNotFoundError(
                f"Dataset not found at {config.RAW_FILE}. "
                "Run: python scripts/download_dataset.py"
            )
        path = str(config.RAW_FILE)
        logger.info("Reading data from local file: %s", path)

    df = (
        spark.read
        .option("header", "false")
        .option("sep", ";")
        .option("nullValue", config.MISSING_VALUE)
        .schema(RAW_SCHEMA)
        .csv(path)
    )

    count = df.count()
    logger.info("Loaded %d raw records", count)
    return df


def get_dataset_stats(df: DataFrame) -> dict:
    """Compute basic dataset statistics for dashboard and documentation."""
    from pyspark.sql import functions as F

    stats = df.agg(
        F.count("*").alias("total_records"),
        F.sum(F.when(F.col("Global_active_power").isNull(), 1).otherwise(0)).alias(
            "missing_global_active_power"
        ),
        F.min("Global_active_power").alias("min_power"),
        F.max("Global_active_power").alias("max_power"),
        F.mean("Global_active_power").alias("mean_power"),
        F.stddev("Global_active_power").alias("std_power"),
    ).collect()[0]

    return {
        "total_records": stats["total_records"],
        "columns": df.columns,
        "missing_global_active_power": stats["missing_global_active_power"],
        "min_power": stats["min_power"],
        "max_power": stats["max_power"],
        "mean_power": stats["mean_power"],
        "std_power": stats["std_power"],
    }
