"""
PySpark data cleaning, timestamp parsing, and hourly aggregation.

Aggregation: Global_active_power (kW, minute-level) -> hourly_energy_consumption (kWh)
Formula: hourly_energy = mean(Global_active_power) * 60 minutes / 60 = mean(kW) * 1 hour
         Since power is in kW and samples are per minute, energy per hour = sum(kW)/60 * 60 min
         = sum of (power * time_interval). With 1-min samples: energy_kWh = sum(kW) / 60
"""

from __future__ import annotations

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

from src.config import Config, get_config
from src.utils import logger, save_json


def parse_timestamp(df: DataFrame, config: Config | None = None) -> DataFrame:
    """Combine Date and Time columns into a proper timestamp."""
    config = config or get_config()
    logger.info("Parsing timestamps")

    # Spark 3+ can reject legacy EU-format timestamps unless compatibility mode
    # is explicitly enabled for the session. This is configured centrally in the
    # Spark builder, but we keep this conversion explicit for clarity.
    date_time = F.concat(F.col("Date"), F.lit(" "), F.col("Time"))
    df = df.withColumn(
        config.TIMESTAMP_COL,
        F.to_timestamp(date_time, config.DATE_FORMAT),
    ).filter(F.col(config.TIMESTAMP_COL).isNotNull())

    return df.drop("Date", "Time")


def clean_data(df: DataFrame, config: Config | None = None) -> DataFrame:
    """
    Clean raw data: handle missing values, invalid measurements, duplicates.

    Decisions:
    - Drop rows with null timestamp or null target after numeric cast
    - Filter negative power values (physically invalid)
    - Remove exact duplicate timestamp records (keep first)
    - Do NOT blindly impute missing power; drop null target rows
    """
    config = config or get_config()
    logger.info("Cleaning data")

    # Ensure numeric columns are valid
    numeric_cols = [
        "Global_active_power", "Global_reactive_power", "Voltage",
        "Global_intensity", "Sub_metering_1", "Sub_metering_2", "Sub_metering_3",
    ]
    for col in numeric_cols:
        df = df.withColumn(col, F.col(col).cast("double"))

    df = parse_timestamp(df, config)

    initial_count = df.count()

    # Remove invalid power readings
    df = df.filter(
        (F.col(config.TARGET_COLUMN).isNotNull())
        & (F.col(config.TARGET_COLUMN) >= 0)
        & (F.col(config.TARGET_COLUMN) < 100)  # sanity upper bound for household kW
    )

    # Remove duplicate timestamps
    df = df.dropDuplicates([config.TIMESTAMP_COL])

    # Sort chronologically
    df = df.orderBy(config.TIMESTAMP_COL)

    final_count = df.count()
    removed = initial_count - final_count
    logger.info("Cleaning removed %d rows (%d -> %d)", removed, initial_count, final_count)

    return df


def aggregate_hourly(df: DataFrame, config: Config | None = None) -> DataFrame:
    """
    Aggregate minute-level data to hourly energy consumption.

    Global_active_power is in kilowatts (kW). With 1-minute sampling:
        hourly_energy_consumption (kWh) = sum(Global_active_power) / 60

    Each minute sample represents power at that instant; integrating over
    60 one-minute intervals gives energy in kWh.
    """
    config = config or get_config()
    logger.info("Creating hourly aggregates")

    df = df.withColumn("hour_ts", F.date_trunc("hour", F.col(config.TIMESTAMP_COL)))

    hourly = df.groupBy("hour_ts").agg(
        (F.sum(config.TARGET_COLUMN) / 60.0).alias(config.HOURLY_TARGET),
        F.avg("Global_reactive_power").alias("avg_reactive_power"),
        F.avg("Voltage").alias("avg_voltage"),
        F.avg("Global_intensity").alias("avg_intensity"),
        F.sum("Sub_metering_1").alias("sub_metering_1_kwh"),
        F.sum("Sub_metering_2").alias("sub_metering_2_kwh"),
        F.sum("Sub_metering_3").alias("sub_metering_3_kwh"),
        F.count("*").alias("minute_samples"),
    ).withColumnRenamed("hour_ts", config.TIMESTAMP_COL)

    # Filter hours with too few samples (likely incomplete hours)
    hourly = hourly.filter(F.col("minute_samples") >= 30)

    hourly = hourly.orderBy(config.TIMESTAMP_COL)
    count = hourly.count()
    logger.info("Created %d hourly records", count)

    return hourly


def save_processed(df: DataFrame, config: Config | None = None) -> str:
    """Write processed hourly data to Parquet (HDFS or local)."""
    config = config or get_config()
    path = config.processed_hourly_path
    logger.info("Saving processed data to %s", path)

    df.write.mode("overwrite").parquet(path)
    return path


def run_preprocessing(spark: SparkSession, config: Config | None = None) -> DataFrame:
    """Execute full preprocessing pipeline."""
    from src.ingestion import read_raw_data

    config = config or get_config()
    logger.info("Starting preprocessing pipeline")

    raw_df = read_raw_data(spark, config)
    cleaned = clean_data(raw_df, config)
    hourly = aggregate_hourly(cleaned, config)
    save_processed(hourly, config)

    # Save preprocessing metadata
    stats = {
        "hourly_records": hourly.count(),
        "date_range": {
            "start": str(hourly.agg(F.min(config.TIMESTAMP_COL)).collect()[0][0]),
            "end": str(hourly.agg(F.max(config.TIMESTAMP_COL)).collect()[0][0]),
        },
        "aggregation": (
            "hourly_energy_consumption (kWh) = sum(Global_active_power) / 60 "
            "from 1-minute kW readings"
        ),
    }
    save_json(stats, config.METRICS_PATH / "preprocessing_stats.json")

    return hourly
