# Technical Documentation

## PySpark Concepts (Educational Section)

### RDD (Resilient Distributed Dataset)
- Fundamental Spark data structure
- Immutable, partitioned collection of records
- Supports map, filter, reduce operations
- Fault-tolerant through lineage graph

### DataFrame
- Structured API built on RDDs with named columns and schema
- Optimized by Catalyst query optimizer
- Used throughout this project for all data operations

### Transformation vs Action
- **Transformations** (lazy): `select`, `filter`, `groupBy`, `withColumn` — build execution plan
- **Actions** (eager): `count`, `collect`, `save`, `show` — trigger computation

### Lazy Evaluation
Spark builds a DAG (Directed Acyclic Graph) of transformations and executes only when an action is called. This enables optimization across the entire pipeline.

### Distributed Processing
Data is partitioned across Spark executors. Transformations run in parallel on each partition. Shuffles occur when data must be redistributed (e.g., `groupBy`).

## Module Reference

### src/config.py
Central configuration with paths, HDFS settings, feature columns, and split ratios. Environment variable `USE_HDFS=1` enables HDFS mode.

### src/ingestion.py
Reads raw semicolon-delimited data with explicit schema. Supports HDFS and local paths.

### src/preprocessing.py
- Parses timestamps from Date + Time columns
- Removes invalid/duplicate records
- Aggregates minute data to hourly: `sum(kW) / 60 = kWh`

### src/feature_engineering.py
Creates 11 features using Spark window functions. Removes rows without valid lag values. Performs chronological train/val/test split.

### src/train_models.py
Trains Linear Regression, Random Forest, and GBT using Spark MLlib. Saves models and validation metrics.

### src/evaluate.py
Builds model comparison table from saved metrics.

### src/forecast.py
Recursive 24-hour forecast using best model. Features computed from history + prior predictions only.

### src/eda.py
Generates 7 EDA figures and computes data insights.

## Hourly Aggregation Formula

```
Global_active_power = instantaneous power in kilowatts (kW)
Sampling interval = 1 minute

hourly_energy_consumption (kWh) = sum(Global_active_power over 60 minutes) / 60
```

Each minute reading represents power at that instant. Summing 60 readings and dividing by 60 converts average kW to kWh for one hour.

## Lag Feature Definitions (Hourly Data)

| Feature | Offset | Meaning |
|---------|--------|---------|
| lag_1 | 1 hour | Previous hour's consumption |
| lag_24 | 24 hours | Same hour yesterday |
| lag_168 | 168 hours | Same hour last week |
| rolling_mean_24 | Previous 24 hours | Smoothed recent trend |

## Why Chronological Split?

Random shuffling in time-series causes:
1. **Data leakage**: future values appear in training set
2. **Invalid lag features**: temporal order is destroyed
3. **Optimistic metrics**: model appears better than it would in production

Our 70/15/15 chronological split simulates real deployment where the model predicts truly unseen future data.
