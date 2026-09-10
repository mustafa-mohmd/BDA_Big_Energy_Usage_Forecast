# System Architecture

## Overview

The Energy Consumption Forecasting system follows a layered Big Data architecture designed for scalability and reproducibility.

## Architecture Diagram

```mermaid
graph TB
    subgraph "Data Sources"
        UCI[UCI Dataset<br/>household_power_consumption.txt]
    end

    subgraph "Storage Layer"
        HDFS[(Hadoop HDFS)]
        LOCAL[(Local Filesystem<br/>Fallback)]
    end

    subgraph "Processing Layer"
        SPARK[Apache Spark / PySpark]
        INGEST[Data Ingestion]
        PREPROC[Preprocessing]
        FEAT[Feature Engineering]
    end

    subgraph "Analytics Layer"
        EDA[Exploratory Analysis]
        ML[Spark MLlib]
        LR[Linear Regression]
        RF[Random Forest]
        GBT[Gradient Boosting]
    end

    subgraph "Output Layer"
        METRICS[Metrics & Results]
        FORECAST[24-Hour Forecast]
        DASH[Streamlit Dashboard]
    end

    UCI --> INGEST
    INGEST --> HDFS
    INGEST --> LOCAL
    HDFS --> SPARK
    LOCAL --> SPARK
    SPARK --> PREPROC
    PREPROC --> FEAT
    FEAT --> EDA
    FEAT --> ML
    ML --> LR
    ML --> RF
    ML --> GBT
    LR --> METRICS
    RF --> METRICS
    GBT --> METRICS
    METRICS --> FORECAST
    METRICS --> DASH
    FORECAST --> DASH
```

## Component Descriptions

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Raw Storage | HDFS / Local | Store original and processed datasets |
| Processing | PySpark | Distributed data cleaning and transformation |
| ML Training | Spark MLlib | Train regression models at scale |
| Results | JSON/CSV/Parquet | Persist metrics, predictions, models |
| Visualization | Streamlit + Plotly | Interactive dashboard |

## Scalability Note

The prototype uses a publicly available dataset (~2M records) that fits on a single machine. The architecture demonstrates how the same pipeline extends to smart-meter datasets with billions of records by scaling HDFS storage and Spark executors.
