# Energy Consumption Forecasting Using Big Data Technologies

## Final Year Project Report

**Department:** Computer Science Engineering  
**Academic Year:** 2025–2026

---

# CHAPTER 1 – INTRODUCTION

## 1.1 Background

Electricity is a fundamental resource for modern society. With the growth of smart grids and smart meters, enormous volumes of energy consumption data are generated continuously. Accurate forecasting of electricity demand enables utilities to optimize power generation, reduce operational costs, and integrate renewable energy sources effectively.

Big Data technologies such as Hadoop and Apache Spark provide the infrastructure needed to store, process, and analyze these large-scale datasets efficiently.

## 1.2 Problem Statement

Household electricity consumption exhibits complex temporal patterns influenced by time of day, day of week, and seasonal variations. Traditional single-machine analysis tools cannot scale to process millions of smart-meter readings. There is a need for a distributed, automated pipeline that can ingest historical data, preprocess it, train forecasting models, and deliver actionable predictions.

## 1.3 Motivation

- Rising global energy demand requires smarter grid management
- Smart meters generate data at unprecedented scale
- Big Data tools (HDFS, Spark) are industry standard for energy analytics
- Machine learning improves forecast accuracy over statistical methods

## 1.4 Objectives

1. Design a scalable Big Data architecture for energy forecasting
2. Implement data ingestion and storage using HDFS
3. Process data using PySpark for cleaning, aggregation, and feature engineering
4. Train and evaluate multiple regression models using Spark MLlib
5. Generate 24-hour ahead consumption forecasts
6. Build an interactive dashboard for visualization and demonstration

## 1.5 Scope

- Dataset: UCI Individual Household Electric Power Consumption (~2M records)
- Forecast target: Hourly energy consumption (kWh)
- Forecast horizon: 24 hours
- Models: Linear Regression, Random Forest, Gradient Boosting
- Platform: Local Spark with optional HDFS

---

# CHAPTER 2 – LITERATURE / TECHNOLOGY REVIEW

## 2.1 Energy Forecasting

Energy demand forecasting has been studied extensively. Approaches range from statistical methods (ARIMA, exponential smoothing) to machine learning (regression, random forests, neural networks). Time-series features such as lag values and rolling statistics are commonly used.

## 2.2 Big Data

Big Data is characterized by Volume, Velocity, and Variety. Energy smart-meter data exhibits all three: millions of readings per day (volume), real-time streaming (velocity), and heterogeneous formats (variety).

## 2.3 Hadoop and HDFS

Apache Hadoop provides distributed storage (HDFS) and processing (MapReduce). HDFS splits files into blocks, replicates them across DataNodes, and provides fault tolerance through the NameNode metadata service.

## 2.4 Apache Spark and PySpark

Apache Spark offers in-memory distributed processing that is significantly faster than MapReduce. PySpark provides a Python API for Spark, enabling data scientists to leverage distributed computing with familiar syntax.

## 2.5 Machine Learning with Spark MLlib

Spark MLlib provides scalable ML algorithms including Linear Regression, Random Forest, and Gradient Boosted Trees. Models integrate directly with Spark DataFrames, enabling end-to-end pipelines.

## 2.6 Existing Approaches

| Approach | Limitation |
|----------|-----------|
| Spreadsheet analysis | Cannot handle millions of records |
| Single-machine Python | No distributed processing |
| Simple moving averages | Cannot capture complex patterns |
| Deep learning only | Requires large data and GPU resources |

---

# CHAPTER 3 – SYSTEM ANALYSIS

## 3.1 Existing System Limitations

- Manual data handling
- No distributed storage
- Limited to small datasets
- No automated ML pipeline
- No interactive visualization

## 3.2 Proposed System

A modular, reproducible pipeline:
Raw Data → HDFS → PySpark → Preprocessing → Features → ML → Forecast → Dashboard

## 3.3 Advantages

- Scalable architecture (HDFS + Spark)
- Reproducible pipeline with configuration management
- Multiple model comparison
- Interactive dashboard for demonstration
- Comprehensive documentation

## 3.4 Functional Requirements

| ID | Requirement |
|----|------------|
| FR1 | Download and ingest UCI dataset |
| FR2 | Store raw data in HDFS |
| FR3 | Clean and preprocess data with PySpark |
| FR4 | Aggregate to hourly consumption |
| FR5 | Engineer time-series features |
| FR6 | Train Linear Regression and Random Forest |
| FR7 | Evaluate models with MAE, RMSE, MAPE, R² |
| FR8 | Generate 24-hour forecast |
| FR9 | Display results in Streamlit dashboard |

## 3.5 Non-Functional Requirements

- Reproducibility: same input produces same output
- Configurability: paths and parameters in central config
- Cross-platform: Windows and Linux support
- Logging: pipeline progress tracking
- Error handling: meaningful error messages

## 3.6 Feasibility

| Aspect | Assessment |
|--------|-----------|
| Technical | Feasible with open-source tools |
| Economic | No licensing costs |
| Operational | Runs on standard hardware |
| Schedule | Achievable in one semester |

---

# CHAPTER 4 – SYSTEM DESIGN

## 4.1 Architecture

See `docs/architecture/system_architecture.md` for the full Mermaid diagram.

## 4.2 Data Flow

See `docs/architecture/data_flow.md`.

## 4.3 Module Design

| Module | File | Responsibility |
|--------|------|---------------|
| Config | src/config.py | Central configuration |
| Ingestion | src/ingestion.py | Read raw data from HDFS/local |
| Preprocessing | src/preprocessing.py | Clean, aggregate hourly |
| Features | src/feature_engineering.py | Lag and time features |
| Training | src/train_models.py | MLlib model training |
| Evaluation | src/evaluate.py | Metrics and comparison |
| Forecast | src/forecast.py | 24-hour recursive forecast |
| EDA | src/eda.py | Exploratory analysis figures |
| Dashboard | dashboard/app.py | Streamlit UI |

## 4.4 Data Storage Design

| Path | Format | Content |
|------|--------|---------|
| data/raw/ | TXT | Original dataset |
| data/processed/hourly/ | Parquet | Hourly aggregated data |
| data/processed/features/ | Parquet | Feature-engineered data |
| models/ | Spark ML | Trained models |
| results/metrics/ | JSON/CSV | Evaluation metrics |
| results/predictions/ | CSV | Forecast output |

## 4.5 ML Pipeline

See `docs/architecture/ml_pipeline.md`.

---

# CHAPTER 5 – IMPLEMENTATION

## 5.1 Data Ingestion

Raw semicolon-delimited data is read using PySpark with explicit schema definition. Missing values ("?") are handled as nulls.

## 5.2 HDFS Integration

HDFS directories are created under `/energy_forecasting/`. Raw data is uploaded via `hdfs dfs -put`. PySpark reads using `hdfs://localhost:9000` URIs. Local fallback uses file paths when HDFS is unavailable.

## 5.3 Preprocessing

- Timestamp parsing: combine Date + Time columns
- Invalid value filtering: remove negative power readings
- Duplicate removal: drop duplicate timestamps
- Hourly aggregation: `sum(Global_active_power) / 60`

## 5.4 Feature Engineering

11 features created using Spark window functions:
hour, day, day_of_week, day_of_month, month, year, is_weekend, lag_1, lag_24, lag_168, rolling_mean_24

## 5.5 Model Training

Three Spark MLlib models trained on 70% chronological training data. Validation on next 15%. Best model selected by validation RMSE.

## 5.6 Evaluation

Metrics computed on validation and test sets. Comparison table saved as CSV.

## 5.7 Forecasting

Recursive one-step-ahead prediction for 24 hours. Each step uses history + prior predictions for lag features.

## 5.8 Dashboard

Streamlit application with 6 sections reading actual generated artifacts.

---

# CHAPTER 6 – RESULTS

> **Note:** Populate this section after running the pipeline.

## 6.1 Dataset Statistics

| Metric | Value |
|--------|-------|
| Raw records | [Run pipeline to populate] |
| Hourly records | [Run pipeline to populate] |
| Date range | [Run pipeline to populate] |

## 6.2 Model Comparison

| Model | MAE | RMSE | MAPE | R² |
|-------|-----|------|------|-----|
| Linear Regression | [TBD] | [TBD] | [TBD] | [TBD] |
| Random Forest | [TBD] | [TBD] | [TBD] | [TBD] |
| Gradient Boosting | [TBD] | [TBD] | [TBD] | [TBD] |

## 6.3 Best Model Test Performance

| Metric | Value |
|--------|-------|
| Best Model | [TBD] |
| Test MAE | [TBD] |
| Test RMSE | [TBD] |
| Test MAPE | [TBD] |
| Test R² | [TBD] |

## 6.4 Key Insights

- Peak consumption hour: [TBD]
- Minimum consumption hour: [TBD]
- Weekday vs weekend difference: [TBD]

---

# CHAPTER 7 – CONCLUSION AND FUTURE SCOPE

## 7.1 Conclusion

This project successfully demonstrates an end-to-end Big Data pipeline for energy consumption forecasting. By integrating HDFS, Apache Spark, and Spark MLlib, the system provides a scalable foundation that can be extended to production smart-meter deployments.

## 7.2 Future Scope

- Real-time streaming with Apache Kafka
- Cluster deployment on YARN
- Weather and calendar features
- Deep learning models (LSTM, Transformer)
- Multi-household aggregation
- Cloud deployment (AWS EMR, Azure HDInsight)

## References

1. UCI Machine Learning Repository — Individual Household Electric Power Consumption Dataset
2. Apache Hadoop Documentation — https://hadoop.apache.org/docs/
3. Apache Spark Documentation — https://spark.apache.org/docs/latest/
4. Spark MLlib Guide — https://spark.apache.org/docs/latest/ml-guide.html
5. Streamlit Documentation — https://docs.streamlit.io/

## Appendix

- A: HDFS Commands (`hadoop/hdfs_commands_windows.txt`, `hadoop/hdfs_commands_linux.txt`)
- B: Project Structure (see README.md)
- C: Viva Questions (`docs/viva/viva_questions.md`)
- D: Configuration Reference (`src/config.py`)
