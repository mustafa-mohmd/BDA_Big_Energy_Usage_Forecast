# Viva Questions and Answers

## Project Overview

**Q1: Why did you choose this project?**
A: Energy consumption forecasting is critical for smart grids and sustainable energy management. This project combines Big Data technologies with machine learning to solve a real-world problem relevant to the energy sector.

**Q2: Why energy forecasting specifically?**
A: Electricity demand fluctuates hourly based on human activity patterns. Accurate forecasting helps utilities plan generation, reduce waste, and optimize costs.

**Q3: What is the primary objective of your project?**
A: To build an end-to-end Big Data pipeline that ingests historical electricity data, processes it with Spark, trains forecasting models, and delivers 24-hour predictions through an interactive dashboard.

## Big Data Concepts

**Q4: What is Big Data?**
A: Big Data refers to datasets characterized by high Volume, Velocity, and Variety (the 3 Vs) that exceed the processing capacity of traditional systems.

**Q5: Is your dataset truly Big Data?**
A: The UCI dataset has ~2 million records, which is moderately large. Our architecture uses HDFS and Spark to demonstrate scalability. The same pipeline extends to billions of smart-meter records in production.

**Q6: Why use Hadoop?**
A: Hadoop provides distributed, fault-tolerant storage (HDFS) and processing (MapReduce). It enables horizontal scaling across commodity hardware.

**Q7: What is HDFS?**
A: Hadoop Distributed File System — a distributed file system that stores data across multiple machines with replication for fault tolerance.

**Q8: What is NameNode vs DataNode?**
A: NameNode manages metadata (file locations, directory structure). DataNodes store actual data blocks on local disks.

**Q9: What are HDFS blocks?**
A: Files are split into fixed-size blocks (default 128 MB). Each block is stored on a DataNode and replicated (default 3 copies).

**Q10: What is HDFS replication?**
A: Each block is copied to multiple DataNodes (default factor 3) to ensure data availability if a node fails.

## Apache Spark

**Q11: Why Spark over MapReduce?**
A: Spark is 10-100x faster due to in-memory processing, supports iterative algorithms (ML), and provides higher-level APIs (DataFrames, MLlib).

**Q12: What is PySpark?**
A: Python API for Apache Spark, allowing distributed data processing using Python syntax and Spark's execution engine.

**Q13: What is an RDD?**
A: Resilient Distributed Dataset — Spark's fundamental immutable, partitioned data structure. Supports transformations and actions.

**Q14: RDD vs DataFrame?**
A: RDDs are low-level, untyped collections. DataFrames are structured, schema-aware tables with Catalyst optimizer for better performance. We use DataFrames.

**Q15: What is lazy evaluation?**
A: Transformations (map, filter) are not executed immediately. Spark builds a DAG and executes only when an action (count, collect) is called.

**Q16: Transformations vs Actions?**
A: Transformations create new RDDs/DataFrames (lazy). Actions trigger computation and return results (count, save, collect).

**Q17: Why Parquet format?**
A: Columnar storage with compression, schema embedding, and efficient column pruning. Ideal for analytical workloads and Spark integration.

## Data Processing

**Q18: How do you handle missing values?**
A: The dataset uses "?" for missing values. We drop rows with null target (Global_active_power) rather than imputing, to avoid introducing bias.

**Q19: How is hourly aggregation performed?**
A: Global_active_power is in kW with 1-minute sampling. Hourly energy (kWh) = sum(Global_active_power) / 60.

**Q20: Why aggregate to hourly?**
A: Minute-level forecasting is noisy and computationally expensive. Hourly aggregation captures daily patterns while reducing data volume by 60x.

**Q21: What preprocessing steps did you perform?**
A: Timestamp parsing, missing value removal, invalid value filtering (negative power), duplicate removal, and chronological sorting.

## Feature Engineering

**Q22: What features did you create?**
A: Time features (hour, day, month, is_weekend), lag features (lag_1, lag_24, lag_168), and rolling_mean_24.

**Q23: What is lag_1?**
A: Energy consumption from the previous hour. Captures short-term temporal dependency.

**Q24: What is lag_24?**
A: Consumption from the same hour yesterday. Captures daily seasonality.

**Q25: What is lag_168?**
A: Consumption from the same hour last week (168 hours). Captures weekly patterns.

**Q26: What is rolling_mean_24?**
A: Average consumption over the previous 24 hours (excluding current). Smooths short-term fluctuations.

**Q27: What is data leakage?**
A: Using future information during training that wouldn't be available at prediction time. We prevent this by using only past values for lags and rolling stats.

## Machine Learning

**Q28: Why not random train/test split?**
A: Time-series data has temporal dependencies. Random splitting causes data leakage and unrealistic performance estimates. We use chronological splitting.

**Q29: What is your train/validation/test split?**
A: 70% training, 15% validation, 15% test — all chronological with no shuffling.

**Q30: Explain Linear Regression.**
A: Models the relationship between features and target as a linear equation. Fast, interpretable, good baseline. Assumes linear relationships.

**Q31: Explain Random Forest.**
A: Ensemble of decision trees trained on random subsets. Handles non-linearity, robust to outliers. Less interpretable than linear models.

**Q32: How do you select the best model?**
A: Compare validation set RMSE across all models. Select the model with lowest validation RMSE, then evaluate on unseen test set.

## Evaluation Metrics

**Q33: What is MAE?**
A: Mean Absolute Error — average absolute difference between predicted and actual values. Same units as target (kWh).

**Q34: What is RMSE?**
A: Root Mean Squared Error — square root of average squared errors. Penalizes large errors more than MAE.

**Q35: What is MAPE?**
A: Mean Absolute Percentage Error — average percentage error. We mask near-zero values where MAPE becomes unstable.

**Q36: What is R²?**
A: Coefficient of determination — proportion of variance in target explained by the model. Range: 0 to 1 (higher is better).

## Forecasting

**Q37: How do you generate 24-hour forecasts?**
A: Recursive one-step-ahead forecasting. For each hour, build features from history + prior predictions, predict, append to history.

**Q38: Do you use future actual values during forecasting?**
A: No. Only past actual values and previously predicted values are used for lag features.

**Q39: Why not LSTM/neural networks?**
A: Spark MLlib models are sufficient for this academic prototype, integrate natively with our pipeline, and are easier to explain in viva.

## System Design

**Q40: What is the technology stack?**
A: Python, Hadoop HDFS, Apache Spark/PySpark, Spark MLlib, Plotly, Streamlit, Parquet.

**Q41: What happens if HDFS is unavailable?**
A: The system falls back to local filesystem storage. HDFS implementation remains intact for production deployment.

**Q42: What are the limitations?**
A: Single-household data (not grid-level), local Spark mode (not cluster), recursive forecast error accumulation, no external factors (weather, holidays).

**Q43: What is future scope?**
A: Kafka for streaming ingestion, cluster deployment, weather features, deep learning models, multi-household aggregation, real-time dashboard updates.

**Q44: How is your project reproducible?**
A: Central config, versioned dependencies, automated pipeline script, documented setup, saved models and metrics as artifacts.

**Q45: Explain Spark's distributed processing.**
A: Spark partitions data across executors, applies transformations in parallel, and shuffles data only when needed (e.g., groupBy). The driver coordinates execution.
