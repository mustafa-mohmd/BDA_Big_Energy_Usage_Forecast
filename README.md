# Energy Consumption Forecasting Using Big Data Technologies

An end-to-end Big Data pipeline for forecasting household electricity consumption using **Hadoop HDFS**, **Apache Spark/PySpark**, and **Spark MLlib**, with an interactive **Streamlit** dashboard.

## Overview

This project demonstrates a scalable Big Data architecture for energy consumption forecasting. The pipeline ingests the UCI Individual Household Electric Power Consumption dataset (~2 million minute-level records), processes it with PySpark, trains multiple regression models, and generates 24-hour ahead forecasts.

> **Scalability Note:** The prototype uses a publicly available dataset of manageable size. The HDFS + Spark architecture can be extended to much larger smart-meter datasets in production deployments.

## Problem Statement

Electricity consumption varies significantly by hour, day, and season. Utilities need accurate demand forecasts to optimize generation and reduce costs. Traditional single-machine tools cannot scale to process millions of smart-meter readings. This project addresses that gap with a distributed Big Data pipeline.

## Objectives

1. Store raw energy data in Hadoop HDFS
2. Process data using Apache Spark/PySpark
3. Clean, aggregate, and engineer time-series features
4. Train and compare forecasting models (Linear Regression, Random Forest, GBT)
5. Evaluate models with MAE, RMSE, MAPE, R²
6. Generate 24-hour consumption forecasts
7. Visualize results in an interactive Streamlit dashboard

## Architecture

```
RAW DATA → INGESTION → HDFS → PySpark → CLEANING → FEATURE ENGINEERING
    → EDA → MODEL TRAINING → EVALUATION → BEST MODEL → 24-HOUR FORECAST → DASHBOARD
```

See [System Architecture](docs/architecture/system_architecture.md) for detailed diagrams.

## Technology Stack

| Layer | Technology |
|-------|-----------|
| Storage | Hadoop HDFS |
| Processing | Apache Spark / PySpark |
| ML | Spark MLlib |
| Visualization | Plotly, Matplotlib |
| Dashboard | Streamlit |
| Language | Python 3.10+ |
| Data Format | Parquet |

## Dataset

**UCI Individual Household Electric Power Consumption Dataset**

- Source: [UCI ML Repository](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption)
- ~2 million minute-level measurements
- Nearly 4 years (Dec 2006 – Nov 2010)
- Semicolon-delimited, missing values as "?"

## Project Structure

```
├── data/
│   ├── raw/                    # Raw dataset (download separately)
│   └── processed/              # Parquet files (generated)
├── hadoop/
│   ├── hdfs_commands_windows.txt
│   └── hdfs_commands_linux.txt
├── src/
│   ├── config.py               # Central configuration
│   ├── ingestion.py            # Data loading
│   ├── preprocessing.py        # Cleaning + hourly aggregation
│   ├── feature_engineering.py  # Lag and time features
│   ├── train_models.py         # MLlib model training
│   ├── evaluate.py             # Metrics and comparison
│   ├── forecast.py             # 24-hour forecasting
│   ├── eda.py                  # Exploratory analysis
│   └── utils.py                # Spark session, metrics, logging
├── dashboard/
│   └── app.py                  # Streamlit dashboard
├── scripts/
│   ├── download_dataset.py     # Dataset download
│   ├── run_pipeline.py         # Full pipeline runner
│   └── run_pipeline.bat        # Windows batch script
├── models/                     # Trained models (generated)
├── results/                    # Metrics, figures, predictions
├── docs/                       # Documentation, report, viva
├── tests/                      # pytest unit tests
├── requirements.txt
└── README.md
```

## System Requirements

- **OS:** Windows 10/11 (WSL2 recommended for Hadoop) or Linux
- **Python:** 3.10 or 3.11
- **Java:** JDK 8 or 11 (required for Spark)
- **RAM:** 8 GB minimum (16 GB recommended)
- **Disk:** 5 GB free

## Installation

### Windows (PowerShell)

```powershell
# Clone/navigate to project
cd G:\vscode\bda4th

# Create virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Verify Java (required for Spark)
java -version
```

### WSL / Linux

```bash
cd /path/to/bda4th
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Hadoop Setup

Hadoop on Windows is challenging. **WSL2 + Ubuntu is recommended.**

### WSL2 Setup

1. Install WSL2: `wsl --install`
2. Install Hadoop in Ubuntu ([Single Node Setup](https://hadoop.apache.org/docs/stable/hadoop-project-dist/hadoop-common/SingleCluster.html))
3. Start HDFS: `start-dfs.sh`
4. Create directories and upload data — see `hadoop/hdfs_commands_linux.txt`

### Without Hadoop

The pipeline runs in **local mode** by default. HDFS integration is available when Hadoop is configured:

```powershell
python scripts/run_pipeline.py --use-hdfs
```

## Spark Setup

Spark is included via PySpark (`pip install pyspark`). No separate Spark installation needed for local mode.

Ensure Java is installed:

```powershell
# Set JAVA_HOME if needed (Windows)
$env:JAVA_HOME = "C:\Program Files\Eclipse Adoptium\jdk-11.0.x"
```

## Dataset Setup

```powershell
python scripts/download_dataset.py
```

**Manual download** if automatic fails:
1. Visit [UCI Dataset Page](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption)
2. Download the ZIP file
3. Extract `household_power_consumption.txt` to `data/raw/`

## HDFS Setup

See command reference files:
- Windows: `hadoop/hdfs_commands_windows.txt`
- Linux/WSL: `hadoop/hdfs_commands_linux.txt`

Quick start (Linux/WSL):

```bash
hdfs dfs -mkdir -p /energy_forecasting/raw
hdfs dfs -put data/raw/household_power_consumption.txt /energy_forecasting/raw/
hdfs dfs -ls /energy_forecasting/raw/
```

## Running the Pipeline

```powershell
# Full pipeline (download → preprocess → train → forecast)
python scripts/run_pipeline.py

# Or use Windows batch script
scripts\run_pipeline.bat

# Skip download if dataset already exists
python scripts/run_pipeline.py --skip-download

# With HDFS
python scripts/run_pipeline.py --use-hdfs
```

Pipeline steps:
1. Download dataset
2. Preprocessing (clean + hourly aggregation)
3. Exploratory data analysis
4. Feature engineering
5. Model training (Linear Regression, Random Forest, GBT)
6. Model evaluation
7. 24-hour forecast generation

## Model Training

Three Spark MLlib models are trained:

| Model | Algorithm | Key Hyperparameters |
|-------|-----------|-------------------|
| Linear Regression | OLS with L2 reg | maxIter=100, regParam=0.1 |
| Random Forest | Ensemble trees | numTrees=50, maxDepth=10 |
| Gradient Boosting | Sequential boosting | maxIter=50, maxDepth=5 |

Data split: 70% train / 15% validation / 15% test (chronological, no shuffling).

## Forecasting

Recursive one-step-ahead forecasting for 24 hours:
- Uses lag features from history + prior predictions
- Does NOT use future actual values
- Output: `results/predictions/forecast_24h.csv`

## Running the Dashboard

```powershell
streamlit run dashboard/app.py
```

Open http://localhost:8501

Dashboard sections:
- **Home** — Project overview and pipeline
- **Dataset Overview** — Record counts, date range, statistics
- **Energy Analysis** — Interactive Plotly charts
- **Model Performance** — Comparison table and metrics
- **Forecast** — 24-hour prediction visualization
- **Insights** — Peak hours, weekday/weekend patterns

## Results

> Run the pipeline to generate actual results. Metrics are saved to `results/metrics/`.

After execution, find:
- `results/metrics/model_comparison.csv` — Model comparison table
- `results/metrics/model_results.json` — Full metrics
- `results/metrics/eda_insights.json` — Data insights
- `results/predictions/forecast_24h.csv` — 24-hour forecast
- `results/figures/` — EDA visualizations

## Screenshots

> Capture screenshots after running the dashboard. Place in `docs/screenshots/`.

## Testing

```powershell
pytest tests/ -v
```

## Limitations

- Single-household data (not grid-level aggregation)
- Local Spark mode (not multi-node cluster)
- Recursive forecast accumulates error over horizon
- No external features (weather, holidays)
- Dataset is moderate size (~2M records), not true "Big Data" volume

## Future Scope

- Apache Kafka for real-time streaming ingestion
- Multi-node Spark cluster deployment
- Weather and calendar feature integration
- Deep learning models (LSTM, Transformer)
- Multi-household grid-level forecasting
- Cloud deployment (AWS EMR, Azure HDInsight)

## Viva Topics

See [docs/viva/viva_questions.md](docs/viva/viva_questions.md) for 45+ questions covering:
- Big Data concepts (HDFS, Spark, PySpark)
- Data preprocessing and feature engineering
- Machine learning models and evaluation
- Time-series forecasting methodology
- System architecture and scalability

## Authors

- [Your Name] — Computer Science, 4th Year
- [College Name]

## License

MIT License — see [LICENSE](LICENSE)
