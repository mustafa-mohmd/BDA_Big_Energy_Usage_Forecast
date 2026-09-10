# Project Workflow

```mermaid
flowchart TD
    START([Start]) --> SETUP[Install Dependencies]
    SETUP --> DOWNLOAD[Download Dataset]
    DOWNLOAD --> HDFS_SETUP{HDFS Setup?}
    HDFS_SETUP -->|Yes| UPLOAD[Upload to HDFS]
    HDFS_SETUP -->|No| LOCAL[Use Local Storage]
    UPLOAD --> PIPELINE[Run Pipeline]
    LOCAL --> PIPELINE
    PIPELINE --> PREPROC[Preprocessing]
    PREPROC --> FEAT[Feature Engineering]
    FEAT --> EDA[EDA]
    EDA --> TRAIN[Model Training]
    TRAIN --> EVAL[Evaluation]
    EVAL --> FORECAST[24h Forecast]
    FORECAST --> DASHBOARD[Launch Dashboard]
    DASHBOARD --> END([Demo Ready])
```

## Execution Commands

```bash
# 1. Setup
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt

# 2. Download data
python scripts/download_dataset.py

# 3. Run pipeline (local mode)
python scripts/run_pipeline.py

# 4. Launch dashboard
streamlit run dashboard/app.py
```
