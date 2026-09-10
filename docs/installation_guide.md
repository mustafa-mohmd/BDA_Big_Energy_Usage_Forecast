# Installation Guide

## System Requirements

- **OS**: Windows 10/11 (with WSL2 recommended for Hadoop) or Linux
- **Python**: 3.10 or 3.11
- **Java**: JDK 8 or 11 (required for Spark)
- **RAM**: 8 GB minimum (16 GB recommended)
- **Disk**: 5 GB free space

## Step 1: Python Environment

### Windows (PowerShell)

```powershell
cd G:\vscode\bda4th
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### WSL / Linux

```bash
cd /mnt/g/vscode/bda4th
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Step 2: Java Setup

Spark requires Java. Verify:

```powershell
java -version
```

If not installed, download JDK 11 from [Adoptium](https://adoptium.net/) and set:

```powershell
$env:JAVA_HOME = "C:\Program Files\Eclipse Adoptium\jdk-11.0.x"
```

## Step 3: Dataset Download

```powershell
python scripts/download_dataset.py
```

Manual fallback: Download from [UCI](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption) and place at `data/raw/household_power_consumption.txt`.

## Step 4: Hadoop Setup (Optional)

### Recommended: WSL2 + Ubuntu

1. Install WSL2: `wsl --install`
2. Install Hadoop in Ubuntu following [Apache docs](https://hadoop.apache.org/docs/stable/hadoop-project-dist/hadoop-common/SingleCluster.html)
3. Start HDFS: `start-dfs.sh`
4. Create directories: see `hadoop/hdfs_commands_linux.txt`
5. Upload data: `hdfs dfs -put data/raw/household_power_consumption.txt /energy_forecasting/raw/`
6. Run with HDFS: `python scripts/run_pipeline.py --use-hdfs`

### Without Hadoop

The pipeline works in local mode by default. No Hadoop setup required for development.

## Step 5: Run Pipeline

```powershell
python scripts/run_pipeline.py
```

Or use the batch script:

```powershell
scripts\run_pipeline.bat
```

## Step 6: Launch Dashboard

```powershell
streamlit run dashboard/app.py
```

Open http://localhost:8501 in your browser.

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Java not found | Install JDK 11, set JAVA_HOME |
| Spark memory error | Reduce data or increase driver memory in config |
| Dataset not found | Run download script or manual download |
| Port 8501 in use | `streamlit run dashboard/app.py --server.port 8502` |
