@echo off
REM Windows batch script to run the Energy Forecasting pipeline
REM Usage: scripts\run_pipeline.bat

echo ============================================
echo Energy Consumption Forecasting Pipeline
echo ============================================

cd /d "%~dp0.."

REM Activate virtual environment if it exists
if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
)

python scripts\download_dataset.py
if errorlevel 1 (
    echo Dataset download failed. Check manual instructions.
    pause
    exit /b 1
)

python scripts\run_pipeline.py
if errorlevel 1 (
    echo Pipeline failed.
    pause
    exit /b 1
)

echo.
echo Pipeline complete! Starting dashboard...
streamlit run dashboard\app.py

pause
