@echo off
title Sales Prediction Model — Setup & Launch
color 0A

echo ============================================================
echo   Sales Prediction Model — Windows Launcher
echo ============================================================
echo.

:: Check Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python not found. Please install Python 3.9+ from https://python.org
    pause
    exit /b 1
)

echo [1/4] Installing Python dependencies...
python -m pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo [ERROR] Dependency installation failed. Check requirements.txt.
    pause
    exit /b 1
)
echo       Done.
echo.

echo [2/4] Running model training pipeline (main.py)...
python main.py
if %errorlevel% neq 0 (
    echo [ERROR] Training pipeline failed. See error output above.
    pause
    exit /b 1
)
echo       Training complete.
echo.

echo [3/4] Verifying output files...
if not exist "output\best_sales_model.joblib" (
    echo [WARNING] Model file not found — training may have failed silently.
)
if not exist "output\graphs\actual_vs_predicted.png" (
    echo [WARNING] Graph files not found.
)
echo       Verification done.
echo.

echo [4/4] Launching Streamlit dashboard...
echo       Open http://localhost:8501 in your browser.
echo       Press Ctrl+C in this window to stop the server.
echo.
streamlit run app.py

pause
