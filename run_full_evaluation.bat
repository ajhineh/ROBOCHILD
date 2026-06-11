@echo off
echo ============================================================
echo   ROBOCHILD AI Evaluator & Analyzer Launcher
echo ============================================================
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1

echo 1. Verifying environment...
if not exist venv\Scripts\python.exe (
    echo [ERROR] Virtual environment not found at .\venv
    pause
    exit /b 1
)

echo 2. Running AI Analyzer evaluator...
venv\Scripts\python.exe src/analysis/training_evaluator.py
if %errorlevel% neq 0 (
    echo [ERROR] Evaluator run failed. Please check connection or configs.
    pause
    exit /b %errorlevel%
)

echo [SUCCESS] Evaluation completed successfully.
pause
