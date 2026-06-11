#!/bin/bash
echo "============================================================"
echo "  ROBOCHILD AI Evaluator & Analyzer Launcher"
echo "============================================================"
export PYTHONIOENCODING=utf-8
export PYTHONUNBUFFERED=1

echo "1. Verifying environment..."
if [ ! -f "venv/Scripts/python" ] && [ ! -f "venv/bin/python" ]; then
    echo "[ERROR] Virtual environment not found."
    exit 1
fi

PYTHON_EXEC="venv/Scripts/python"
if [ -f "venv/bin/python" ]; then
    PYTHON_EXEC="venv/bin/python"
fi

echo "2. Running AI Analyzer evaluator..."
$PYTHON_EXEC src/analysis/training_evaluator.py
if [ $? -ne 0 ]; then
    echo "[ERROR] Evaluator run failed."
    exit 1
fi

echo "[SUCCESS] Evaluation completed successfully."
