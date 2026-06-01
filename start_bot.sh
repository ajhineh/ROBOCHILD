#!/bin/bash
# ROBORDER Autostart Script with Concurrency Protection & Background Daemon Support

# Get directory of this script
CDIR="$(cd "$(dirname "$0")" && pwd)"
cd "$CDIR"

# Detect if port 3000 is already active
if netstat -ano 2>/dev/null | grep -q "LISTENING" | grep -q ":3000" || lsof -i :3000 -t >/dev/null 2>&1; then
    echo "[INFO] ROBORDER-X is already running in the background!"
    echo "[INFO] Opening the interactive dashboard in your browser..."
    if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" ]]; then
        start http://localhost:3000
    else
        xdg-open http://localhost:3000 || open http://localhost:3000
    fi
    exit 0
fi

echo "[INFO] Starting ROBORDER-X in the background..."

# Check OS type
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" ]]; then
    # Windows Git Bash / Cygwin
    PYTHON_BIN="pythonw"
    if [ -f "venv/Scripts/pythonw.exe" ]; then
        PYTHON_BIN="venv/Scripts/pythonw.exe"
    elif [ -f ".venv/Scripts/pythonw.exe" ]; then
        PYTHON_BIN=".venv/Scripts/pythonw.exe"
    fi
    
    # Start completely detached
    "$PYTHON_BIN" -m src.main &
    sleep 3
    start http://localhost:3000
else
    # Linux / macOS
    PYTHON_BIN="python3"
    if [ -d "venv" ]; then
        source venv/bin/activate
        PYTHON_BIN="venv/bin/python"
    elif [ -d ".venv" ]; then
        source .venv/bin/activate
        PYTHON_BIN=".venv/bin/python"
    fi
    
    # Run in background detached with disown and stdin redirection
    nohup "$PYTHON_BIN" -m src.main > roborder_x.log 2>&1 </dev/null & disown
    sleep 3
    # تنها در صورتی مرورگر را باز کن که سرور دارای رابط گرافیکی باشد (نه سرور ابری AWS)
    if [ -n "$DISPLAY" ] || [ -n "$WAYLAND_DISPLAY" ]; then
        xdg-open http://localhost:3000 >/dev/null 2>&1 || open http://localhost:3000 >/dev/null 2>&1
    fi
fi

echo "[SUCCESS] ROBORDER-X has been successfully launched in the background!"
