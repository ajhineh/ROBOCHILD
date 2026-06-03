@echo off
title ROBOCHILD TensorBoard Launcher
color 0e
echo =========================================================================
echo                  📊 ROBOCHILD TENSORBOARD LAUNCHER 📊
echo =========================================================================
echo.

:: Check if port 7007 is already in use
netstat -ano | findstr LISTENING | findstr :7007 >nul
if %errorlevel% equ 0 (
    echo [INFO] TensorBoard is already running in the background!
    echo [INFO] Opening TensorBoard dashboard in your browser...
    echo.
    start http://localhost:7007
    timeout /t 3 >nul
    exit
)

echo [INFO] Port 7007 is free. Starting TensorBoard...
echo.

:: Detect virtual environment if it exists, otherwise fall back to global python
set PYTHON_BIN=python
if exist venv\Scripts\python.exe (
    set PYTHON_BIN=venv\Scripts\python.exe
) else if exist .venv\Scripts\python.exe (
    set PYTHON_BIN=.venv\Scripts\python.exe
)

:: Start TensorBoard in a minimized or background window using Python module
start /min "TensorBoard Server" %PYTHON_BIN% -m tensorboard.main --logdir=tb_logs --port=7007 --host=localhost

:: Wait a brief moment for TensorBoard to initialize
echo [INFO] Initializing TensorBoard server...
timeout /t 3 >nul

:: Open the browser to display the dashboard
echo [INFO] Opening TensorBoard in your browser...
start http://localhost:7007

echo.
echo =========================================================================
echo [SUCCESS] TensorBoard is running in the background!
echo [SUCCESS] You can safely close this launcher window.
echo [SUCCESS] (The actual TensorBoard process is running minimized in your taskbar)
echo =========================================================================
timeout /t 5 >nul
exit
