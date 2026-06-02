@echo off
title ROBOCHILD Launcher
color 0b
echo =========================================================================
echo                       🤖 ROBOCHILD LAUNCHER 🤖
echo =========================================================================
echo.

:: Check if port 6006 is already in use by ROBORDER-X
netstat -ano | findstr LISTENING | findstr :6006 >nul
if %errorlevel% equ 0 (
    echo [INFO] ROBOCHILD is already running in the background!
    echo [INFO] Opening the interactive dashboard in your browser...
    echo.
    start http://localhost:6006
    timeout /t 3 >nul
    exit
)

echo [INFO] Port 6006 is free. Starting ROBOCHILD in the background...
echo.

:: Detect virtual environment if it exists, otherwise fall back to global pythonw
set PYTHON_BIN=pythonw
if exist venv\Scripts\pythonw.exe (
    set PYTHON_BIN=venv\Scripts\pythonw.exe
) else if exist .venv\Scripts\pythonw.exe (
    set PYTHON_BIN=.venv\Scripts\pythonw.exe
)

:: Start the bot in the background using pythonw (detached and windowless)
start "" %PYTHON_BIN% -m src.main

:: Wait a brief moment for the server to initialize and bind to the port
echo [INFO] Initializing trading engine and web dashboard...
timeout /t 3 >nul

:: Open the browser to display the dashboard
echo [INFO] Opening the interactive dashboard...
start http://localhost:6006

echo.
echo =========================================================================
echo [SUCCESS] ROBOCHILD is running in the background!
echo [SUCCESS] You can safely close this terminal window.
echo [SUCCESS] To stop the bot process completely, click the "Shutdown"
echo           button inside the web dashboard.
echo =========================================================================
timeout /t 5 >nul
exit
