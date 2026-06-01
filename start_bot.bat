@echo off
title ROBORDER-X Launcher
color 0b
echo =========================================================================
echo                       🤖 ROBORDER-X LAUNCHER 🤖
echo =========================================================================
echo.

:: Check if port 3000 is already in use by ROBORDER-X
netstat -ano | findstr LISTENING | findstr :3000 >nul
if %errorlevel% equ 0 (
    echo [INFO] ROBORDER-X is already running in the background!
    echo [INFO] Opening the interactive dashboard in your browser...
    echo.
    start http://localhost:3000
    timeout /t 3 >nul
    exit
)

echo [INFO] Port 3000 is free. Starting ROBORDER-X in the background...
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
start http://localhost:3000

echo.
echo =========================================================================
echo [SUCCESS] ROBORDER-X is running in the background!
echo [SUCCESS] You can safely close this terminal window.
echo [SUCCESS] To stop the bot process completely, click the "Shutdown"
echo           button inside the web dashboard.
echo =========================================================================
timeout /t 5 >nul
exit
