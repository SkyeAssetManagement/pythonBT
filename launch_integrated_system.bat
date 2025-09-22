@echo off
REM Launch Integrated Trading System
REM =================================

echo ==========================================
echo   SAM - INTEGRATED BACKTESTING SYSTEM
echo ==========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8 or higher
    pause
    exit /b 1
)

echo Starting SAM - Integrated Backtesting System...
echo.

REM Change to the script directory
cd /d "%~dp0"

REM Launch the integrated GUI
python integrated_trading_launcher.py

if errorlevel 1 (
    echo.
    echo ERROR: Failed to launch the integrated system
    echo Please check the error messages above
    pause
) else (
    echo.
    echo System closed successfully
)