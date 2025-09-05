@echo off
echo ========================================
echo OMtree Trading System - Setup and Run
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8 or higher from https://www.python.org
    pause
    exit /b 1
)

echo Step 1: Installing dependencies...
echo.
pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install dependencies
    pause
    exit /b 1
)

echo.
echo Step 2: Setting up directories and sample data...
echo.
python setup.py
if errorlevel 1 (
    echo WARNING: Setup script encountered issues
    echo You may need to manually configure some settings
)

echo.
echo ========================================
echo Setup Complete!
echo ========================================
echo.
echo What would you like to do?
echo 1. Run GUI Interface
echo 2. Run Walk-Forward Validation
echo 3. View Documentation
echo 4. Exit
echo.
set /p choice="Enter your choice (1-4): "

if "%choice%"=="1" (
    echo Starting GUI...
    python OMtree_gui.py
) else if "%choice%"=="2" (
    echo Running Walk-Forward Validation...
    python OMtree_walkforward.py
) else if "%choice%"=="3" (
    echo Opening documentation...
    start HOW-TO-GUIDE.md
) else (
    echo Exiting...
)

pause