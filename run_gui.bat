@echo off
echo ============================================================
echo OMtree Trading Model - GUI Launcher
echo ============================================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8 or higher
    pause
    exit /b 1
)

REM Launch the GUI
echo Starting GUI...
echo.
python OMtree_gui.py

REM Check if GUI exited with error
if errorlevel 1 (
    echo.
    echo ============================================================
    echo ERROR: GUI exited with an error
    echo ============================================================
    echo.
    echo Possible issues:
    echo 1. Missing dependencies - try: pip install -r requirements.txt
    echo 2. File not found - ensure you're in the OM3 directory
    echo 3. Import errors - check the error message above
    echo.
    pause
    exit /b 1
)

REM GUI closed normally
echo.
echo ============================================================
echo GUI closed successfully
echo ============================================================
exit /b 0