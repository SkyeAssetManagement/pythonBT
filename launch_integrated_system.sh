#!/bin/bash
# Launch Integrated Trading System
# =================================

echo "======================================="
echo "  INTEGRATED TRADING SYSTEM LAUNCHER"
echo "======================================="
echo

# Check if Python is installed
if ! command -v python3 &> /dev/null
then
    echo "ERROR: Python 3 is not installed"
    echo "Please install Python 3.8 or higher"
    exit 1
fi

echo "Starting Integrated Trading System..."
echo

# Launch the integrated GUI
python3 integrated_trading_launcher.py

if [ $? -ne 0 ]; then
    echo
    echo "ERROR: Failed to launch the integrated system"
    echo "Please check the error messages above"
else
    echo
    echo "System closed successfully"
fi