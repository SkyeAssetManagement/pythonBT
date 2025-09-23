"""
Test script to verify hover data and trade generation
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PyQt5.QtWidgets import QApplication
from launch_pyqtgraph_with_selector import ConfiguredChart
import pandas as pd

def test_hover_and_trades():
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    # Configure with real data
    import glob
    csv_files = glob.glob(r"C:\code\PythonBT\dataRaw\**\*.csv", recursive=True)
    if not csv_files:
        print("ERROR: No data files found")
        return False

    config = {
        'data_file': csv_files[0],
        'trade_source': 'system',
        'system': 'SMA Crossover'
    }

    print(f"Testing with: {config['data_file']}")

    # Create chart
    chart = ConfiguredChart(config)

    # Test hover data
    if chart.full_data is None:
        print("ERROR: full_data is None!")
        return False

    print(f"SUCCESS: full_data loaded with keys: {list(chart.full_data.keys())}")

    # Test data access
    if 'timestamp' in chart.full_data:
        print(f"  timestamp[0]: {chart.full_data['timestamp'][0]}")
    if 'open' in chart.full_data:
        print(f"  open[0]: {chart.full_data['open'][0]}")

    # Test trade generation
    if hasattr(chart, 'current_trades'):
        print(f"Trades generated: {len(chart.current_trades)}")

    return True

if __name__ == "__main__":
    success = test_hover_and_trades()
    print(f"\nTest result: {'SUCCESS' if success else 'FAILED'}")
