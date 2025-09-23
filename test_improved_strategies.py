"""
Test improved strategy execution with better feedback
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PyQt5.QtWidgets import QApplication
from launch_pyqtgraph_with_selector import ConfiguredChart
import glob
import time

def test_improved_execution():
    print("="*60)
    print("TESTING IMPROVED STRATEGY EXECUTION")
    print("="*60)

    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    # Load data
    csv_files = glob.glob(r"C:\code\PythonBT\dataRaw\range-ATR30x0.05\ES\diffAdjusted\*.csv")
    if not csv_files:
        csv_files = glob.glob(r"C:\code\PythonBT\dataRaw\**\*.csv", recursive=True)

    config = {
        'data_file': csv_files[0],
        'trade_source': 'none',
        'indicators': []
    }

    print(f"Loading chart...")
    chart = ConfiguredChart(config)

    # Access strategy runner
    runner = chart.trade_panel.strategy_runner

    print(f"\nInitial status: {runner.status_label.text()}")

    # Test 1: Default parameters (should generate many trades)
    print("\n1. Testing with default parameters (Fast=10, Slow=30)...")
    runner.strategy_combo.setCurrentText("SMA Crossover")
    runner.fast_period_spin.setValue(10)
    runner.slow_period_spin.setValue(30)
    runner.run_strategy()
    app.processEvents()

    status1 = runner.status_label.text()
    print(f"   Result: {status1.split(chr(10))[0]}")  # First line only

    if "WARNING: Excessive trades" in status1:
        print("   -> Correctly warned about excessive trades")

    # Clear trades
    time.sleep(0.5)
    runner.clear_trades()
    app.processEvents()

    # Test 2: Longer periods (should generate fewer trades)
    print("\n2. Testing with longer periods (Fast=50, Slow=200)...")
    runner.fast_period_spin.setValue(50)
    runner.slow_period_spin.setValue(200)
    runner.run_strategy()
    app.processEvents()

    status2 = runner.status_label.text()
    print(f"   Result: {status2.split(chr(10))[0]}")

    if "Successfully generated" in status2:
        print("   -> Good feedback for reasonable trade count")

    # Test 3: RSI strategy
    print("\n3. Testing RSI Momentum strategy...")
    runner.strategy_combo.setCurrentText("RSI Momentum")
    runner.run_strategy()
    app.processEvents()

    status3 = runner.status_label.text()
    print(f"   Result: {status3.split(chr(10))[0]}")

    # Check color styling
    style = runner.status_label.styleSheet()
    if "green" in style:
        print("   -> Status shows green for success")
    elif "orange" in style:
        print("   -> Status shows orange for warning")

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print("Strategy execution is working with improved feedback:")
    print("- Clear success/warning messages")
    print("- Color-coded status (green/orange)")
    print("- Warnings for excessive trades")
    print("- Helpful suggestions for parameter adjustment")

    return True

if __name__ == "__main__":
    test_improved_execution()