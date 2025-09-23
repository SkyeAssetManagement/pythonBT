"""
Test strategy execution from user perspective
"""
import sys
import os
import time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PyQt5.QtWidgets import QApplication, QMessageBox
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtTest import QTest
from launch_pyqtgraph_with_selector import ConfiguredChart
import glob

def simulate_user_workflow():
    """Simulate what a user would do to run a strategy"""
    print("="*60)
    print("SIMULATING USER WORKFLOW FOR STRATEGY EXECUTION")
    print("="*60)

    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    # Load data like user would
    csv_files = glob.glob(r"C:\code\PythonBT\dataRaw\range-ATR30x0.05\ES\diffAdjusted\*.csv")
    if not csv_files:
        csv_files = glob.glob(r"C:\code\PythonBT\dataRaw\**\*.csv", recursive=True)

    config = {
        'data_file': csv_files[0],
        'trade_source': 'none',  # User starts with no trades
        'indicators': []
    }

    print(f"1. Loading chart with data: {os.path.basename(config['data_file'])}")
    chart = ConfiguredChart(config)
    chart.show()

    # Allow chart to render
    app.processEvents()
    time.sleep(0.5)

    print(f"   - Data loaded: {chart.total_bars} bars")
    print(f"   - Initial trades: {len(chart.current_trades) if hasattr(chart, 'current_trades') else 0}")

    # User would click on "Run Strategy" tab
    print("\n2. Accessing 'Run Strategy' tab...")

    if hasattr(chart.trade_panel, 'tab_widget'):
        tab_widget = chart.trade_panel.tab_widget

        # Find the "Run Strategy" tab
        run_strategy_index = -1
        for i in range(tab_widget.count()):
            if "Run Strategy" in tab_widget.tabText(i):
                run_strategy_index = i
                break

        if run_strategy_index >= 0:
            print(f"   Found 'Run Strategy' tab at index {run_strategy_index}")
            tab_widget.setCurrentIndex(run_strategy_index)
            app.processEvents()

            # Get the strategy runner
            runner = chart.trade_panel.strategy_runner

            # Check initial state
            print("\n3. Checking strategy runner state...")
            print(f"   - Has data: {runner.chart_data is not None}")
            if runner.chart_data is not None:
                print(f"   - Data shape: {runner.chart_data.shape}")

            # User selects strategy and parameters
            print("\n4. Setting strategy parameters...")
            print("   - Strategy: SMA Crossover")
            print("   - Fast Period: 10")
            print("   - Slow Period: 30")
            print("   - Long Only: True")

            runner.strategy_combo.setCurrentText("SMA Crossover")
            runner.fast_period_spin.setValue(10)
            runner.slow_period_spin.setValue(30)
            runner.long_only_check.setChecked(True)
            app.processEvents()

            # User clicks "Run Strategy" button
            print("\n5. Clicking 'Run Strategy' button...")

            initial_status = runner.status_label.text()
            print(f"   Initial status: {initial_status}")

            # Click the button
            runner.run_button.click()
            app.processEvents()
            time.sleep(0.5)  # Give time for strategy to execute

            # Check results
            final_status = runner.status_label.text()
            print(f"   Final status: {final_status}")

            # Check if trades were generated
            if hasattr(chart, 'current_trades'):
                print(f"\n6. Results:")
                print(f"   - Trades generated: {len(chart.current_trades)}")
                if len(chart.current_trades) > 0:
                    print(f"   - First trade: Bar {chart.current_trades[0].bar_index} @ ${chart.current_trades[0].price:.2f}")
                    print(f"   - Last trade: Bar {chart.current_trades[-1].bar_index} @ ${chart.current_trades[-1].price:.2f}")

            # Check trade panel display
            print("\n7. Checking trade panel display...")
            trade_list = chart.trade_panel.trades
            print(f"   - Trades in panel: {len(trade_list)}")

            # Switch back to trades tab to see them
            trades_index = -1
            for i in range(tab_widget.count()):
                if tab_widget.tabText(i) == "Trades":
                    trades_index = i
                    break

            if trades_index >= 0:
                tab_widget.setCurrentIndex(trades_index)
                app.processEvents()
                print("   - Switched to Trades tab")

            # Check if there's an error or warning
            if "Error" in final_status:
                print(f"\n   ERROR DETECTED: {final_status}")
                return False
            elif len(chart.current_trades) == 0:
                print("\n   WARNING: No trades generated!")
                return False
            else:
                print(f"\n   SUCCESS: {len(chart.current_trades)} trades generated and displayed!")
                return True

        else:
            print("   ERROR: 'Run Strategy' tab not found!")
            return False
    else:
        print("   ERROR: No tab widget found!")
        return False

    chart.close()

def test_strategy_feedback():
    """Test if strategy provides proper feedback to user"""
    print("\n" + "="*60)
    print("TESTING USER FEEDBACK")
    print("="*60)

    from PyQt5.QtWidgets import QApplication
    from src.trading.visualization.strategy_runner import StrategyRunner
    import pandas as pd
    import numpy as np

    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    # Create runner with minimal data
    runner = StrategyRunner()

    # Test with very small dataset
    small_data = {
        'timestamp': pd.date_range(start='2024-01-01', periods=50, freq='1min'),
        'open': np.ones(50) * 100,
        'high': np.ones(50) * 101,
        'low': np.ones(50) * 99,
        'close': np.ones(50) * 100,
        'volume': np.ones(50) * 1000
    }

    runner.set_chart_data(small_data)

    print("Testing with small dataset (50 bars)...")

    # Test each strategy
    strategies = ["SMA Crossover", "RSI Momentum"]

    for strategy_name in strategies:
        print(f"\n{strategy_name}:")
        runner.strategy_combo.setCurrentText(strategy_name)

        # Run strategy
        runner.run_strategy()

        # Get status
        status = runner.status_label.text()
        print(f"  Status: {status}")

        if "Error" in status:
            print(f"  ERROR found in status!")
        elif "0 trades" in status or "Generated 0" in status:
            print(f"  WARNING: No trades generated")
        else:
            print(f"  OK: Status looks good")

    return True

def main():
    success1 = simulate_user_workflow()
    success2 = test_strategy_feedback()

    print("\n" + "="*60)
    print("USER EXPERIENCE TEST SUMMARY")
    print("="*60)

    if success1 and success2:
        print("Strategy execution works from user perspective!")
        print("\nIf user reports issues, possible causes:")
        print("1. User expects immediate visual feedback")
        print("2. Too many trades cluttering the display")
        print("3. Status message not clear enough")
        print("4. User doesn't see 'Run Strategy' tab")
    else:
        print("Issues found - see details above")

    return success1 and success2

if __name__ == "__main__":
    main()