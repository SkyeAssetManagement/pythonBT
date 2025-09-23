"""
Test that strategy runner works in unified system
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PyQt5.QtWidgets import QApplication
import time

def test_unified_system_strategy_runner():
    """Test strategy runner in unified system"""
    print("="*60)
    print("TESTING STRATEGY RUNNER IN UNIFIED SYSTEM")
    print("="*60)

    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    # Import and run the unified system
    from launch_unified_system import main as unified_main, UnifiedConfiguredChart

    # Find a test data file
    import glob
    csv_files = glob.glob(r"C:\code\PythonBT\dataRaw\range-ATR30x0.05\ES\diffAdjusted\*.csv")
    if not csv_files:
        csv_files = glob.glob(r"C:\code\PythonBT\dataRaw\**\*.csv", recursive=True)

    config = {
        'data_file': csv_files[0] if csv_files else None,
        'trade_source': 'none',
        'indicators': []
    }

    print(f"\n1. Creating unified chart with data...")
    if config['data_file']:
        print(f"   Using data file: {os.path.basename(config['data_file'])}")
    else:
        print("   Using sample data (no data file found)")

    chart = UnifiedConfiguredChart(config)

    # Let it initialize
    app.processEvents()
    time.sleep(0.5)

    print(f"\n2. Checking chart data loaded...")
    print(f"   - Chart has data: {chart.full_data is not None}")
    if chart.full_data:
        print(f"   - Number of bars: {len(chart.full_data.get('timestamp', []))}")

    print(f"\n3. Checking trade panel...")
    print(f"   - Trade panel exists: {chart.trade_panel is not None}")

    if chart.trade_panel:
        # Check if strategy runner has data
        if hasattr(chart.trade_panel, 'strategy_runner'):
            runner = chart.trade_panel.strategy_runner

            print(f"\n4. Checking strategy runner...")
            print(f"   - Strategy runner exists: {runner is not None}")

            if runner:
                print(f"   - Has chart_data: {runner.chart_data is not None}")

                if runner.chart_data is not None:
                    print(f"   - Data shape: {runner.chart_data.shape}")
                    print(f"   - Data columns: {runner.chart_data.columns.tolist()}")

                    # Get initial status
                    status = runner.status_label.text()
                    print(f"\n5. Strategy runner status:")
                    print(f"   {status}")

                    # Try to run a strategy
                    print(f"\n6. Testing strategy execution...")
                    runner.strategy_combo.setCurrentText("SMA Crossover")
                    runner.fast_period_spin.setValue(20)
                    runner.slow_period_spin.setValue(50)

                    # Click run
                    runner.run_button.click()
                    app.processEvents()
                    time.sleep(1)

                    # Check final status
                    final_status = runner.status_label.text()
                    print(f"   Result: {final_status}")

                    if "No chart data available" in final_status:
                        print("\n   ERROR: Still showing 'No chart data available'!")
                        return False
                    else:
                        print("\n   SUCCESS: Strategy runner has data and can execute!")
                        return True
                else:
                    print("\n   ERROR: Strategy runner has no chart_data!")
                    return False
            else:
                print("\n   ERROR: No strategy runner found!")
                return False
        else:
            print("\n   ERROR: Trade panel has no strategy_runner attribute!")
            return False
    else:
        print("\n   ERROR: No trade panel found!")
        return False

def main():
    success = test_unified_system_strategy_runner()

    print("\n" + "="*60)
    if success:
        print("SUCCESS: Strategy runner works in unified system!")
        print("\nThe fix applied:")
        print("- Added pass_data_to_trade_panel() method")
        print("- Calls trade_panel.set_bar_data() with chart data")
        print("- This connects the chart data to the strategy runner")
    else:
        print("FAILED: Strategy runner still not working")
        print("Check the error messages above for details")

    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)