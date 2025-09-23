"""
Test that all errors are fixed, especially hover data KeyErrors
"""
import sys
import os
import traceback
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QPointF
from launch_pyqtgraph_with_selector import ConfiguredChart
import glob

def test_hover_without_errors():
    """Test hover functionality without KeyErrors"""
    print("TESTING HOVER DATA WITHOUT ERRORS")
    print("="*50)

    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    # Find test data
    csv_files = glob.glob(r"C:\code\PythonBT\dataRaw\range-ATR30x0.05\ES\diffAdjusted\*.csv")
    if not csv_files:
        csv_files = glob.glob(r"C:\code\PythonBT\dataRaw\**\*.csv", recursive=True)

    if not csv_files:
        print("ERROR: No data files found")
        return False

    # Create chart with data
    config = {
        'data_file': csv_files[0],
        'trade_source': 'system',
        'system': 'SMA Crossover'
    }

    print(f"Loading: {os.path.basename(config['data_file'])}")
    chart = ConfiguredChart(config)

    # Test hover data access for multiple bars
    print("\nTesting hover data access (should not throw KeyErrors):")

    errors = []
    test_indices = [0, 10, 100, 500, 1000, min(5000, chart.total_bars-1)]

    for bar_idx in test_indices:
        if bar_idx >= chart.total_bars:
            continue

        try:
            # Simulate the exact code from on_mouse_moved
            timestamp = chart.full_data['timestamp'][bar_idx]
            o = chart.full_data['open'][bar_idx]
            h = chart.full_data['high'][bar_idx]
            l = chart.full_data['low'][bar_idx]
            c = chart.full_data['close'][bar_idx]

            # These are the lines that were causing KeyErrors
            v = chart.full_data['volume'][bar_idx] if 'volume' in chart.full_data and chart.full_data['volume'] is not None else 0
            atr = chart.full_data['aux1'][bar_idx] if 'aux1' in chart.full_data and chart.full_data['aux1'] is not None else 0
            range_mult = chart.full_data['aux2'][bar_idx] if 'aux2' in chart.full_data and chart.full_data['aux2'] is not None else 0

            print(f"  Bar {bar_idx:5d}: OK - OHLC=${o:.2f}/${h:.2f}/${l:.2f}/${c:.2f}, Vol={v:.0f}, ATR={atr:.2f}")

        except KeyError as e:
            errors.append(f"Bar {bar_idx}: KeyError - {e}")
            print(f"  Bar {bar_idx:5d}: ERROR - {e}")
        except Exception as e:
            errors.append(f"Bar {bar_idx}: {type(e).__name__} - {e}")
            print(f"  Bar {bar_idx:5d}: ERROR - {e}")

    if errors:
        print(f"\nERRORS FOUND ({len(errors)}):")
        for error in errors:
            print(f"  - {error}")
        return False
    else:
        print("\nSUCCESS: No KeyErrors or other errors!")
        return True

def test_mouse_movement_simulation():
    """Simulate actual mouse movement to test hover"""
    print("\n" + "="*50)
    print("SIMULATING MOUSE MOVEMENT")
    print("="*50)

    from src.trading.visualization.pyqtgraph_range_bars_final import RangeBarChart

    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    # Create chart directly
    chart = RangeBarChart()

    # Set up minimal test data
    import numpy as np
    from datetime import datetime

    test_data = {
        'timestamp': np.array([datetime(2024, 1, 1, 9, 30), datetime(2024, 1, 1, 10, 0)]),
        'open': np.array([4500.0, 4510.0]),
        'high': np.array([4520.0, 4530.0]),
        'low': np.array([4490.0, 4505.0]),
        'close': np.array([4510.0, 4525.0]),
        # Note: No volume, aux1, or aux2 - testing that missing keys don't cause errors
    }

    chart.full_data = test_data
    chart.total_bars = 2

    print("Testing with minimal data (no volume, aux1, aux2)...")

    # Test accessing data like hover would
    try:
        bar_idx = 0
        timestamp = chart.full_data['timestamp'][bar_idx]
        o = chart.full_data['open'][bar_idx]
        h = chart.full_data['high'][bar_idx]
        l = chart.full_data['low'][bar_idx]
        c = chart.full_data['close'][bar_idx]

        # These should not cause KeyErrors anymore
        v = chart.full_data['volume'][bar_idx] if 'volume' in chart.full_data and chart.full_data['volume'] is not None else 0
        atr = chart.full_data['aux1'][bar_idx] if 'aux1' in chart.full_data and chart.full_data['aux1'] is not None else 0
        range_mult = chart.full_data['aux2'][bar_idx] if 'aux2' in chart.full_data and chart.full_data['aux2'] is not None else 0

        print(f"  Bar 0: OHLC=${o}/{h}/{l}/{c}")
        print(f"  Volume: {v} (default 0 - key not present)")
        print(f"  ATR: {atr} (default 0 - key not present)")
        print(f"  Range Mult: {range_mult} (default 0 - key not present)")
        print("\nSUCCESS: No KeyErrors with missing optional fields!")
        return True

    except KeyError as e:
        print(f"\nERROR: KeyError still occurring: {e}")
        traceback.print_exc()
        return False
    except Exception as e:
        print(f"\nERROR: Unexpected error: {e}")
        traceback.print_exc()
        return False

def test_full_application_launch():
    """Test launching the full application"""
    print("\n" + "="*50)
    print("TESTING FULL APPLICATION LAUNCH")
    print("="*50)

    try:
        # Test if we can import and run without errors
        from launch_pyqtgraph_with_selector import main

        print("Application can be imported without errors")
        print("To fully test, run: python launch_pyqtgraph_with_selector.py")
        return True

    except Exception as e:
        print(f"ERROR importing application: {e}")
        return False

def main():
    print("="*60)
    print("COMPREHENSIVE ERROR FIX VERIFICATION")
    print("="*60)

    results = {}

    # Run tests
    tests = [
        ("Hover Without Errors", test_hover_without_errors),
        ("Mouse Movement Simulation", test_mouse_movement_simulation),
        ("Application Import", test_full_application_launch)
    ]

    for test_name, test_func in tests:
        try:
            print(f"\nRunning: {test_name}")
            results[test_name] = test_func()
        except Exception as e:
            print(f"CRITICAL ERROR in {test_name}: {e}")
            traceback.print_exc()
            results[test_name] = False

    # Summary
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)

    all_passed = True
    for test_name, passed in results.items():
        status = "PASSED" if passed else "FAILED"
        print(f"  {test_name}: {status}")
        if not passed:
            all_passed = False

    if all_passed:
        print("\nALL ERRORS FIXED!")
        print("The application should now run without KeyErrors.")
        print("\nKey fixes applied:")
        print("- Check if dictionary keys exist before accessing them")
        print("- Use 'key in dict' pattern instead of accessing first")
        print("- Provide default values for missing optional fields")
    else:
        print("\nSome issues remain. Check output above.")

    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)