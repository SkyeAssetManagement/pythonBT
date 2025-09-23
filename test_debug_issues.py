"""
Debug script to test hover data and trade generation issues
"""
import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime

# Add the project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_data_loading():
    """Test if data loads correctly"""
    print("\n=== TESTING DATA LOADING ===")

    # Import the launcher module
    from launch_pyqtgraph_with_selector import load_data

    # Test with a known data file
    test_file = r"C:\code\PythonBT\dataRaw\range-ATR30x0.05\ES\diffAdjusted\ES_ATR30x0.05_diff2024.csv"

    if not os.path.exists(test_file):
        print(f"ERROR: Test file not found: {test_file}")
        # Try to find any CSV file
        import glob
        csv_files = glob.glob(r"C:\code\PythonBT\dataRaw\**\*.csv", recursive=True)
        if csv_files:
            test_file = csv_files[0]
            print(f"Using alternative file: {test_file}")
        else:
            print("ERROR: No CSV files found in dataRaw directory")
            return None, None

    print(f"Loading data from: {test_file}")

    # Load the data
    full_data, df = load_data(test_file)

    # Check results
    if full_data is None:
        print("ERROR: full_data is None!")
    else:
        print(f"SUCCESS: full_data loaded with keys: {list(full_data.keys())}")
        print(f"Data shape: {len(full_data.get('DateTime', []))} bars")

        # Check for essential columns
        required = ['DateTime', 'Open', 'High', 'Low', 'Close']
        for col in required:
            if col in full_data:
                print(f"  - {col}: {type(full_data[col])} with {len(full_data[col])} values")
                if len(full_data[col]) > 0:
                    print(f"    First value: {full_data[col][0]}")
            else:
                print(f"  - {col}: MISSING!")

    if df is None:
        print("ERROR: DataFrame is None!")
    else:
        print(f"SUCCESS: DataFrame loaded with shape: {df.shape}")
        print(f"DataFrame columns: {df.columns.tolist()}")
        print(f"First few rows:\n{df.head()}")

    return full_data, df

def test_trade_generation(df):
    """Test if trades can be generated"""
    print("\n=== TESTING TRADE GENERATION ===")

    if df is None or df.empty:
        print("ERROR: No DataFrame available for trade generation")
        return None

    # Import strategy modules
    from src.trading.strategies.sma_crossover import SMACrossoverStrategy

    print(f"DataFrame info before strategy:")
    print(f"  - Shape: {df.shape}")
    print(f"  - Columns: {df.columns.tolist()}")
    print(f"  - Index type: {type(df.index)}")
    print(f"  - First index: {df.index[0] if len(df) > 0 else 'N/A'}")

    # Try to generate trades with SMA strategy
    try:
        strategy = SMACrossoverStrategy(fast_period=10, slow_period=30)
        print("Strategy created successfully")

        # Execute the strategy
        trades = strategy.execute(df)

        if trades:
            print(f"SUCCESS: Generated {len(trades)} trades")
            for i, trade in enumerate(trades[:5]):  # Show first 5 trades
                print(f"  Trade {i+1}: bar_index={trade.bar_index}, price={trade.price:.2f}, type={trade.trade_type}")
        else:
            print("WARNING: No trades generated")

            # Debug the strategy execution
            print("\nDebugging strategy execution:")

            # Check if SMA can be calculated
            if 'Close' in df.columns:
                close_prices = df['Close'].values
                print(f"Close prices shape: {close_prices.shape}")
                print(f"First 5 close prices: {close_prices[:5]}")

                # Calculate SMAs manually
                sma_fast = pd.Series(close_prices).rolling(window=10).mean()
                sma_slow = pd.Series(close_prices).rolling(window=30).mean()

                print(f"SMA Fast (10) - first valid: {sma_fast.dropna().iloc[0] if not sma_fast.dropna().empty else 'N/A'}")
                print(f"SMA Slow (30) - first valid: {sma_slow.dropna().iloc[0] if not sma_slow.dropna().empty else 'N/A'}")

                # Check for crossovers
                valid_idx = ~(sma_fast.isna() | sma_slow.isna())
                if valid_idx.sum() > 0:
                    crossovers = ((sma_fast > sma_slow) & (sma_fast.shift(1) <= sma_slow.shift(1))) | \
                                ((sma_fast < sma_slow) & (sma_fast.shift(1) >= sma_slow.shift(1)))
                    num_crossovers = crossovers.sum()
                    print(f"Number of crossovers detected: {num_crossovers}")
                else:
                    print("ERROR: No valid SMA values calculated")
            else:
                print("ERROR: 'Close' column not found in DataFrame")

        return trades

    except Exception as e:
        print(f"ERROR generating trades: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def test_hover_functionality():
    """Test hover data structure"""
    print("\n=== TESTING HOVER DATA STRUCTURE ===")

    # Import the chart module
    from src.trading.visualization.pyqtgraph_range_bars_final import RangeBarChart
    from PyQt5.QtWidgets import QApplication
    import sys

    # Create a QApplication if it doesn't exist
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    # Create a chart instance
    chart = RangeBarChart()

    # Check initial state
    print(f"Chart full_data initial state: {chart.full_data}")

    # Create test data
    test_data = {
        'DateTime': np.array([datetime(2024, 1, 1, 9, 30), datetime(2024, 1, 1, 10, 0)]),
        'Open': np.array([4500.0, 4510.0]),
        'High': np.array([4520.0, 4530.0]),
        'Low': np.array([4490.0, 4505.0]),
        'Close': np.array([4510.0, 4525.0]),
        'Volume': np.array([1000, 1500])
    }

    # Set the data
    chart.full_data = test_data
    print(f"Chart full_data after setting: {chart.full_data is not None}")

    # Test the hover functionality
    if hasattr(chart, 'on_mouse_moved'):
        print("Chart has on_mouse_moved method")

        # Check what the method expects
        import inspect
        sig = inspect.signature(chart.on_mouse_moved)
        print(f"on_mouse_moved signature: {sig}")
    else:
        print("ERROR: Chart missing on_mouse_moved method")

    return chart

def main():
    """Main test execution"""
    print("="*60)
    print("DEBUGGING HOVER DATA AND TRADE GENERATION ISSUES")
    print("="*60)

    # Test 1: Data Loading
    full_data, df = test_data_loading()

    # Test 2: Trade Generation
    if df is not None:
        trades = test_trade_generation(df)
    else:
        print("\nSKIPPING TRADE GENERATION: No data loaded")

    # Test 3: Hover Functionality
    chart = test_hover_functionality()

    print("\n" + "="*60)
    print("DEBUGGING COMPLETE")
    print("="*60)

    # Summary
    print("\nSUMMARY:")
    print(f"1. Data Loading: {'SUCCESS' if full_data is not None else 'FAILED'}")
    print(f"2. Trade Generation: {'SUCCESS' if df is not None else 'FAILED'}")
    print(f"3. Hover Structure: {'READY' if chart else 'FAILED'}")

if __name__ == "__main__":
    main()