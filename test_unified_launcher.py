"""
Test the unified launcher with real data
"""

import sys
import os
sys.path.insert(0, 'src')
sys.path.insert(0, 'src/trading/visualization')
sys.path.insert(0, 'src/trading')

import pandas as pd
import numpy as np
from pathlib import Path

# Test data loading
def test_data_loading():
    """Test that we can load ES data properly"""

    # Try to find ES data file
    data_paths = [
        "dataRaw\\range-ATR30x0.05\\ES\\diffAdjusted\\ES-DIFF-range-ATR30x0.05-dailyATR.csv",
        "C:\\code\\PythonBT\\dataRaw\\range-ATR30x0.05\\ES\\diffAdjusted\\ES-DIFF-range-ATR30x0.05-dailyATR.csv"
    ]

    file_path = None
    for path in data_paths:
        if os.path.exists(path):
            file_path = path
            break

    if not file_path:
        print("No ES data file found, using sample data")
        return None

    print(f"Loading from: {file_path}")
    df = pd.read_csv(file_path)

    print(f"Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")

    # Test datetime combination
    if 'Date' in df.columns and 'Time' in df.columns:
        print("\nTesting datetime combination...")
        # Convert Date to datetime
        if df['Date'].dtype == 'object':
            df['Date'] = pd.to_datetime(df['Date'])

        # Check Time column type
        print(f"Time column dtype: {df['Time'].dtype}")
        print(f"First 5 Time values:\n{df['Time'].head()}")

        # Handle Time column
        if pd.api.types.is_timedelta64_dtype(df['Time']):
            df['DateTime'] = df['Date'] + df['Time']
        else:
            df['DateTime'] = pd.to_datetime(df['Date'].astype(str) + ' ' + df['Time'].astype(str))

        print(f"\nFirst 5 DateTime values:\n{df['DateTime'].head()}")

    # Test data structure for chart
    data_dict = {
        'DateTime': df['DateTime'].values if 'DateTime' in df.columns else None,
        'open': df['Open'].values.astype(np.float64),
        'high': df['High'].values.astype(np.float64),
        'low': df['Low'].values.astype(np.float64),
        'close': df['Close'].values.astype(np.float64),
        'volume': df['Volume'].values.astype(np.float64) if 'Volume' in df.columns else np.zeros(len(df))
    }

    print(f"\nData dict keys: {list(data_dict.keys())}")
    print(f"Number of bars: {len(data_dict['open'])}")
    print(f"First OHLC values: O={data_dict['open'][0]:.2f}, H={data_dict['high'][0]:.2f}, L={data_dict['low'][0]:.2f}, C={data_dict['close'][0]:.2f}")

    return data_dict

# Test unified execution
def test_unified_execution():
    """Test that unified execution works with the data"""

    from core.standalone_execution import ExecutionConfig
    from strategies.strategy_wrapper import StrategyFactory

    print("\n" + "="*60)
    print("Testing Unified Execution")
    print("="*60)

    # Create sample data
    df = pd.DataFrame({
        'DateTime': pd.date_range('2024-01-01', periods=100, freq='5min'),
        'Open': 100 + np.random.randn(100) * 0.5,
        'High': 101 + np.random.randn(100) * 0.5,
        'Low': 99 + np.random.randn(100) * 0.5,
        'Close': 100 + np.cumsum(np.random.randn(100) * 0.2),
        'Volume': np.random.randint(1000, 5000, 100)
    })

    # Load execution config
    exec_config = ExecutionConfig.from_yaml()
    print(f"Execution config loaded:")
    print(f"  Signal lag: {exec_config.signal_lag}")
    print(f"  Execution price: {exec_config.execution_price}")
    print(f"  Buy formula: {exec_config.buy_execution_formula}")

    # Create strategy
    strategy = StrategyFactory.create_sma_crossover(
        fast_period=5,
        slow_period=10,
        execution_config=exec_config
    )

    # Execute trades
    trades = strategy.execute_trades(df)
    print(f"\nGenerated {len(trades)} trades")

    if len(trades) > 0:
        for trade in trades[:3]:
            print(f"  Trade {trade.trade_id}: {trade.trade_type}")
            print(f"    Signal bar: {trade.signal_bar}, Execution bar: {trade.execution_bar}")
            print(f"    Lag: {trade.lag} bars")
            print(f"    Price: ${trade.price:.2f}")

if __name__ == "__main__":
    print("UNIFIED LAUNCHER TEST")
    print("=" * 60)

    # Test data loading
    data = test_data_loading()

    if data:
        print("\n[OK] Data loading works")

    # Test execution
    try:
        test_unified_execution()
        print("\n[OK] Unified execution works")
    except Exception as e:
        print(f"\n[ERROR] Unified execution failed: {e}")
        import traceback
        traceback.print_exc()