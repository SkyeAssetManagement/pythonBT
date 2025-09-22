#!/usr/bin/env python3
"""
Test DateTime propagation through the trading system
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src', 'trading'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'src', 'trading', 'visualization'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'src', 'trading', 'strategies'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'src', 'trading', 'data'))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Import the classes we need to test
from strategies.sma_crossover import SMACrossoverStrategy
from data.trade_data import TradeData, TradeCollection

def test_datetime_flow():
    """Test DateTime propagation from chart data to trades"""
    print("=" * 80)
    print("Testing DateTime Flow")
    print("=" * 80)

    # 1. Create sample data with timestamps
    n = 100
    start_date = pd.Timestamp('2022-01-18')
    timestamps = pd.date_range(start_date, periods=n, freq='5min')

    # Create price data
    base_price = 5185.50
    prices = base_price + np.cumsum(np.random.randn(n) * 2)

    # Create DataFrame as StrategyRunner would
    df = pd.DataFrame({
        'DateTime': timestamps,
        'Open': prices + np.random.randn(n) * 0.5,
        'High': prices + np.abs(np.random.randn(n) * 2),
        'Low': prices - np.abs(np.random.randn(n) * 2),
        'Close': prices,
        'Volume': np.random.randint(100, 1000, n)
    })

    print(f"\n1. Created DataFrame with columns: {df.columns.tolist()}")
    print(f"   First DateTime: {df['DateTime'].iloc[0]}")
    print(f"   DateTime type: {df['DateTime'].dtype}")

    # 2. Test strategy signal generation
    strategy = SMACrossoverStrategy(fast_period=5, slow_period=15)
    print(f"\n2. Running {strategy.name} strategy...")

    signals = strategy.generate_signals(df)
    print(f"   Generated {len(signals)} signals")
    print(f"   Signal values: {signals.value_counts().to_dict()}")

    # 3. Test trade generation with DateTime
    print(f"\n3. Converting signals to trades...")
    trades = strategy.signals_to_trades(signals, df)

    print(f"   Generated {len(trades)} trades")

    # 4. Check if trades have timestamps
    print(f"\n4. Checking trade timestamps:")
    for i, trade in enumerate(trades[:5]):
        print(f"   Trade {i}: bar_index={trade.bar_index}, "
              f"type={trade.trade_type}, price={trade.price:.2f}, "
              f"timestamp={trade.timestamp}")
        if trade.timestamp:
            print(f"           timestamp type: {type(trade.timestamp)}")

    # 5. Test TradeCollection to DataFrame conversion
    print(f"\n5. Testing TradeCollection.to_dataframe():")
    trades_df = trades.to_dataframe()
    print(f"   DataFrame columns: {trades_df.columns.tolist()}")
    if 'timestamp' in trades_df.columns:
        print(f"   First timestamp in DataFrame: {trades_df['timestamp'].iloc[0] if len(trades_df) > 0 else 'EMPTY'}")
        print(f"   Timestamp dtype: {trades_df['timestamp'].dtype}")
    else:
        print(f"   WARNING: No timestamp column in trades DataFrame!")

    # 6. Simulate what happens in trade_panel
    print(f"\n6. Simulating trade_panel display:")
    for i, trade in enumerate(trades[:3]):
        # This is what trade_panel tries to do
        value = trade.timestamp
        if value is not None:
            try:
                if hasattr(value, 'strftime'):
                    formatted = value.strftime('%y-%m-%d %H:%M:%S')
                    print(f"   Trade {i} DateTime display: {formatted}")
                else:
                    dt = pd.to_datetime(value)
                    formatted = dt.strftime('%y-%m-%d %H:%M:%S')
                    print(f"   Trade {i} DateTime display (converted): {formatted}")
            except Exception as e:
                print(f"   Trade {i} ERROR formatting: {e}")
        else:
            print(f"   Trade {i} DateTime display: -")

    print("\n" + "=" * 80)
    print("Test Complete")
    print("=" * 80)

if __name__ == "__main__":
    test_datetime_flow()