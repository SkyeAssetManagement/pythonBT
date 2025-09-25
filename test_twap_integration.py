#!/usr/bin/env python3
"""
Quick TWAP Integration Test
==========================
Test if the TWAP system is properly integrated with the strategy runner
"""

import sys
import os
sys.path.insert(0, 'src')
sys.path.insert(0, 'src/trading')

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def test_twap_integration():
    """Test TWAP system integration with strategy runner"""
    print("=== TWAP INTEGRATION TEST ===")

    # Create mock ES 0.05 range bar data
    np.random.seed(42)

    base_time = datetime(2023, 1, 4, 9, 30, 0)
    data = []

    for i in range(50):
        # Variable time intervals (range bars characteristic)
        time_increment = np.random.choice([1, 2, 3, 5, 8])
        base_time += timedelta(minutes=int(time_increment))

        # ES-like prices
        base_price = 4230.0 + i * 0.25

        bar_data = {
            'datetime': base_time,
            'open': base_price,
            'high': base_price + 0.25,
            'low': base_price,
            'close': base_price + 0.25,
            'volume': np.random.randint(800, 3000)
        }
        data.append(bar_data)

    df = pd.DataFrame(data)
    df.index = pd.to_datetime(df['datetime'])

    print(f"Created {len(df)} test range bars")
    print(f"Time range: {df.index[0]} to {df.index[-1]}")
    print(f"Price range: {df['low'].min():.2f} to {df['high'].max():.2f}")

    # Test strategy runner adapter
    try:
        from core.strategy_runner_adapter import StrategyRunnerAdapter

        adapter = StrategyRunnerAdapter()
        print(f"Strategy adapter initialized")
        print(f"Use TWAP: {adapter.use_twap}")
        print(f"Use unified: {adapter.use_unified_engine}")

        if adapter.use_twap:
            print("[OK] TWAP system is ENABLED and ready!")

            # Test strategy execution
            parameters = {
                'fast_period': 10,
                'slow_period': 20,
                'long_only': False
            }

            print("Testing TWAP execution with SMA strategy...")
            trades = adapter.run_strategy('sma_crossover', parameters, df)

            print(f"Generated {len(trades)} trades with TWAP execution")

            # Check if execBars data is present
            if len(trades) > 0:
                first_trade = trades[0]
                if hasattr(first_trade, 'metadata') and 'exec_bars' in first_trade.metadata:
                    print(f"[OK] execBars data found: {first_trade.metadata['exec_bars']} bars")
                    print(f"[OK] Execution time: {first_trade.metadata.get('execution_time_minutes', 'N/A')} minutes")
                    print(f"[OK] Natural phases: {first_trade.metadata.get('num_phases', 'N/A')}")
                else:
                    print("[ERROR] execBars metadata not found in trades")
            else:
                print("[WARNING] No trades generated (may need more data or different parameters)")

        else:
            print("[ERROR] TWAP system is NOT enabled")
            print("Check config.yaml: time_based_twap.enabled should be true")

    except ImportError as e:
        print(f"[ERROR] Import error: {e}")
        print("TWAP system not available")
    except Exception as e:
        print(f"[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_twap_integration()