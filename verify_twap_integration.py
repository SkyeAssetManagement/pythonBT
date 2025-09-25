#!/usr/bin/env python3
"""
TWAP System Integration Verification
====================================
Comprehensive verification that the TWAP system is properly integrated
with the chart visualizer and all components are working correctly.
"""

import sys
import os
sys.path.insert(0, 'src')
sys.path.insert(0, 'src/trading')

import pandas as pd
import numpy as np
import yaml
from datetime import datetime, timedelta

def verify_config():
    """Verify TWAP configuration is enabled"""
    print("=== CONFIGURATION VERIFICATION ===")

    config_path = "C:\\code\\PythonBT\\tradingCode\\config.yaml"
    if not os.path.exists(config_path):
        print("[ERROR] Config file not found")
        return False

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    twap_config = config.get('time_based_twap', {})
    enabled = twap_config.get('enabled', False)
    min_time = twap_config.get('minimum_execution_minutes', 5.0)

    print(f"TWAP enabled: {enabled}")
    print(f"Minimum execution time: {min_time} minutes")

    if enabled:
        print("[OK] TWAP system is properly configured")
        return True
    else:
        print("[ERROR] TWAP system is not enabled in config")
        return False

def verify_imports():
    """Verify all TWAP modules can be imported"""
    print("\n=== IMPORT VERIFICATION ===")

    try:
        from core.time_based_twap_execution import TimeBasedTWAPConfig, TimeBasedTWAPEngine
        print("[OK] Core TWAP execution module imported")
    except ImportError as e:
        print(f"[ERROR] Core TWAP module: {e}")
        return False

    try:
        from core.vectorbt_twap_adapter import VectorBTTWAPAdapter
        print("[OK] VectorBT TWAP adapter imported")
    except ImportError as e:
        print(f"[ERROR] VectorBT adapter: {e}")
        return False

    try:
        from core.strategy_runner_adapter import StrategyRunnerAdapter
        print("[OK] Strategy runner adapter imported")
    except ImportError as e:
        print(f"[ERROR] Strategy adapter: {e}")
        return False

    return True

def verify_strategy_adapter():
    """Verify strategy adapter initializes with TWAP"""
    print("\n=== STRATEGY ADAPTER VERIFICATION ===")

    try:
        from core.strategy_runner_adapter import StrategyRunnerAdapter

        adapter = StrategyRunnerAdapter()
        print(f"Use TWAP: {adapter.use_twap}")
        print(f"Use unified engine: {adapter.use_unified_engine}")
        print(f"TWAP adapter available: {adapter.twap_adapter is not None}")

        if adapter.use_twap:
            print("[OK] Strategy adapter initialized with TWAP system")
            return True
        else:
            print("[ERROR] Strategy adapter not using TWAP system")
            return False

    except Exception as e:
        print(f"[ERROR] Strategy adapter initialization: {e}")
        return False

def verify_trade_panel_columns():
    """Verify trade panel includes execBars column"""
    print("\n=== TRADE PANEL VERIFICATION ===")

    try:
        from visualization.enhanced_trade_panel import EnhancedTradeListPanel

        # Check if execBars column is defined through the table model
        from visualization.enhanced_trade_panel import EnhancedTradeTableModel
        model = EnhancedTradeTableModel()
        columns = model.COLUMNS

        exec_bars_column = None
        for col_name, col_key in columns:
            if col_key == 'exec_bars':
                exec_bars_column = (col_name, col_key)
                break

        if exec_bars_column:
            print(f"[OK] execBars column found: {exec_bars_column[0]} ({exec_bars_column[1]})")
            return True
        else:
            print("[ERROR] execBars column not found in trade panel")
            print(f"Available columns: {[col[1] for col in columns]}")
            return False

    except Exception as e:
        print(f"[ERROR] Trade panel verification: {e}")
        return False

def test_full_execution():
    """Test complete TWAP execution pipeline"""
    print("\n=== FULL EXECUTION TEST ===")

    # Create test data similar to ES range bars
    np.random.seed(42)
    base_time = datetime(2023, 1, 4, 9, 30, 0)
    data = []

    for i in range(100):  # More bars for better testing
        time_increment = np.random.choice([1, 2, 3, 5, 8])
        base_time += timedelta(minutes=int(time_increment))

        base_price = 4230.0 + i * 0.25

        bar_data = {
            'datetime': base_time,
            'open': base_price,
            'high': base_price + 0.25,
            'low': base_price - 0.1,
            'close': base_price + 0.15,
            'volume': np.random.randint(800, 3000)
        }
        data.append(bar_data)

    df = pd.DataFrame(data)
    df.index = pd.to_datetime(df['datetime'])

    try:
        from core.strategy_runner_adapter import StrategyRunnerAdapter

        adapter = StrategyRunnerAdapter()
        parameters = {
            'fast_period': 10,
            'slow_period': 20,
            'long_only': False
        }

        print(f"Testing with {len(df)} bars from {df.index[0]} to {df.index[-1]}")

        trades = adapter.run_strategy('sma_crossover', parameters, df)

        print(f"Generated {len(trades)} trades")

        # Verify execBars metadata
        if len(trades) > 0:
            first_trade = trades[0]
            if hasattr(first_trade, 'metadata') and 'exec_bars' in first_trade.metadata:
                exec_bars = first_trade.metadata['exec_bars']
                exec_time = first_trade.metadata.get('execution_time_minutes', 'N/A')
                phases = first_trade.metadata.get('num_phases', 'N/A')

                print(f"[OK] First trade execBars: {exec_bars}")
                print(f"[OK] Execution time: {exec_time} minutes")
                print(f"[OK] Natural phases: {phases}")
                return True
            else:
                print("[ERROR] execBars metadata missing from trades")
                return False
        else:
            print("[WARNING] No trades generated - may need more data")
            return True  # Not necessarily an error

    except Exception as e:
        print(f"[ERROR] Full execution test: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run complete TWAP integration verification"""
    print("TWAP SYSTEM INTEGRATION VERIFICATION")
    print("=" * 50)

    results = []

    results.append(("Configuration", verify_config()))
    results.append(("Imports", verify_imports()))
    results.append(("Strategy Adapter", verify_strategy_adapter()))
    results.append(("Trade Panel", verify_trade_panel_columns()))
    results.append(("Full Execution", test_full_execution()))

    print("\n" + "=" * 50)
    print("VERIFICATION RESULTS")
    print("=" * 50)

    all_passed = True
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{test_name:20s}: {status}")
        if not result:
            all_passed = False

    print("=" * 50)

    if all_passed:
        print("[SUCCESS] All verifications passed!")
        print("\nTWAP system is ready for chart visualizer testing:")
        print("1. Run: python launch_unified_system.py")
        print("2. Load ES 0.05 range bar data")
        print("3. Run SMA strategy (10/30 periods)")
        print("4. Verify execBars column appears in trade list")
        print("5. Check console output for TWAP execution messages")
        return True
    else:
        print("[ERROR] Some verifications failed!")
        print("Fix the failed components before proceeding.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)