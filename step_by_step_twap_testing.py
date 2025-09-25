#!/usr/bin/env python3
"""
Step-by-Step TWAP Testing Protocol
==================================
Run individual tests to isolate where the TWAP system is hanging
"""

import sys
import os
sys.path.insert(0, 'src')
sys.path.insert(0, 'src/trading')

import pandas as pd
import numpy as np
import yaml
from datetime import datetime, timedelta

def create_test_data(num_bars=50):
    """Create minimal test data for debugging"""
    print(f"Creating {num_bars} test bars...")

    np.random.seed(42)
    base_time = datetime(2023, 1, 4, 9, 30, 0)
    data = []

    for i in range(num_bars):
        time_increment = np.random.choice([1, 2, 3])  # Shorter increments
        base_time += timedelta(minutes=int(time_increment))

        base_price = 4230.0 + i * 0.25
        bar_data = {
            'datetime': base_time,
            'open': base_price,
            'high': base_price + 0.25,
            'low': base_price,
            'close': base_price + 0.25,
            'volume': np.random.randint(500, 1500)  # Smaller volumes
        }
        data.append(bar_data)

    df = pd.DataFrame(data)
    df.index = pd.to_datetime(df['datetime'])

    print(f"Data created: {len(df)} bars from {df.index[0]} to {df.index[-1]}")
    return df

def test_1_config_verification():
    """TEST 1: Verify TWAP configuration is loaded correctly"""
    print("\n" + "="*60)
    print("TEST 1: Configuration Verification")
    print("="*60)

    try:
        config_path = "C:\\code\\PythonBT\\tradingCode\\config.yaml"
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        twap_config = config.get('time_based_twap', {})
        print(f"TWAP enabled: {twap_config.get('enabled', False)}")
        print(f"Minimum execution minutes: {twap_config.get('minimum_execution_minutes', 'Not set')}")
        print(f"Batch size: {twap_config.get('batch_size', 'Not set')}")

        if twap_config.get('enabled', False):
            print("[PASS] TEST 1 PASSED: Configuration properly loaded")
            return True
        else:
            print("[FAIL] TEST 1 FAILED: TWAP not enabled in config")
            return False

    except Exception as e:
        print(f"[FAIL] TEST 1 FAILED: {e}")
        return False

def test_2_strategy_adapter_init():
    """TEST 2: Strategy adapter initialization"""
    print("\n" + "="*60)
    print("TEST 2: Strategy Adapter Initialization")
    print("="*60)

    try:
        from core.strategy_runner_adapter import StrategyRunnerAdapter

        print("Initializing strategy adapter...")
        adapter = StrategyRunnerAdapter()

        print(f"TWAP available: {adapter.use_twap}")
        print(f"TWAP adapter object: {adapter.twap_adapter is not None}")

        if adapter.use_twap and adapter.twap_adapter is not None:
            print("[PASS] TEST 2 PASSED: Strategy adapter initialized with TWAP")
            return adapter
        else:
            print("[FAIL] TEST 2 FAILED: TWAP not properly initialized")
            return None

    except Exception as e:
        print(f"[FAIL] TEST 2 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_3_signal_generation(adapter, df):
    """TEST 3: Signal generation only"""
    print("\n" + "="*60)
    print("TEST 3: Signal Generation")
    print("="*60)

    try:
        # Generate signals manually without TWAP execution
        from strategies.sma_crossover import SMACrossoverStrategy

        strategy = SMACrossoverStrategy(fast_period=10, slow_period=20, long_only=False)
        signals = strategy.generate_signals(df)

        long_signals = (signals > 0).sum()
        short_signals = (signals < 0).sum()

        print(f"Generated signals - Long: {long_signals}, Short: {short_signals}")

        if long_signals > 0 or short_signals > 0:
            print("[PASS] TEST 3 PASSED: Signals generated successfully")
            return signals, long_signals, short_signals
        else:
            print("[FAIL] TEST 3 FAILED: No signals generated")
            return None, 0, 0

    except Exception as e:
        print(f"[FAIL] TEST 3 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return None, 0, 0

def test_4_twap_engine_execution(adapter, df, signals):
    """TEST 4: TWAP engine execution without VectorBT"""
    print("\n" + "="*60)
    print("TEST 4: TWAP Engine Execution (No VectorBT)")
    print("="*60)

    try:
        long_signals = (signals > 0)
        short_signals = (signals < 0)

        print("Testing TWAP engine execution...")

        # Use the TWAP engine directly
        twap_engine = adapter.twap_adapter.twap_engine

        execution_results = twap_engine.execute_signals_with_volume_weighted_twap(
            df=df,
            signals_long=long_signals,
            signals_short=short_signals,
            signal_lag=2,
            target_position_size=1.0
        )

        print(f"Long execution data: {len(execution_results['long_execution_data'])}")
        print(f"Short execution data: {len(execution_results['short_execution_data'])}")
        print(f"Total natural phases: {execution_results['total_natural_phases']}")

        if execution_results['long_execution_data'] or execution_results['short_execution_data']:
            print("[PASS] TEST 4 PASSED: TWAP engine executed successfully")
            return execution_results
        else:
            print("[FAIL] TEST 4 FAILED: No execution results")
            return None

    except Exception as e:
        print(f"[FAIL] TEST 4 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_5_vectorbt_portfolio_creation(adapter, execution_results):
    """TEST 5: VectorBT portfolio creation"""
    print("\n" + "="*60)
    print("TEST 5: VectorBT Portfolio Creation")
    print("="*60)

    try:
        print("Creating VectorBT portfolio parameters...")

        # This is where it might hang - test just the parameter preparation
        df_for_vbt = create_test_data(30)  # Need df for vectorbt preparation
        vectorbt_params = adapter.twap_adapter._prepare_vectorbt_data(
            df=df_for_vbt, execution_results=execution_results, size=1.0, fees=0.0
        )

        print(f"VectorBT parameters prepared:")
        print(f"- Entries shape: {vectorbt_params['entries'].shape}")
        print(f"- Sizes shape: {vectorbt_params['size'].shape}")
        print(f"- Custom price array shape: {vectorbt_params['price'].shape}")

        print("[PASS] TEST 5 PASSED: VectorBT parameters prepared")
        return vectorbt_params

    except Exception as e:
        print(f"[FAIL] TEST 5 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_6_minimal_vectorbt_creation(vectorbt_params):
    """TEST 6: Create minimal VectorBT portfolio"""
    print("\n" + "="*60)
    print("TEST 6: Minimal VectorBT Portfolio Creation")
    print("="*60)

    try:
        import vectorbtpro as vbt

        print("Creating VectorBT portfolio (this might hang)...")
        print("Portfolio parameters:")
        for key, value in vectorbt_params.items():
            if hasattr(value, 'shape'):
                print(f"- {key}: {value.shape}")
            else:
                print(f"- {key}: {type(value)}")

        # Try to create the portfolio - this is likely where it hangs
        portfolio = vbt.Portfolio.from_signals(**vectorbt_params)

        print(f"Portfolio created successfully")
        print(f"Portfolio stats available: {hasattr(portfolio, 'stats')}")

        print("[PASS] TEST 6 PASSED: VectorBT portfolio created")
        return portfolio

    except Exception as e:
        print(f"[FAIL] TEST 6 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return None

def run_step_by_step_tests():
    """Run all tests step by step"""
    print("STEP-BY-STEP TWAP TESTING PROTOCOL")
    print("=" * 80)
    print("This will test each component individually to find where it hangs")

    # Create small test dataset
    df = create_test_data(num_bars=30)  # Very small dataset

    # Test 1: Configuration
    if not test_1_config_verification():
        print("\n[WARNING] Fix configuration before proceeding")
        return

    # Test 2: Adapter initialization
    adapter = test_2_strategy_adapter_init()
    if adapter is None:
        print("\n[WARNING] Fix adapter initialization before proceeding")
        return

    # Test 3: Signal generation
    signals, long_count, short_count = test_3_signal_generation(adapter, df)
    if signals is None:
        print("\n[WARNING] Fix signal generation before proceeding")
        return

    # Test 4: TWAP execution
    execution_results = test_4_twap_engine_execution(adapter, df, signals)
    if execution_results is None:
        print("\n[WARNING] Fix TWAP execution before proceeding")
        return

    # Test 5: VectorBT parameter preparation
    vectorbt_params = test_5_vectorbt_portfolio_creation(adapter, execution_results)
    if vectorbt_params is None:
        print("\n[WARNING] Fix VectorBT parameter preparation before proceeding")
        return

    # Test 6: VectorBT portfolio creation (likely culprit)
    portfolio = test_6_minimal_vectorbt_creation(vectorbt_params)
    if portfolio is None:
        print("\n[WARNING] VectorBT portfolio creation failed - this is likely where it hangs")
        return

    print("\n" + "="*80)
    print("[SUCCESS] ALL TESTS PASSED - TWAP system working correctly")
    print("="*80)

if __name__ == "__main__":
    run_step_by_step_tests()