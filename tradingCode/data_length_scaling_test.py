# tradingCode/data_length_scaling_test.py
# Test execution time scaling with data length vs parameter count
# Execution time should NOT increase with data length for fixed parameters

#!/usr/bin/env python3
"""
Data Length Scaling Test

Tests the fundamental requirement that execution time should be constant 
with data length for a fixed number of parameters, but can scale with parameter count.

Expected behavior:
- Single run 2020 vs 2022: Similar execution time (data length independent)
- 20 param run 2020 vs 2022: Similar execution time (data length independent)  
- Single vs 20 param: 20x longer time (parameter count dependent)
"""

import time
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from src.data.parquet_converter import ParquetConverter
from strategies.time_window_strategy_vectorized import TimeWindowVectorizedStrategy

def create_reduced_parameter_strategy():
    """
    Create a strategy with exactly 20 parameter combinations for testing.
    This allows us to test parameter scaling vs data length scaling.
    """
    
    class ReducedParameterStrategy(TimeWindowVectorizedStrategy):
        def __init__(self):
            super().__init__()
            # Override parameters to get exactly 20 combinations
            self.parameters = {
                'entry_time': {
                    'default': "09:30",
                    'type': str,
                    'description': "Entry time in HH:MM format",
                    'values': ["09:30", "10:00", "10:30", "14:00"]  # 4 values
                },
                'direction': {
                    'default': "long", 
                    'type': str,
                    'description': "Trade direction",
                    'values': ["long", "short"]  # 2 values
                },
                'hold_time': {
                    'default': 60,
                    'type': int,
                    'description': "Minutes to hold position", 
                    'values': [30, 60, 90]  # 3 values -> 4*2*3 = 24 combinations
                },
                'entry_spread': {
                    'default': 5,
                    'type': int,
                    'description': "Entry window minutes",
                    'values': [5]  # Fixed to 5 for consistency
                }
            }
        
        def get_parameter_combinations(self, use_defaults_only: bool = False) -> list:
            """Override to get exactly 20 combinations"""
            if use_defaults_only:
                return super().get_parameter_combinations(use_defaults_only)
            
            # Generate all combinations and take first 20
            combinations = []
            for entry_time in self.parameters['entry_time']['values']:
                for direction in self.parameters['direction']['values']:
                    for hold_time in self.parameters['hold_time']['values']:
                        combinations.append({
                            'entry_time': entry_time,
                            'direction': direction, 
                            'hold_time': hold_time,
                            'entry_spread': 5,
                            'exit_spread': 5,
                            'max_trades_per_day': 1
                        })
                        if len(combinations) >= 20:  # Stop at exactly 20
                            break
                    if len(combinations) >= 20:
                        break
                if len(combinations) >= 20:
                    break
            
            return combinations[:20]  # Ensure exactly 20
    
    return ReducedParameterStrategy()

def test_data_length_scaling():
    """
    Test the core requirement: execution time should be constant with data length
    for fixed parameter count, but can scale with parameter count.
    """
    
    print("DATA LENGTH SCALING TEST")
    print("=" * 70)
    print("Testing fundamental scaling requirement:")
    print("- Execution time should NOT increase with data length")
    print("- Execution time CAN increase with parameter count")
    print()
    
    # Load datasets
    parquet_converter = ParquetConverter()
    
    print("Loading datasets...")
    
    # Dataset 1: Since 2022 (smaller)
    print("  Loading data since 2022-01-01...")
    start_time = time.time()
    data_2022 = parquet_converter.load_or_convert("ES", "1m", "diffAdjusted")
    data_2022 = parquet_converter.filter_data_by_date(data_2022, "2022-01-01", None)
    load_time_2022 = time.time() - start_time
    bars_2022 = len(data_2022['close'])
    print(f"    SUCCESS: {bars_2022:,} bars in {load_time_2022:.2f}s")
    
    # Dataset 2: Since 2020 (larger)  
    print("  Loading data since 2020-01-01...")
    start_time = time.time()
    data_2020 = parquet_converter.load_or_convert("ES", "1m", "diffAdjusted")
    data_2020 = parquet_converter.filter_data_by_date(data_2020, "2020-01-01", None)
    load_time_2020 = time.time() - start_time
    bars_2020 = len(data_2020['close'])
    print(f"    SUCCESS: {bars_2020:,} bars in {load_time_2020:.2f}s")
    
    data_ratio = bars_2020 / bars_2022
    print(f"  Data size ratio: {data_ratio:.1f}x more data in 2020 dataset")
    
    # Create strategies
    single_strategy = TimeWindowVectorizedStrategy()
    param20_strategy = create_reduced_parameter_strategy()
    
    config = {'initial_capital': 100000, 'commission': 0.001, 'slippage': 0.0001}
    
    # Test matrix
    test_results = []
    
    print(f"\n{'='*70}")
    print("EXECUTION TIME TESTS")
    print(f"{'='*70}")
    
    # Test 1: Single run on 2022 data
    print("\n1. Single Run - 2022 Data (Since 2022-01-01)")
    start_time = time.time()
    pf1 = single_strategy.run_vectorized_backtest(data_2022, config, use_defaults_only=True)
    time1 = time.time() - start_time
    trades1 = len(pf1.trades.records_readable)
    
    print(f"   Time: {time1:.3f}s | Trades: {trades1} | Throughput: {bars_2022/time1:,.0f} bars/sec")
    test_results.append(('Single 2022', bars_2022, time1, 1, trades1))
    
    # Test 2: Single run on 2020 data  
    print("\n2. Single Run - 2020 Data (Since 2020-01-01)")
    start_time = time.time()
    pf2 = single_strategy.run_vectorized_backtest(data_2020, config, use_defaults_only=True)
    time2 = time.time() - start_time
    trades2 = len(pf2.trades.records_readable)
    
    print(f"   Time: {time2:.3f}s | Trades: {trades2} | Throughput: {bars_2020/time2:,.0f} bars/sec")
    test_results.append(('Single 2020', bars_2020, time2, 1, trades2))
    
    # Test 3: 20 param run on 2022 data
    print("\n3. 20 Parameter Run - 2022 Data")
    param_combos = param20_strategy.get_parameter_combinations(use_defaults_only=False)
    print(f"   Parameter combinations: {len(param_combos)}")
    
    start_time = time.time()
    pf3 = param20_strategy.run_vectorized_backtest(data_2022, config, use_defaults_only=False)
    time3 = time.time() - start_time
    trades3 = len(pf3.trades.records_readable)
    
    print(f"   Time: {time3:.3f}s | Trades: {trades3} | Throughput: {bars_2022/time3:,.0f} bars/sec")
    test_results.append(('20-Param 2022', bars_2022, time3, len(param_combos), trades3))
    
    # Test 4: 20 param run on 2020 data
    print("\n4. 20 Parameter Run - 2020 Data")
    start_time = time.time()
    pf4 = param20_strategy.run_vectorized_backtest(data_2020, config, use_defaults_only=False)
    time4 = time.time() - start_time
    trades4 = len(pf4.trades.records_readable)
    
    print(f"   Time: {time4:.3f}s | Trades: {trades4} | Throughput: {bars_2020/time4:,.0f} bars/sec")
    test_results.append(('20-Param 2020', bars_2020, time4, len(param_combos), trades4))
    
    # Analysis
    print(f"\n{'='*70}")
    print("SCALING ANALYSIS")
    print(f"{'='*70}")
    
    print(f"{'Test':<15} {'Bars':<10} {'Time(s)':<8} {'Params':<7} {'Trades':<7} {'B/s':<10}")
    print("-" * 70)
    
    for test_name, bars, exec_time, params, trades in test_results:
        throughput = bars / exec_time
        print(f"{test_name:<15} {bars:<10,} {exec_time:<8.3f} {params:<7} {trades:<7} {throughput:<10,.0f}")
    
    # Critical analysis
    print(f"\nCRITICAL SCALING ANALYSIS:")
    print("-" * 40)
    
    # Data length scaling (should be constant)
    single_ratio = time2 / time1  # 2020 vs 2022 single run
    param20_ratio = time4 / time3  # 2020 vs 2022 20-param run
    
    print(f"Data Length Scaling (Should be ~1.0):")
    print(f"  Single run: 2020 vs 2022 = {single_ratio:.2f}x slower")
    print(f"  20-param run: 2020 vs 2022 = {param20_ratio:.2f}x slower")
    
    # Parameter count scaling (should be ~20x)
    param_scaling_2022 = time3 / time1  # 20-param vs single on 2022 data
    param_scaling_2020 = time4 / time2  # 20-param vs single on 2020 data
    
    print(f"\nParameter Count Scaling (Should be ~20x):")
    print(f"  2022 data: 20-param vs single = {param_scaling_2022:.1f}x slower")  
    print(f"  2020 data: 20-param vs single = {param_scaling_2020:.1f}x slower")
    
    # Verdict
    print(f"\nVERDICT:")
    print("=" * 20)
    
    data_scaling_ok = max(single_ratio, param20_ratio) < 1.5  # Should be near 1.0
    param_scaling_ok = 15 <= min(param_scaling_2022, param_scaling_2020) <= 25  # Should be ~20
    
    if data_scaling_ok:
        print("SUCCESS: DATA LENGTH SCALING: PASS - Time constant with data size")
    else:
        print("FAIL: DATA LENGTH SCALING: FAIL - Time increases with data size")
        print(f"   BOTTLENECK DETECTED: Need to fix algorithm/memory allocation")
    
    if param_scaling_ok:
        print("SUCCESS: PARAMETER SCALING: PASS - Time scales properly with parameter count")
    else:
        print("WARNING: PARAMETER SCALING: Suboptimal but acceptable")
    
    if not data_scaling_ok:
        print(f"\nCRITICAL: Data length scaling failure detected!")
        print(f"   Single run scaling: {single_ratio:.2f}x (should be ~1.0x)")
        print(f"   20-param scaling: {param20_ratio:.2f}x (should be ~1.0x)")
        print(f"   Data size ratio: {data_ratio:.1f}x")
        print(f"   Expected time ratio: ~1.0x")
        print(f"   Actual time ratios: {single_ratio:.2f}x and {param20_ratio:.2f}x")
        
        if single_ratio > 2.0 or param20_ratio > 2.0:
            print(f"\n   RECOMMENDATION: Go back and solve the bottleneck")
            print(f"   Root cause: Algorithm/memory operations not O(1) with data length")
    
    return {
        'data_scaling_ok': data_scaling_ok,
        'param_scaling_ok': param_scaling_ok,
        'single_ratio': single_ratio,
        'param20_ratio': param20_ratio,
        'param_scaling_2022': param_scaling_2022,
        'param_scaling_2020': param_scaling_2020
    }

if __name__ == "__main__":
    results = test_data_length_scaling()