"""
Performance test for phased entry system
Tests scalability with different dataset sizes
"""

import sys
import os
import pandas as pd
import numpy as np
import time
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def create_test_data(bars=1000, start_price=100.0):
    """Create synthetic price data for testing"""
    np.random.seed(42)  # For reproducible tests
    dates = pd.date_range(start='2023-01-01', periods=bars, freq='1H')

    returns = np.random.normal(0.001, 0.02, bars)
    prices = [start_price]

    for i in range(1, bars):
        price = prices[-1] * (1 + returns[i])
        prices.append(max(price, 1.0))

    df = pd.DataFrame({
        'DateTime': dates,
        'Open': prices,
        'High': [p * (1 + abs(np.random.normal(0, 0.005))) for p in prices],
        'Low': [p * (1 - abs(np.random.normal(0, 0.005))) for p in prices],
        'Close': prices,
        'Volume': np.random.randint(1000, 10000, bars)
    })

    df['High'] = np.maximum(df['High'], df[['Open', 'Close']].max(axis=1))
    df['Low'] = np.minimum(df['Low'], df[['Open', 'Close']].min(axis=1))

    return df

def create_simple_signals(df, signal_frequency=0.05):
    """Create simple signals for testing"""
    np.random.seed(42)
    signals = pd.Series(0, index=range(len(df)))

    # Generate some long signals
    signal_bars = np.random.choice(len(df), int(len(df) * signal_frequency), replace=False)
    for bar in signal_bars:
        if bar < len(df) - 50:  # Leave room for exits
            signals.iloc[bar] = 1
            # Add exit after some bars
            exit_bar = min(bar + np.random.randint(10, 50), len(df) - 1)
            signals.iloc[exit_bar] = 0

    return signals

def test_original_execution(df, signals):
    """Test original execution engine performance"""
    try:
        from trading.core.standalone_execution import StandaloneExecutionEngine, ExecutionConfig

        config = ExecutionConfig()
        engine = StandaloneExecutionEngine(config)

        start_time = time.time()
        trades = engine.execute_signals(signals, df)
        end_time = time.time()

        return end_time - start_time, len(trades)
    except Exception as e:
        print(f"Error in original execution: {e}")
        return None, 0

def test_phased_execution(df, signals):
    """Test phased execution engine performance"""
    try:
        from trading.core.phased_execution_engine import PhasedExecutionEngine, PhasedExecutionConfig

        config = PhasedExecutionConfig()
        config.phased_config.enabled = True
        config.phased_config.max_phases = 3
        engine = PhasedExecutionEngine(config)

        start_time = time.time()
        trades = engine.execute_signals_with_phases(signals, df, "TEST")
        end_time = time.time()

        return end_time - start_time, len(trades)
    except Exception as e:
        print(f"Error in phased execution: {e}")
        return None, 0

def run_performance_comparison():
    """Run performance comparison across different dataset sizes"""
    dataset_sizes = [1000, 5000, 10000, 25000, 50000]
    results = []

    print("=== Phased Entry Performance Analysis ===")
    print(f"{'Size':>8} | {'Original':>12} | {'Phased':>12} | {'Ratio':>8} | {'Linear?':>8}")
    print("-" * 60)

    for size in dataset_sizes:
        print(f"Testing dataset size: {size}")

        # Create test data
        df = create_test_data(size)
        signals = create_simple_signals(df)

        # Test original execution
        original_time, original_trades = test_original_execution(df, signals)

        # Test phased execution
        phased_time, phased_trades = test_phased_execution(df, signals)

        if original_time is not None and phased_time is not None:
            ratio = phased_time / original_time if original_time > 0 else float('inf')

            # Check if scaling is roughly linear (bad) vs constant (good)
            is_linear = "YES" if ratio > size / 1000 else "NO"

            results.append({
                'size': size,
                'original_time': original_time,
                'phased_time': phased_time,
                'ratio': ratio,
                'original_trades': original_trades,
                'phased_trades': phased_trades
            })

            print(f"{size:>8} | {original_time:>10.4f}s | {phased_time:>10.4f}s | {ratio:>6.2f}x | {is_linear:>8}")
        else:
            print(f"{size:>8} | ERROR")

    return results

def analyze_scaling(results):
    """Analyze scaling characteristics"""
    print("\n=== Scaling Analysis ===")

    if len(results) < 2:
        print("Not enough data points for scaling analysis")
        return

    # Calculate scaling factors
    base_result = results[0]

    print("Time scaling factors (relative to 1k dataset):")
    print(f"{'Size':>8} | {'Original':>12} | {'Phased':>12} | {'Expected Linear':>15}")
    print("-" * 55)

    for result in results:
        size_factor = result['size'] / base_result['size']

        original_factor = result['original_time'] / base_result['original_time'] if base_result['original_time'] > 0 else 0
        phased_factor = result['phased_time'] / base_result['phased_time'] if base_result['phased_time'] > 0 else 0

        print(f"{result['size']:>8} | {original_factor:>10.2f}x | {phased_factor:>10.2f}x | {size_factor:>13.2f}x")

    # Check if phased is scaling linearly (bad)
    largest = results[-1]
    size_increase = largest['size'] / base_result['size']
    time_increase = largest['phased_time'] / base_result['phased_time'] if base_result['phased_time'] > 0 else 0

    print(f"\nScaling Assessment:")
    print(f"Dataset size increased: {size_increase:.1f}x")
    print(f"Phased time increased: {time_increase:.1f}x")

    if time_increase > size_increase * 0.8:  # If time scales > 80% of size increase
        print("❌ CRITICAL: Phased execution is scaling LINEARLY (O(n))")
        print("   This will cause severe performance issues with large datasets")
        print("   Original vectorized approach should scale roughly O(1)")
    else:
        print("✅ GOOD: Phased execution scaling is reasonable")

def main():
    """Run performance analysis"""
    print("Phased Entry Performance Analysis")
    print("=" * 50)

    results = run_performance_comparison()
    analyze_scaling(results)

    print("\n" + "=" * 50)
    if any(r['phased_time'] > r['original_time'] * 10 for r in results):
        print("⚠️  WARNING: Phased execution is significantly slower than original")
        print("   This suggests the implementation is not using vectorized operations")
        print("   and may not be suitable for large datasets.")
    else:
        print("✅ Performance looks reasonable")

if __name__ == "__main__":
    main()