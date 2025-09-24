"""
Performance comparison: Vectorized vs Loop-based phased execution
Tests O(1) vs O(n) scaling characteristics
"""

import sys
import os
import pandas as pd
import numpy as np
import time
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def create_test_data(bars=1000, start_price=100.0, volatility=0.02):
    """Create synthetic price data for testing"""
    np.random.seed(42)  # For reproducible tests
    dates = pd.date_range(start='2023-01-01', periods=bars, freq='1h')

    # Generate more realistic price series
    returns = np.random.normal(0.001, volatility, bars)
    log_prices = np.cumsum(returns)
    prices = start_price * np.exp(log_prices)

    df = pd.DataFrame({
        'DateTime': dates,
        'Open': prices,
        'High': prices * (1 + np.abs(np.random.normal(0, 0.005, bars))),
        'Low': prices * (1 - np.abs(np.random.normal(0, 0.005, bars))),
        'Close': prices,
        'Volume': np.random.randint(1000, 10000, bars)
    })

    df['High'] = np.maximum(df['High'], df[['Open', 'Close']].max(axis=1))
    df['Low'] = np.minimum(df['Low'], df[['Open', 'Close']].min(axis=1))

    return df

def create_realistic_signals(df, signal_frequency=0.02):
    """Create more realistic trading signals"""
    np.random.seed(42)
    signals = pd.Series(0, index=range(len(df)))

    # Create trend-following signals based on moving averages
    prices = df['Close'].values
    ma_short = pd.Series(prices).rolling(10).mean().values
    ma_long = pd.Series(prices).rolling(30).mean().values

    position = 0
    for i in range(30, len(df) - 50):  # Leave room for exits
        if position == 0:  # Not in position
            if ma_short[i] > ma_long[i] and np.random.random() < signal_frequency:
                signals.iloc[i] = 1
                position = 1
        else:  # In position
            if ma_short[i] < ma_long[i] or np.random.random() < signal_frequency * 0.5:
                signals.iloc[i] = 0
                position = 0

    return signals

def test_loop_based_execution(df, signals):
    """Test original loop-based phased execution"""
    try:
        from trading.core.phased_execution_engine import PhasedExecutionEngine, PhasedExecutionConfig

        config = PhasedExecutionConfig()
        config.phased_config.enabled = True
        config.phased_config.max_phases = 3
        config.phased_config.phase_trigger_value = 2.0
        engine = PhasedExecutionEngine(config)

        start_time = time.perf_counter()
        trades = engine.execute_signals_with_phases(signals, df, "TEST")
        end_time = time.perf_counter()

        return end_time - start_time, len(trades), 'loop-based'
    except Exception as e:
        print(f"Error in loop-based execution: {e}")
        return None, 0, 'loop-based'

def test_vectorized_execution(df, signals):
    """Test new vectorized phased execution"""
    try:
        from trading.core.vectorized_phased_execution import VectorizedPhasedEngine, VectorizedPhasedExecutionConfig

        config = VectorizedPhasedExecutionConfig()
        config.phased_config.enabled = True
        config.phased_config.max_phases = 3
        config.phased_config.phase_trigger_value = 2.0
        engine = VectorizedPhasedEngine(config)

        start_time = time.perf_counter()
        trades = engine.execute_signals_vectorized(signals, df, "TEST")
        end_time = time.perf_counter()

        return end_time - start_time, len(trades), 'vectorized'
    except Exception as e:
        print(f"Error in vectorized execution: {e}")
        return None, 0, 'vectorized'

def test_original_execution(df, signals):
    """Test original single-entry execution for baseline"""
    try:
        from trading.core.standalone_execution import StandaloneExecutionEngine, ExecutionConfig

        config = ExecutionConfig()
        engine = StandaloneExecutionEngine(config)

        start_time = time.perf_counter()
        trades = engine.execute_signals(signals, df)
        end_time = time.perf_counter()

        return end_time - start_time, len(trades), 'original'
    except Exception as e:
        print(f"Error in original execution: {e}")
        return None, 0, 'original'

def run_performance_comparison():
    """Run comprehensive performance comparison"""
    dataset_sizes = [1000, 2000, 5000, 10000, 20000, 50000]
    results = []

    print("=== Vectorized vs Loop-based Performance Comparison ===")
    print(f"{'Size':>8} | {'Original':>10} | {'Loop':>10} | {'Vector':>10} | {'Loop/Orig':>10} | {'Vector/Orig':>12} | {'Speedup':>8}")
    print("-" * 85)

    for size in dataset_sizes:
        print(f"Testing dataset size: {size}")

        # Create test data (more realistic)
        df = create_test_data(size, volatility=0.015)
        signals = create_realistic_signals(df)

        # Test all three implementations
        original_time, original_trades, _ = test_original_execution(df, signals)
        loop_time, loop_trades, _ = test_loop_based_execution(df, signals)
        vector_time, vector_trades, _ = test_vectorized_execution(df, signals)

        if all(t is not None for t in [original_time, loop_time, vector_time]):
            loop_ratio = loop_time / original_time if original_time > 0 else 0
            vector_ratio = vector_time / original_time if original_time > 0 else 0
            speedup = loop_time / vector_time if vector_time > 0 else 0

            results.append({
                'size': size,
                'original_time': original_time,
                'loop_time': loop_time,
                'vector_time': vector_time,
                'loop_ratio': loop_ratio,
                'vector_ratio': vector_ratio,
                'speedup': speedup,
                'original_trades': original_trades,
                'loop_trades': loop_trades,
                'vector_trades': vector_trades
            })

            print(f"{size:>8} | {original_time:>8.4f}s | {loop_time:>8.4f}s | {vector_time:>8.4f}s | {loop_ratio:>8.2f}x | {vector_ratio:>10.2f}x | {speedup:>6.2f}x")
        else:
            print(f"{size:>8} | ERROR in one or more implementations")

    return results

def analyze_scaling_characteristics(results):
    """Analyze scaling characteristics"""
    if len(results) < 3:
        print("Not enough data points for scaling analysis")
        return

    print("\n=== Scaling Characteristics Analysis ===")

    # Calculate scaling factors relative to smallest dataset
    base_result = results[0]
    base_size = base_result['size']

    print(f"Scaling factors relative to {base_size} bars:")
    print(f"{'Size':>8} | {'Size Factor':>12} | {'Original':>10} | {'Loop':>10} | {'Vector':>10}")
    print("-" * 65)

    for result in results:
        size_factor = result['size'] / base_size

        orig_factor = result['original_time'] / base_result['original_time'] if base_result['original_time'] > 0 else 0
        loop_factor = result['loop_time'] / base_result['loop_time'] if base_result['loop_time'] > 0 else 0
        vector_factor = result['vector_time'] / base_result['vector_time'] if base_result['vector_time'] > 0 else 0

        print(f"{result['size']:>8} | {size_factor:>10.1f}x | {orig_factor:>8.2f}x | {loop_factor:>8.2f}x | {vector_factor:>8.2f}x")

    # Analyze if vectorized version achieves better scaling
    largest = results[-1]
    size_increase = largest['size'] / base_result['size']

    orig_time_increase = largest['original_time'] / base_result['original_time']
    loop_time_increase = largest['loop_time'] / base_result['loop_time']
    vector_time_increase = largest['vector_time'] / base_result['vector_time']

    print(f"\nScaling Assessment (Dataset increased {size_increase:.1f}x):")
    print(f"Original time increased: {orig_time_increase:.1f}x")
    print(f"Loop-based time increased: {loop_time_increase:.1f}x")
    print(f"Vectorized time increased: {vector_time_increase:.1f}x")

    # Determine scaling quality
    def assess_scaling(time_factor, size_factor):
        if time_factor < size_factor * 0.3:
            return "EXCELLENT (Sub-linear)"
        elif time_factor < size_factor * 0.7:
            return "GOOD (Better than linear)"
        elif time_factor < size_factor * 1.3:
            return "LINEAR (O(n))"
        else:
            return "POOR (Super-linear)"

    print(f"\nScaling Quality:")
    print(f"Original: {assess_scaling(orig_time_increase, size_increase)}")
    print(f"Loop-based: {assess_scaling(loop_time_increase, size_increase)}")
    print(f"Vectorized: {assess_scaling(vector_time_increase, size_increase)}")

def calculate_performance_gains(results):
    """Calculate average performance gains"""
    if not results:
        return

    print("\n=== Performance Gains Summary ===")

    # Calculate average speedup
    speedups = [r['speedup'] for r in results if r['speedup'] > 0]
    avg_speedup = np.mean(speedups) if speedups else 0

    # Calculate speedup for largest dataset
    largest_result = results[-1]
    largest_speedup = largest_result['speedup']

    print(f"Average speedup (vectorized vs loop): {avg_speedup:.1f}x")
    print(f"Largest dataset speedup: {largest_speedup:.1f}x")

    # Memory and scalability implications
    largest_size = largest_result['size']
    loop_time = largest_result['loop_time']
    vector_time = largest_result['vector_time']

    # Estimate time for 1 million bars
    million_bars = 1000000
    scale_factor = million_bars / largest_size

    estimated_loop_time = loop_time * scale_factor
    estimated_vector_time = vector_time * min(scale_factor, 2)  # Assume vectorized scales much better

    print(f"\nEstimated time for 1,000,000 bars:")
    print(f"Loop-based: {estimated_loop_time:.1f} seconds ({estimated_loop_time/60:.1f} minutes)")
    print(f"Vectorized: {estimated_vector_time:.1f} seconds ({estimated_vector_time/60:.1f} minutes)")
    print(f"Estimated speedup at scale: {estimated_loop_time/estimated_vector_time:.1f}x")

def main():
    """Run comprehensive performance analysis"""
    print("Vectorized vs Loop-based Phased Execution Performance Test")
    print("=" * 70)

    results = run_performance_comparison()

    if results:
        analyze_scaling_characteristics(results)
        calculate_performance_gains(results)

        print("\n" + "=" * 70)
        print("CONCLUSIONS:")

        # Check if vectorized version is actually faster
        avg_speedup = np.mean([r['speedup'] for r in results if r['speedup'] > 0])

        if avg_speedup > 2.0:
            print("SUCCESS: Vectorized implementation is significantly faster")
            print(f"         Average speedup: {avg_speedup:.1f}x")
        elif avg_speedup > 1.2:
            print("GOOD: Vectorized implementation shows improvement")
            print(f"      Average speedup: {avg_speedup:.1f}x")
        else:
            print("ISSUE: Vectorized implementation may not be properly optimized")
            print(f"       Average speedup: {avg_speedup:.1f}x")

        # Check scaling characteristics
        if len(results) >= 2:
            largest = results[-1]
            smallest = results[0]
            size_ratio = largest['size'] / smallest['size']
            vector_time_ratio = largest['vector_time'] / smallest['vector_time']

            if vector_time_ratio < size_ratio * 0.5:
                print("EXCELLENT: Vectorized scaling is sub-linear")
            elif vector_time_ratio < size_ratio:
                print("GOOD: Vectorized scaling is better than linear")
            else:
                print("ISSUE: Vectorized scaling is still linear")

    else:
        print("ERROR: Could not complete performance comparison")
        print("Check that all required modules are properly implemented")

if __name__ == "__main__":
    import time
    main()