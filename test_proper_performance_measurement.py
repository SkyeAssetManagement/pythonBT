"""
Proper performance measurement excluding compilation time
Fair comparison between loop-based and vectorized implementations
"""

import sys
import os
import pandas as pd
import numpy as np
import time

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def create_test_data(bars=1000):
    """Create synthetic price data for testing"""
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', periods=bars, freq='1h')

    returns = np.random.normal(0.001, 0.02, bars)
    prices = [100.0]
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

def create_signals(df, frequency=0.02):
    """Create trading signals"""
    np.random.seed(42)
    signals = pd.Series(0, index=range(len(df)))

    # Generate entry/exit pairs
    for i in range(50, len(df) - 50, 100):
        if np.random.random() < frequency * 10:
            signals.iloc[i] = 1
            exit_bar = min(i + np.random.randint(10, 30), len(df) - 1)
            signals.iloc[exit_bar] = 0

    return signals

def warm_up_engines():
    """Warm up both engines to trigger compilation"""
    print("Warming up engines (triggering compilation)...")

    # Small dataset for warmup
    warmup_df = create_test_data(100)
    warmup_signals = create_signals(warmup_df)

    # Warm up original engine
    try:
        from trading.core.standalone_execution import StandaloneExecutionEngine, ExecutionConfig
        config = ExecutionConfig()
        engine = StandaloneExecutionEngine(config)
        engine.execute_signals(warmup_signals, warmup_df)
        print("  + Original engine warmed up")
    except Exception as e:
        print(f"  - Original engine warmup failed: {e}")

    # Warm up loop-based engine
    try:
        from trading.core.phased_execution_engine import PhasedExecutionEngine, PhasedExecutionConfig
        config = PhasedExecutionConfig()
        config.phased_config.enabled = False
        engine = PhasedExecutionEngine(config)
        engine.execute_signals_with_phases(warmup_signals, warmup_df, "TEST")
        print("  + Loop-based engine warmed up")
    except Exception as e:
        print(f"  - Loop-based engine warmup failed: {e}")

    # Warm up vectorized engine (most important for numba)
    try:
        from trading.core.truly_vectorized_execution import TrulyVectorizedEngine, ExecutionConfig
        config = ExecutionConfig()
        engine = TrulyVectorizedEngine(config)
        engine.phased_config.enabled = False
        # Run multiple times to ensure numba compilation is complete
        for _ in range(3):
            engine.execute_signals_truly_vectorized(warmup_signals, warmup_df)
        print("  + Vectorized engine warmed up (numba compiled)")
    except Exception as e:
        print(f"  - Vectorized engine warmup failed: {e}")

    print("Warmup complete - all compilation done\n")

def test_original_engine(df, signals, runs=3):
    """Test original engine with multiple runs"""
    try:
        from trading.core.standalone_execution import StandaloneExecutionEngine, ExecutionConfig

        config = ExecutionConfig()
        engine = StandaloneExecutionEngine(config)

        times = []
        for _ in range(runs):
            start_time = time.perf_counter()
            trades = engine.execute_signals(signals, df)
            end_time = time.perf_counter()
            times.append(end_time - start_time)

        avg_time = sum(times) / len(times)
        return avg_time, len(trades), "Original"
    except Exception as e:
        print(f"Error in original: {e}")
        return None, 0, "Original"

def test_loop_engine(df, signals, runs=3):
    """Test loop-based engine with multiple runs"""
    try:
        from trading.core.phased_execution_engine import PhasedExecutionEngine, PhasedExecutionConfig

        config = PhasedExecutionConfig()
        config.phased_config.enabled = False
        engine = PhasedExecutionEngine(config)

        times = []
        for _ in range(runs):
            start_time = time.perf_counter()
            trades = engine.execute_signals_with_phases(signals, df, "TEST")
            end_time = time.perf_counter()
            times.append(end_time - start_time)

        avg_time = sum(times) / len(times)
        return avg_time, len(trades), "Loop-based"
    except Exception as e:
        print(f"Error in loop-based: {e}")
        return None, 0, "Loop-based"

def test_vectorized_engine(df, signals, runs=3):
    """Test vectorized engine with multiple runs"""
    try:
        from trading.core.truly_vectorized_execution import TrulyVectorizedEngine, ExecutionConfig

        config = ExecutionConfig()
        engine = TrulyVectorizedEngine(config)
        engine.phased_config.enabled = False

        times = []
        for _ in range(runs):
            start_time = time.perf_counter()
            trades = engine.execute_signals_truly_vectorized(signals, df)
            end_time = time.perf_counter()
            times.append(end_time - start_time)

        avg_time = sum(times) / len(times)
        return avg_time, len(trades), "Vectorized"
    except Exception as e:
        print(f"Error in vectorized: {e}")
        return None, 0, "Vectorized"

def run_proper_performance_test():
    """Run proper performance test excluding compilation time"""

    # First, warm up all engines
    warm_up_engines()

    # Test with increasing dataset sizes
    dataset_sizes = [1000, 5000, 10000, 25000, 50000]
    results = []

    print("=== Fair Performance Comparison (Post-Compilation) ===")
    print(f"{'Size':>8} | {'Original':>10} | {'Loop':>10} | {'Vector':>10} | {'Speedup':>10}")
    print("-" * 65)

    for size in dataset_sizes:
        print(f"Testing {size} bars (averaging 3 runs)...")

        # Create test data
        df = create_test_data(size)
        signals = create_signals(df)

        # Test all implementations (multiple runs each)
        orig_time, orig_trades, _ = test_original_engine(df, signals, runs=3)
        loop_time, loop_trades, _ = test_loop_engine(df, signals, runs=3)
        vec_time, vec_trades, _ = test_vectorized_engine(df, signals, runs=3)

        if all(t is not None for t in [orig_time, loop_time, vec_time]):
            speedup = orig_time / vec_time if vec_time > 0 else 0

            result = {
                'size': size,
                'original_time': orig_time,
                'loop_time': loop_time,
                'vector_time': vec_time,
                'speedup': speedup,
                'original_trades': orig_trades,
                'loop_trades': loop_trades,
                'vector_trades': vec_trades
            }
            results.append(result)

            print(f"{size:>8} | {orig_time:>8.4f}s | {loop_time:>8.4f}s | {vec_time:>8.4f}s | {speedup:>8.2f}x")
        else:
            print(f"{size:>8} | ERROR in one or more implementations")

    return results

def analyze_proper_scaling(results):
    """Analyze scaling characteristics properly"""
    print("\n=== Proper Scaling Analysis ===")

    if len(results) < 2:
        print("Not enough data points")
        return

    base_result = results[0]
    largest_result = results[-1]
    size_ratio = largest_result['size'] / base_result['size']

    print(f"Dataset size increased: {size_ratio:.1f}x")
    print(f"Time scaling factors:")

    implementations = [
        ('Original', 'original_time'),
        ('Loop-based', 'loop_time'),
        ('Vectorized', 'vector_time')
    ]

    for name, time_key in implementations:
        base_time = base_result[time_key]
        large_time = largest_result[time_key]

        if base_time and large_time and base_time > 0:
            time_ratio = large_time / base_time

            # Classification
            if time_ratio < size_ratio * 0.3:
                classification = "EXCELLENT (Sub-linear)"
            elif time_ratio < size_ratio * 0.8:
                classification = "GOOD (Better than linear)"
            elif time_ratio < size_ratio * 1.3:
                classification = "LINEAR (O(n))"
            else:
                classification = "POOR (Super-linear)"

            print(f"  {name:>12}: {time_ratio:>6.1f}x - {classification}")

    # Performance summary
    print(f"\n=== Performance Summary ===")
    avg_speedup = sum(r['speedup'] for r in results) / len(results)
    print(f"Average vectorized speedup: {avg_speedup:.1f}x")

    # Check if vectorized is actually faster
    if avg_speedup > 2.0:
        print("+ Vectorized implementation is significantly faster")
    elif avg_speedup > 1.2:
        print("+ Vectorized implementation shows improvement")
    elif avg_speedup < 0.8:
        print("- Vectorized implementation is actually slower!")
    else:
        print("~ Vectorized implementation shows marginal improvement")

def main():
    """Run proper performance measurement"""
    print("Proper Performance Measurement")
    print("Excluding compilation time from all measurements")
    print("=" * 60)

    results = run_proper_performance_test()

    if results:
        analyze_proper_scaling(results)

        print("\n" + "=" * 60)
        print("CONCLUSIONS:")
        print("- All measurements exclude compilation/initialization time")
        print("- Each measurement is average of 3 runs")
        print("- Fair comparison between all implementations")

        # Final assessment
        if results:
            final_result = results[-1]  # Largest dataset
            if final_result['vector_time'] < final_result['original_time']:
                improvement = final_result['original_time'] / final_result['vector_time']
                print(f"- Vectorized is {improvement:.1f}x faster on largest dataset")
            else:
                degradation = final_result['vector_time'] / final_result['original_time']
                print(f"- Vectorized is {degradation:.1f}x slower on largest dataset")
    else:
        print("ERROR: Could not complete performance measurement")

if __name__ == "__main__":
    main()