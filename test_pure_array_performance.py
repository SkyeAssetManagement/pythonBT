"""
Test pure array processing implementation
Compare true O(1) scaling vs previous O(n) implementations
"""

import sys
import os
import pandas as pd
import numpy as np
import time

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def create_test_data(bars=1000, start_price=100.0):
    """Create synthetic price data for testing"""
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', periods=bars, freq='1h')

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

def create_simple_signals(df, signal_frequency=0.02):
    """Create simple signals for testing"""
    np.random.seed(42)
    signals = pd.Series(0, index=range(len(df)))

    # Generate some long signals
    for i in range(50, len(df) - 50, 100):  # Every 100 bars
        if np.random.random() < signal_frequency * 10:  # Higher chance
            signals.iloc[i] = 1
            # Add exit after some bars
            exit_bar = min(i + np.random.randint(10, 30), len(df) - 1)
            signals.iloc[exit_bar] = 0

    return signals

def warm_up_all_engines():
    """Warm up all engines to exclude compilation time"""
    print("Warming up all engines...")

    warmup_df = create_test_data(100)
    warmup_signals = create_simple_signals(warmup_df)

    # Warm up original
    try:
        from trading.core.standalone_execution import StandaloneExecutionEngine, ExecutionConfig
        config = ExecutionConfig()
        engine = StandaloneExecutionEngine(config)
        engine.execute_signals(warmup_signals, warmup_df)
        print("  + Original engine warmed up")
    except Exception as e:
        print(f"  - Original engine error: {e}")

    # Warm up vectorized (has loops)
    try:
        from trading.core.truly_vectorized_execution import TrulyVectorizedEngine, ExecutionConfig
        config = ExecutionConfig()
        engine = TrulyVectorizedEngine(config)
        engine.phased_config.enabled = False
        for _ in range(3):  # Multiple runs to ensure numba compilation
            engine.execute_signals_truly_vectorized(warmup_signals, warmup_df)
        print("  + Vectorized (with loops) engine warmed up")
    except Exception as e:
        print(f"  - Vectorized engine error: {e}")

    # Warm up pure array
    try:
        from trading.core.pure_array_execution import PureArrayExecutionEngine, ExecutionConfig
        config = ExecutionConfig()
        engine = PureArrayExecutionEngine(config)
        for _ in range(3):  # Multiple runs for numba compilation
            engine.execute_signals_pure_array(warmup_signals, warmup_df)
        print("  + Pure Array engine warmed up")
    except Exception as e:
        print(f"  - Pure Array engine error: {e}")

    print("All engines warmed up\n")

def test_original_engine(df, signals, runs=3):
    """Test original engine"""
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

def test_vectorized_engine(df, signals, runs=3):
    """Test vectorized engine (still has loops)"""
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
        return avg_time, len(trades), "Vectorized+Loops"
    except Exception as e:
        print(f"Error in vectorized: {e}")
        return None, 0, "Vectorized+Loops"

def test_pure_array_engine(df, signals, runs=3):
    """Test pure array engine"""
    try:
        from trading.core.pure_array_execution import PureArrayExecutionEngine, ExecutionConfig

        config = ExecutionConfig()
        engine = PureArrayExecutionEngine(config)

        times = []
        for _ in range(runs):
            start_time = time.perf_counter()
            trades = engine.execute_signals_pure_array(signals, df)
            end_time = time.perf_counter()
            times.append(end_time - start_time)

        avg_time = sum(times) / len(times)
        return avg_time, len(trades), "Pure Array"
    except Exception as e:
        print(f"Error in pure array: {e}")
        return None, 0, "Pure Array"

def run_scaling_comparison():
    """Compare scaling characteristics"""

    # Warm up first
    warm_up_all_engines()

    dataset_sizes = [1000, 5000, 10000, 25000, 50000]
    results = []

    print("=== Scaling Comparison: Original vs Vectorized+Loops vs Pure Array ===")
    print(f"{'Size':>8} | {'Original':>10} | {'Vec+Loops':>10} | {'PureArray':>10} | {'Array Speedup':>12}")
    print("-" * 75)

    for size in dataset_sizes:
        print(f"Testing {size} bars...")

        # Create test data
        df = create_test_data(size)
        signals = create_simple_signals(df)

        # Test all implementations
        orig_time, orig_trades, _ = test_original_engine(df, signals, runs=3)
        vec_time, vec_trades, _ = test_vectorized_engine(df, signals, runs=3)
        array_time, array_trades, _ = test_pure_array_engine(df, signals, runs=3)

        if all(t is not None for t in [orig_time, vec_time, array_time]):
            # Calculate speedup of pure array vs original
            array_speedup = orig_time / array_time if array_time > 0 else 0

            result = {
                'size': size,
                'original_time': orig_time,
                'vectorized_time': vec_time,
                'array_time': array_time,
                'original_trades': orig_trades,
                'vectorized_trades': vec_trades,
                'array_trades': array_trades,
                'array_speedup': array_speedup
            }
            results.append(result)

            print(f"{size:>8} | {orig_time:>8.4f}s | {vec_time:>8.4f}s | {array_time:>8.4f}s | {array_speedup:>10.1f}x")
        else:
            print(f"{size:>8} | ERROR in one or more implementations")

    return results

def analyze_true_scaling(results):
    """Analyze if we achieved true O(1) scaling"""
    print("\n=== True Scaling Analysis ===")

    if len(results) < 2:
        print("Not enough data points")
        return

    base_result = results[0]
    largest_result = results[-1]
    size_ratio = largest_result['size'] / base_result['size']

    print(f"Dataset size increased: {size_ratio:.1f}x")
    print(f"Time scaling analysis:")

    implementations = [
        ('Original', 'original_time'),
        ('Vectorized+Loops', 'vectorized_time'),
        ('Pure Array', 'array_time')
    ]

    for name, time_key in implementations:
        base_time = base_result[time_key]
        large_time = largest_result[time_key]

        if base_time and large_time and base_time > 0:
            time_ratio = large_time / base_time
            scaling_factor = time_ratio / size_ratio

            # Classification
            if scaling_factor < 0.1:
                classification = "EXCELLENT O(1) - TRUE CONSTANT TIME"
            elif scaling_factor < 0.3:
                classification = "VERY GOOD - SUB-LINEAR"
            elif scaling_factor < 0.7:
                classification = "GOOD - BETTER THAN LINEAR"
            elif scaling_factor < 1.3:
                classification = "LINEAR O(n) - PROBLEM"
            else:
                classification = "POOR - SUPER LINEAR"

            print(f"  {name:>15}: {time_ratio:>6.1f}x time | {scaling_factor:>5.2f} factor | {classification}")

    # Final assessment
    array_base = base_result['array_time']
    array_large = largest_result['array_time']
    if array_base and array_large and array_base > 0:
        array_scaling = (array_large / array_base) / size_ratio

        print(f"\n=== FINAL ASSESSMENT ===")
        if array_scaling < 0.1:
            print("+ SUCCESS: Pure Array achieved TRUE O(1) constant time scaling!")
            print("+ Performance independent of dataset size")
        elif array_scaling < 0.5:
            print("+ GOOD: Pure Array achieved sub-linear scaling")
            print("+ Significant improvement over linear implementations")
        else:
            print("- ISSUE: Pure Array still shows linear-like behavior")
            print("- Further optimization needed")

def main():
    """Run complete pure array performance analysis"""
    print("Pure Array Processing Performance Analysis")
    print("Testing true O(1) scaling vs previous implementations")
    print("=" * 70)

    results = run_scaling_comparison()

    if results:
        analyze_true_scaling(results)

        print("\n" + "=" * 70)
        print("KEY FINDINGS:")

        # Compare average performance
        avg_array_speedup = sum(r['array_speedup'] for r in results) / len(results)
        print(f"Average Pure Array speedup vs Original: {avg_array_speedup:.1f}x")

        # Check final scaling
        if len(results) >= 2:
            base = results[0]
            final = results[-1]
            size_increase = final['size'] / base['size']

            for impl_name, time_key in [('Pure Array', 'array_time'), ('Vectorized+Loops', 'vectorized_time')]:
                base_time = base[time_key]
                final_time = final[time_key]
                if base_time and final_time and base_time > 0:
                    time_increase = final_time / base_time
                    scaling_quality = time_increase / size_increase

                    if scaling_quality < 0.2:
                        quality = "EXCELLENT O(1)"
                    elif scaling_quality < 0.5:
                        quality = "GOOD sub-linear"
                    elif scaling_quality < 1.2:
                        quality = "LINEAR O(n)"
                    else:
                        quality = "POOR super-linear"

                    print(f"{impl_name}: {scaling_quality:.2f} scaling factor - {quality}")

    else:
        print("ERROR: Could not complete analysis")

if __name__ == "__main__":
    main()