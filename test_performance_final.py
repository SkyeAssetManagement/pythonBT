"""
Final performance comparison between all implementations
Tests O(1) vs O(n) scaling with realistic datasets
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

def test_original_engine(df, signals):
    """Test original standalone execution engine"""
    try:
        from trading.core.standalone_execution import StandaloneExecutionEngine, ExecutionConfig

        config = ExecutionConfig()
        engine = StandaloneExecutionEngine(config)

        start_time = time.perf_counter()
        trades = engine.execute_signals(signals, df)
        end_time = time.perf_counter()

        return end_time - start_time, len(trades), "Original"
    except Exception as e:
        print(f"Error in original: {e}")
        return None, 0, "Original"

def test_loop_phased_engine(df, signals):
    """Test loop-based phased execution engine"""
    try:
        from trading.core.phased_execution_engine import PhasedExecutionEngine, PhasedExecutionConfig

        config = PhasedExecutionConfig()
        config.phased_config.enabled = False  # Test without phases first
        engine = PhasedExecutionEngine(config)

        start_time = time.perf_counter()
        trades = engine.execute_signals_with_phases(signals, df, "TEST")
        end_time = time.perf_counter()

        return end_time - start_time, len(trades), "Loop-based"
    except Exception as e:
        print(f"Error in loop-based: {e}")
        return None, 0, "Loop-based"

def test_vectorized_engine(df, signals):
    """Test truly vectorized execution engine"""
    try:
        from trading.core.truly_vectorized_execution import TrulyVectorizedEngine, ExecutionConfig

        config = ExecutionConfig()
        engine = TrulyVectorizedEngine(config)
        engine.phased_config.enabled = False  # Test without phases first

        start_time = time.perf_counter()
        trades = engine.execute_signals_truly_vectorized(signals, df)
        end_time = time.perf_counter()

        return end_time - start_time, len(trades), "Vectorized"
    except Exception as e:
        print(f"Error in vectorized: {e}")
        return None, 0, "Vectorized"

def run_comprehensive_comparison():
    """Run comprehensive performance comparison"""

    # Test with increasing dataset sizes
    dataset_sizes = [1000, 5000, 10000, 25000, 50000]
    results = []

    print("=== Comprehensive Performance Comparison ===")
    print("Testing Original vs Loop-based vs Vectorized implementations")
    print()
    print(f"{'Size':>8} | {'Original':>10} | {'Loop':>10} | {'Vector':>10} | {'Loop/Orig':>10} | {'Vector/Orig':>12}")
    print("-" * 75)

    for size in dataset_sizes:
        print(f"Testing {size} bars...")

        # Create test data
        df = create_test_data(size)
        signals = create_simple_signals(df)

        # Test all implementations
        orig_time, orig_trades, _ = test_original_engine(df, signals)
        loop_time, loop_trades, _ = test_loop_phased_engine(df, signals)
        vec_time, vec_trades, _ = test_vectorized_engine(df, signals)

        # Calculate ratios
        loop_ratio = loop_time / orig_time if orig_time and orig_time > 0 else 0
        vec_ratio = vec_time / orig_time if orig_time and orig_time > 0 else 0

        result = {
            'size': size,
            'original_time': orig_time,
            'loop_time': loop_time,
            'vector_time': vec_time,
            'original_trades': orig_trades,
            'loop_trades': loop_trades,
            'vector_trades': vec_trades,
            'loop_ratio': loop_ratio,
            'vector_ratio': vec_ratio
        }
        results.append(result)

        print(f"{size:>8} | {orig_time:>8.4f}s | {loop_time:>8.4f}s | {vec_time:>8.4f}s | {loop_ratio:>8.2f}x | {vec_ratio:>10.2f}x")

    return results

def analyze_scaling(results):
    """Analyze scaling characteristics"""
    print("\n=== Scaling Analysis ===")

    if len(results) < 2:
        print("Not enough data points")
        return

    base_result = results[0]
    largest_result = results[-1]

    size_increase = largest_result['size'] / base_result['size']

    print(f"Dataset size increased: {size_increase:.1f}x")
    print(f"Time increases:")

    for impl in ['original', 'loop', 'vector']:
        base_time = base_result[f'{impl}_time']
        large_time = largest_result[f'{impl}_time']

        if base_time and large_time and base_time > 0:
            time_increase = large_time / base_time

            # Assess scaling quality
            if time_increase < size_increase * 0.3:
                quality = "EXCELLENT (Sub-linear)"
            elif time_increase < size_increase * 0.7:
                quality = "GOOD (Better than linear)"
            elif time_increase < size_increase * 1.3:
                quality = "LINEAR (O(n)) - PROBLEM"
            else:
                quality = "POOR (Super-linear) - MAJOR PROBLEM"

            print(f"  {impl.capitalize():>12}: {time_increase:>6.1f}x - {quality}")

def main():
    """Run comprehensive analysis"""
    print("Comprehensive Performance Analysis")
    print("Testing array processing vs loop processing")
    print("=" * 60)

    results = run_comprehensive_comparison()

    if results:
        analyze_scaling(results)

        print("\n" + "=" * 60)
        print("KEY FINDINGS:")

        # Check if any implementation scales properly
        base = results[0]
        largest = results[-1]
        size_ratio = largest['size'] / base['size']

        for impl_name, time_key in [('Original', 'original_time'), ('Loop-based', 'loop_time'), ('Vectorized', 'vector_time')]:
            base_time = base[time_key]
            large_time = largest[time_key]

            if base_time and large_time and base_time > 0:
                time_ratio = large_time / base_time

                if time_ratio < size_ratio * 0.5:
                    print(f"+ {impl_name}: GOOD scaling (sub-linear)")
                elif time_ratio > size_ratio * 1.2:
                    print(f"- {impl_name}: POOR scaling (linear or worse)")
                else:
                    print(f"~ {impl_name}: Acceptable scaling")

        print("\nFor truly efficient array processing, we should see sub-linear")
        print("scaling where time increase is much less than dataset size increase.")

    else:
        print("ERROR: Could not complete performance comparison")

if __name__ == "__main__":
    main()