"""
CRITICAL PERFORMANCE TEST: mplfinance with 7M+ datapoints
This is the make-or-break test for the mplfinance solution
"""
import sys
import time
import os
import gc
import psutil
from pathlib import Path
import numpy as np
import pandas as pd
import datetime
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for performance


def create_large_dataset(n_points):
    """Create large OHLC dataset for performance testing"""
    print(f"Creating {n_points:,} datapoint test dataset...")
    
    # Use memory-efficient data generation
    start_time = pd.Timestamp('2020-01-01 09:30:00')
    
    # Generate timestamps efficiently
    timestamps = pd.date_range(start_time, periods=n_points, freq='1min')
    
    # Generate realistic price data with trends
    np.random.seed(42)  # Reproducible results
    base_price = 4000.0
    
    # Use cumulative sum for realistic price movement
    price_changes = np.random.normal(0, 0.5, n_points)
    prices = base_price + np.cumsum(price_changes)
    
    # Create OHLC with realistic spread
    opens = prices.copy()
    closes = prices + np.random.normal(0, 0.2, n_points)
    
    # Highs and lows
    hl_spread = np.abs(np.random.normal(0, 1, n_points))
    highs = np.maximum(opens, closes) + hl_spread
    lows = np.minimum(opens, closes) - hl_spread
    
    # Volume data
    volumes = np.random.randint(100, 10000, n_points)
    
    # Create DataFrame
    df = pd.DataFrame({
        'Open': opens,
        'High': highs, 
        'Low': lows,
        'Close': closes,
        'Volume': volumes
    }, index=timestamps)
    
    # Memory usage check
    memory_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
    print(f"Dataset created: {n_points:,} bars, {memory_mb:.1f} MB memory")
    
    return df


def test_memory_usage():
    """Check current memory usage"""
    process = psutil.Process(os.getpid())
    memory_mb = process.memory_info().rss / 1024 / 1024
    return memory_mb


def performance_test_series():
    """Run performance tests with increasing dataset sizes"""
    
    # Create timestamped test directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    test_dir = Path("performance_testing") / f"mplfinance_7m_test_{timestamp}"
    test_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"=== MPLFINANCE 7M+ PERFORMANCE TEST ===")
    print(f"Test directory: {test_dir}")
    
    # Test datasets - progressive sizing up to 7M+
    test_sizes = [
        10000,      # 10k - baseline
        100000,     # 100k - medium
        500000,     # 500k - large
        1000000,    # 1M - very large
        3000000,    # 3M - huge
        7000000,    # 7M - target size
    ]
    
    results = []
    
    try:
        import mplfinance as mpf
        
        for size in test_sizes:
            print(f"\n--- TESTING {size:,} DATAPOINTS ---")
            
            # Memory before
            memory_before = test_memory_usage()
            print(f"Memory before: {memory_before:.1f} MB")
            
            # Create dataset
            start_create = time.time()
            df = create_large_dataset(size)
            create_time = time.time() - start_create
            
            # Memory after data creation
            memory_after_data = test_memory_usage()
            print(f"Memory after data creation: {memory_after_data:.1f} MB")
            
            # Test chart rendering
            print(f"Rendering chart for {size:,} datapoints...")
            start_render = time.time()
            
            try:
                # Configure for performance
                style = mpf.make_mpf_style(
                    base_mpf_style='charles',
                    gridstyle='-',
                    gridcolor='lightgray'
                )
                
                # Render chart - save to file for performance
                chart_file = test_dir / f"performance_test_{size}.png"
                
                mpf.plot(
                    df,
                    type='candle',
                    style=style,
                    volume=False,  # Disable volume for performance
                    title=f'Performance Test - {size:,} bars',
                    savefig=chart_file,
                    figsize=(12, 8),
                    warn_too_much_data=size + 1000,  # Suppress warnings
                    show_nontrading=False
                )
                
                render_time = time.time() - start_render
                render_success = True
                
                # Memory after rendering
                memory_after_render = test_memory_usage()
                
                print(f"SUCCESS: {size:,} bars rendered in {render_time:.2f}s")
                print(f"Memory after render: {memory_after_render:.1f} MB")
                
                # Calculate performance metrics
                bars_per_second = size / render_time
                memory_per_bar = (memory_after_render - memory_before) / size * 1024  # KB per bar
                
                print(f"Performance: {bars_per_second:,.0f} bars/second")
                print(f"Memory efficiency: {memory_per_bar:.3f} KB/bar")
                
            except Exception as e:
                render_time = time.time() - start_render
                render_success = False
                memory_after_render = test_memory_usage()
                bars_per_second = 0
                memory_per_bar = 0
                
                print(f"FAILED: {size:,} bars failed after {render_time:.2f}s")
                print(f"Error: {e}")
            
            # Record results
            result = {
                'size': size,
                'create_time': create_time,
                'render_time': render_time,
                'render_success': render_success,
                'memory_before': memory_before,
                'memory_after_data': memory_after_data,
                'memory_after_render': memory_after_render,
                'bars_per_second': bars_per_second,
                'memory_per_bar': memory_per_bar
            }
            results.append(result)
            
            # Clean up memory
            del df
            gc.collect()
            
            # If rendering failed, stop testing larger sizes
            if not render_success:
                print(f"STOPPING: Failed at {size:,} datapoints")
                break
                
            # Check if we're approaching memory limits
            if memory_after_render > 8000:  # 8GB limit
                print(f"WARNING: High memory usage ({memory_after_render:.1f} MB)")
        
        # Generate performance report
        generate_performance_report(test_dir, results, timestamp)
        
        return results, test_dir
        
    except ImportError:
        print("ERROR: mplfinance not available")
        return [], test_dir
    except Exception as e:
        print(f"ERROR: Performance test failed: {e}")
        import traceback
        traceback.print_exc()
        return [], test_dir


def generate_performance_report(test_dir, results, timestamp):
    """Generate comprehensive performance report"""
    
    report_file = test_dir / "performance_report.txt"
    
    with open(report_file, 'w') as f:
        f.write("MPLFINANCE 7M+ PERFORMANCE TEST REPORT\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Test timestamp: {timestamp}\n")
        f.write("Purpose: Determine if mplfinance can handle 7M+ datapoints\n\n")
        
        f.write("PERFORMANCE RESULTS:\n")
        f.write("-" * 30 + "\n")
        
        max_successful_size = 0
        
        for result in results:
            size = result['size']
            success = "SUCCESS" if result['render_success'] else "FAILED"
            
            f.write(f"\n{size:,} datapoints: {success}\n")
            f.write(f"  Data creation: {result['create_time']:.2f}s\n")
            f.write(f"  Chart rendering: {result['render_time']:.2f}s\n")
            f.write(f"  Memory usage: {result['memory_after_render']:.1f} MB\n")
            
            if result['render_success']:
                f.write(f"  Performance: {result['bars_per_second']:,.0f} bars/second\n")
                f.write(f"  Memory efficiency: {result['memory_per_bar']:.3f} KB/bar\n")
                max_successful_size = max(max_successful_size, size)
        
        f.write(f"\nMAXIMUM SUCCESSFUL SIZE: {max_successful_size:,} datapoints\n")
        
        # Verdict
        f.write(f"\nVERDICT:\n")
        if max_successful_size >= 7000000:
            f.write("[OK] PASS: mplfinance can handle 7M+ datapoints\n")
            f.write("  Recommendation: Proceed with mplfinance implementation\n")
        elif max_successful_size >= 1000000:
            f.write("[WARNING] PARTIAL: mplfinance can handle large datasets but may need optimization for 7M+\n")
            f.write(f"  Maximum tested: {max_successful_size:,} datapoints\n")
            f.write("  Recommendation: Consider data decimation or chunking for 7M+ datasets\n")
        else:
            f.write("[X] FAIL: mplfinance cannot handle large datasets efficiently\n")
            f.write("  Recommendation: Look for alternative charting solutions\n")
        
        f.write(f"\nRECOMMENDATIONS:\n")
        if max_successful_size >= 3000000:
            f.write("- mplfinance is viable for large trading datasets\n")
            f.write("- Consider LOD (Level of Detail) for 7M+ datasets\n")
            f.write("- Use data decimation for initial chart load\n")
            f.write("- Implement progressive loading for user interaction\n")
        else:
            f.write("- mplfinance may not be suitable for very large datasets\n")
            f.write("- Consider alternative solutions (Plotly, Bokeh, custom WebGL)\n")
    
    print(f"\nPerformance report written: {report_file}")


if __name__ == "__main__":
    print("MPLFINANCE 7M+ PERFORMANCE TEST")
    print("This is the make-or-break test for the mplfinance solution")
    print("Testing progressive dataset sizes up to 7M+ datapoints")
    print()
    
    # Run performance test series
    results, test_dir = performance_test_series()
    
    if results:
        max_size = max(r['size'] for r in results if r['render_success'])
        
        print(f"\n=== PERFORMANCE TEST COMPLETE ===")
        print(f"Maximum successful size: {max_size:,} datapoints")
        
        if max_size >= 7000000:
            print("[SUCCESS] SUCCESS: mplfinance can handle 7M+ datapoints!")
            print("[OK] Proceed with mplfinance implementation")
        elif max_size >= 1000000:
            print("[WARNING] PARTIAL SUCCESS: Large datasets supported, 7M+ may need optimization")
            print("Consider LOD/decimation strategies")
        else:
            print("[X] FAILED: mplfinance not suitable for large datasets")
            print("Need alternative charting solution")
        
        print(f"\nDetailed results in: {test_dir}")
    else:
        print("[X] PERFORMANCE TEST FAILED")
        print("Check error messages above")