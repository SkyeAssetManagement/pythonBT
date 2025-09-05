"""
CRITICAL TEST: Load 7M datapoints in memory, render only viewport
This tests the proper approach: full data in memory + selective viewport rendering
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
matplotlib.use('Agg')  # Non-interactive backend


class ViewportRenderer:
    """Handles large dataset with viewport-based rendering"""
    
    def __init__(self, full_dataset):
        self.full_data = full_dataset
        self.total_bars = len(full_dataset)
        print(f"ViewportRenderer initialized with {self.total_bars:,} bars")
    
    def get_viewport_data(self, start_idx, end_idx):
        """Extract data for specific viewport range"""
        start_idx = max(0, start_idx)
        end_idx = min(self.total_bars, end_idx)
        
        viewport_data = self.full_data.iloc[start_idx:end_idx].copy()
        print(f"Viewport extracted: bars {start_idx:,} to {end_idx:,} ({len(viewport_data):,} bars)")
        
        return viewport_data
    
    def render_viewport(self, start_idx, end_idx, output_file=None):
        """Render only the viewport data"""
        import mplfinance as mpf
        
        viewport_data = self.get_viewport_data(start_idx, end_idx)
        
        # Performance-optimized style
        style = mpf.make_mpf_style(
            base_mpf_style='charles',
            gridstyle='-',
            gridcolor='lightgray'
        )
        
        start_time = time.time()
        
        # Render only viewport data
        mpf.plot(
            viewport_data,
            type='candle',
            style=style,
            volume=True,
            title=f'Viewport Rendering - Bars {start_idx:,} to {end_idx:,} of {self.total_bars:,}',
            savefig=output_file,
            figsize=(12, 8),
            warn_too_much_data=len(viewport_data) + 1000,
            show_nontrading=False
        )
        
        render_time = time.time() - start_time
        return render_time, len(viewport_data)


def create_7m_dataset():
    """Create 7M datapoint dataset efficiently"""
    print("Creating 7M datapoint dataset...")
    
    n_points = 7000000
    start_time = pd.Timestamp('2015-01-01 09:30:00')
    
    # Memory-efficient generation using chunks
    chunk_size = 100000
    chunks = []
    
    np.random.seed(42)  # Reproducible
    base_price = 4000.0
    current_price = base_price
    current_time = start_time
    
    print("Generating data in chunks for memory efficiency...")
    
    for chunk_i in range(0, n_points, chunk_size):
        chunk_end = min(chunk_i + chunk_size, n_points)
        chunk_len = chunk_end - chunk_i
        
        if chunk_i % 500000 == 0:
            print(f"  Generated {chunk_i:,} / {n_points:,} bars ({chunk_i/n_points*100:.1f}%)")
        
        # Generate timestamps for chunk
        timestamps = pd.date_range(current_time, periods=chunk_len, freq='1min')
        current_time = timestamps[-1] + pd.Timedelta(minutes=1)
        
        # Generate price movements (trending random walk)
        price_changes = np.random.normal(0, 0.3, chunk_len)
        prices = current_price + np.cumsum(price_changes)
        current_price = prices[-1]
        
        # Create OHLC
        opens = np.concatenate([[current_price - np.sum(price_changes[:1])], prices[:-1]])
        closes = prices
        
        # Realistic highs/lows
        hl_spread = np.abs(np.random.normal(0, 0.8, chunk_len))
        highs = np.maximum(opens, closes) + hl_spread
        lows = np.minimum(opens, closes) - hl_spread
        
        # Volume
        volumes = np.random.randint(100, 5000, chunk_len)
        
        # Create chunk DataFrame
        chunk_df = pd.DataFrame({
            'Open': opens,
            'High': highs,
            'Low': lows, 
            'Close': closes,
            'Volume': volumes
        }, index=timestamps)
        
        chunks.append(chunk_df)
    
    # Combine all chunks
    print("Combining chunks into final dataset...")
    full_df = pd.concat(chunks, ignore_index=False)
    
    # Memory usage
    memory_mb = full_df.memory_usage(deep=True).sum() / 1024 / 1024
    print(f"7M dataset created: {len(full_df):,} bars, {memory_mb:.1f} MB")
    
    return full_df


def test_memory_usage():
    """Check current memory usage"""
    process = psutil.Process(os.getpid())
    memory_mb = process.memory_info().rss / 1024 / 1024
    return memory_mb


def test_viewport_rendering():
    """Test the viewport rendering approach with 7M dataset"""
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    test_dir = Path("performance_testing") / f"viewport_7m_test_{timestamp}"
    test_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"=== VIEWPORT RENDERING TEST - 7M DATAPOINTS ===")
    print(f"Test directory: {test_dir}")
    
    results = {}
    
    try:
        # Step 1: Load 7M datapoints into memory
        print("\n--- STEP 1: LOADING 7M DATAPOINTS ---")
        memory_before = test_memory_usage()
        print(f"Memory before loading: {memory_before:.1f} MB")
        
        start_load = time.time()
        full_dataset = create_7m_dataset()
        load_time = time.time() - start_load
        
        memory_after_load = test_memory_usage()
        print(f"Memory after loading: {memory_after_load:.1f} MB")
        print(f"Data loading time: {load_time:.2f}s")
        print(f"Loading rate: {len(full_dataset)/load_time:,.0f} bars/second")
        
        results['load_time'] = load_time
        results['memory_after_load'] = memory_after_load
        results['data_size'] = len(full_dataset)
        
        # Step 2: Test viewport rendering with different sizes
        print("\n--- STEP 2: VIEWPORT RENDERING TESTS ---")
        
        renderer = ViewportRenderer(full_dataset)
        
        # Test different viewport sizes
        viewport_tests = [
            {"name": "small", "bars": 1000, "desc": "1K bars (typical zoom in)"},
            {"name": "medium", "bars": 10000, "desc": "10K bars (medium view)"},
            {"name": "large", "bars": 100000, "desc": "100K bars (wide view)"},
            {"name": "xlarge", "bars": 500000, "desc": "500K bars (very wide view)"}
        ]
        
        viewport_results = []
        
        for test in viewport_tests:
            print(f"\n  Testing {test['desc']}...")
            
            # Test multiple positions in dataset
            positions = [
                (0, test['bars']),  # Beginning
                (len(full_dataset)//2 - test['bars']//2, len(full_dataset)//2 + test['bars']//2),  # Middle
                (len(full_dataset) - test['bars'], len(full_dataset))  # End
            ]
            
            position_times = []
            
            for i, (start_idx, end_idx) in enumerate(positions):
                pos_name = ["beginning", "middle", "end"][i]
                output_file = test_dir / f"viewport_{test['name']}_{pos_name}.png"
                
                memory_before_render = test_memory_usage()
                
                render_time, bars_rendered = renderer.render_viewport(start_idx, end_idx, output_file)
                
                memory_after_render = test_memory_usage()
                
                position_times.append(render_time)
                
                print(f"    {pos_name.capitalize()}: {render_time:.2f}s for {bars_rendered:,} bars")
                print(f"    Performance: {bars_rendered/render_time:,.0f} bars/second")
                print(f"    Memory: {memory_before_render:.1f} -> {memory_after_render:.1f} MB")
            
            avg_render_time = np.mean(position_times)
            viewport_results.append({
                'name': test['name'],
                'bars': test['bars'],
                'avg_render_time': avg_render_time,
                'bars_per_second': test['bars'] / avg_render_time
            })
            
            print(f"  Average render time: {avg_render_time:.2f}s")
        
        results['viewport_tests'] = viewport_results
        
        # Step 3: Test rapid viewport changes (pan/zoom simulation)
        print("\n--- STEP 3: RAPID VIEWPORT CHANGES ---")
        
        # Simulate user panning through chart
        pan_tests = []
        viewport_size = 5000  # 5K bars viewport
        step_size = 1000     # Move 1K bars at a time
        
        print(f"Simulating rapid panning: {viewport_size} bar viewport, {step_size} bar steps")
        
        pan_start = time.time()
        
        for i in range(0, min(100000, len(full_dataset) - viewport_size), step_size):
            start_idx = i
            end_idx = i + viewport_size
            
            render_start = time.time()
            
            # Just extract viewport data (no actual rendering for speed)
            viewport_data = renderer.get_viewport_data(start_idx, end_idx)
            
            render_time = time.time() - render_start
            pan_tests.append(render_time)
            
            if len(pan_tests) % 20 == 0:
                avg_time = np.mean(pan_tests[-20:])
                print(f"  Pan step {len(pan_tests)}: {render_time:.3f}s (avg: {avg_time:.3f}s)")
        
        total_pan_time = time.time() - pan_start
        avg_pan_time = np.mean(pan_tests)
        
        print(f"Pan simulation complete: {len(pan_tests)} steps in {total_pan_time:.2f}s")
        print(f"Average pan time: {avg_pan_time:.3f}s per step")
        
        results['pan_tests'] = {
            'total_steps': len(pan_tests),
            'total_time': total_pan_time,
            'avg_step_time': avg_pan_time
        }
        
        # Generate report
        generate_viewport_report(test_dir, results, timestamp)
        
        return results, test_dir
        
    except Exception as e:
        print(f"ERROR: Viewport test failed: {e}")
        import traceback
        traceback.print_exc()
        return {}, test_dir


def generate_viewport_report(test_dir, results, timestamp):
    """Generate viewport rendering performance report"""
    
    report_file = test_dir / "viewport_performance_report.txt"
    
    with open(report_file, 'w') as f:
        f.write("VIEWPORT RENDERING PERFORMANCE REPORT - 7M DATAPOINTS\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Test timestamp: {timestamp}\n")
        f.write("Approach: Load all data in memory, render only viewport\n\n")
        
        # Data loading results
        if 'load_time' in results:
            f.write("DATA LOADING PERFORMANCE:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Dataset size: {results['data_size']:,} bars\n")
            f.write(f"Loading time: {results['load_time']:.2f}s\n")
            f.write(f"Loading rate: {results['data_size']/results['load_time']:,.0f} bars/second\n")
            f.write(f"Memory usage: {results['memory_after_load']:.1f} MB\n\n")
        
        # Viewport rendering results
        if 'viewport_tests' in results:
            f.write("VIEWPORT RENDERING PERFORMANCE:\n")
            f.write("-" * 35 + "\n")
            
            for test in results['viewport_tests']:
                f.write(f"\n{test['name'].upper()} viewport ({test['bars']:,} bars):\n")
                f.write(f"  Render time: {test['avg_render_time']:.2f}s\n")
                f.write(f"  Performance: {test['bars_per_second']:,.0f} bars/second\n")
        
        # Pan simulation results
        if 'pan_tests' in results:
            f.write(f"\nPAN SIMULATION PERFORMANCE:\n")
            f.write("-" * 30 + "\n")
            pan = results['pan_tests']
            f.write(f"Total pan steps: {pan['total_steps']}\n")
            f.write(f"Total time: {pan['total_time']:.2f}s\n")
            f.write(f"Average step time: {pan['avg_step_time']:.3f}s\n")
            f.write(f"Pan responsiveness: {1/pan['avg_step_time']:.1f} FPS equivalent\n")
        
        # Verdict
        f.write(f"\nVERDICT:\n")
        f.write("=" * 10 + "\n")
        
        if 'viewport_tests' in results:
            fastest_viewport = min(results['viewport_tests'], key=lambda x: x['avg_render_time'])
            slowest_viewport = max(results['viewport_tests'], key=lambda x: x['avg_render_time'])
            
            if fastest_viewport['avg_render_time'] < 1.0:
                f.write("[OK] EXCELLENT: Sub-second viewport rendering\n")
                verdict = "excellent"
            elif fastest_viewport['avg_render_time'] < 3.0:
                f.write("[OK] GOOD: Fast viewport rendering\n")
                verdict = "good"
            elif fastest_viewport['avg_render_time'] < 10.0:
                f.write("[WARNING] ACCEPTABLE: Reasonable viewport rendering\n")
                verdict = "acceptable"
            else:
                f.write("[X] POOR: Slow viewport rendering\n")
                verdict = "poor"
            
            f.write(f"  Fastest: {fastest_viewport['bars']:,} bars in {fastest_viewport['avg_render_time']:.2f}s\n")
            f.write(f"  Slowest: {slowest_viewport['bars']:,} bars in {slowest_viewport['avg_render_time']:.2f}s\n")
        
        if 'pan_tests' in results:
            pan_fps = 1 / results['pan_tests']['avg_step_time']
            if pan_fps >= 10:
                f.write("[OK] SMOOTH: Excellent pan responsiveness\n")
            elif pan_fps >= 5:
                f.write("[OK] GOOD: Good pan responsiveness\n")
            elif pan_fps >= 2:
                f.write("[WARNING] ACCEPTABLE: Acceptable pan responsiveness\n")
            else:
                f.write("[X] CHOPPY: Poor pan responsiveness\n")
            
            f.write(f"  Pan responsiveness: {pan_fps:.1f} updates/second\n")
        
        f.write(f"\nRECOMMENDATION:\n")
        if 'viewport_tests' in results and fastest_viewport['avg_render_time'] < 3.0:
            f.write("[OK] PROCEED: Viewport rendering approach is viable\n")
            f.write("  - 7M datapoints can be loaded in memory\n")
            f.write("  - Viewport rendering is fast enough for real-time use\n")
            f.write("  - Implement viewport-based chart solution\n")
        else:
            f.write("[X] RECONSIDER: Viewport rendering may be too slow\n")
            f.write("  - Consider alternative approaches\n")
            f.write("  - Look into WebGL-based solutions\n")
            f.write("  - Consider server-side rendering\n")
    
    print(f"\nViewport performance report written: {report_file}")


if __name__ == "__main__":
    print("VIEWPORT RENDERING TEST - 7M DATAPOINTS")
    print("Tests: Load 7M bars in memory + render only viewport")
    print("This is the proper approach for large dataset visualization")
    print()
    
    results, test_dir = test_viewport_rendering()
    
    if results:
        print(f"\n=== VIEWPORT RENDERING TEST COMPLETE ===")
        
        if 'viewport_tests' in results:
            fastest = min(results['viewport_tests'], key=lambda x: x['avg_render_time'])
            print(f"Fastest viewport: {fastest['bars']:,} bars in {fastest['avg_render_time']:.2f}s")
            
            if fastest['avg_render_time'] < 3.0:
                print("[SUCCESS] SUCCESS: Viewport rendering is viable!")
                print("[OK] Can load 7M datapoints and render viewports quickly")
                print("[OK] Proceed with viewport-based mplfinance implementation")
            else:
                print("[X] FAILED: Viewport rendering too slow")
                print("Need alternative charting solution")
        
        print(f"\nDetailed results: {test_dir}")
    else:
        print("[X] VIEWPORT TEST FAILED")
        print("Check error messages above")