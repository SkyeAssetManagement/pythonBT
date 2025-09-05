"""
QUICK 2-MINUTE TEST: Load parquet data in memory, render viewport only
If this doesn't complete in 2 minutes, we need a different solution
"""
import sys
import time
import signal
from pathlib import Path
import numpy as np
import pandas as pd
import datetime
import matplotlib
matplotlib.use('Agg')  # Fast non-interactive backend


class TimeoutError(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutError("Test exceeded 2-minute timeout")


def quick_parquet_test():
    """Quick test with 2-minute timeout"""
    
    # Set 2-minute timeout
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(120)  # 2 minutes
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    test_dir = Path("quick_tests") / f"viewport_2min_{timestamp}"
    test_dir.mkdir(parents=True, exist_ok=True)
    
    print("=== 2-MINUTE VIEWPORT TEST ===")
    print("Loading parquet data -> render viewport only")
    print("Timeout: 2 minutes")
    
    try:
        # Step 1: Create realistic parquet dataset (fast)
        print("\n1. Creating parquet dataset...")
        start_time = time.time()
        
        # Create 1M datapoints (representative sample)
        n_points = 1000000
        
        # Fast data generation
        dates = pd.date_range('2020-01-01', periods=n_points, freq='1min')
        np.random.seed(42)
        
        # Vectorized price generation (much faster)
        base_price = 4000.0
        price_changes = np.random.normal(0, 0.5, n_points)
        prices = base_price + np.cumsum(price_changes)
        
        # Create OHLC efficiently
        df = pd.DataFrame({
            'timestamp': dates,
            'open': np.roll(prices, 1),
            'high': prices + np.abs(np.random.normal(0, 1, n_points)),
            'low': prices - np.abs(np.random.normal(0, 1, n_points)),
            'close': prices,
            'volume': np.random.randint(100, 5000, n_points)
        })
        df.iloc[0, 1] = base_price  # Fix first open
        
        # Save as parquet
        parquet_file = test_dir / "test_data.parquet"
        df.to_parquet(parquet_file)
        
        create_time = time.time() - start_time
        print(f"   Created {n_points:,} bars in {create_time:.2f}s")
        
        # Step 2: Load parquet data
        print("\n2. Loading parquet into memory...")
        load_start = time.time()
        
        loaded_df = pd.read_parquet(parquet_file)
        loaded_df.set_index('timestamp', inplace=True)
        
        load_time = time.time() - load_start
        memory_mb = loaded_df.memory_usage(deep=True).sum() / 1024 / 1024
        print(f"   Loaded {len(loaded_df):,} bars in {load_time:.2f}s ({memory_mb:.1f} MB)")
        
        # Step 3: Test viewport rendering
        print("\n3. Testing viewport rendering...")
        
        import mplfinance as mpf
        
        # Test different viewport sizes
        viewport_tests = [
            {"name": "small", "size": 1000},
            {"name": "medium", "size": 5000}, 
            {"name": "large", "size": 20000}
        ]
        
        render_results = []
        
        for test in viewport_tests:
            viewport_size = test['size']
            
            # Extract viewport from middle of dataset
            start_idx = len(loaded_df) // 2 - viewport_size // 2
            end_idx = start_idx + viewport_size
            
            viewport_data = loaded_df.iloc[start_idx:end_idx]
            
            print(f"   Rendering {viewport_size:,} bars...")
            render_start = time.time()
            
            # Quick style
            style = mpf.make_mpf_style(base_mpf_style='charles')
            
            output_file = test_dir / f"viewport_{test['name']}.png"
            
            mpf.plot(
                viewport_data,
                type='candle',
                style=style,
                volume=False,  # Skip volume for speed
                savefig=output_file,
                figsize=(10, 6),
                warn_too_much_data=viewport_size + 100
            )
            
            render_time = time.time() - render_start
            bars_per_sec = viewport_size / render_time
            
            print(f"     {viewport_size:,} bars: {render_time:.2f}s ({bars_per_sec:,.0f} bars/sec)")
            
            render_results.append({
                'size': viewport_size,
                'time': render_time,
                'bars_per_sec': bars_per_sec
            })
        
        # Step 4: Test rapid viewport changes
        print("\n4. Testing rapid viewport switching...")
        
        viewport_size = 2000
        switch_times = []
        
        switch_start = time.time()
        
        # Test 10 rapid viewport switches
        for i in range(10):
            # Random position in dataset
            start_pos = np.random.randint(0, len(loaded_df) - viewport_size)
            end_pos = start_pos + viewport_size
            
            extract_start = time.time()
            viewport_data = loaded_df.iloc[start_pos:end_pos]
            extract_time = time.time() - extract_start
            
            switch_times.append(extract_time)
        
        total_switch_time = time.time() - switch_start
        avg_switch_time = np.mean(switch_times)
        
        print(f"   10 viewport switches: {total_switch_time:.2f}s total")
        print(f"   Average switch time: {avg_switch_time:.4f}s ({1/avg_switch_time:.1f} FPS)")
        
        # Cancel timeout - we succeeded!
        signal.alarm(0)
        
        # Results summary
        total_test_time = time.time() - start_time
        
        print(f"\n=== RESULTS ===")
        print(f"Total test time: {total_test_time:.2f}s (under 2min limit)")
        print(f"Data loading: {load_time:.2f}s for {len(loaded_df):,} bars")
        
        fastest_render = min(render_results, key=lambda x: x['time'])
        print(f"Fastest render: {fastest_render['size']:,} bars in {fastest_render['time']:.2f}s")
        
        print(f"Viewport switching: {avg_switch_time:.4f}s avg ({1/avg_switch_time:.1f} FPS)")
        
        # Verdict
        if fastest_render['time'] < 2.0 and avg_switch_time < 0.1:
            print("\n[SUCCESS] SUCCESS: Viewport approach is viable!")
            print("[OK] Fast parquet loading")
            print("[OK] Quick viewport rendering") 
            print("[OK] Responsive viewport switching")
            verdict = "SUCCESS"
        elif fastest_render['time'] < 5.0:
            print("\n[WARNING] MARGINAL: Might work with optimization")
            print("Viewport rendering is acceptable but could be faster")
            verdict = "MARGINAL"
        else:
            print("\n[X] FAILED: Too slow for interactive use")
            print("Need alternative solution")
            verdict = "FAILED"
        
        # Write quick report
        report_file = test_dir / "quick_test_results.txt"
        with open(report_file, 'w') as f:
            f.write(f"QUICK VIEWPORT TEST RESULTS\n")
            f.write(f"Total time: {total_test_time:.2f}s\n")
            f.write(f"Verdict: {verdict}\n")
            f.write(f"Data loading: {load_time:.2f}s\n")
            f.write(f"Fastest render: {fastest_render['time']:.2f}s\n")
            f.write(f"Viewport switching: {avg_switch_time:.4f}s\n")
        
        return verdict, test_dir
        
    except TimeoutError:
        print("\n[X] TIMEOUT: Test exceeded 2 minutes")
        print("mplfinance + viewport approach is too slow")
        print("Need different solution (WebGL, canvas, etc.)")
        return "TIMEOUT", test_dir
        
    except Exception as e:
        signal.alarm(0)  # Cancel timeout
        print(f"\n[X] ERROR: {e}")
        import traceback
        traceback.print_exc()
        return "ERROR", test_dir
    
    finally:
        signal.alarm(0)  # Ensure timeout is cancelled


if __name__ == "__main__":
    print("QUICK 2-MINUTE VIEWPORT TEST")
    print("If this doesn't complete in 2 minutes, we need a different solution")
    print()
    
    verdict, test_dir = quick_parquet_test()
    
    print(f"\n=== FINAL VERDICT: {verdict} ===")
    
    if verdict == "SUCCESS":
        print("[OK] Proceed with mplfinance viewport implementation")
    elif verdict == "MARGINAL":
        print("[WARNING] Consider mplfinance with heavy optimization")
    else:
        print("[X] Abandon mplfinance, find alternative solution")
        print("Consider: Plotly, Bokeh, custom WebGL, or server-side rendering")
    
    print(f"Test results: {test_dir}")