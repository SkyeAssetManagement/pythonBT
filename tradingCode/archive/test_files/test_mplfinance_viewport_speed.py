"""
Quick viewport performance test for mplfinance integration
Tests progressive dataset sizes with 120-second timeout per test
"""
import sys
import time
import numpy as np
import pandas as pd
import datetime
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import mplfinance as mpf

def create_test_data(n_bars):
    """Create test OHLC data"""
    timestamps = pd.date_range('2020-01-01', periods=n_bars, freq='1min')
    base_price = 4000.0
    
    # Realistic price movement
    changes = np.random.normal(0, 0.5, n_bars)
    prices = base_price + np.cumsum(changes)
    
    # OHLC with realistic spreads
    opens = prices.copy()
    closes = prices + np.random.normal(0, 0.2, n_bars)
    highs = np.maximum(opens, closes) + np.abs(np.random.normal(0, 1, n_bars))
    lows = np.minimum(opens, closes) - np.abs(np.random.normal(0, 1, n_bars))
    volumes = np.random.randint(100, 1000, n_bars)
    
    return pd.DataFrame({
        'Open': opens,
        'High': highs,
        'Low': lows, 
        'Close': closes,
        'Volume': volumes
    }, index=timestamps)

def ohlc_decimate(data, factor):
    """OHLC-aware decimation"""
    if factor <= 1:
        return data
    
    # Group into chunks and create OHLC
    n_groups = len(data) // factor
    decimated_data = []
    
    for i in range(n_groups):
        start_idx = i * factor
        end_idx = min((i + 1) * factor, len(data))
        chunk = data.iloc[start_idx:end_idx]
        
        if len(chunk) == 0:
            continue
            
        ohlc_row = {
            'Open': chunk['Open'].iloc[0],
            'High': chunk['High'].max(),
            'Low': chunk['Low'].min(),
            'Close': chunk['Close'].iloc[-1],
            'Volume': chunk['Volume'].sum()
        }
        decimated_data.append(ohlc_row)
    
    # Create new dataframe with decimated data
    decimated_indices = data.index[::factor][:len(decimated_data)]
    return pd.DataFrame(decimated_data, index=decimated_indices)

def test_viewport_performance():
    """Test viewport rendering performance"""
    
    print("=== MPLFINANCE VIEWPORT PERFORMANCE TEST ===")
    
    # Create output directory
    output_dir = Path("performance_testing") / f"viewport_test_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Chart style
    style = mpf.make_mpf_style(
        base_mpf_style='charles',
        gridstyle='-',
        gridcolor='lightgray'
    )
    
    # Test progressive dataset sizes with viewport management
    test_scenarios = [
        # (total_bars, viewport_size, decimation_factor, description)
        (10000, 2000, 1, "10K dataset, 2K viewport, no decimation"),
        (100000, 2000, 1, "100K dataset, 2K viewport, no decimation"),
        (500000, 2000, 1, "500K dataset, 2K viewport, no decimation"),
        (1000000, 2000, 1, "1M dataset, 2K viewport, no decimation"),
        (3000000, 2000, 1, "3M dataset, 2K viewport, no decimation"),
        (7000000, 2000, 1, "7M dataset, 2K viewport, no decimation"),
        (7000000, 5000, 1, "7M dataset, 5K viewport, no decimation"),
        (7000000, 2000, 2, "7M dataset, 2K viewport, 2x decimation"),
        (7000000, 2000, 5, "7M dataset, 2K viewport, 5x decimation"),
    ]
    
    results = []
    
    for total_bars, viewport_size, decimation_factor, description in test_scenarios:
        print(f"\n--- {description} ---")
        
        try:
            # Create full dataset
            print(f"Creating {total_bars:,} bars...")
            start_create = time.time()
            full_data = create_test_data(total_bars)
            create_time = time.time() - start_create
            print(f"Data created in {create_time:.2f}s")
            
            # Extract viewport (middle section)
            start_idx = max(0, (total_bars - viewport_size) // 2)
            end_idx = start_idx + viewport_size
            viewport_data = full_data.iloc[start_idx:end_idx].copy()
            
            # Apply decimation if needed
            if decimation_factor > 1:
                print(f"Applying {decimation_factor}x decimation...")
                start_decimate = time.time()
                viewport_data = ohlc_decimate(viewport_data, decimation_factor)
                decimate_time = time.time() - start_decimate
                print(f"Decimation completed in {decimate_time:.2f}s")
                print(f"Decimated size: {len(viewport_data):,} bars")
            else:
                decimate_time = 0
            
            # Memory check
            memory_mb = full_data.memory_usage(deep=True).sum() / 1024 / 1024
            viewport_memory_mb = viewport_data.memory_usage(deep=True).sum() / 1024 / 1024
            print(f"Memory: Full={memory_mb:.1f}MB, Viewport={viewport_memory_mb:.1f}MB")
            
            # Render viewport with timeout
            print(f"Rendering {len(viewport_data):,} bars...")
            chart_file = output_dir / f"test_{total_bars}_{viewport_size}_{decimation_factor}.png"
            
            start_render = time.time()
            timeout_start = time.time()
            
            # Render with timeout protection
            try:
                mpf.plot(
                    viewport_data,
                    type='candle',
                    style=style,
                    volume=True,
                    title=description,
                    savefig=str(chart_file),
                    figsize=(16, 10),
                    warn_too_much_data=len(viewport_data) + 1000
                )
                
                render_time = time.time() - start_render
                success = True
                
            except Exception as e:
                render_time = time.time() - start_render
                success = False
                print(f"Render failed: {e}")
            
            # Calculate performance metrics
            if success:
                bars_per_second = len(viewport_data) / render_time if render_time > 0 else 0
                total_time = create_time + decimate_time + render_time
                
                print(f"SUCCESS: Rendered in {render_time:.2f}s ({bars_per_second:,.0f} bars/s)")
                print(f"Total time: {total_time:.2f}s")
                
                # Performance verdict
                if render_time <= 5.0:
                    verdict = "EXCELLENT"
                elif render_time <= 15.0:
                    verdict = "GOOD"
                elif render_time <= 60.0:
                    verdict = "ACCEPTABLE"
                else:
                    verdict = "TOO_SLOW"
                    
                print(f"Performance: {verdict}")
            else:
                total_time = create_time + decimate_time + render_time
                bars_per_second = 0
                verdict = "FAILED"
            
            # Record results
            result = {
                'total_bars': total_bars,
                'viewport_size': viewport_size,
                'decimation_factor': decimation_factor,
                'description': description,
                'create_time': create_time,
                'decimate_time': decimate_time,
                'render_time': render_time,
                'total_time': total_time,
                'success': success,
                'bars_per_second': bars_per_second,
                'memory_mb': memory_mb,
                'viewport_memory_mb': viewport_memory_mb,
                'verdict': verdict
            }
            results.append(result)
            
            # Stop if we're getting too slow
            if render_time > 120:  # 2 minute timeout
                print(f"STOPPING: Render time ({render_time:.1f}s) exceeded 120s limit")
                break
                
        except Exception as e:
            print(f"Test failed: {e}")
            break
    
    # Generate summary report
    print(f"\n=== PERFORMANCE SUMMARY ===")
    
    report_file = output_dir / "viewport_performance_report.txt"
    with open(report_file, 'w') as f:
        f.write("MPLFINANCE VIEWPORT PERFORMANCE REPORT\n")
        f.write("=" * 50 + "\n\n")
        
        successful_tests = [r for r in results if r['success']]
        
        if successful_tests:
            max_bars = max(r['total_bars'] for r in successful_tests)
            best_performance = min(r['render_time'] for r in successful_tests)
            
            f.write(f"Maximum dataset size: {max_bars:,} bars\n") 
            f.write(f"Best render time: {best_performance:.2f}s\n\n")
            
            f.write("DETAILED RESULTS:\n")
            f.write("-" * 30 + "\n")
            
            for result in results:
                f.write(f"\n{result['description']}\n")
                f.write(f"  Total bars: {result['total_bars']:,}\n")
                f.write(f"  Viewport size: {result['viewport_size']:,}\n")
                f.write(f"  Create time: {result['create_time']:.2f}s\n")
                f.write(f"  Render time: {result['render_time']:.2f}s\n")
                f.write(f"  Success: {result['success']}\n")
                f.write(f"  Verdict: {result['verdict']}\n")
                if result['success']:
                    f.write(f"  Performance: {result['bars_per_second']:,.0f} bars/s\n")
            
            # Final verdict
            f.write(f"\nFINAL VERDICT:\n")
            if max_bars >= 7000000:
                f.write("PASS: Can handle 7M+ bars with viewport approach\n")
                f.write("Recommendation: Implement viewport-based mplfinance dashboard\n")
            elif max_bars >= 1000000:
                f.write("PARTIAL: Can handle large datasets, may need optimization for 7M+\n")
                f.write("Recommendation: Use viewport + decimation for very large datasets\n")
            else:
                f.write("FAIL: Cannot handle large datasets efficiently\n")
                f.write("Recommendation: Consider alternative charting solutions\n")
        else:
            f.write("NO SUCCESSFUL TESTS\n")
    
    print(f"Report saved: {report_file}")
    
    # Print summary to console
    successful_tests = [r for r in results if r['success']]
    if successful_tests:
        max_bars = max(r['total_bars'] for r in successful_tests)
        print(f"Maximum successful size: {max_bars:,} bars")
        
        if max_bars >= 7000000:
            print("SUCCESS: 7M+ bars supported with viewport approach!")
        elif max_bars >= 1000000:
            print("PARTIAL SUCCESS: Large datasets supported, optimization needed for 7M+")
        else:
            print("INSUFFICIENT: Need alternative approach for very large datasets")
    else:
        print("FAILED: No successful renders")
    
    return results, output_dir

if __name__ == "__main__":
    results, output_dir = test_viewport_performance()