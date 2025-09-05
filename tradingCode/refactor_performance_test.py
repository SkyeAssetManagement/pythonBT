"""
Refactoring Performance Baseline Test
Tests array processing efficiency at 1yr, 4yr, 20yr scales
"""

import time
import numpy as np
import pandas as pd
import json
from datetime import datetime

def test_array_scaling():
    """Test that array operations scale efficiently."""
    
    print("="*60)
    print("ARRAY PROCESSING EFFICIENCY TEST")
    print("="*60)
    
    results = {}
    
    # Test different data sizes (minutes in a year)
    sizes = {
        "1yr": 365 * 24 * 60,        # ~525k points
        "4yr": 4 * 365 * 24 * 60,    # ~2.1M points
        "20yr": 20 * 365 * 24 * 60   # ~10.5M points
    }
    
    for name, n_points in sizes.items():
        print(f"\nTesting {name} ({n_points:,} points)...")
        
        # Generate test data
        start = time.time()
        price = 100.0 + np.cumsum(np.random.randn(n_points) * 0.01).astype(np.float32)
        gen_time = time.time() - start
        
        # Test typical backtest operations
        start = time.time()
        
        # 1. Calculate SMAs (vectorized)
        close_series = pd.Series(price)
        sma_20 = close_series.rolling(window=20, min_periods=1).mean().values
        sma_100 = close_series.rolling(window=100, min_periods=1).mean().values
        
        # 2. Generate signals (vectorized)
        fast_above = sma_20 > sma_100
        fast_above_prev = np.roll(fast_above, 1)
        fast_above_prev[0] = False
        
        entries = fast_above & ~fast_above_prev
        exits = ~fast_above & fast_above_prev
        
        # 3. Calculate returns (vectorized)
        returns = np.diff(price) / price[:-1]
        cumulative = np.cumprod(1 + returns)
        
        calc_time = time.time() - start
        
        # Store results
        results[name] = {
            'points': n_points,
            'gen_time': gen_time,
            'calc_time': calc_time,
            'time_per_1k': (calc_time / n_points) * 1000,  # Time per 1000 points
            'total_time': gen_time + calc_time
        }
        
        print(f"  Generation: {gen_time:.3f}s")
        print(f"  Calculations: {calc_time:.3f}s")  
        print(f"  Time per 1K points: {results[name]['time_per_1k']*1000:.3f}ms")
    
    # Analyze scaling
    print("\n" + "="*60)
    print("SCALING ANALYSIS")
    print("="*60)
    
    t1 = results["1yr"]["time_per_1k"]
    t4 = results["4yr"]["time_per_1k"]
    t20 = results["20yr"]["time_per_1k"]
    
    # Calculate ratios
    ratio_4yr = t4 / t1 if t1 > 0 else float('inf')
    ratio_20yr = t20 / t1 if t1 > 0 else float('inf')
    
    print(f"\nTime per 1K points:")
    print(f"  1yr:  {t1*1000:.3f}ms (baseline)")
    print(f"  4yr:  {t4*1000:.3f}ms ({ratio_4yr:.2f}x)")
    print(f"  20yr: {t20*1000:.3f}ms ({ratio_20yr:.2f}x)")
    
    # Determine performance status
    if ratio_20yr < 1.2:
        status = "✅ EXCELLENT: Near-perfect array processing!"
        good = True
    elif ratio_20yr < 1.5:
        status = "✅ GOOD: Efficient array processing"
        good = True
    elif ratio_20yr < 2.0:
        status = "⚠️ OK: Some linear scaling detected"
        good = False
    else:
        status = f"❌ POOR: Linear scaling detected ({ratio_20yr:.1f}x slowdown)"
        good = False
    
    print(f"\n{status}")
    
    # Save baseline
    baseline = {
        "timestamp": datetime.now().isoformat(),
        "results": results,
        "scaling_ratios": {
            "4yr_vs_1yr": ratio_4yr,
            "20yr_vs_1yr": ratio_20yr
        },
        "status": "good" if good else "needs_attention"
    }
    
    with open("refactor_baseline.json", "w") as f:
        json.dump(baseline, f, indent=2)
    
    print(f"\nBaseline saved to refactor_baseline.json")
    
    return good

if __name__ == "__main__":
    print("REFACTORING PERFORMANCE BASELINE")
    print("Testing array processing efficiency before refactoring...")
    print()
    
    is_good = test_array_scaling()
    
    if is_good:
        print("\n" + "="*60)
        print("✅ SAFE TO PROCEED WITH REFACTORING")
        print("Array processing is efficient - refactoring won't break performance")
        print("="*60)
    else:
        print("\n" + "="*60)
        print("⚠️ CAUTION: Performance issues detected")
        print("Review array operations before major refactoring")
        print("="*60)