"""
Performance Baseline Testing Script
Tests backtesting and chart rendering performance for 1yr, 4yr, and 20yr periods
Ensures array processing efficiency (non-linear scaling check)
"""

import time
import sys
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import json

# Add path for imports
sys.path.insert(0, str(Path(__file__).parent))

def test_backtest_performance(symbol="ES", strategy="simpleSMA", periods=[1, 4, 20]):
    """Test backtesting performance for different time periods."""
    
    results = {}
    
    for years in periods:
        print(f"\n{'='*60}")
        print(f"Testing {years} year(s) of data...")
        print(f"{'='*60}")
        
        # Calculate date range
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=365*years)).strftime("%Y-%m-%d")
        
        # Import main function
        from main import main
        
        # Time the backtest
        start_time = time.time()
        
        try:
            # Run backtest without visualization
            results_dir = main(
                symbol=symbol,
                strategy_name=strategy,
                config_path="config.yaml",
                start_date=start_date,
                end_date=end_date,
                launch_viz=False,
                use_defaults=True,
                intraday_performance=False,
                use_plotly=False
            )
            
            elapsed_time = time.time() - start_time
            
            # Get number of data points from results
            equity_file = Path(results_dir) / "equity_curve.csv"
            if equity_file.exists():
                df = pd.read_csv(equity_file)
                n_points = len(df)
            else:
                n_points = 0
            
            results[f"{years}yr"] = {
                "elapsed_time": elapsed_time,
                "data_points": n_points,
                "time_per_point": elapsed_time / n_points if n_points > 0 else 0,
                "start_date": start_date,
                "end_date": end_date
            }
            
            print(f"SUCCESS: {years}yr backtest completed in {elapsed_time:.2f}s")
            print(f"Data points: {n_points:,}")
            print(f"Time per point: {results[f'{years}yr']['time_per_point']*1000:.4f}ms")
            
        except Exception as e:
            print(f"ERROR: {years}yr backtest failed: {e}")
            results[f"{years}yr"] = {"error": str(e)}
    
    return results

def test_chart_rendering_performance(periods=[1, 4, 20]):
    """Test chart rendering performance for different data sizes."""
    
    results = {}
    
    for years in periods:
        print(f"\n{'='*60}")
        print(f"Testing chart rendering for {years} year(s)...")
        print(f"{'='*60}")
        
        # Generate test data
        n_points = years * 365 * 24 * 60  # Minute bars
        
        print(f"Generating {n_points:,} data points...")
        
        start_time = time.time()
        
        # Create test OHLCV data
        dates = pd.date_range(start='2020-01-01', periods=n_points, freq='1min')
        price = 100 * np.ones(n_points)
        
        ohlcv_data = {
            'datetime': dates.values.astype(np.int64),
            'open': price * (1 + np.random.uniform(-0.002, 0.002, n_points)),
            'high': price * (1 + np.random.uniform(0, 0.005, n_points)),
            'low': price * (1 + np.random.uniform(-0.005, 0, n_points)),
            'close': price,
            'volume': np.random.uniform(1000, 10000, n_points)
        }
        
        data_gen_time = time.time() - start_time
        
        # Test chart creation (without actually running the server)
        try:
            from plotly_dashboard_enhanced import EnhancedPlotlyDashboard
            
            start_time = time.time()
            
            dashboard = EnhancedPlotlyDashboard(
                ohlcv_data=ohlcv_data,
                symbol="TEST",
                strategy_name="Performance Test"
            )
            
            # Create the figure (this is where rendering logic happens)
            fig = dashboard.create_main_figure()
            
            render_time = time.time() - start_time
            
            results[f"{years}yr"] = {
                "data_points": n_points,
                "data_generation_time": data_gen_time,
                "render_time": render_time,
                "time_per_point": render_time / n_points,
                "total_time": data_gen_time + render_time
            }
            
            print(f"SUCCESS: Chart rendering completed")
            print(f"Data generation: {data_gen_time:.2f}s")
            print(f"Render time: {render_time:.2f}s")
            print(f"Time per point: {results[f'{years}yr']['time_per_point']*1000:.6f}ms")
            
        except Exception as e:
            print(f"ERROR: Chart rendering failed: {e}")
            results[f"{years}yr"] = {"error": str(e)}
    
    return results

def analyze_scaling(results):
    """Analyze if performance scales linearly or not."""
    
    print(f"\n{'='*60}")
    print("SCALING ANALYSIS")
    print(f"{'='*60}")
    
    if "1yr" in results and "4yr" in results and "20yr" in results:
        if all("time_per_point" in results[k] for k in ["1yr", "4yr", "20yr"]):
            
            # Calculate scaling factors
            t1 = results["1yr"]["time_per_point"]
            t4 = results["4yr"]["time_per_point"]
            t20 = results["20yr"]["time_per_point"]
            
            # Check if time per point is relatively constant (good array processing)
            variation_4yr = abs(t4 - t1) / t1 * 100 if t1 > 0 else 0
            variation_20yr = abs(t20 - t1) / t1 * 100 if t1 > 0 else 0
            
            print(f"Time per data point:")
            print(f"  1yr:  {t1*1000:.6f}ms")
            print(f"  4yr:  {t4*1000:.6f}ms ({variation_4yr:+.1f}% vs 1yr)")
            print(f"  20yr: {t20*1000:.6f}ms ({variation_20yr:+.1f}% vs 1yr)")
            
            if variation_20yr < 50:
                print("\n✅ GOOD: Performance scales well with data size!")
                print("   Array processing is working efficiently.")
            else:
                print("\n⚠️ WARNING: Performance may be scaling linearly!")
                print("   Array processing might be compromised.")
                
            return variation_20yr < 50
    
    return None

def save_baseline(backtest_results, render_results):
    """Save baseline results to file."""
    
    baseline = {
        "timestamp": datetime.now().isoformat(),
        "backtest": backtest_results,
        "render": render_results
    }
    
    with open("performance_baseline.json", "w") as f:
        json.dump(baseline, f, indent=2)
    
    print(f"\nBaseline saved to performance_baseline.json")

def main():
    """Run all performance tests."""
    
    print("="*60)
    print("PERFORMANCE BASELINE TESTING")
    print("="*60)
    
    # Test backtesting performance
    print("\n1. BACKTESTING PERFORMANCE")
    backtest_results = test_backtest_performance(periods=[1, 4, 20])
    backtest_scaling_good = analyze_scaling(backtest_results)
    
    # Test chart rendering performance
    print("\n2. CHART RENDERING PERFORMANCE")
    render_results = test_chart_rendering_performance(periods=[1, 4, 20])
    render_scaling_good = analyze_scaling(render_results)
    
    # Save baseline
    save_baseline(backtest_results, render_results)
    
    # Summary
    print("\n" + "="*60)
    print("PERFORMANCE BASELINE SUMMARY")
    print("="*60)
    
    if backtest_scaling_good is not None:
        print(f"Backtesting scaling: {'✅ GOOD' if backtest_scaling_good else '⚠️ NEEDS ATTENTION'}")
    
    if render_scaling_good is not None:
        print(f"Rendering scaling: {'✅ GOOD' if render_scaling_good else '⚠️ NEEDS ATTENTION'}")
    
    if backtest_scaling_good and render_scaling_good:
        print("\n✅ All systems performing well! Safe to proceed with refactoring.")
    else:
        print("\n⚠️ Performance issues detected. Review before refactoring.")

if __name__ == "__main__":
    main()