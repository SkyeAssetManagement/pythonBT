"""
Test lightweight-charts-python with large datasets
"""
import pandas as pd
import numpy as np
import datetime
import time
from lightweight_charts import Chart


def generate_large_ohlc_data(n_bars=100000):
    """Generate large OHLC dataset efficiently"""
    print(f"Generating {n_bars:,} OHLC bars...")
    
    start_time = time.time()
    
    # Create date range
    start_date = datetime.datetime(2020, 1, 1)
    dates = pd.date_range(start=start_date, periods=n_bars, freq='1min')
    
    # Efficient vectorized price generation
    np.random.seed(42)
    base_price = 3750.0
    
    # Generate random returns (vectorized)
    returns = np.random.normal(0, 0.001, n_bars)
    close_prices = base_price * np.exp(np.cumsum(returns))
    
    # Generate volatility array
    volatility = np.abs(np.random.normal(0, 0.002, n_bars))
    
    # Generate opens (with gaps)
    open_prices = np.zeros(n_bars)
    open_prices[0] = close_prices[0] + np.random.normal(0, 0.001) * close_prices[0]
    open_prices[1:] = close_prices[:-1] + np.random.normal(0, 0.0005, n_bars-1) * close_prices[1:]
    
    # Generate highs and lows (vectorized)
    highs = np.maximum(open_prices, close_prices) + volatility * close_prices
    lows = np.minimum(open_prices, close_prices) - volatility * close_prices
    
    # Generate volumes
    volumes = np.random.randint(100, 10000, n_bars)
    
    # Create DataFrame directly (much faster than append)
    df = pd.DataFrame({
        'time': dates,
        'open': np.round(open_prices, 2),
        'high': np.round(highs, 2),
        'low': np.round(lows, 2),
        'close': np.round(close_prices, 2),
        'volume': volumes
    })
    
    generation_time = time.time() - start_time
    print(f"Data generation completed in {generation_time:.3f} seconds")
    print(f"Generation rate: {n_bars/generation_time:,.0f} bars/second")
    
    return df


def test_with_real_data_size():
    """Test with realistic trading data sizes"""
    
    # Test sizes that match real trading scenarios
    test_sizes = [
        (100000, "100K bars (~70 days 1min data)"),
        (500000, "500K bars (~350 days 1min data)"),
        (1000000, "1M bars (~2 years 1min data)"),
        (2000000, "2M bars (~4 years 1min data)"),
    ]
    
    print("=== Large Dataset Performance Test ===")
    
    results = []
    
    for size, description in test_sizes:
        print(f"\n--- {description} ---")
        
        try:
            # Generate data
            df = generate_large_ohlc_data(size)
            
            print(f"Data size: {df.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB")
            print(f"Date range: {df['time'].min()} to {df['time'].max()}")
            print(f"Price range: ${df['low'].min():.2f} - ${df['high'].max():.2f}")
            
            # Test chart creation and data loading
            start_time = time.time()
            chart = Chart()
            chart_time = time.time() - start_time
            print(f"Chart creation: {chart_time:.3f} seconds")
            
            # Load data into chart
            start_time = time.time()
            chart.set(df)
            load_time = time.time() - start_time
            print(f"Data loading: {load_time:.3f} seconds")
            print(f"Loading rate: {size/load_time:,.0f} bars/second")
            
            # Show chart (non-blocking)
            start_time = time.time()
            chart.show(block=False)
            show_time = time.time() - start_time
            print(f"Chart display: {show_time:.3f} seconds")
            
            total_time = chart_time + load_time + show_time
            overall_rate = size / total_time
            
            print(f"TOTAL CHART TIME: {total_time:.3f} seconds")
            print(f"OVERALL RATE: {overall_rate:,.0f} bars/second")
            print(f"SUCCESS: Chart with {size:,} bars created")
            
            results.append({
                'size': size,
                'description': description,
                'total_time': total_time,
                'rate': overall_rate,
                'status': 'SUCCESS'
            })
            
            # Brief pause between tests
            time.sleep(3)
            
        except Exception as e:
            print(f"FAILED: {e}")
            results.append({
                'size': size,
                'description': description,
                'total_time': None,
                'rate': None,
                'status': f'FAILED: {e}'
            })
            # Continue with next size even if this one failed
            continue
    
    # Summary
    print("\n" + "="*60)
    print("PERFORMANCE SUMMARY")
    print("="*60)
    
    for result in results:
        print(f"{result['description']}")
        if result['status'] == 'SUCCESS':
            print(f"  Time: {result['total_time']:.3f}s, Rate: {result['rate']:,.0f} bars/sec")
        else:
            print(f"  {result['status']}")
        print()
    
    # Determine maximum successful size
    successful_results = [r for r in results if r['status'] == 'SUCCESS']
    if successful_results:
        max_successful = max(successful_results, key=lambda x: x['size'])
        print(f"MAXIMUM SUCCESSFUL SIZE: {max_successful['size']:,} bars")
        print(f"PERFORMANCE AT MAX: {max_successful['rate']:,.0f} bars/second")


if __name__ == "__main__":
    test_with_real_data_size()
    
    print("\nTest completed. Charts should be visible if successful.")
    print("This demonstrates lightweight-charts-python capability with large datasets.")
    
    # Keep alive briefly
    time.sleep(5)