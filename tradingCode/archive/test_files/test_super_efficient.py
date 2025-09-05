"""
Test the super-efficient chart implementation
"""
import sys
import numpy as np
import pandas as pd
from PyQt5.QtWidgets import QApplication
from src.dashboard.super_efficient_chart import create_super_efficient_chart, SuperEfficientChartWidget
from src.dashboard.data_structures import TradeData
import time


def generate_test_data(n_bars=100000):
    """Generate test OHLC data"""
    print(f"Generating {n_bars:,} test bars...")
    
    start_time = time.time()
    
    # Generate timestamps (nanoseconds)
    start_timestamp = pd.Timestamp("2020-01-01").value
    timestamps = np.arange(start_timestamp, start_timestamp + n_bars * 60 * 1e9, 60 * 1e9, dtype=np.int64)
    
    # Generate realistic price data
    np.random.seed(42)
    base_price = 3750.0
    returns = np.random.normal(0, 0.001, n_bars)
    close_prices = base_price * np.exp(np.cumsum(returns))
    
    # Generate OHLC
    open_prices = np.roll(close_prices, 1)
    open_prices[0] = base_price
    
    volatility = np.abs(np.random.normal(0, 0.002, n_bars))
    high_prices = np.maximum(open_prices, close_prices) + volatility * close_prices
    low_prices = np.minimum(open_prices, close_prices) - volatility * close_prices
    
    volumes = np.random.randint(100, 10000, n_bars)
    
    generation_time = time.time() - start_time
    print(f"Data generation: {generation_time:.3f}s ({n_bars/generation_time:,.0f} bars/sec)")
    
    return {
        'datetime': timestamps,
        'open': open_prices.astype(np.float64),
        'high': high_prices.astype(np.float64),
        'low': low_prices.astype(np.float64),
        'close': close_prices.astype(np.float64),
        'volume': volumes.astype(np.float64)
    }


def generate_test_trades(n_trades=100):
    """Generate test trades"""
    trades = []
    base_time = pd.Timestamp("2020-01-01").value
    
    for i in range(n_trades):
        entry_time = base_time + i * 3600 * 1e9  # 1 hour apart
        side = 'buy' if i % 2 == 0 else 'sell'
        price = 3750 + np.random.normal(0, 50)
        
        trade = TradeData(
            trade_id=f"T{i+1}",
            timestamp=int(entry_time),
            side=side,
            price=price,
            quantity=1.0,
            pnl=None
        )
        trades.append(trade)
    
    return trades


def test_super_efficient_chart():
    """Test the super-efficient chart with different dataset sizes"""
    
    print("=== SUPER-EFFICIENT CHART TEST ===")
    
    test_sizes = [
        1000,     # Small
        10000,    # Medium  
        100000,   # Large
        500000,   # Very large
        1000000   # Massive
    ]
    
    # Create Qt application
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    
    for size in test_sizes:
        print(f"\n--- Testing {size:,} bars ---")
        
        try:
            # Generate data
            data = generate_test_data(size)
            trades = generate_test_trades(size // 1000)  # 1 trade per 1000 bars
            
            print(f"Memory usage: {sum(arr.nbytes for arr in data.values()) / 1024 / 1024:.1f} MB")
            
            # Create chart
            start_time = time.time()
            chart = create_super_efficient_chart(data, trades) 
            
            # Add trades
            if trades:
                chart.add_trades(trades)
            
            creation_time = time.time() - start_time
            print(f"Chart creation: {creation_time:.3f}s")
            print(f"Performance: {size/creation_time:,.0f} bars/second")
            
            # Get performance stats
            stats = chart.get_performance_stats()
            if stats:
                print(f"Render stats: {stats}")
            
            print(f"SUCCESS: {size:,} bars rendered")
            
            # Clean up
            chart.deleteLater()
            
        except Exception as e:
            print(f"FAILED: {size:,} bars - {e}")
            import traceback
            traceback.print_exc()
            break
    
    print("\n=== TEST COMPLETE ===")


if __name__ == "__main__":
    test_super_efficient_chart()