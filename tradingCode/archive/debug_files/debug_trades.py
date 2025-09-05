#!/usr/bin/env python3
"""
Trade Arrow Positioning Diagnostic Tool
"""

import sys
import numpy as np
from pathlib import Path

# Add src to path  
src_path = Path(__file__).parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from PyQt5 import QtWidgets
from src.dashboard.chart_widget import TradingChart
from src.dashboard.data_structures import ChartDataBuffer, TradeData

def test_trade_positioning():
    """Test trade arrow positioning with mock data"""
    
    print("=== TRADE ARROW POSITIONING TEST ===")
    
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(sys.argv)
    
    chart = TradingChart()
    
    # Create mock OHLCV data
    n_bars = 100
    timestamps = np.arange(n_bars, dtype=np.int64) * 1000000000  # Nanoseconds
    open_prices = np.random.rand(n_bars) * 0.01 + 0.65  # 0.65-0.66 range
    high_prices = open_prices + np.random.rand(n_bars) * 0.005  # Add up to 0.005
    low_prices = open_prices - np.random.rand(n_bars) * 0.005   # Subtract up to 0.005
    close_prices = open_prices + (np.random.rand(n_bars) - 0.5) * 0.01
    volume = np.random.randint(1000, 10000, n_bars).astype(float)
    
    # Create data buffer
    data_buffer = ChartDataBuffer(
        timestamps=timestamps,
        open=open_prices, 
        high=high_prices,
        low=low_prices,
        close=close_prices,
        volume=volume
    )
    
    print(f"Created data buffer with {len(data_buffer)} bars")
    print(f"Price range: {np.min(low_prices):.5f} - {np.max(high_prices):.5f}")
    
    # Set data on chart
    chart.set_data(data_buffer)
    
    print(f"Chart data_buffer exists: {chart.data_buffer is not None}")
    if chart.data_buffer:
        print(f"Chart data_buffer length: {len(chart.data_buffer)}")
        print(f"Chart has low data: {hasattr(chart.data_buffer, 'low')}")
        print(f"Chart has high data: {hasattr(chart.data_buffer, 'high')}")
    
    # Create mock trades
    trades = []
    
    # Buy trade at bar 20
    buy_trade = TradeData(
        trade_id="BUY_001",
        timestamp=timestamps[20],
        side="buy", 
        price=close_prices[20],
        quantity=100,
        pnl=None
    )
    trades.append(buy_trade)
    
    # Sell trade at bar 50
    sell_trade = TradeData(
        trade_id="SELL_001", 
        timestamp=timestamps[50],
        side="sell",
        price=close_prices[50], 
        quantity=100,
        pnl=50.0
    )
    trades.append(sell_trade)
    
    print(f"\nCreated {len(trades)} test trades")
    
    # Test positioning calculation manually
    for trade in trades:
        idx = chart._timestamp_to_index(trade.timestamp)
        print(f"\n--- {trade.side.upper()} Trade Analysis ---")
        print(f"Trade timestamp: {trade.timestamp}")
        print(f"Resolved to index: {idx}")
        print(f"Trade price: {trade.price:.5f}")
        
        if idx is not None and idx < len(chart.data_buffer.low):
            candle_low = chart.data_buffer.low[idx]
            candle_high = chart.data_buffer.high[idx]
            candle_range = candle_high - candle_low
            
            print(f"Candle low: {candle_low:.5f}")
            print(f"Candle high: {candle_high:.5f}")
            print(f"Candle range: {candle_range:.5f}")
            
            if trade.side == 'buy':
                offset = candle_range * 0.02
                arrow_y = candle_low - offset
                print(f"Buy arrow offset: {offset:.5f}")
                print(f"Buy arrow Y position: {arrow_y:.5f}")
            else:
                offset = candle_range * 0.02  
                arrow_y = candle_high + offset
                print(f"Sell arrow offset: {offset:.5f}")
                print(f"Sell arrow Y position: {arrow_y:.5f}")
    
    # Add trades to chart
    print(f"\nAdding trades to chart...")
    chart.add_trades(trades)
    
    print(f"Trade markers created: {list(chart.trade_markers.keys())}")
    
    app.quit()

if __name__ == "__main__":
    test_trade_positioning()