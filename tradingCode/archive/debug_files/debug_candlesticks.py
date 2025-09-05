#!/usr/bin/env python3
"""
Debug Candlestick Rendering
Test if candlesticks are being generated properly
"""

import sys
import numpy as np
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from PyQt5 import QtWidgets, QtCore, QtGui
from src.dashboard.chart_widget import TradingChart, CandlestickItem
from src.dashboard.data_structures import ChartDataBuffer

def test_candlestick_rendering():
    """Test candlestick rendering in isolation"""
    
    print("=== CANDLESTICK RENDERING DEBUG ===")
    
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(sys.argv)
    
    # Create simple test data
    n_bars = 20
    
    # Create clear price patterns
    base_price = 0.65400
    close_prices = base_price + np.random.uniform(-0.00050, 0.00050, n_bars)
    open_prices = np.roll(close_prices, 1)
    open_prices[0] = base_price
    
    # Make sure high > max(open, close) and low < min(open, close)
    high_prices = np.maximum(open_prices, close_prices) + np.random.uniform(0.00010, 0.00030, n_bars)
    low_prices = np.minimum(open_prices, close_prices) - np.random.uniform(0.00010, 0.00030, n_bars)
    
    volume = np.full(n_bars, 10000.0)
    timestamps = np.arange(n_bars, dtype=np.int64) * 60 * 1000000000
    
    print(f"Test data created:")
    print(f"  Bars: {n_bars}")
    print(f"  Price range: {np.min(low_prices):.5f} - {np.max(high_prices):.5f}")
    print(f"  Sample OHLC[0]: O={open_prices[0]:.5f}, H={high_prices[0]:.5f}, L={low_prices[0]:.5f}, C={close_prices[0]:.5f}")
    
    # Create data buffer
    data_buffer = ChartDataBuffer(
        timestamps=timestamps,
        open=open_prices,
        high=high_prices,
        low=low_prices,
        close=close_prices,
        volume=volume
    )
    
    print(f"Data buffer created: {len(data_buffer)} bars")
    
    # Test CandlestickItem directly
    print("\\nTesting CandlestickItem...")
    candle_item = CandlestickItem(data_buffer)
    
    print(f"CandlestickItem created:")
    print(f"  Has data: {candle_item.data is not None}")
    print(f"  Data length: {len(candle_item.data) if candle_item.data else 0}")
    print(f"  Has picture: {candle_item.picture is not None}")
    
    if candle_item.picture:
        rect = candle_item.picture.boundingRect()
        print(f"  Picture bounds: {rect.width()}x{rect.height()}")
    
    # Test bounding rect
    bounding = candle_item.boundingRect()
    print(f"  Bounding rect: x={bounding.x()}, y={bounding.y()}, w={bounding.width()}, h={bounding.height()}")
    
    # Create chart and add candlestick item
    print("\\nTesting TradingChart...")
    chart = TradingChart()
    chart.set_data(data_buffer)
    
    print(f"Chart created:")
    print(f"  Has data_buffer: {chart.data_buffer is not None}")
    print(f"  Has candle_item: {hasattr(chart, 'candle_item') and chart.candle_item is not None}")
    
    if hasattr(chart, 'candle_item') and chart.candle_item:
        print(f"  Candle item added to chart: {chart.candle_item in chart.scene().items()}")
        candle_bounds = chart.candle_item.boundingRect()
        print(f"  Chart candle bounds: w={candle_bounds.width()}, h={candle_bounds.height()}")
    
    # Show chart in a window
    window = QtWidgets.QMainWindow()
    window.setCentralWidget(chart)
    window.setWindowTitle("Candlestick Debug")
    window.resize(800, 400)
    window.show()
    
    # Process events and take screenshot
    app.processEvents()
    
    screenshot_path = Path(__file__).parent / "candlestick_debug.png"
    pixmap = window.grab()
    pixmap.save(str(screenshot_path))
    print(f"\\nScreenshot saved: {screenshot_path}")
    
    print(f"\\n{'='*50}")
    print(f"If candlesticks don't appear in the screenshot:")
    print(f"1. Check if picture is None")
    print(f"2. Check if bounding rect is valid")
    print(f"3. Check if paint method is being called")
    print(f"4. Check if data has valid OHLC relationships")
    print(f"{'='*50}")
    
    # Keep window open briefly
    QtCore.QTimer.singleShot(3000, app.quit)
    app.exec_()

if __name__ == "__main__":
    test_candlestick_rendering()