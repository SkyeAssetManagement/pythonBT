#!/usr/bin/env python3
"""
Trade Graphics - Fixed PyQtGraph implementation using built-in ScatterPlotItem
=============================================================================

High-performance trade visualization using PyQtGraph's optimized ScatterPlotItem
with built-in arrow symbols instead of custom GraphicsObjects.
"""

import numpy as np
import pandas as pd
import pyqtgraph as pg
from PyQt5 import QtCore, QtGui, QtWidgets
from typing import List, Dict, Optional, Tuple
import logging

from trade_data import TradeData, TradeCollection

logger = logging.getLogger(__name__)

class TradeVisualization:
    """
    Efficient trade visualization using PyQtGraph's built-in ScatterPlotItem
    
    Uses optimized scatter plots with built-in symbols:
    - arrow_up: Green for BUY/COVER (below bars)
    - arrow_down: Red for SHORT/SELL (above bars)  
    - circle: White dots for exact execution prices
    """
    
    def __init__(self, trades: TradeCollection, bar_data: Dict[str, np.ndarray]):
        """
        Initialize trade visualization
        
        Args:
            trades: TradeCollection with trade data
            bar_data: Dictionary with 'high', 'low' arrays for positioning
        """
        self.trades = trades
        self.bar_data = bar_data
        
        # Create scatter plot items
        self.trade_arrows = None
        self.price_dots = None
        
        # Prepare data
        self._prepare_visualization_data()
        
        logger.info(f"Created TradeVisualization with {len(trades)} trades")
    
    def _prepare_visualization_data(self):
        """Prepare data for scatter plot visualization"""
        if len(self.trades) == 0:
            return
        
        # Lists for arrow data
        arrow_x = []
        arrow_y = []  
        arrow_symbols = []
        arrow_colors = []
        arrow_sizes = []
        
        # Lists for price dot data
        dot_x = []
        dot_y = []
        
        # Color scheme - using RGB tuples for PyQtGraph
        colors = {
            'BUY': (0, 255, 136, 200),      # Bright green
            'COVER': (0, 255, 136, 120),    # Bright green, more transparent
            'SHORT': (255, 68, 68, 200),    # Bright red  
            'SELL': (255, 68, 68, 120)      # Bright red, more transparent
        }
        
        for trade in self.trades:
            bar_idx = trade.bar_index
            
            # Skip trades without bar data
            if (bar_idx < 0 or 
                bar_idx >= len(self.bar_data['high']) or
                bar_idx >= len(self.bar_data['low'])):
                continue
            
            # Get bar high/low for positioning
            bar_high = self.bar_data['high'][bar_idx]
            bar_low = self.bar_data['low'][bar_idx]
            bar_range = bar_high - bar_low
            
            # Position arrows and determine symbol
            if trade.trade_type in ['BUY', 'COVER']:
                # Below bar for long entries/exits
                arrow_y_pos = bar_low - (bar_range * 0.1)  # Closer to bar
                symbol = 'arrow_up'
            else:  # SHORT, SELL
                # Above bar for short entries/exits
                arrow_y_pos = bar_high + (bar_range * 0.1)  # Closer to bar  
                symbol = 'arrow_down'
            
            # Add arrow data
            arrow_x.append(bar_idx)
            arrow_y.append(arrow_y_pos)
            arrow_symbols.append(symbol)
            arrow_colors.append(colors[trade.trade_type])
            
            # Size based on trade size (smaller, reasonable bounds)
            size = min(max(trade.size * 4, 8), 16)  # Smaller arrows
            arrow_sizes.append(size)
            
            # Add price dot data  
            dot_x.append(bar_idx)
            dot_y.append(trade.price)
        
        # Store processed data
        self.arrow_data = {
            'x': np.array(arrow_x),
            'y': np.array(arrow_y),
            'symbols': arrow_symbols,
            'colors': arrow_colors, 
            'sizes': np.array(arrow_sizes)
        }
        
        self.dot_data = {
            'x': np.array(dot_x),
            'y': np.array(dot_y)
        }
    
    def create_arrow_scatter(self) -> pg.ScatterPlotItem:
        """Create optimized scatter plot for trade arrows using spots parameter"""
        if not hasattr(self, 'arrow_data') or len(self.arrow_data['x']) == 0:
            # Return empty scatter plot
            return pg.ScatterPlotItem([])
        
        # FIXED: Create individual spot dictionaries for mixed symbols
        # This is the correct way to handle different symbols per point
        spots = []
        for i in range(len(self.arrow_data['x'])):
            spot = {
                'pos': (self.arrow_data['x'][i], self.arrow_data['y'][i]),
                'symbol': self.arrow_data['symbols'][i],  # Individual symbol per point
                'size': self.arrow_data['sizes'][i],
                'brush': pg.mkBrush(color=self.arrow_data['colors'][i]),
                'pen': pg.mkPen(color='black', width=1)  # Black outline for visibility
            }
            spots.append(spot)
        
        # Create scatter plot using spots parameter (supports mixed symbols)
        scatter = pg.ScatterPlotItem(
            spots=spots,  # CORRECT: Use spots for mixed symbols
            pxMode=True,  # CRITICAL: Size in pixels for proper updates
            hoverable=True  # Enable hover for tooltips
        )
        
        return scatter
    
    def create_dots_scatter(self) -> pg.ScatterPlotItem:
        """Create optimized scatter plot for price dots using spots parameter"""
        if not hasattr(self, 'dot_data') or len(self.dot_data['x']) == 0:
            # Return empty scatter plot
            return pg.ScatterPlotItem([])
        
        # Create spots for consistent approach (even though single symbol)
        spots = []
        for i in range(len(self.dot_data['x'])):
            spot = {
                'pos': (self.dot_data['x'][i], self.dot_data['y'][i]),
                'symbol': 'o',  # Circle symbol
                'size': 6,  # Small, consistent size
                'brush': pg.mkBrush(color='white'),
                'pen': pg.mkPen(color='black', width=1)  # Black outline for visibility
            }
            spots.append(spot)
        
        # Create scatter plot using spots parameter for consistency
        scatter = pg.ScatterPlotItem(
            spots=spots,  # Use spots approach
            pxMode=True,  # Size in pixels
            hoverable=True  # Enable hover for trade details
        )
        
        return scatter
    
    def get_trade_at_point(self, x: float, y: float, tolerance: float = 10.0) -> Optional[TradeData]:
        """
        Get trade at specific point for hover detection
        
        Args:
            x, y: Point coordinates  
            tolerance: Search tolerance in pixels
            
        Returns:
            TradeData if found within tolerance, None otherwise
        """
        if not hasattr(self, 'dot_data') or len(self.dot_data['x']) == 0:
            return None
        
        # Find closest dot
        distances = np.sqrt((self.dot_data['x'] - x)**2 + (self.dot_data['y'] - y)**2)
        min_idx = np.argmin(distances)
        
        if distances[min_idx] <= tolerance:
            # Find corresponding trade
            target_x = self.dot_data['x'][min_idx]
            target_y = self.dot_data['y'][min_idx]
            
            for trade in self.trades:
                if (abs(trade.bar_index - target_x) < 0.5 and 
                    abs(trade.price - target_y) < 0.01):
                    return trade
        
        return None


def create_test_scatter_graphics():
    """Create test trade graphics using ScatterPlotItem approach"""
    from trade_data import create_sample_trades
    
    # Create sample data
    trades = create_sample_trades(50, 0, 200)
    
    # Create fake bar data
    n_bars = 200
    bar_data = {
        'high': 4000 + np.random.randn(n_bars) * 10 + np.arange(n_bars) * 0.1,
        'low': 3990 + np.random.randn(n_bars) * 10 + np.arange(n_bars) * 0.1
    }
    
    # Ensure high > low
    for i in range(n_bars):
        if bar_data['high'][i] <= bar_data['low'][i]:
            bar_data['high'][i] = bar_data['low'][i] + 5
    
    # Create visualization
    viz = TradeVisualization(trades, bar_data)
    arrows = viz.create_arrow_scatter()
    dots = viz.create_dots_scatter()
    
    print(f"Created scatter plot graphics:")
    print(f"  - {len(viz.arrow_data['x']) if hasattr(viz, 'arrow_data') else 0} trade arrows")
    print(f"  - {len(viz.dot_data['x']) if hasattr(viz, 'dot_data') else 0} price dots")
    print(f"  - Using built-in PyQtGraph ScatterPlotItem for efficiency")
    
    return viz, arrows, dots, trades, bar_data


if __name__ == "__main__":
    # Test the fixed graphics approach
    print("Testing Fixed Trade Graphics with ScatterPlotItem...")
    
    try:
        viz, arrows, dots, trades, bar_data = create_test_scatter_graphics()
        
        # Test trade lookup
        if len(trades) > 0:
            first_trade = trades[0]
            found_trade = viz.get_trade_at_point(
                first_trade.bar_index, 
                first_trade.price, 
                tolerance=1.0
            )
            
            if found_trade:
                print(f"[OK] Trade lookup working: Found trade at bar {found_trade.bar_index}")
            else:
                print("[FAIL] Trade lookup failed")
        
        # Test performance  
        import time
        from trade_data import create_sample_trades
        start_time = time.perf_counter()
        large_trades = create_sample_trades(5000, 0, 10000)
        large_bar_data = {
            'high': 4000 + np.random.randn(10000) * 10,
            'low': 3990 + np.random.randn(10000) * 10
        }
        large_viz = TradeVisualization(large_trades, large_bar_data)
        large_arrows = large_viz.create_arrow_scatter()
        large_dots = large_viz.create_dots_scatter()
        creation_time = (time.perf_counter() - start_time) * 1000
        
        print(f"[OK] Performance test: {creation_time:.2f}ms for 5000 trades")
        
        # Test statistics
        stats = trades.get_statistics()
        print(f"Trade statistics: {stats['trade_types']}")
        
        print("Fixed trade graphics test completed successfully!")
        
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()