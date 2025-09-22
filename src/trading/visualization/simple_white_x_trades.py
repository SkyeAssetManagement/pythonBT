#!/usr/bin/env python3
"""
Simple White X Trades - Replace complex arrows with simple white X marks
=======================================================================
This replaces trade_graphics_fixed.py with a simple white X implementation
"""

import numpy as np
import pyqtgraph as pg
from typing import Dict, List, Optional
from trade_data import TradeCollection, TradeData

class SimpleWhiteXTrades:
    """Simple white X marks for trades - size 14, positioned at actual trade prices"""
    
    def __init__(self, trades: TradeCollection, bar_data: Dict[str, np.ndarray]):
        """
        Initialize with trades and bar data
        
        Args:
            trades: TradeCollection with trade data  
            bar_data: Dictionary with 'high', 'low' arrays for validation
        """
        self.trades = trades
        self.bar_data = bar_data
        
        # Prepare trade positions
        self.x_positions = []
        self.y_positions = []
        self._prepare_positions()
        
        print(f"SimpleWhiteXTrades: Created {len(self.x_positions)} white X marks")
    
    def _prepare_positions(self):
        """Prepare X mark positions at trade prices"""
        if len(self.trades) == 0:
            return
            
        for trade in self.trades:
            bar_idx = trade.bar_index
            
            # Validate bar index
            if (bar_idx < 0 or 
                bar_idx >= len(self.bar_data['high']) or
                bar_idx >= len(self.bar_data['low'])):
                print(f"Skipping trade at invalid bar {bar_idx}")
                continue
            
            # Validate trade price is within bar range
            bar_high = self.bar_data['high'][bar_idx]
            bar_low = self.bar_data['low'][bar_idx]
            
            trade_price = trade.price
            
            # Ensure trade price is reasonable (within bar range + some tolerance)
            price_tolerance = (bar_high - bar_low) * 0.5  # 50% tolerance
            if not (bar_low - price_tolerance <= trade_price <= bar_high + price_tolerance):
                # Adjust trade price to be within bar range
                trade_price = np.clip(trade_price, bar_low, bar_high)
                print(f"Adjusted trade price from {trade.price:.2f} to {trade_price:.2f} for bar {bar_idx}")
            
            self.x_positions.append(bar_idx)
            self.y_positions.append(trade_price)
        
        print(f"Prepared {len(self.x_positions)} trade positions within price ranges")
    
    def create_arrow_scatter(self) -> pg.ScatterPlotItem:
        """Create white X marks instead of arrows - this replaces the arrow visualization"""
        if len(self.x_positions) == 0:
            return pg.ScatterPlotItem([])

        # Create simple white X marks - size increased by 25% (14 -> 17.5 -> 18)
        scatter = pg.ScatterPlotItem(
            pos=list(zip(self.x_positions, self.y_positions)),
            symbol='x',           # Simple X symbol
            size=18,             # Size increased by 25% from 14
            brush=pg.mkBrush(color='white'),
            pen=pg.mkPen(color='white', width=3),  # Bold white pen (width 3)
            pxMode=True,         # Size in pixels
            hoverable=True,      # Enable hover
            zValue=1000         # High z-value to render on top of candles
        )

        print(f"Created white X scatter (arrows replacement): {len(self.x_positions)} marks, size=18")
        return scatter
    
    def create_dots_scatter(self) -> pg.ScatterPlotItem:
        """Create empty dots scatter - we only want X marks"""
        return pg.ScatterPlotItem([])  # No dots, just X marks
    
    def get_trade_at_point(self, x: float, y: float, tolerance: float = 10.0) -> Optional[TradeData]:
        """Get trade at specific point for hover detection"""
        if len(self.x_positions) == 0:
            return None
        
        # Find closest mark
        distances = np.sqrt((np.array(self.x_positions) - x)**2 + 
                           (np.array(self.y_positions) - y)**2)
        min_idx = np.argmin(distances)
        
        if distances[min_idx] <= tolerance:
            # Return the corresponding trade
            if min_idx < len(self.trades):
                return self.trades[min_idx]
        
        return None

# For compatibility - this is the class the main chart expects
class TradeVisualization(SimpleWhiteXTrades):
    """Compatibility wrapper - same interface as trade_graphics_fixed"""
    pass