"""
Trade Marker Renderer - Specialized module for rendering trade triangles
Handles trade visualization with proper triangle markers.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class TradeMarkerRenderer:
    """Renders trade markers as visible triangles on the chart."""
    
    def __init__(self):
        """Initialize the trade marker renderer."""
        self.trades_data = []
        self.datetime_data = None
        self.price_data = None
        self.viewport_start = 0
        self.viewport_end = 500
        
    def set_trades(self, trades_data: List[Dict]):
        """Set trades data for rendering."""
        self.trades_data = trades_data
        logger.info(f"Loaded {len(trades_data)} trades for rendering")
        
    def set_datetime_data(self, datetime_data: np.ndarray):
        """Set datetime array for timestamp conversion."""
        self.datetime_data = datetime_data
        
    def set_price_data(self, ohlcv_data: Dict[str, np.ndarray]):
        """Set price data for calculating marker positions."""
        self.price_data = ohlcv_data
        
    def generate_triangle_vertices(self, viewport_start: int, viewport_end: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate triangle vertices for trade markers.
        Returns (vertices, colors) as numpy arrays.
        """
        if not self.trades_data or self.price_data is None:
            logger.warning("No trades or price data available")
            return None, None
            
        self.viewport_start = viewport_start
        self.viewport_end = viewport_end
        
        vertices = []
        colors = []
        
        # Calculate price range for offset
        highs = self.price_data['high'][viewport_start:viewport_end]
        lows = self.price_data['low'][viewport_start:viewport_end]
        price_range = np.max(highs) - np.min(lows) if len(highs) > 0 else 1.0
        offset = price_range * 0.04  # 4% offset for visibility
        
        for trade in self.trades_data:
            # Convert timestamps to bar indices
            entry_idx = self._timestamp_to_index(trade['entry_time'])
            exit_idx = self._timestamp_to_index(trade['exit_time'])
            
            # Get trade direction
            is_long = self._is_long_trade(trade)
            
            # Generate entry marker if in viewport
            if viewport_start <= entry_idx < viewport_end:
                entry_vertices, entry_color = self._create_triangle(
                    entry_idx, 
                    trade['entry_price'],
                    offset,
                    is_entry=True,
                    is_long=is_long
                )
                vertices.extend(entry_vertices)
                colors.extend([entry_color] * len(entry_vertices))
                
            # Generate exit marker if in viewport
            if viewport_start <= exit_idx < viewport_end:
                exit_vertices, exit_color = self._create_triangle(
                    exit_idx,
                    trade['exit_price'],
                    offset,
                    is_entry=False,
                    is_long=is_long
                )
                vertices.extend(exit_vertices)
                colors.extend([exit_color] * len(exit_vertices))
        
        if vertices:
            vertices_array = np.array(vertices, dtype=np.float32)
            colors_array = np.array(colors, dtype=np.float32)
            logger.info(f"Generated {len(vertices)} triangle vertices for trades")
            return vertices_array, colors_array
        else:
            logger.warning("No trade markers in current viewport")
            return None, None
    
    def _timestamp_to_index(self, timestamp) -> int:
        """Convert timestamp to bar index."""
        # Handle nanosecond timestamps
        if timestamp > 1e15:
            if self.datetime_data is not None:
                # Find closest timestamp
                diffs = np.abs(self.datetime_data - timestamp)
                idx = int(np.argmin(diffs))
                return idx
            else:
                logger.warning("No datetime data for timestamp conversion")
                return 0
        # Handle direct indices
        elif timestamp < 100000:
            return int(timestamp)
        else:
            logger.warning(f"Unknown timestamp format: {timestamp}")
            return 0
    
    def _is_long_trade(self, trade: Dict) -> bool:
        """Determine if trade is long or short."""
        # Check various possible field names
        if 'direction' in trade:
            return trade['direction'].lower() in ['long', 'buy']
        elif 'side' in trade:
            return trade['side'].lower() in ['long', 'buy']
        elif 'type' in trade:
            return trade['type'].lower() in ['long', 'buy']
        else:
            # Default to long if not specified
            return True
    
    def _create_triangle(self, x: int, y: float, offset: float, 
                        is_entry: bool, is_long: bool) -> Tuple[List, Tuple]:
        """
        Create triangle vertices for a trade marker.
        Returns (vertices, color).
        """
        triangle_size = 1.5  # Width of triangle in bars
        triangle_height = offset * 0.8  # Height of triangle
        
        if is_long:
            if is_entry:
                # Long entry: Green triangle pointing up, below price
                base_y = y - offset
                vertices = [
                    [x, base_y - triangle_height],  # Bottom point
                    [x - triangle_size/2, base_y],  # Top left
                    [x + triangle_size/2, base_y],  # Top right
                ]
                color = (0.0, 0.8, 0.0, 1.0)  # Green
            else:
                # Long exit: Red triangle pointing down, above price
                base_y = y + offset
                vertices = [
                    [x, base_y + triangle_height],  # Top point
                    [x - triangle_size/2, base_y],  # Bottom left
                    [x + triangle_size/2, base_y],  # Bottom right
                ]
                color = (0.8, 0.0, 0.0, 1.0)  # Red
        else:
            if is_entry:
                # Short entry: Red triangle pointing down, above price
                base_y = y + offset
                vertices = [
                    [x, base_y + triangle_height],  # Top point
                    [x - triangle_size/2, base_y],  # Bottom left
                    [x + triangle_size/2, base_y],  # Bottom right
                ]
                color = (0.8, 0.0, 0.0, 1.0)  # Red
            else:
                # Short exit: Green triangle pointing up, below price
                base_y = y - offset
                vertices = [
                    [x, base_y - triangle_height],  # Bottom point
                    [x - triangle_size/2, base_y],  # Top left
                    [x + triangle_size/2, base_y],  # Top right
                ]
                color = (0.0, 0.8, 0.0, 1.0)  # Green
        
        return vertices, color
    
    def get_trade_at_position(self, bar_index: int) -> Optional[Dict]:
        """Get trade information at a specific bar index."""
        for trade in self.trades_data:
            entry_idx = self._timestamp_to_index(trade['entry_time'])
            exit_idx = self._timestamp_to_index(trade['exit_time'])
            
            if entry_idx <= bar_index <= exit_idx:
                return trade
        
        return None
    
    def highlight_trade(self, trade_id: str) -> Optional[Tuple[int, int]]:
        """
        Get the bar indices for a specific trade to highlight it.
        Returns (entry_index, exit_index) or None.
        """
        for trade in self.trades_data:
            if trade.get('trade_id') == trade_id:
                entry_idx = self._timestamp_to_index(trade['entry_time'])
                exit_idx = self._timestamp_to_index(trade['exit_time'])
                return entry_idx, exit_idx
        
        return None