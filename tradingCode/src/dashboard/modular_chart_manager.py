"""
Modular Chart Manager - Orchestrates all chart components
Clean, modular architecture for the trading dashboard.
"""

import numpy as np
from typing import Dict, List, Optional, Callable
from PyQt5.QtCore import QObject, pyqtSignal
import logging

from .vispy_chart_engine import VispyChartEngine
from .trade_marker_renderer import TradeMarkerRenderer

logger = logging.getLogger(__name__)


class ModularChartManager(QObject):
    """
    Main chart manager that coordinates all rendering components.
    Provides a clean interface for the dashboard.
    """
    
    # Signals for component communication
    viewport_changed = pyqtSignal(int, int)  # start, end
    trade_clicked = pyqtSignal(str)  # trade_id
    hover_update = pyqtSignal(int, dict)  # bar_index, data
    crosshair_update = pyqtSignal(int, float)  # bar_index, price
    
    def __init__(self, width=1400, height=700):
        """Initialize the modular chart manager."""
        super().__init__()
        
        # Core components
        self.engine = VispyChartEngine(width, height)
        self.trade_renderer = TradeMarkerRenderer()
        
        # Data storage
        self.ohlcv_data = None
        self.trades_data = []
        self.datetime_data = None
        
        # State
        self.viewport_start = 0
        self.viewport_end = 500
        self.data_length = 0
        
        # Callbacks
        self.callbacks = {
            'viewport_change': [],
            'hover': [],
            'crosshair': [],
            'trade_click': []
        }
        
        # Connect engine events
        self._setup_event_handlers()
        
        logger.info("Modular chart manager initialized")
    
    def _setup_event_handlers(self):
        """Set up event handlers for user interactions."""
        canvas = self.engine.canvas
        
        # Mouse events
        canvas.connect(self.on_mouse_move)
        canvas.connect(self.on_mouse_press)
        canvas.connect(self.on_mouse_release)
        canvas.connect(self.on_mouse_wheel)
        
        # Keyboard events
        canvas.connect(self.on_key_press)
    
    # ==================== Data Loading ====================
    
    def load_ohlcv_data(self, ohlcv_data: Dict[str, np.ndarray]) -> bool:
        """
        Load OHLCV data into the chart.
        
        Args:
            ohlcv_data: Dictionary with keys 'open', 'high', 'low', 'close', 'volume', 'datetime'
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.ohlcv_data = ohlcv_data
            self.data_length = len(ohlcv_data['open'])
            
            # Extract datetime if available
            if 'datetime' in ohlcv_data:
                self.datetime_data = ohlcv_data['datetime']
                self.trade_renderer.set_datetime_data(self.datetime_data)
            
            # Update components
            self.engine.set_ohlcv_data(ohlcv_data)
            self.engine.datetime_data = self.datetime_data  # Pass datetime to engine
            self.trade_renderer.set_price_data(ohlcv_data)
            
            # Set initial viewport
            self.viewport_end = min(500, self.data_length)
            self.set_viewport(0, self.viewport_end)
            
            logger.info(f"Loaded {self.data_length} bars of OHLCV data")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load OHLCV data: {e}")
            return False
    
    def load_trades(self, trades_data: List[Dict]) -> bool:
        """
        Load trades data for rendering markers.
        
        Args:
            trades_data: List of trade dictionaries
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.trades_data = trades_data
            self.trade_renderer.set_trades(trades_data)
            
            # Update trade markers in engine
            self._update_trade_markers()
            
            logger.info(f"Loaded {len(trades_data)} trades")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load trades: {e}")
            return False
    
    # ==================== Viewport Management ====================
    
    def set_viewport(self, start: int, end: int):
        """Set the visible range of bars."""
        self.viewport_start = max(0, start)
        self.viewport_end = min(self.data_length, end)
        
        # Update engine
        self.engine.set_viewport(self.viewport_start, self.viewport_end)
        
        # Update trade markers
        self._update_trade_markers()
        
        # Emit signal
        self.viewport_changed.emit(self.viewport_start, self.viewport_end)
        
        # Call callbacks
        for callback in self.callbacks['viewport_change']:
            callback(self.viewport_start, self.viewport_end)
        
        logger.debug(f"Viewport set to [{self.viewport_start}:{self.viewport_end}]")
    
    def zoom(self, factor: float, center: Optional[int] = None):
        """Zoom in/out by factor."""
        if center is None:
            center = (self.viewport_start + self.viewport_end) // 2
        
        current_width = self.viewport_end - self.viewport_start
        new_width = int(current_width * factor)
        new_width = max(10, min(new_width, self.data_length))
        
        # Center the zoom
        new_start = center - new_width // 2
        new_end = new_start + new_width
        
        # Adjust bounds
        if new_start < 0:
            new_start = 0
            new_end = new_width
        elif new_end > self.data_length:
            new_end = self.data_length
            new_start = new_end - new_width
        
        self.set_viewport(new_start, new_end)
    
    def pan(self, delta: int):
        """Pan the viewport by delta bars."""
        new_start = self.viewport_start + delta
        new_end = self.viewport_end + delta
        
        # Check bounds
        if new_start < 0:
            delta = -self.viewport_start
        elif new_end > self.data_length:
            delta = self.data_length - self.viewport_end
        
        if delta != 0:
            self.set_viewport(self.viewport_start + delta, self.viewport_end + delta)
    
    # ==================== Trade Management ====================
    
    def _update_trade_markers(self):
        """Update trade marker rendering."""
        if not self.trades_data:
            return
        
        # Generate triangle vertices
        vertices, colors = self.trade_renderer.generate_triangle_vertices(
            self.viewport_start, self.viewport_end
        )
        
        if vertices is not None and len(vertices) > 0:
            logger.info(f"Updating engine with {len(vertices)} trade vertices")
            logger.info(f"First triangle vertices: {vertices[:3] if len(vertices) >= 3 else vertices}")
            logger.info(f"Vertex data types: {vertices.dtype}, {colors.dtype}")
            
            # Pass to engine for rendering
            self.engine.trade_vertices = vertices
            self.engine.trade_colors = colors
            
            # Update shader
            if self.engine.trade_marker_program:
                self.engine.trade_marker_program['a_position'] = vertices
                self.engine.trade_marker_program['a_color'] = colors
                
                # Also ensure projection is set
                if hasattr(self.engine, 'projection') and self.engine.projection is not None:
                    self.engine.trade_marker_program['u_projection'] = self.engine.projection
                    logger.info("Projection matrix updated for trade markers")
            else:
                logger.error("Trade marker program not initialized!")
            
            self.engine.canvas.update()
        else:
            logger.warning("No vertices to update")
    
    def navigate_to_trade(self, trade_id: str):
        """Navigate the viewport to show a specific trade."""
        result = self.trade_renderer.highlight_trade(trade_id)
        if result:
            entry_idx, exit_idx = result
            
            # Calculate viewport to center on trade
            trade_width = exit_idx - entry_idx
            padding = max(50, trade_width * 2)
            
            new_start = max(0, entry_idx - padding)
            new_end = min(self.data_length, exit_idx + padding)
            
            self.set_viewport(new_start, new_end)
            logger.info(f"Navigated to trade {trade_id} at bars [{entry_idx}:{exit_idx}]")
    
    # ==================== Event Handlers ====================
    
    def on_mouse_move(self, event):
        """Handle mouse move events."""
        if event.pos is None:
            return
        
        x, y = event.pos
        bar_index = self._screen_to_bar_index(x)
        
        if 0 <= bar_index < self.data_length:
            # Get data at this bar
            data = self._get_bar_data(bar_index)
            
            # Emit hover signal
            self.hover_update.emit(bar_index, data)
            
            # Call hover callbacks
            for callback in self.callbacks['hover']:
                callback(bar_index, data)
    
    def on_mouse_press(self, event):
        """Handle mouse press events."""
        if event.button == 1:  # Left click
            x, y = event.pos
            bar_index = self._screen_to_bar_index(x)
            
            # Check if clicking on a trade
            trade = self.trade_renderer.get_trade_at_position(bar_index)
            if trade:
                trade_id = trade.get('trade_id', '')
                self.trade_clicked.emit(trade_id)
                
                for callback in self.callbacks['trade_click']:
                    callback(trade_id)
    
    def on_mouse_release(self, event):
        """Handle mouse release events."""
        pass
    
    def on_mouse_wheel(self, event):
        """Handle mouse wheel events for zooming."""
        delta = event.delta[1]
        factor = 1.1 if delta > 0 else 0.9
        
        # Get bar index at mouse position
        x, y = event.pos
        center = self._screen_to_bar_index(x)
        
        self.zoom(factor, center)
    
    def on_key_press(self, event):
        """Handle keyboard events."""
        if event.key == 'Left':
            self.pan(-10)
        elif event.key == 'Right':
            self.pan(10)
        elif event.key == 'Up':
            self.zoom(0.9)
        elif event.key == 'Down':
            self.zoom(1.1)
        elif event.key == 'Home':
            self.set_viewport(0, min(500, self.data_length))
        elif event.key == 'End':
            self.set_viewport(max(0, self.data_length - 500), self.data_length)
    
    # ==================== Helper Methods ====================
    
    def _screen_to_bar_index(self, x: float) -> int:
        """Convert screen x coordinate to bar index."""
        if self.engine.width <= 0:
            return 0
        
        viewport_width = self.viewport_end - self.viewport_start
        bar_index = self.viewport_start + int((x / self.engine.width) * viewport_width)
        return max(0, min(bar_index, self.data_length - 1))
    
    def _get_bar_data(self, bar_index: int) -> Dict:
        """Get OHLCV data for a specific bar."""
        if self.ohlcv_data is None or bar_index < 0 or bar_index >= self.data_length:
            return {}
        
        data = {
            'index': bar_index,
            'open': self.ohlcv_data['open'][bar_index],
            'high': self.ohlcv_data['high'][bar_index],
            'low': self.ohlcv_data['low'][bar_index],
            'close': self.ohlcv_data['close'][bar_index],
            'volume': self.ohlcv_data['volume'][bar_index]
        }
        
        if self.datetime_data is not None:
            data['datetime'] = self.datetime_data[bar_index]
        
        return data
    
    # ==================== Callback Registration ====================
    
    def register_callback(self, event_type: str, callback: Callable):
        """Register a callback for an event type."""
        if event_type in self.callbacks:
            self.callbacks[event_type].append(callback)
            logger.debug(f"Registered callback for {event_type}")
    
    def unregister_callback(self, event_type: str, callback: Callable):
        """Unregister a callback for an event type."""
        if event_type in self.callbacks and callback in self.callbacks[event_type]:
            self.callbacks[event_type].remove(callback)
            logger.debug(f"Unregistered callback for {event_type}")
    
    # ==================== Public Interface ====================
    
    def get_canvas(self):
        """Get the VisPy canvas for embedding in Qt."""
        return self.engine.canvas.native
    
    def update(self):
        """Force a redraw of the chart."""
        self.engine.canvas.update()
    
    def get_viewport_info(self) -> Dict:
        """Get current viewport information."""
        return {
            'start': self.viewport_start,
            'end': self.viewport_end,
            'width': self.viewport_end - self.viewport_start,
            'total_bars': self.data_length
        }