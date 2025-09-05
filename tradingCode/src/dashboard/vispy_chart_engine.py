"""
VisPy Chart Engine - Core rendering engine for GPU-accelerated charts
This module handles all OpenGL rendering operations for the trading dashboard.
"""

import numpy as np
from vispy import app, gloo
from vispy.util.transforms import ortho
from typing import Dict, Optional, Tuple, List
import logging

logger = logging.getLogger(__name__)


class VispyChartEngine:
    """Core VisPy chart rendering engine with OpenGL shaders."""
    
    # Vertex shader for candlesticks
    CANDLESTICK_VERTEX_SHADER = """
    uniform mat4 u_projection;
    attribute vec2 a_position;
    attribute vec4 a_color;
    varying vec4 v_color;
    
    void main() {
        gl_Position = u_projection * vec4(a_position, 0.0, 1.0);
        v_color = a_color;
    }
    """
    
    # Fragment shader for candlesticks
    CANDLESTICK_FRAGMENT_SHADER = """
    varying vec4 v_color;
    
    void main() {
        gl_FragColor = v_color;
    }
    """
    
    # Vertex shader for trade markers (same as candlesticks for triangles)
    TRADE_MARKER_VERTEX_SHADER = """
    uniform mat4 u_projection;
    attribute vec2 a_position;
    attribute vec4 a_color;
    varying vec4 v_color;
    
    void main() {
        gl_Position = u_projection * vec4(a_position, 0.0, 1.0);
        v_color = a_color;
    }
    """
    
    # Fragment shader for trade markers
    TRADE_MARKER_FRAGMENT_SHADER = """
    varying vec4 v_color;
    
    void main() {
        gl_FragColor = v_color;
    }
    """
    
    def __init__(self, width=1400, height=700):
        """Initialize the VisPy chart engine."""
        self.width = width
        self.height = height
        
        # Initialize canvas
        app.use_app('PyQt5')
        self.canvas = app.Canvas(
            size=(width, height),
            show=False,
            keys='interactive'
        )
        
        # Shader programs
        self.candlestick_program = None
        self.trade_marker_program = None
        self.grid_program = None
        
        # Data storage
        self.ohlcv_data = None
        self.trades_data = []
        self.datetime_data = None
        
        # Data buffers
        self.candlestick_vertices = None
        self.candlestick_colors = None
        self.trade_vertices = None
        self.trade_colors = None
        
        # Viewport
        self.viewport_start = 0
        self.viewport_end = 500
        self.data_length = 0
        
        # Projection matrix
        self.projection = None
        
        # Initialize shaders
        self._init_shaders()
        
        # Connect canvas events
        self.canvas.connect(self.on_draw)
        self.canvas.connect(self.on_resize)
        
    def _init_shaders(self):
        """Initialize all shader programs."""
        try:
            # Candlestick shader
            self.candlestick_program = gloo.Program(
                self.CANDLESTICK_VERTEX_SHADER,
                self.CANDLESTICK_FRAGMENT_SHADER
            )
            
            # Trade marker shader
            self.trade_marker_program = gloo.Program(
                self.TRADE_MARKER_VERTEX_SHADER,
                self.TRADE_MARKER_FRAGMENT_SHADER
            )
            
            # Grid shader (simple lines)
            self.grid_program = gloo.Program(
                self.CANDLESTICK_VERTEX_SHADER,  # Reuse for simplicity
                self.CANDLESTICK_FRAGMENT_SHADER
            )
            
            logger.info("Shader programs initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize shaders: {e}")
            raise
    
    def set_ohlcv_data(self, ohlcv_data: Dict[str, np.ndarray]):
        """Set OHLCV data for rendering."""
        self.ohlcv_data = ohlcv_data
        self.data_length = len(ohlcv_data['open'])
        self.viewport_end = min(self.viewport_end, self.data_length)
        self._update_candlestick_geometry()
        
    def set_trades_data(self, trades_data: List[Dict]):
        """Set trades data for rendering markers."""
        self.trades_data = trades_data
        self._update_trade_markers()
        
    def _update_candlestick_geometry(self):
        """Generate candlestick geometry for current viewport."""
        if self.ohlcv_data is None:
            return
            
        start = self.viewport_start
        end = self.viewport_end
        
        # Get viewport data
        opens = self.ohlcv_data['open'][start:end]
        highs = self.ohlcv_data['high'][start:end]
        lows = self.ohlcv_data['low'][start:end]
        closes = self.ohlcv_data['close'][start:end]
        
        num_bars = end - start
        vertices = []
        colors = []
        
        bar_width = 0.6
        
        for i in range(num_bars):
            x = start + i
            o, h, l, c = opens[i], highs[i], lows[i], closes[i]
            
            # Determine color
            color = [0.0, 1.0, 0.0, 1.0] if c >= o else [1.0, 0.0, 0.0, 1.0]
            
            # Wick (vertical line from low to high)
            vertices.extend([
                [x, l], [x, h]
            ])
            colors.extend([color, color])
            
            # Body (rectangle)
            body_top = max(o, c)
            body_bottom = min(o, c)
            
            # Create filled rectangle using triangles
            vertices.extend([
                [x - bar_width/2, body_bottom],
                [x + bar_width/2, body_bottom],
                [x + bar_width/2, body_top],
                
                [x - bar_width/2, body_bottom],
                [x + bar_width/2, body_top],
                [x - bar_width/2, body_top]
            ])
            colors.extend([color] * 6)
        
        if vertices:
            self.candlestick_vertices = np.array(vertices, dtype=np.float32)
            self.candlestick_colors = np.array(colors, dtype=np.float32)
            
            # Update shader data
            self.candlestick_program['a_position'] = self.candlestick_vertices
            self.candlestick_program['a_color'] = self.candlestick_colors
            
            logger.debug(f"Generated {len(vertices)} candlestick vertices")
    
    def _update_trade_markers(self):
        """Update trade marker rendering - called externally by chart manager."""
        # This method is now just a placeholder since the chart manager
        # handles trade marker updates directly
        pass
    
    def _get_bar_index(self, timestamp):
        """Convert timestamp to bar index."""
        if timestamp > 1e15:  # Nanosecond timestamp
            if hasattr(self, 'datetime_data') and self.datetime_data is not None:
                # Find closest timestamp
                diffs = np.abs(self.datetime_data - timestamp)
                return int(np.argmin(diffs))
        elif timestamp < 100000:  # Already an index
            return int(timestamp)
        return 0
    
    def _get_price_range(self):
        """Get price range for current viewport."""
        if self.ohlcv_data is None:
            return 1.0
            
        start = self.viewport_start
        end = self.viewport_end
        
        highs = self.ohlcv_data['high'][start:end]
        lows = self.ohlcv_data['low'][start:end]
        
        return np.max(highs) - np.min(lows)
    
    def _update_projection(self):
        """Update projection matrix for current viewport."""
        if self.ohlcv_data is None:
            return
            
        start = self.viewport_start
        end = self.viewport_end
        
        # Get price range
        highs = self.ohlcv_data['high'][start:end]
        lows = self.ohlcv_data['low'][start:end]
        
        y_min = np.min(lows) * 0.99
        y_max = np.max(highs) * 1.01
        
        # Create projection matrix
        self.projection = ortho(
            start - 3, end + 3,
            y_min, y_max,
            -1, 1
        )
        
        # Update all programs
        if self.candlestick_program:
            self.candlestick_program['u_projection'] = self.projection
        if self.trade_marker_program:
            self.trade_marker_program['u_projection'] = self.projection
        if self.grid_program:
            self.grid_program['u_projection'] = self.projection
    
    def on_draw(self, event):
        """Handle draw event."""
        gloo.clear(color=(0.1, 0.1, 0.1, 1.0))
        
        # Draw candlesticks
        if self.candlestick_vertices is not None:
            try:
                self.candlestick_program.draw('lines', indices=self._get_wick_indices())
                self.candlestick_program.draw('triangles', indices=self._get_body_indices())
            except:
                # Fallback to simple line drawing
                self.candlestick_program.draw('lines')
        
        # Draw trade markers as triangles
        if self.trade_vertices is not None and len(self.trade_vertices) > 0:
            try:
                gloo.set_state(
                    blend=True,
                    blend_func=('src_alpha', 'one_minus_src_alpha'),
                    depth_test=False,
                    cull_face=False
                )
                # Draw as triangles - every 3 vertices forms a triangle
                self.trade_marker_program.draw('triangles')
                gloo.set_state(blend=False)
                
                # Log once per second to avoid spam
                import time
                current_time = time.time()
                if not hasattr(self, '_last_log_time') or current_time - self._last_log_time > 1.0:
                    logger.info(f"Drew {len(self.trade_vertices)} trade vertices as triangles")
                    self._last_log_time = current_time
            except Exception as e:
                logger.error(f"Error drawing trade triangles: {e}")
    
    def _get_wick_indices(self):
        """Get indices for drawing wicks as lines."""
        if self.candlestick_vertices is None:
            return None
        
        num_bars = (self.viewport_end - self.viewport_start)
        indices = []
        for i in range(num_bars):
            base = i * 8  # 8 vertices per candlestick
            indices.extend([base, base + 1])  # Wick line
        
        return np.array(indices, dtype=np.uint32)
    
    def _get_body_indices(self):
        """Get indices for drawing bodies as triangles."""
        if self.candlestick_vertices is None:
            return None
            
        num_bars = (self.viewport_end - self.viewport_start)
        indices = []
        for i in range(num_bars):
            base = i * 8 + 2  # Skip wick vertices
            indices.extend(range(base, base + 6))  # 6 vertices for body
        
        return np.array(indices, dtype=np.uint32)
    
    def on_resize(self, event):
        """Handle resize event."""
        self.width, self.height = event.size
        gloo.set_viewport(0, 0, self.width, self.height)
        self._update_projection()
        self.canvas.update()
    
    def set_viewport(self, start: int, end: int):
        """Set the viewport range."""
        self.viewport_start = max(0, start)
        self.viewport_end = min(self.data_length, end)
        self._update_candlestick_geometry()
        # Don't call _update_trade_markers here - it's handled by chart manager
        self._update_projection()
        self.canvas.update()
    
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
            new_start = 0
            new_end = self.viewport_end - self.viewport_start
        elif new_end > self.data_length:
            new_end = self.data_length
            new_start = new_end - (self.viewport_end - self.viewport_start)
            
        self.set_viewport(new_start, new_end)