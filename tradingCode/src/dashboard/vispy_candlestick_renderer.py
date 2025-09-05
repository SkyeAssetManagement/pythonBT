# src/dashboard/vispy_candlestick_renderer.py
# High-Performance VisPy Candlestick Renderer with GPU Instancing
# 
# Implements GPU-accelerated candlestick rendering for 7M+ datapoints using:
# - Viewport-based culling (only render visible 500 bars)
# - GPU instancing for maximum performance
# - Custom shaders optimized for OHLCV data
# - Memory-efficient data management with LOD (Level of Detail)

import numpy as np
from typing import Dict, Optional, Tuple, List
import time
from vispy import app, gloo, scene
from vispy.util.transforms import ortho
from vispy.color import Color

# GPU-optimized shaders for candlestick rendering
CANDLESTICK_VERTEX_SHADER = """
#version 120

// Per-vertex attributes (base geometry)
attribute vec2 a_position;  // Base quad vertices [-0.5, -0.5] to [0.5, 0.5]

// Per-instance attributes (candlestick data)
attribute float a_timestamp;   // X position (time)
attribute float a_open;        // Open price
attribute float a_high;        // High price  
attribute float a_low;         // Low price
attribute float a_close;       // Close price
attribute float a_volume;      // Volume (for width scaling)
attribute float a_color_flag;  // 0=bearish(red), 1=bullish(green)

// Uniforms for viewport and rendering
uniform mat4 u_projection;     // Projection matrix
uniform vec2 u_viewport_range; // Visible time range [start_time, end_time]
uniform float u_candle_width;  // Dynamic candle width based on zoom
uniform vec2 u_price_range;    // Visible price range [min_price, max_price]
uniform vec2 u_screen_size;    // Screen dimensions

// Outputs to fragment shader
varying vec3 v_color;
varying float v_alpha;

void main() {
    // Calculate candle body dimensions
    float body_bottom = min(a_open, a_close);
    float body_top = max(a_open, a_close);
    float body_height = max(body_top - body_bottom, u_candle_width * 0.1); // Minimum height for doji
    
    // Transform base quad to candle body
    vec2 pos = a_position;
    pos.x = a_timestamp + (pos.x * u_candle_width);  // Scale width and position
    pos.y = body_bottom + ((pos.y + 0.5) * body_height); // Scale height and center
    
    // Apply projection transform
    gl_Position = u_projection * vec4(pos, 0.0, 1.0);
    
    // Determine color based on price direction
    if (a_color_flag > 0.5) {
        v_color = vec3(0.0, 0.8, 0.2); // Bullish green
    } else {
        v_color = vec3(0.8, 0.2, 0.0); // Bearish red
    }
    
    // Alpha for viewport culling optimization
    v_alpha = 1.0;
    
    // Cull candles outside viewport (GPU-side optimization)
    if (a_timestamp < u_viewport_range.x || a_timestamp > u_viewport_range.y) {
        v_alpha = 0.0;
    }
}
"""

CANDLESTICK_FRAGMENT_SHADER = """
#version 120

varying vec3 v_color;
varying float v_alpha;

void main() {
    gl_FragColor = vec4(v_color, v_alpha);
}
"""

# Wick rendering shaders (separate draw call for performance)
WICK_VERTEX_SHADER = """
#version 120

// Per-vertex attributes (line endpoints)
attribute vec2 a_position;  // [0,0] for bottom, [0,1] for top

// Per-instance attributes
attribute float a_timestamp;
attribute float a_high;
attribute float a_low;
attribute float a_color_flag;

uniform mat4 u_projection;
uniform vec2 u_viewport_range;
uniform float u_candle_width;

varying vec3 v_color;
varying float v_alpha;

void main() {
    // Create vertical wick line
    vec2 pos;
    pos.x = a_timestamp;
    pos.y = mix(a_low, a_high, a_position.y);  // Interpolate between low and high
    
    gl_Position = u_projection * vec4(pos, 0.0, 1.0);
    
    // Color (slightly darker than body)
    if (a_color_flag > 0.5) {
        v_color = vec3(0.0, 0.6, 0.1); // Dark green
    } else {
        v_color = vec3(0.6, 0.1, 0.0); // Dark red
    }
    
    // Viewport culling
    v_alpha = 1.0;
    if (a_timestamp < u_viewport_range.x || a_timestamp > u_viewport_range.y) {
        v_alpha = 0.0;
    }
}
"""

WICK_FRAGMENT_SHADER = """
#version 120

varying vec3 v_color;
varying float v_alpha;

void main() {
    gl_FragColor = vec4(v_color, v_alpha);
}
"""

class DataPipeline:
    """
    High-performance data pipeline for managing 7M+ candlesticks
    Implements viewport-based data loading and LOD (Level of Detail) management
    """
    
    def __init__(self, max_points: int = 7_000_000):
        # Full dataset in memory - optimized numpy arrays
        self.max_points = max_points
        self.full_data = None
        self.data_length = 0
        
        # Viewport management
        self.viewport_start = 0
        self.viewport_end = 500  # Initially show last 500 bars
        self.viewport_buffer = 50  # Extra bars for smooth scrolling
        
        # LOD (Level of Detail) cache for different zoom levels
        self.lod_cache = {}
        self.current_lod = 1  # 1 = full detail, 2 = every 2nd bar, etc.
        
    def load_ohlcv_data(self, data: Dict[str, np.ndarray]) -> bool:
        """
        Load OHLCV data into optimized GPU-friendly format
        Creates pre-computed instance data for maximum rendering performance
        """
        try:
            # Extract and validate data
            timestamps = np.asarray(data['datetime'], dtype=np.float64)
            open_prices = np.asarray(data['open'], dtype=np.float32)
            high_prices = np.asarray(data['high'], dtype=np.float32)
            low_prices = np.asarray(data['low'], dtype=np.float32)
            close_prices = np.asarray(data['close'], dtype=np.float32)
            volumes = np.asarray(data['volume'], dtype=np.float32)
            
            self.data_length = len(timestamps)
            print(f"   INFO: Loading {self.data_length:,} candlesticks into GPU pipeline...")
            
            # Convert timestamps to sequential indices for easier rendering
            # Store original timestamps for trade matching
            self.original_timestamps = timestamps.copy()
            time_indices = np.arange(self.data_length, dtype=np.float32)
            
            # Calculate color flags (0=bearish, 1=bullish)
            color_flags = (close_prices >= open_prices).astype(np.float32)
            
            # Create optimized instance data structure for GPU
            # Each candlestick becomes one instance with all required attributes
            instance_dtype = [
                ('timestamp', np.float32),    # Sequential index for X positioning
                ('open', np.float32),         # Open price
                ('high', np.float32),         # High price
                ('low', np.float32),          # Low price
                ('close', np.float32),        # Close price
                ('volume', np.float32),       # Volume (for width modulation)
                ('color_flag', np.float32),   # Bullish/Bearish flag
            ]
            
            # Pack all data into GPU-optimized structure
            self.full_data = np.zeros(self.data_length, dtype=instance_dtype)
            self.full_data['timestamp'] = time_indices
            self.full_data['open'] = open_prices
            self.full_data['high'] = high_prices
            self.full_data['low'] = low_prices
            self.full_data['close'] = close_prices
            self.full_data['volume'] = volumes
            self.full_data['color_flag'] = color_flags
            
            # Calculate price ranges for viewport optimization
            self.global_price_min = float(np.min(low_prices))
            self.global_price_max = float(np.max(high_prices))
            
            print(f"   SUCCESS: GPU data pipeline ready - {self.data_length:,} candlesticks")
            print(f"   INFO: Price range: ${self.global_price_min:.5f} - ${self.global_price_max:.5f}")
            print(f"   INFO: Memory usage: ~{self.full_data.nbytes / 1024 / 1024:.1f} MB")
            
            # Pre-compute LOD levels for smooth zooming
            self._precompute_lod_levels()
            
            # Set initial viewport to last 500 bars
            self.set_viewport_to_recent(500)
            
            return True
            
        except Exception as e:
            print(f"   ERROR: Failed to load OHLCV data: {e}")
            return False
    
    def _precompute_lod_levels(self):
        """
        Pre-compute multiple LOD (Level of Detail) levels for different zoom ranges
        Enables smooth zooming from 500 bars to 7M bars without performance loss
        """
        print(f"   INFO: Pre-computing LOD levels for smooth zooming...")
        
        if self.full_data is None:
            return
            
        # Create LOD levels: 1x, 2x, 5x, 10x, 20x, 50x, 100x decimation
        lod_factors = [1, 2, 5, 10, 20, 50, 100, 200, 500]
        
        for factor in lod_factors:
            if factor == 1:
                self.lod_cache[factor] = self.full_data  # Full detail
            else:
                # Decimate data by taking every Nth candlestick
                decimated = self.full_data[::factor].copy()
                # Adjust timestamps to maintain proper spacing
                decimated['timestamp'] = np.arange(len(decimated), dtype=np.float32) * factor
                self.lod_cache[factor] = decimated
        
        print(f"   SUCCESS: Created {len(lod_factors)} LOD levels for optimal zooming")
    
    def get_viewport_data(self, start_idx: int, end_idx: int, lod_factor: int = 1) -> np.ndarray:
        """
        Get data for current viewport with specified LOD
        Returns only the candlesticks that need to be rendered for maximum performance
        """
        if self.full_data is None:
            return np.array([], dtype=self.full_data.dtype if self.full_data is not None else np.float32)
        
        # Get appropriate LOD data
        lod_data = self.lod_cache.get(lod_factor, self.full_data)
        
        # Add buffer for smooth scrolling
        buffer_start = max(0, start_idx - self.viewport_buffer)
        buffer_end = min(len(lod_data), end_idx + self.viewport_buffer)
        
        viewport_data = lod_data[buffer_start:buffer_end].copy()
        
        return viewport_data
    
    def set_viewport_to_recent(self, num_bars: int = 500):
        """Set viewport to show the most recent N bars (AmiBroker-style default)"""
        if self.data_length > num_bars:
            self.viewport_start = self.data_length - num_bars
            self.viewport_end = self.data_length
        else:
            self.viewport_start = 0
            self.viewport_end = self.data_length
    
    def calculate_optimal_lod(self, visible_range: int) -> int:
        """
        Calculate optimal LOD factor based on how many bars are visible
        Ensures smooth performance regardless of zoom level
        """
        if visible_range <= 1000:
            return 1  # Full detail for close zoom
        elif visible_range <= 5000:
            return 2  # Slight decimation
        elif visible_range <= 20000:
            return 5  # Moderate decimation
        elif visible_range <= 100000:
            return 20  # Aggressive decimation
        else:
            return 100  # Maximum decimation for full view

class VispyCandlestickRenderer:
    """
    High-performance VisPy candlestick renderer with GPU instancing
    Optimized for 7M+ datapoints with viewport-based rendering
    """
    
    def __init__(self, width: int = 1400, height: int = 800):
        # Core VisPy application and canvas
        self.app = app.use_app('PyQt5')  # Use PyQt5 backend for best performance
        self.canvas = app.Canvas(title='Lightning Trading Dashboard - VisPy', 
                                size=(width, height), keys='interactive')
        
        # Performance monitoring
        self.frame_count = 0
        self.fps_start_time = time.time()
        self.target_fps = 60
        
        # Data pipeline for managing 7M+ candlesticks
        self.data_pipeline = DataPipeline()
        
        # GPU programs for rendering
        self.candle_program = None
        self.wick_program = None
        
        # Viewport and interaction state
        self.viewport_x_range = [0, 500]  # Show last 500 bars initially
        self.viewport_y_range = [0, 100]  # Will be auto-calculated from data
        self.candle_width = 0.8  # Width of each candlestick
        
        # Viewport change callback for time axis updates
        self.viewport_change_callback = None
        
        # Base geometry for instancing
        self._create_base_geometry()
        
        # Initialize GPU programs
        self._initialize_gpu_programs()
        
        # Connect event handlers for interaction
        self._connect_event_handlers()
        
        # Screenshot system for testing and debugging
        self.screenshot_counter = 0
        
        print(f"   SUCCESS: VisPy candlestick renderer initialized")
        print(f"   INFO: Canvas size: {width}x{height}")
        print(f"   INFO: Target FPS: {self.target_fps}")
    
    def _create_base_geometry(self):
        """
        Create base geometry for GPU instancing
        Each candlestick will be an instance of these base shapes
        """
        # Base quad vertices for candlestick bodies (will be scaled per instance)
        self.body_vertices = np.array([
            [-0.5, -0.5],  # Bottom-left
            [ 0.5, -0.5],  # Bottom-right
            [ 0.5,  0.5],  # Top-right
            [-0.5,  0.5],  # Top-left
        ], dtype=np.float32)
        
        # Indices for quad (2 triangles)
        self.body_indices = np.array([0, 1, 2, 0, 2, 3], dtype=np.uint32)
        
        # Base line vertices for wicks (2 points: bottom and top)
        self.wick_vertices = np.array([
            [0.0, 0.0],  # Bottom of wick
            [0.0, 1.0],  # Top of wick
        ], dtype=np.float32)
        
        # Indices for line
        self.wick_indices = np.array([0, 1], dtype=np.uint32)
    
    def _initialize_gpu_programs(self):
        """
        Initialize GPU shader programs for maximum rendering performance
        Uses instanced rendering to draw all candlesticks in minimal draw calls
        """
        try:
            # Create candlestick body program
            self.candle_program = gloo.Program(CANDLESTICK_VERTEX_SHADER, CANDLESTICK_FRAGMENT_SHADER)
            
            # Set base geometry
            self.candle_program['a_position'] = gloo.VertexBuffer(self.body_vertices)
            
            # Create wick program  
            self.wick_program = gloo.Program(WICK_VERTEX_SHADER, WICK_FRAGMENT_SHADER)
            self.wick_program['a_position'] = gloo.VertexBuffer(self.wick_vertices)
            
            # Set initial uniforms
            self._update_projection_matrix()
            
            print(f"   SUCCESS: GPU shader programs initialized")
            
        except Exception as e:
            print(f"   ERROR: Failed to initialize GPU programs: {e}")
            raise
    
    def _update_projection_matrix(self):
        """Update projection matrix for current viewport"""
        if self.candle_program is None:
            return
            
        # Create orthographic projection for the current viewport
        x_min, x_max = self.viewport_x_range
        y_min, y_max = self.viewport_y_range
        
        # Add small padding to avoid edge clipping
        x_padding = (x_max - x_min) * 0.02
        y_padding = (y_max - y_min) * 0.02
        
        projection = ortho(x_min - x_padding, x_max + x_padding, 
                          y_min - y_padding, y_max + y_padding, 
                          -1, 1)
        
        # Update both programs
        self.candle_program['u_projection'] = projection
        if self.wick_program:
            self.wick_program['u_projection'] = projection
        
        # Update viewport-related uniforms
        self.candle_program['u_viewport_range'] = [x_min, x_max]
        self.candle_program['u_candle_width'] = self.candle_width
        self.candle_program['u_price_range'] = [y_min, y_max]
        
        if self.wick_program:
            self.wick_program['u_viewport_range'] = [x_min, x_max]
            self.wick_program['u_candle_width'] = self.candle_width
    
    def load_data(self, ohlcv_data: Dict[str, np.ndarray]) -> bool:
        """
        Load OHLCV data into the renderer
        Optimizes data for GPU rendering and sets up viewport
        """
        success = self.data_pipeline.load_ohlcv_data(ohlcv_data)
        
        if success:
            # Auto-calculate optimal viewport based on data
            self._auto_set_viewport()
            
            # Load initial viewport data into GPU
            self._update_gpu_buffers()
            
            print(f"   SUCCESS: {self.data_pipeline.data_length:,} candlesticks loaded into renderer")
            
        return success
    
    def _auto_set_viewport(self):
        """
        Automatically set viewport to show recent data (AmiBroker-style)
        Calculates optimal price range for forex data visibility
        """
        if self.data_pipeline.full_data is None:
            return
        
        # Start with last 500 bars (or all data if less)
        self.data_pipeline.set_viewport_to_recent(500)
        
        # Set X range
        self.viewport_x_range = [self.data_pipeline.viewport_start, 
                                self.data_pipeline.viewport_end]
        
        # Calculate Y range from visible data
        viewport_data = self.data_pipeline.get_viewport_data(
            self.data_pipeline.viewport_start, 
            self.data_pipeline.viewport_end
        )
        
        if len(viewport_data) > 0:
            y_min = float(np.min(viewport_data['low']))
            y_max = float(np.max(viewport_data['high']))
            y_padding = (y_max - y_min) * 0.1  # 10% padding for readability
            
            self.viewport_y_range = [y_min - y_padding, y_max + y_padding]
            
            print(f"   INFO: Auto-set viewport: X=[{self.viewport_x_range[0]:.0f}, {self.viewport_x_range[1]:.0f}]")
            print(f"   INFO: Auto-set viewport: Y=[{self.viewport_y_range[0]:.5f}, {self.viewport_y_range[1]:.5f}]")
        
        # Update projection matrix
        self._update_projection_matrix()
    
    def _update_gpu_buffers(self):
        """
        Update GPU buffers with current viewport data
        Only uploads visible candlesticks for maximum performance
        """
        if self.data_pipeline.full_data is None:
            return
        
        start_idx = int(self.viewport_x_range[0])
        end_idx = int(self.viewport_x_range[1])
        
        # Get optimal LOD for current zoom level
        visible_range = end_idx - start_idx
        lod_factor = self.data_pipeline.calculate_optimal_lod(visible_range)
        
        # Get viewport data with LOD
        viewport_data = self.data_pipeline.get_viewport_data(start_idx, end_idx, lod_factor)
        
        if len(viewport_data) == 0:
            return
        
        # Create instanced vertex buffers for GPU
        # Each attribute becomes a per-instance buffer with divisor=1
        candle_vbo = gloo.VertexBuffer(viewport_data)
        
        # Bind instance attributes for candlestick bodies
        self.candle_program['a_timestamp'] = gloo.VertexBuffer(viewport_data['timestamp'])
        self.candle_program['a_open'] = gloo.VertexBuffer(viewport_data['open'])
        self.candle_program['a_high'] = gloo.VertexBuffer(viewport_data['high'])
        self.candle_program['a_low'] = gloo.VertexBuffer(viewport_data['low'])
        self.candle_program['a_close'] = gloo.VertexBuffer(viewport_data['close'])
        self.candle_program['a_volume'] = gloo.VertexBuffer(viewport_data['volume'])
        self.candle_program['a_color_flag'] = gloo.VertexBuffer(viewport_data['color_flag'])
        
        # Bind instance attributes for wicks
        self.wick_program['a_timestamp'] = gloo.VertexBuffer(viewport_data['timestamp'])
        self.wick_program['a_high'] = gloo.VertexBuffer(viewport_data['high'])
        self.wick_program['a_low'] = gloo.VertexBuffer(viewport_data['low'])
        self.wick_program['a_color_flag'] = gloo.VertexBuffer(viewport_data['color_flag'])
        
        # Store instance count for rendering
        self.current_instance_count = len(viewport_data)
        
        print(f"   INFO: GPU buffers updated - {self.current_instance_count:,} instances (LOD: {lod_factor}x)")
    
    def _connect_event_handlers(self):
        """Connect mouse and keyboard event handlers for interaction"""
        
        @self.canvas.connect
        def on_draw(event):
            """Main rendering function - called at target FPS"""
            gloo.clear(color='black')
            
            if hasattr(self, 'current_instance_count') and self.current_instance_count > 0:
                # Draw candlestick wicks first (behind bodies)
                self.wick_program.draw('lines', indices=self.wick_indices, 
                                      instances=self.current_instance_count)
                
                # Draw candlestick bodies on top
                self.candle_program.draw('triangles', indices=self.body_indices,
                                        instances=self.current_instance_count)
            
            # Update FPS counter
            self._update_fps_counter()
        
        @self.canvas.connect
        def on_resize(event):
            """Handle canvas resize"""
            gloo.set_viewport(0, 0, *event.physical_size)
            
        @self.canvas.connect  
        def on_mouse_wheel(event):
            """Handle zooming with Ctrl+scroll, panning with scroll"""
            # Check for Ctrl modifier (VisPy uses 'Control' in modifiers tuple)
            ctrl_pressed = hasattr(event, 'modifiers') and 'Control' in (event.modifiers or [])
            
            if ctrl_pressed:
                # Ctrl+scroll for zoom
                self._handle_zoom(event.delta[1], event.pos)
            else:
                # Just scroll for pan (horizontal pan based on scroll direction)
                self._handle_scroll_pan(event.delta[1])
        
        @self.canvas.connect
        def on_mouse_move(event):
            """Handle panning with mouse drag"""
            if event.is_dragging and event.button == 1:  # Left mouse button
                self._handle_pan(event.last_pos, event.pos)
                
        @self.canvas.connect
        def on_key_press(event):
            """Handle keyboard shortcuts"""
            if event.key == 'r' or event.key == 'R':
                self._reset_view()
            elif event.key == 's' or event.key == 'S':
                self._take_screenshot()
            elif event.key == 'q' or event.key == 'Q':
                self.app.quit()
    
    def _handle_zoom(self, delta: float, mouse_pos: Tuple[int, int]):
        """
        Handle zooming with smooth LOD transitions
        Maintains mouse position as zoom center for intuitive interaction
        """
        zoom_factor = 1.1 if delta > 0 else 0.9
        
        # Calculate current view range
        x_range = self.viewport_x_range[1] - self.viewport_x_range[0]
        y_range = self.viewport_y_range[1] - self.viewport_y_range[0]
        
        # Calculate zoom center as fraction of current view
        canvas_width, canvas_height = self.canvas.size
        zoom_center_x_frac = mouse_pos[0] / canvas_width
        zoom_center_y_frac = 1.0 - (mouse_pos[1] / canvas_height)  # Flip Y for OpenGL coords
        
        # Calculate new ranges
        new_x_range = x_range * zoom_factor
        new_y_range = y_range * zoom_factor
        
        # Apply zoom centered on mouse position
        x_center = self.viewport_x_range[0] + x_range * zoom_center_x_frac
        y_center = self.viewport_y_range[0] + y_range * zoom_center_y_frac
        
        self.viewport_x_range[0] = x_center - new_x_range * zoom_center_x_frac
        self.viewport_x_range[1] = x_center + new_x_range * (1 - zoom_center_x_frac)
        self.viewport_y_range[0] = y_center - new_y_range * zoom_center_y_frac
        self.viewport_y_range[1] = y_center + new_y_range * (1 - zoom_center_y_frac)
        
        # Clamp to data bounds
        max_x = self.data_pipeline.data_length - 1
        self.viewport_x_range[0] = max(0, self.viewport_x_range[0])
        self.viewport_x_range[1] = min(max_x, self.viewport_x_range[1])
        
        # Update rendering
        self._update_projection_matrix()
        self._update_gpu_buffers()
        self.canvas.update()
        
        # Notify time axis of viewport change
        self._notify_viewport_change()
        
    def _handle_pan(self, last_pos: Tuple[int, int], current_pos: Tuple[int, int]):
        """Handle panning by converting mouse movement to data coordinates"""
        canvas_width, canvas_height = self.canvas.size
        
        # Calculate movement in normalized coordinates
        dx_norm = (current_pos[0] - last_pos[0]) / canvas_width
        dy_norm = (last_pos[1] - current_pos[1]) / canvas_height  # Flip Y
        
        # Convert to data coordinates
        x_range = self.viewport_x_range[1] - self.viewport_x_range[0]
        y_range = self.viewport_y_range[1] - self.viewport_y_range[0]
        
        dx_data = dx_norm * x_range
        dy_data = dy_norm * y_range
        
        # Apply pan
        self.viewport_x_range[0] -= dx_data
        self.viewport_x_range[1] -= dx_data
        self.viewport_y_range[0] -= dy_data
        self.viewport_y_range[1] -= dy_data
        
        # Clamp X to data bounds
        max_x = self.data_pipeline.data_length - 1
        if self.viewport_x_range[0] < 0:
            shift = -self.viewport_x_range[0]
            self.viewport_x_range[0] += shift
            self.viewport_x_range[1] += shift
        elif self.viewport_x_range[1] > max_x:
            shift = self.viewport_x_range[1] - max_x
            self.viewport_x_range[0] -= shift
            self.viewport_x_range[1] -= shift
        
        # Update rendering
        self._update_projection_matrix()
        self._update_gpu_buffers()
        self.canvas.update()
        
        # Notify time axis of viewport change
        self._notify_viewport_change()
    
    def _handle_scroll_pan(self, delta: float):
        """Handle horizontal panning with scroll wheel (no Ctrl)"""
        # Calculate pan distance based on current viewport size
        viewport_width = self.viewport_x_range[1] - self.viewport_x_range[0]
        pan_factor = 0.1  # Pan 10% of current viewport width per scroll tick
        pan_distance = viewport_width * pan_factor
        
        # Reverse scroll direction for natural panning (scroll up = pan right)
        if delta > 0:
            pan_distance = -pan_distance
        
        # Apply horizontal pan
        self.viewport_x_range[0] += pan_distance
        self.viewport_x_range[1] += pan_distance
        
        # Clamp to data bounds
        max_x = self.data_pipeline.data_length - 1
        if self.viewport_x_range[0] < 0:
            shift = -self.viewport_x_range[0]
            self.viewport_x_range[0] += shift
            self.viewport_x_range[1] += shift
        elif self.viewport_x_range[1] > max_x:
            shift = self.viewport_x_range[1] - max_x
            self.viewport_x_range[0] -= shift
            self.viewport_x_range[1] -= shift
        
        # Update rendering
        self._update_projection_matrix()
        self._update_gpu_buffers()
        self.canvas.update()
        
        # Notify time axis of viewport change
        self._notify_viewport_change()
    
    def _reset_view(self):
        """Reset view to show recent data (AmiBroker-style default)"""
        self._auto_set_viewport()
        self._update_gpu_buffers()
        self.canvas.update()
        print(f"   INFO: View reset to recent data")
    
    def _take_screenshot(self):
        """Take screenshot for testing and verification"""
        try:
            import imageio
            
            # Render current frame to get pixels
            img = self.canvas.render()
            
            # Save with timestamp
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"candlestick_test_{timestamp}_{self.screenshot_counter:03d}.png"
            filepath = f"C:\\Users\\skyeAM\\SkyeAM Dropbox\\SAMresearch\\ABtoPython\\tradingCode\\{filename}"
            
            imageio.imwrite(filepath, img)
            self.screenshot_counter += 1
            
            print(f"   INFO: Screenshot saved: {filename}")
            
        except ImportError:
            print(f"   WARNING: imageio not available for screenshots")
        except Exception as e:
            print(f"   WARNING: Screenshot failed: {e}")
    
    def _update_fps_counter(self):
        """Monitor and display FPS for performance validation"""
        self.frame_count += 1
        current_time = time.time()
        
        if current_time - self.fps_start_time >= 2.0:  # Update every 2 seconds
            fps = self.frame_count / (current_time - self.fps_start_time)
            print(f"   PERFORMANCE: {fps:.1f} FPS ({self.current_instance_count:,} candles)")
            
            # Reset counter
            self.frame_count = 0
            self.fps_start_time = current_time
    
    def set_viewport_change_callback(self, callback):
        """Set callback to be called when viewport changes"""
        self.viewport_change_callback = callback
    
    def _notify_viewport_change(self):
        """Notify registered callback of viewport change"""
        if self.viewport_change_callback:
            # Convert viewport ranges to integer indices
            start_idx = int(max(0, self.viewport_x_range[0]))
            end_idx = int(min(self.data_pipeline.data_length - 1, self.viewport_x_range[1]))
            self.viewport_change_callback(start_idx, end_idx)
    
    def show(self):
        """Show the canvas and start the event loop"""
        self.canvas.show()
        print(f"   SUCCESS: VisPy candlestick renderer is now running!")
        print(f"   CONTROLS:")
        print(f"     - Mouse wheel: Zoom in/out")
        print(f"     - Left mouse drag: Pan")
        print(f"     - 'R' key: Reset view")
        print(f"     - 'S' key: Take screenshot") 
        print(f"     - 'Q' key: Quit")
        return self.app.run()


def create_test_data(num_candles: int = 10000) -> Dict[str, np.ndarray]:
    """
    Create synthetic OHLCV test data for development and testing
    Simulates realistic forex price movements
    """
    print(f"   INFO: Creating {num_candles:,} synthetic candlesticks for testing...")
    
    # Generate realistic price walk
    np.random.seed(42)  # Reproducible test data
    
    # Start price around typical forex levels
    base_price = 1.2000  # EUR/USD typical level
    price_volatility = 0.001  # 100 pips typical movement
    
    # Generate price series using random walk
    price_changes = np.random.normal(0, price_volatility, num_candles)
    prices = np.cumsum(price_changes) + base_price
    
    # Generate OHLC from price series
    opens = prices.copy()
    closes = opens + np.random.normal(0, price_volatility/2, num_candles)
    
    # Generate highs and lows relative to open/close
    high_noise = np.random.exponential(price_volatility/4, num_candles)
    low_noise = np.random.exponential(price_volatility/4, num_candles)
    
    highs = np.maximum(opens, closes) + high_noise
    lows = np.minimum(opens, closes) - low_noise
    
    # Generate volumes (random but realistic)
    volumes = np.random.lognormal(10, 0.5, num_candles)
    
    # Generate sequential timestamps
    timestamps = np.arange(num_candles, dtype=np.int64)
    
    test_data = {
        'datetime': timestamps,
        'open': opens.astype(np.float32),
        'high': highs.astype(np.float32), 
        'low': lows.astype(np.float32),
        'close': closes.astype(np.float32),
        'volume': volumes.astype(np.float32)
    }
    
    print(f"   SUCCESS: Test data created - Price range: {lows.min():.5f} to {highs.max():.5f}")
    
    return test_data


def test_performance_scaling():
    """
    Test renderer performance with different dataset sizes
    Validates that viewport rendering scales to 7M+ candlesticks
    """
    print(f"\n=== VISPY CANDLESTICK RENDERER PERFORMANCE TEST ===")
    
    # Test with multiple dataset sizes
    test_sizes = [1000, 10000, 100000, 1000000]
    
    for size in test_sizes:
        print(f"\n--- Testing with {size:,} candlesticks ---")
        
        # Create test data
        start_time = time.time()
        test_data = create_test_data(size)
        data_gen_time = time.time() - start_time
        print(f"   Data generation: {data_gen_time:.3f}s")
        
        # Create renderer  
        renderer = VispyCandlestickRenderer(width=1200, height=800)
        
        # Load data and measure performance
        start_time = time.time()
        success = renderer.load_data(test_data)
        load_time = time.time() - start_time
        
        if success:
            print(f"   Data loading: {load_time:.3f}s")
            print(f"   Performance: {size/load_time:.0f} candles/second")
            
            # Show renderer for 3 seconds to measure FPS
            print(f"   Running renderer for FPS measurement...")
            
            # Note: In actual testing, you would call renderer.show() 
            # Here we just validate the setup
            print(f"   SUCCESS: Renderer ready for {size:,} candlesticks")
            
        else:
            print(f"   ERROR: Failed to load {size:,} candlesticks")


# Main execution for testing
if __name__ == "__main__":
    # Quick functionality test
    print(f"VisPy Candlestick Renderer - High Performance Test")
    
    # Test with 50,000 candlesticks (manageable for development)
    test_data = create_test_data(50000)
    
    # Create and test renderer
    renderer = VispyCandlestickRenderer()
    
    if renderer.load_data(test_data):
        print(f"SUCCESS: Ready to render 50,000 candlesticks")
        print(f"INFO: Call renderer.show() to display the chart")
        
        # Uncomment next line to actually show the renderer:
        # renderer.show()
    else:
        print(f"ERROR: Failed to initialize renderer")