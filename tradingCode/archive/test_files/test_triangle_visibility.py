#!/usr/bin/env python
"""
Direct test to verify triangle visibility in the modular dashboard
"""

import sys
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from vispy import app, gloo
from vispy.util.transforms import ortho
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Simple shaders
VERTEX_SHADER = """
uniform mat4 u_projection;
attribute vec2 a_position;
attribute vec4 a_color;
varying vec4 v_color;

void main() {
    gl_Position = u_projection * vec4(a_position, 0.0, 1.0);
    v_color = a_color;
}
"""

FRAGMENT_SHADER = """
varying vec4 v_color;

void main() {
    gl_FragColor = v_color;
}
"""

class TriangleVisibilityTest:
    def __init__(self):
        app.use_app('PyQt5')
        self.canvas = app.Canvas(size=(1200, 600), title='Triangle Visibility Test', show=True)
        
        # Create shader program
        self.program = gloo.Program(VERTEX_SHADER, FRAGMENT_SHADER)
        
        # Create some sample candlestick-like data
        self.create_chart_data()
        
        # Create trade triangles
        self.create_trade_triangles()
        
        # Connect draw event
        self.canvas.connect(self.on_draw)
        self.canvas.connect(self.on_resize)
        
        # Initial setup
        self.on_resize(None)
        
    def create_chart_data(self):
        """Create simple line chart data to represent price"""
        x = np.arange(0, 100, 1, dtype=np.float32)
        y = 100 + np.cumsum(np.random.randn(100) * 0.5).astype(np.float32)
        
        # Create line vertices (pairs of points)
        vertices = []
        for i in range(len(x)-1):
            vertices.append([x[i], y[i]])
            vertices.append([x[i+1], y[i+1]])
        
        self.line_vertices = np.array(vertices, dtype=np.float32)
        self.line_colors = np.array([[0.5, 0.5, 0.5, 1.0]] * len(vertices), dtype=np.float32)
        
        # Store y values for triangle positioning
        self.y_values = y
        
    def create_trade_triangles(self):
        """Create trade triangle markers"""
        triangles = []
        colors = []
        
        # Create 5 trade markers at different positions
        trade_positions = [10, 25, 40, 60, 80]
        
        for i, pos in enumerate(trade_positions):
            # Get price at this position
            price = self.y_values[pos]
            
            # Calculate offset (5% of price range)
            price_range = np.max(self.y_values) - np.min(self.y_values)
            offset = price_range * 0.1
            
            # Triangle size
            width = 2.0
            height = offset * 0.8
            
            if i % 2 == 0:
                # Long entry - green triangle pointing up below price
                base_y = price - offset
                triangles.extend([
                    [pos, base_y - height],      # Bottom tip
                    [pos - width, base_y],        # Top left
                    [pos + width, base_y],        # Top right
                ])
                # Bright green
                colors.extend([[0.0, 1.0, 0.0, 1.0]] * 3)
            else:
                # Short entry - red triangle pointing down above price
                base_y = price + offset
                triangles.extend([
                    [pos, base_y + height],       # Top tip
                    [pos - width, base_y],        # Bottom left
                    [pos + width, base_y],        # Bottom right
                ])
                # Bright red
                colors.extend([[1.0, 0.0, 0.0, 1.0]] * 3)
        
        self.triangle_vertices = np.array(triangles, dtype=np.float32)
        self.triangle_colors = np.array(colors, dtype=np.float32)
        
        print(f"Created {len(triangles)} triangle vertices")
        print(f"Triangle positions: {trade_positions}")
        print(f"First triangle: {triangles[:3]}")
        
    def on_draw(self, event):
        gloo.clear(color='black')
        
        # Draw price line
        self.program['a_position'] = self.line_vertices
        self.program['a_color'] = self.line_colors
        self.program.draw('lines')
        
        # Draw trade triangles
        gloo.set_state(
            blend=True,
            blend_func=('src_alpha', 'one_minus_src_alpha'),
            depth_test=False,
            cull_face=False
        )
        
        self.program['a_position'] = self.triangle_vertices
        self.program['a_color'] = self.triangle_colors
        self.program.draw('triangles')
        
        gloo.set_state(blend=False)
        
    def on_resize(self, event):
        if event:
            w, h = event.size
        else:
            w, h = self.canvas.size
            
        gloo.set_viewport(0, 0, w, h)
        
        # Set projection to show x: 0-100, y: based on data range
        y_min = np.min(self.y_values) - 5
        y_max = np.max(self.y_values) + 5
        
        projection = ortho(-5, 105, y_min, y_max, -1, 1)
        self.program['u_projection'] = projection
        
        print(f"Viewport: X[0-100], Y[{y_min:.1f}-{y_max:.1f}]")


def main():
    print("="*60)
    print("TRIANGLE VISIBILITY TEST")
    print("="*60)
    print("You should see:")
    print("  - Gray price line")
    print("  - GREEN triangles pointing UP (long entries)")
    print("  - RED triangles pointing DOWN (short entries)")
    print("="*60)
    
    test = TriangleVisibilityTest()
    app.run()


if __name__ == "__main__":
    main()