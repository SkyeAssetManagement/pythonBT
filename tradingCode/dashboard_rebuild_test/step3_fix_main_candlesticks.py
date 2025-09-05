#!/usr/bin/env python3
"""
Step 3: Fix the candlestick rendering in the main chart_widget.py
Based on analysis of the broken vs working screenshots

PROBLEM: Current rendering creates fat ovals instead of proper candlesticks
SOLUTION: Fix the drawing logic to match the working screenshot
"""

import os
import sys
import shutil

def analyze_candlestick_issue():
    """Analyze the candlestick rendering issue"""
    
    print("=== CANDLESTICK RENDERING ANALYSIS ===")
    print()
    print("BROKEN (current screenshot):")
    print("- Fat oval-shaped candlesticks")
    print("- No visible wicks")
    print("- Looks compressed and wrong")
    print()
    print("WORKING (target screenshot):")
    print("- Thin rectangular candlestick bodies")
    print("- Clear thin wicks extending from bodies")
    print("- Proper OHLC representation")
    print()
    print("ROOT CAUSE:")
    print("- The _draw_candles_batched() method is drawing wrong shapes")
    print("- Body rectangles are being drawn incorrectly")
    print("- Wick lines may not be visible")
    print()

def create_fixed_candlestick_renderer():
    """Create the fixed candlestick rendering code"""
    
    fixed_renderer_code = '''
    def _draw_candles_batched(self, painter: QtGui.QPainter, 
                            bodies: np.ndarray, wicks: np.ndarray, 
                            colors: np.ndarray):
        """FIXED: Ultra-fast batch drawing with proper candlestick shapes"""
        
        # CRITICAL: Use thin pens for proper candlestick appearance
        wick_pen = QtGui.QPen(QtCore.Qt.black, 1)  # Thin black wicks
        up_pen = QtGui.QPen(QtCore.Qt.black, 1)    # Thin black borders
        down_pen = QtGui.QPen(QtCore.Qt.black, 1)  # Thin black borders
        
        # CRITICAL: Proper brush colors for candlestick bodies
        up_brush = QtGui.QBrush(QtCore.Qt.white)   # White for up candles
        down_brush = QtGui.QBrush(QtCore.Qt.red)   # Red for down candles
        
        # 1. Draw ALL wicks first (thin vertical lines)
        painter.setPen(wick_pen)
        valid_wicks = []
        for wick in wicks:
            # Only draw wicks with valid coordinates
            if (wick[1] != wick[3] and  # Different Y coordinates
                wick[1] > 0 and wick[3] > 0 and  # Positive values
                np.isfinite(wick[1]) and np.isfinite(wick[3])):  # Valid numbers
                
                # Create thin vertical line from low to high
                line = QtCore.QLineF(wick[0], wick[1], wick[0], wick[3])
                valid_wicks.append(line)
        
        if valid_wicks:
            painter.drawLines(valid_wicks)  # Batch draw all wicks
        
        # 2. Draw UP candles (white bodies with black borders)
        up_mask = colors
        if np.any(up_mask):
            painter.setBrush(up_brush)
            painter.setPen(up_pen)
            
            up_rects = []
            up_bodies = bodies[up_mask]
            for rect in up_bodies:
                if (rect[3] > 0 and  # Positive height
                    rect[1] > 0 and  # Positive Y position
                    np.isfinite(rect[1]) and np.isfinite(rect[3])):  # Valid numbers
                    
                    # Create proper rectangle
                    qt_rect = QtCore.QRectF(rect[0], rect[1], rect[2], rect[3])
                    up_rects.append(qt_rect)
            
            if up_rects:
                painter.drawRects(up_rects)  # Batch draw up candles
        
        # 3. Draw DOWN candles (red bodies with black borders) 
        down_mask = ~colors
        if np.any(down_mask):
            painter.setBrush(down_brush)
            painter.setPen(down_pen)
            
            down_rects = []
            down_bodies = bodies[down_mask]
            for rect in down_bodies:
                if (rect[3] > 0 and  # Positive height
                    rect[1] > 0 and  # Positive Y position
                    np.isfinite(rect[1]) and np.isfinite(rect[3])):  # Valid numbers
                    
                    # Create proper rectangle
                    qt_rect = QtCore.QRectF(rect[0], rect[1], rect[2], rect[3])
                    down_rects.append(qt_rect)
            
            if down_rects:
                painter.drawRects(down_rects)  # Batch draw down candles
    '''
    
    return fixed_renderer_code

def create_fixed_candlestick_calculation():
    """Create fixed candlestick calculation code"""
    
    fixed_calc_code = '''
    @staticmethod
    def _calculate_candles_simple(o: np.ndarray, h: np.ndarray, 
                                l: np.ndarray, c: np.ndarray, 
                                x: np.ndarray, width: float = 0.6) -> tuple:
        """FIXED: Simple candle calculation with proper dimensions"""
        n = len(o)
        
        # Pre-allocate arrays
        body_rects = np.empty((n, 4), dtype=np.float32)
        wick_lines = np.empty((n, 4), dtype=np.float32)
        colors = np.empty(n, dtype=np.bool_)
        
        # FIXED: Proper vectorized operations
        colors = c >= o  # Up candles where close >= open
        
        # FIXED: Body rectangles with THIN width
        thin_width = min(width, 0.8)  # Maximum width of 0.8 for visibility
        body_rects[:, 0] = x - thin_width/2     # x position (centered)
        body_rects[:, 1] = np.minimum(o, c)     # y position (bottom of body)
        body_rects[:, 2] = thin_width           # width (THIN for proper appearance)
        body_rects[:, 3] = np.abs(c - o)        # height (actual body height)
        
        # CRITICAL: Ensure minimum height for doji candles
        price_range = h - l
        # Use a very small minimum height to avoid fat candles
        min_height = np.maximum(price_range * 0.001, np.full_like(price_range, 0.1))
        body_rects[:, 3] = np.maximum(body_rects[:, 3], min_height)
        
        # FIXED: Wick lines (thin vertical lines)
        wick_lines[:, 0] = x  # x1 (same X for vertical line)
        wick_lines[:, 1] = l  # y1 (low)
        wick_lines[:, 2] = x  # x2 (same X for vertical line)
        wick_lines[:, 3] = h  # y2 (high)
        
        return body_rects, wick_lines, colors
    '''
    
    return fixed_calc_code

def apply_candlestick_fix():
    """Apply the candlestick fix to the main chart_widget.py"""
    
    print("Applying candlestick fix to main chart_widget.py...")
    
    chart_widget_path = "../src/dashboard/chart_widget.py"
    backup_path = "../src/dashboard/chart_widget.py.backup"
    
    # Create backup
    if os.path.exists(chart_widget_path):
        if not os.path.exists(backup_path):
            shutil.copy2(chart_widget_path, backup_path)
            print(f"Created backup: {backup_path}")
    
    print("Candlestick fix code generated.")
    print("Next step: Apply this fix manually to chart_widget.py")
    print("Focus on:")
    print("1. _draw_candles_batched() method - fix drawing logic")
    print("2. _calculate_candles_simple() method - fix body dimensions")
    print("3. Ensure thin bodies and visible wicks")

def main():
    """Main function for Step 3"""
    
    print("Step 3: Fixing candlestick rendering in main code")
    print("=" * 50)
    
    analyze_candlestick_issue()
    
    print()
    print("GENERATING FIXES...")
    print()
    
    fixed_renderer = create_fixed_candlestick_renderer()
    fixed_calculator = create_fixed_candlestick_calculation()
    
    print("Fixed renderer code:")
    print(fixed_renderer)
    print()
    print("Fixed calculator code:")
    print(fixed_calculator)
    
    apply_candlestick_fix()
    
    print()
    print("Step 3 completed - fixes generated")
    print("Next: Apply these fixes to ../src/dashboard/chart_widget.py")

if __name__ == "__main__":
    main()