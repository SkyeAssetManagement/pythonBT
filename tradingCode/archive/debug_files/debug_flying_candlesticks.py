#!/usr/bin/env python3
"""
Debug the flying candlesticks issue by examining vertex positions
"""

import sys
import numpy as np
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QTimer
from step6_complete_final import FinalTradingDashboard, create_ultimate_test_data

def debug_candlestick_positions():
    """Debug candlestick vertex positions to find flying candlesticks"""
    print("="*70)
    print("DEBUGGING FLYING CANDLESTICKS ISSUE")
    print("="*70)
    
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    
    # Create test data
    ohlcv_data, trades_csv_path = create_ultimate_test_data()
    
    print(f"Created OHLCV data with {len(ohlcv_data['close'])} bars")
    print(f"First few datetime values: {ohlcv_data['datetime'][:5]}")
    print(f"First few close prices: {ohlcv_data['close'][:5]}")
    
    # Create dashboard
    dashboard = FinalTradingDashboard()
    dashboard.resize(1920, 1200)
    dashboard.show()
    
    # Load data
    success = dashboard.load_ultimate_dataset(ohlcv_data, trades_csv_path)
    print(f"Dataset loaded: {'SUCCESS' if success else 'FAILED'}")
    
    if success and dashboard.final_chart:
        def analyze_geometry():
            """Analyze the generated candlestick geometry"""
            chart = dashboard.final_chart
            print(f"\nChart Analysis:")
            print(f"Data length: {chart.data_length}")
            print(f"Viewport: {chart.viewport_start} - {chart.viewport_end}")
            print(f"Vertex count: {chart.candlestick_vertex_count}")
            
            # Let's regenerate and examine geometry
            if hasattr(chart, 'candlestick_program'):
                try:
                    # Get the current vertex buffer data
                    vertices = chart.candlestick_program['a_position'].get_data()
                    if vertices is not None and len(vertices) > 0:
                        print(f"\nVertex Analysis:")
                        print(f"Total vertices: {len(vertices)}")
                        print(f"Vertex shape: {vertices.shape}")
                        
                        # Analyze X coordinates (should be bar indices)
                        x_coords = vertices[:, 0]
                        y_coords = vertices[:, 1]
                        
                        print(f"X coordinates range: {x_coords.min():.2f} - {x_coords.max():.2f}")
                        print(f"Y coordinates range: {y_coords.min():.6f} - {y_coords.max():.6f}")
                        
                        # Look for outliers (flying candlesticks)
                        expected_x_min = chart.viewport_start - 50
                        expected_x_max = chart.viewport_end + 50
                        
                        flying_vertices = (x_coords < expected_x_min) | (x_coords > expected_x_max)
                        num_flying = np.sum(flying_vertices)
                        
                        print(f"\nExpected X range: {expected_x_min} - {expected_x_max}")
                        print(f"Flying vertices (outside expected range): {num_flying}")
                        
                        if num_flying > 0:
                            print(f"Flying X coordinates: {x_coords[flying_vertices][:10]}")  # First 10
                            print("PROBLEM: Vertices outside expected viewport range!")
                        else:
                            print("All vertices within expected range")
                            
                        # Check for extremely large coordinates (datetime stamps)
                        huge_coords = np.abs(x_coords) > 100000
                        if np.any(huge_coords):
                            print(f"WARNING: Found {np.sum(huge_coords)} vertices with huge coordinates!")
                            print(f"Huge coordinates: {x_coords[huge_coords][:5]}")
                        
                except Exception as e:
                    print(f"Error analyzing geometry: {e}")
            
            # Take screenshot for analysis
            print(f"\nTaking screenshot for visual inspection...")
            import time
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"flying_candlesticks_debug_{timestamp}.png"
            pixmap = dashboard.grab()
            success = pixmap.save(filename)
            if success:
                print(f"Screenshot saved: {filename}")
            
            app.quit()
        
        # Run analysis after 2 seconds
        QTimer.singleShot(2000, analyze_geometry)
        
        print("\nStarting geometry analysis...")
        app.exec_()
    
    print("Flying candlesticks debug complete")

if __name__ == "__main__":
    debug_candlestick_positions()