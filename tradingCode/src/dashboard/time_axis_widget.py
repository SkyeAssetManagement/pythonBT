#!/usr/bin/env python3
"""
Time Axis Widget - Displays HH:MM YYYY-MM-DD labels on X-axis
"""

import numpy as np
import pandas as pd
from typing import Optional, List, Tuple
from PyQt5.QtWidgets import QWidget, QLabel, QHBoxLayout
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QPainter, QFont, QColor

class TimeAxisWidget(QWidget):
    """
    Widget that displays time labels along the bottom of the chart
    Shows HH:MM format with YYYY-MM-DD at day boundaries
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Data storage
        self.datetime_data: Optional[np.ndarray] = None
        self.viewport_start: int = 0
        self.viewport_end: int = 100
        self.chart_width: int = 1200
        
        # Styling
        self.setFixedHeight(30)
        self.setStyleSheet("""
            QWidget {
                background-color: #2b2b2b;
                color: white;
                border-top: 1px solid #555555;
            }
        """)
        
        # Font for labels
        self.time_font = QFont("Arial", 9)
        self.date_font = QFont("Arial", 8)
        
        print("SUCCESS: Time axis widget initialized")
    
    def load_datetime_data(self, datetime_data: np.ndarray) -> bool:
        """Load datetime data for time axis"""
        try:
            self.datetime_data = datetime_data
            self.update()
            print(f"Time axis: Loaded {len(datetime_data)} datetime points")
            return True
        except Exception as e:
            print(f"ERROR: Failed to load datetime data: {e}")
            return False
    
    def update_viewport(self, start: int, end: int, chart_width: int = 1200):
        """Update viewport range"""
        self.viewport_start = start
        self.viewport_end = end  
        self.chart_width = chart_width
        self.update()
    
    def paintEvent(self, event):
        """Paint time labels"""
        if self.datetime_data is None or len(self.datetime_data) == 0 or self.viewport_start >= self.viewport_end:
            return
            
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        
        try:
            # Calculate label positions
            viewport_width = self.viewport_end - self.viewport_start
            widget_width = self.width()
            
            # Generate time labels - aim for 6-8 labels across the viewport
            num_labels = min(8, max(4, viewport_width // 50))
            step = max(1, viewport_width // num_labels)
            
            prev_date = None
            
            for i in range(num_labels + 1):
                bar_idx = self.viewport_start + i * step
                if bar_idx >= len(self.datetime_data):
                    break
                    
                # Convert bar index to screen position
                screen_x = int((bar_idx - self.viewport_start) / viewport_width * widget_width)
                
                # Get timestamp and format
                timestamp = pd.to_datetime(self.datetime_data[bar_idx])
                time_str = timestamp.strftime('%H:%M')
                date_str = timestamp.strftime('%Y-%m-%d')
                
                # Check if we need to show date (day boundary or first label)
                current_date = timestamp.date()
                show_date = (prev_date is None or current_date != prev_date)
                
                # Draw time label
                painter.setFont(self.time_font)
                painter.setPen(QColor(255, 255, 255))  # White
                painter.drawText(screen_x - 15, 15, time_str)
                
                # Draw date label if needed
                if show_date:
                    painter.setFont(self.date_font)
                    painter.setPen(QColor(180, 180, 180))  # Light gray
                    painter.drawText(screen_x - 25, 28, date_str)
                
                prev_date = current_date
                
        except Exception as e:
            print(f"ERROR: Time axis painting failed: {e}")


class TradeListTimeFormatter:
    """Helper class to format trade list times as HH:MM YYYY-MM-DD"""
    
    @staticmethod
    def format_timestamp(timestamp) -> str:
        """Format timestamp as HH:MM YYYY-MM-DD"""
        try:
            if pd.isna(timestamp):
                return "--"
                
            # Handle different timestamp formats
            if isinstance(timestamp, (int, float)):
                if timestamp > 1e15:  # Likely nanoseconds
                    dt = pd.to_datetime(timestamp)
                else:  # Likely bar index or seconds
                    return str(int(timestamp))
            else:
                dt = pd.to_datetime(timestamp)
            
            return dt.strftime('%H:%M %Y-%m-%d')
            
        except Exception as e:
            print(f"ERROR: Timestamp formatting failed: {e}")
            return str(timestamp)


def test_time_axis_widget():
    """Test the time axis widget"""
    from PyQt5.QtWidgets import QApplication, QVBoxLayout, QMainWindow
    import sys
    
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    
    # Create test data
    start_time = pd.Timestamp('2024-01-01 09:00:00')
    datetime_index = pd.date_range(start=start_time, periods=1000, freq='1min')
    datetime_ns = datetime_index.astype(np.int64)
    
    # Create main window
    main_window = QMainWindow()
    central_widget = QWidget()
    main_window.setCentralWidget(central_widget)
    
    layout = QVBoxLayout(central_widget)
    
    # Add placeholder chart area
    chart_placeholder = QWidget()
    chart_placeholder.setMinimumHeight(400)
    chart_placeholder.setStyleSheet("background-color: #1e1e1e; border: 1px solid #444444;")
    layout.addWidget(chart_placeholder)
    
    # Add time axis widget
    time_axis = TimeAxisWidget()
    time_axis.load_datetime_data(datetime_ns)
    time_axis.update_viewport(100, 300, 800)  # Show bars 100-300
    layout.addWidget(time_axis)
    
    # Test formatter
    formatter = TradeListTimeFormatter()
    test_timestamps = [datetime_ns[150], datetime_ns[200], datetime_ns[250]]
    print("Time formatting test:")
    for ts in test_timestamps:
        formatted = formatter.format_timestamp(ts)
        print(f"  {ts} -> {formatted}")
    
    main_window.setWindowTitle("Time Axis Widget Test")
    main_window.resize(1000, 500)
    main_window.show()
    
    print("Time axis widget test ready. Close window to exit.")
    
    return app.exec_()

if __name__ == "__main__":
    success = test_time_axis_widget()
    print(f"Time axis widget test {'completed' if success == 0 else 'failed'}")