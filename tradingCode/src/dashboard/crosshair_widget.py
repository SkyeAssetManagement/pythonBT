# src/dashboard/crosshair_widget.py
# Crosshair with Axis Data Points - Step 5
# Interactive crosshair overlay with precise axis value display

import numpy as np
from typing import Dict, List, Optional, Tuple, Callable
from PyQt5.QtWidgets import QWidget, QLabel, QVBoxLayout, QHBoxLayout, QFrame
from PyQt5.QtCore import Qt, pyqtSignal, QPoint, QTimer
from PyQt5.QtGui import QPainter, QPen, QColor, QFont, QBrush, QFontMetrics
import math

class CrosshairOverlay(QWidget):
    """
    Crosshair overlay widget that displays on top of chart
    Shows crosshair lines with precise axis data points
    """
    
    # Signal emitted when crosshair position changes
    position_changed = pyqtSignal(float, float)  # x_value, y_value
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Configuration
        self.enabled = True
        self.locked = False  # Whether crosshair is locked to a position
        
        # Position data
        self.mouse_x = 0
        self.mouse_y = 0
        self.data_x = 0.0
        self.data_y = 0.0
        
        # Chart bounds for coordinate conversion
        self.chart_bounds = None  # (x_min, x_max, y_min, y_max)
        self.widget_size = None   # (width, height)
        
        # Visual configuration
        self.crosshair_color = QColor(200, 200, 200, 180)
        self.locked_color = QColor(255, 255, 100, 200)
        self.axis_label_bg = QColor(60, 60, 60, 220)
        self.axis_label_text = QColor(255, 255, 255)
        
        # Fonts
        self.axis_font = QFont("Arial", 9, QFont.Bold)
        self.value_font = QFont("Courier New", 8)
        
        # Setup widget
        self.setAttribute(Qt.WA_TransparentForMouseEvents, False)
        self.setMouseTracking(True)
        
        print("SUCCESS: Crosshair overlay initialized")
    
    def set_chart_bounds(self, x_min: float, x_max: float, y_min: float, y_max: float):
        """Set chart data bounds for coordinate conversion"""
        self.chart_bounds = (x_min, x_max, y_min, y_max)
        self.update()
    
    def set_widget_size(self, width: int, height: int):
        """Set widget size for coordinate conversion"""
        self.widget_size = (width, height)
    
    def enable_crosshair(self, enabled: bool = True):
        """Enable or disable crosshair display"""
        self.enabled = enabled
        if not enabled:
            self.locked = False
        self.update()
        print(f"Crosshair {'enabled' if enabled else 'disabled'}")
    
    def lock_crosshair(self, locked: bool = True):
        """Lock or unlock crosshair at current position"""
        self.locked = locked
        self.update()
        print(f"Crosshair {'locked' if locked else 'unlocked'} at ({self.data_x:.5f}, {self.data_y:.5f})")
    
    def toggle_lock(self):
        """Toggle crosshair lock state"""
        self.lock_crosshair(not self.locked)
    
    def mousePressEvent(self, event):
        """Handle mouse press to lock/unlock crosshair"""
        if event.button() == Qt.LeftButton:
            self.toggle_lock()
        super().mousePressEvent(event)
    
    def mouseMoveEvent(self, event):
        """Handle mouse movement to update crosshair position"""
        if not self.locked and self.enabled:
            self.mouse_x = event.x()
            self.mouse_y = event.y()
            self._update_data_coordinates()
            self.update()
        super().mouseMoveEvent(event)
    
    def _update_data_coordinates(self):
        """Convert mouse coordinates to data coordinates"""
        if not self.chart_bounds or not self.widget_size:
            return
        
        x_min, x_max, y_min, y_max = self.chart_bounds
        width, height = self.widget_size
        
        # Convert mouse position to normalized coordinates
        norm_x = self.mouse_x / width
        norm_y = 1.0 - (self.mouse_y / height)  # Flip Y axis
        
        # Convert to data coordinates
        self.data_x = x_min + norm_x * (x_max - x_min)
        self.data_y = y_min + norm_y * (y_max - y_min)
        
        # Emit position change signal
        self.position_changed.emit(self.data_x, self.data_y)
    
    def set_position(self, data_x: float, data_y: float):
        """Set crosshair position in data coordinates"""
        if not self.chart_bounds or not self.widget_size:
            return
        
        x_min, x_max, y_min, y_max = self.chart_bounds
        width, height = self.widget_size
        
        # Convert data coordinates to widget coordinates
        norm_x = (data_x - x_min) / (x_max - x_min)
        norm_y = (data_y - y_min) / (y_max - y_min)
        
        self.mouse_x = int(norm_x * width)
        self.mouse_y = int((1.0 - norm_y) * height)
        
        self.data_x = data_x
        self.data_y = data_y
        
        self.update()
        self.position_changed.emit(self.data_x, self.data_y)
    
    def paintEvent(self, event):
        """Paint the crosshair overlay"""
        if not self.enabled or not self.chart_bounds or not self.widget_size:
            return
        
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        
        width, height = self.widget_size
        
        # Choose color based on lock state
        line_color = self.locked_color if self.locked else self.crosshair_color
        
        # Draw crosshair lines
        pen = QPen(line_color, 1, Qt.DashLine if not self.locked else Qt.SolidLine)
        painter.setPen(pen)
        
        # Vertical line
        painter.drawLine(self.mouse_x, 0, self.mouse_x, height)
        
        # Horizontal line
        painter.drawLine(0, self.mouse_y, width, self.mouse_y)
        
        # Draw axis labels
        self._draw_axis_labels(painter)
        
        # Draw center marker
        self._draw_center_marker(painter)
    
    def _draw_axis_labels(self, painter: QPainter):
        """Draw axis value labels"""
        width, height = self.widget_size
        
        # X-axis label (bottom)
        x_text = self._format_x_value(self.data_x)
        x_label_rect = self._draw_axis_label(
            painter, x_text, self.mouse_x, height - 5, 'bottom'
        )
        
        # Y-axis label (left)
        y_text = self._format_y_value(self.data_y)
        y_label_rect = self._draw_axis_label(
            painter, y_text, 5, self.mouse_y, 'left'
        )
    
    def _draw_axis_label(self, painter: QPainter, text: str, x: int, y: int, position: str):
        """Draw a single axis label with background"""
        painter.setFont(self.value_font)
        
        # Calculate text size
        metrics = QFontMetrics(self.value_font)
        text_rect = metrics.boundingRect(text)
        
        # Adjust position based on alignment
        if position == 'bottom':
            # Center horizontally at x, position at y
            label_x = x - text_rect.width() // 2
            label_y = y - text_rect.height() - 2
        elif position == 'left':
            # Position at x, center vertically at y
            label_x = x
            label_y = y - text_rect.height() // 2
        else:
            label_x = x
            label_y = y
        
        # Create padded rectangle
        padding = 4
        bg_rect = text_rect.adjusted(-padding, -padding, padding, padding)
        bg_rect.moveTopLeft(QPoint(label_x, label_y))
        
        # Draw background
        painter.fillRect(bg_rect, QBrush(self.axis_label_bg))
        
        # Draw border
        pen = QPen(self.crosshair_color, 1)
        painter.setPen(pen)
        painter.drawRect(bg_rect)
        
        # Draw text
        painter.setPen(QPen(self.axis_label_text))
        painter.drawText(bg_rect.adjusted(padding, padding, -padding, -padding), Qt.AlignCenter, text)
        
        return bg_rect
    
    def _draw_center_marker(self, painter: QPainter):
        """Draw center marker at crosshair intersection"""
        # Small circle at intersection
        marker_radius = 3
        color = self.locked_color if self.locked else self.crosshair_color
        
        painter.setPen(QPen(color, 2))
        painter.setBrush(QBrush(QColor(color.red(), color.green(), color.blue(), 100)))
        
        painter.drawEllipse(
            self.mouse_x - marker_radius,
            self.mouse_y - marker_radius,
            marker_radius * 2,
            marker_radius * 2
        )
    
    def _format_x_value(self, x_value: float) -> str:
        """Format X axis value for display as time"""
        try:
            # Convert bar index to timestamp if we have datetime data
            bar_idx = int(round(x_value))
            if hasattr(self, 'datetime_data') and self.datetime_data is not None and 0 <= bar_idx < len(self.datetime_data):
                import pandas as pd
                timestamp = pd.to_datetime(self.datetime_data[bar_idx])
                return timestamp.strftime('%H:%M %Y-%m-%d')
            else:
                return f"{bar_idx}"
        except:
            return f"{int(x_value)}"
    
    def _format_y_value(self, y_value: float) -> str:
        """Format Y axis value for display"""
        # Assume Y is price
        return f"{y_value:.5f}"
    
    def resizeEvent(self, event):
        """Handle widget resize"""
        self.set_widget_size(event.size().width(), event.size().height())
        super().resizeEvent(event)


class CrosshairInfoWidget(QWidget):
    """
    Information widget that shows detailed crosshair position data
    Displays precise coordinates, nearest bar data, and statistics
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Data storage
        self.ohlcv_data = None
        self.current_position = None
        self.nearest_bar = None
        
        # UI components
        self.position_labels = {}
        self.bar_labels = {}
        self.stats_labels = {}
        
        self._setup_ui()
        self._setup_styling()
        
        # Initially hidden
        self.hide()
        
        print("SUCCESS: Crosshair info widget initialized")
    
    def _setup_ui(self):
        """Setup the information widget UI"""
        
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(8, 6, 8, 6)
        main_layout.setSpacing(4)
        
        # Set minimum size to expand vertically
        self.setMinimumSize(220, 200)  # Increased height from default
        
        # Title
        title_label = QLabel("CROSSHAIR POSITION")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("""
            QLabel {
                color: white;
                background-color: #505050;
                font-weight: bold;
                font-size: 9pt;
                padding: 3px;
                border: 1px solid #707070;
                border-radius: 2px;
            }
        """)
        main_layout.addWidget(title_label)
        
        # Position information
        pos_frame = QFrame()
        pos_layout = QVBoxLayout(pos_frame)
        pos_layout.setContentsMargins(6, 6, 6, 6)
        pos_layout.setSpacing(4)
        
        # Coordinate labels
        coord_items = [
            ("X (Bar):", "x_coord", "#4CAF50"),
            ("Y (Price):", "y_coord", "#2196F3"),
            ("Status:", "status", "#FF9800")
        ]
        
        for label_text, key, color in coord_items:
            coord_layout = QHBoxLayout()
            coord_layout.setContentsMargins(0, 0, 0, 0)
            
            label = QLabel(label_text)
            label.setStyleSheet(f"color: {color}; font-size: 8pt; font-weight: bold;")
            coord_layout.addWidget(label)
            
            value_label = QLabel("--")
            value_label.setStyleSheet("color: white; font-size: 8pt; font-family: 'Courier New', monospace;")
            coord_layout.addWidget(value_label)
            coord_layout.addStretch()
            
            pos_layout.addLayout(coord_layout)
            self.position_labels[key] = value_label
        
        main_layout.addWidget(pos_frame)
        
        # Bar information
        bar_frame = QFrame()
        bar_layout = QVBoxLayout(bar_frame)
        bar_layout.setContentsMargins(6, 6, 6, 6)
        bar_layout.setSpacing(4)
        
        bar_title = QLabel("NEAREST BAR DATA")
        bar_title.setAlignment(Qt.AlignCenter)
        bar_title.setStyleSheet("color: #cccccc; font-size: 8pt; font-weight: bold;")
        bar_layout.addWidget(bar_title)
        
        # OHLC data
        ohlc_items = [
            ("Open:", "open", "#888888"),
            ("High:", "high", "#4CAF50"),
            ("Low:", "low", "#F44336"),
            ("Close:", "close", "#2196F3")
        ]
        
        for label_text, key, color in ohlc_items:
            ohlc_layout = QHBoxLayout()
            ohlc_layout.setContentsMargins(0, 0, 0, 0)
            
            label = QLabel(label_text)
            label.setStyleSheet(f"color: {color}; font-size: 8pt; font-weight: bold;")
            ohlc_layout.addWidget(label)
            
            value_label = QLabel("--")
            value_label.setStyleSheet("color: white; font-size: 8pt; font-family: 'Courier New', monospace;")
            ohlc_layout.addWidget(value_label)
            ohlc_layout.addStretch()
            
            bar_layout.addLayout(ohlc_layout)
            self.bar_labels[key] = value_label
        
        main_layout.addWidget(bar_frame)
        
        # Statistics
        stats_frame = QFrame()
        stats_layout = QVBoxLayout(stats_frame)
        stats_layout.setContentsMargins(3, 3, 3, 3)
        stats_layout.setSpacing(2)
        
        stats_title = QLabel("STATISTICS")
        stats_title.setAlignment(Qt.AlignCenter)
        stats_title.setStyleSheet("color: #cccccc; font-size: 8pt; font-weight: bold;")
        stats_layout.addWidget(stats_title)
        
        # Distance from current price
        distance_layout = QHBoxLayout()
        distance_layout.setContentsMargins(0, 0, 0, 0)
        
        distance_label = QLabel("Distance:")
        distance_label.setStyleSheet("color: #9C27B0; font-size: 8pt; font-weight: bold;")
        distance_layout.addWidget(distance_label)
        
        distance_value = QLabel("--")
        distance_value.setStyleSheet("color: white; font-size: 8pt; font-family: 'Courier New', monospace;")
        distance_layout.addWidget(distance_value)
        distance_layout.addStretch()
        
        stats_layout.addLayout(distance_layout)
        self.stats_labels['distance'] = distance_value
        
        main_layout.addWidget(stats_frame)
        
        # Instructions
        instructions = QLabel("Click to lock crosshair\nMove mouse to update")
        instructions.setAlignment(Qt.AlignCenter)
        instructions.setStyleSheet("color: #666666; font-size: 8pt; font-style: italic;")
        main_layout.addWidget(instructions)
        
        print("SUCCESS: Crosshair info UI created")
    
    def _setup_styling(self):
        """Apply styling to the widget"""
        self.setStyleSheet("""
            QWidget {
                background-color: rgba(40, 40, 40, 240);
                border: 2px solid #707070;
                border-radius: 6px;
                color: white;
            }
            QFrame {
                background-color: transparent;
                border: 1px solid #555555;
                border-radius: 3px;
                margin: 1px;
            }
        """)
        
        # Set fixed size
        self.setFixedSize(200, 280)
    
    def load_ohlcv_data(self, ohlcv_data: Dict[str, np.ndarray]):
        """Load OHLCV data for bar information display"""
        try:
            self.ohlcv_data = {
                'open': np.asarray(ohlcv_data['open'], dtype=np.float64),
                'high': np.asarray(ohlcv_data['high'], dtype=np.float64),
                'low': np.asarray(ohlcv_data['low'], dtype=np.float64),
                'close': np.asarray(ohlcv_data['close'], dtype=np.float64)
            }
            
            # Store datetime data for time formatting
            if 'datetime' in ohlcv_data:
                self.datetime_data = ohlcv_data['datetime']
            elif 'datetime_ns' in ohlcv_data:
                self.datetime_data = ohlcv_data['datetime_ns']
            else:
                self.datetime_data = None
                
            print(f"Crosshair info loaded OHLCV data: {len(self.ohlcv_data['close'])} bars")
            return True
            
        except Exception as e:
            print(f"ERROR: Failed to load OHLCV data for crosshair info: {e}")
            return False
    
    def update_position(self, x_value: float, y_value: float):
        """Update crosshair position information"""
        try:
            self.current_position = (x_value, y_value)
            
            # Update coordinate labels with time formatting for X
            x_formatted = self._format_x_coordinate(x_value)
            self.position_labels['x_coord'].setText(x_formatted)
            self.position_labels['y_coord'].setText(f"{y_value:.5f}")
            
            # Find nearest bar
            if self.ohlcv_data is not None:
                bar_index = int(round(x_value))
                if 0 <= bar_index < len(self.ohlcv_data['close']):
                    self.nearest_bar = bar_index
                    self._update_bar_information(bar_index)
                    self._update_statistics(bar_index, y_value)
                    self.position_labels['status'].setText("Valid")
                    self.position_labels['status'].setStyleSheet("color: #4CAF50; font-size: 8pt; font-family: 'Courier New', monospace;")
                else:
                    self.nearest_bar = None
                    self._clear_bar_information()
                    self.position_labels['status'].setText("Out of range")
                    self.position_labels['status'].setStyleSheet("color: #F44336; font-size: 8pt; font-family: 'Courier New', monospace;")
            else:
                self.position_labels['status'].setText("No data")
                self.position_labels['status'].setStyleSheet("color: #FF9800; font-size: 8pt; font-family: 'Courier New', monospace;")
            
        except Exception as e:
            print(f"ERROR: Failed to update crosshair position: {e}")
    
    def _format_x_coordinate(self, x_value: float) -> str:
        """Format X coordinate as time if datetime data is available"""
        try:
            bar_idx = int(round(x_value))
            if hasattr(self, 'datetime_data') and self.datetime_data is not None and 0 <= bar_idx < len(self.datetime_data):
                import pandas as pd
                timestamp = pd.to_datetime(self.datetime_data[bar_idx])
                return timestamp.strftime('%H:%M %Y-%m-%d')
            else:
                return f"{x_value:.2f}"
        except:
            return f"{x_value:.2f}"
    
    def _update_bar_information(self, bar_index: int):
        """Update nearest bar OHLC information"""
        try:
            # Update OHLC values
            ohlc_values = {
                'open': self.ohlcv_data['open'][bar_index],
                'high': self.ohlcv_data['high'][bar_index],
                'low': self.ohlcv_data['low'][bar_index],
                'close': self.ohlcv_data['close'][bar_index]
            }
            
            for key, value in ohlc_values.items():
                self.bar_labels[key].setText(f"{value:.5f}")
            
            # Color coding for close price
            if bar_index > 0:
                prev_close = self.ohlcv_data['close'][bar_index - 1]
                current_close = ohlc_values['close']
                
                if current_close > prev_close:
                    close_color = "#4CAF50"  # Green
                elif current_close < prev_close:
                    close_color = "#F44336"  # Red
                else:
                    close_color = "white"    # No change
                
                self.bar_labels['close'].setStyleSheet(f"color: {close_color}; font-size: 8pt; font-family: 'Courier New', monospace; font-weight: bold;")
            
        except Exception as e:
            print(f"ERROR: Failed to update bar information: {e}")
    
    def _update_statistics(self, bar_index: int, y_value: float):
        """Update statistics for current position"""
        try:
            if self.ohlcv_data is not None and bar_index < len(self.ohlcv_data['close']):
                # Calculate distance from close price
                close_price = self.ohlcv_data['close'][bar_index]
                distance = y_value - close_price
                distance_pct = (distance / close_price) * 100 if close_price != 0 else 0
                
                distance_text = f"{distance:+.5f} ({distance_pct:+.2f}%)"
                self.stats_labels['distance'].setText(distance_text)
                
                # Color coding
                if distance > 0:
                    color = "#4CAF50"  # Above close - green
                elif distance < 0:
                    color = "#F44336"  # Below close - red
                else:
                    color = "white"     # At close - white
                
                self.stats_labels['distance'].setStyleSheet(f"color: {color}; font-size: 8pt; font-family: 'Courier New', monospace;")
            
        except Exception as e:
            print(f"ERROR: Failed to update statistics: {e}")
    
    def _clear_bar_information(self):
        """Clear bar information when out of range"""
        for label in self.bar_labels.values():
            label.setText("--")
            label.setStyleSheet("color: #666666; font-size: 8pt; font-family: 'Courier New', monospace;")
        
        for label in self.stats_labels.values():
            label.setText("--")
            label.setStyleSheet("color: #666666; font-size: 8pt; font-family: 'Courier New', monospace;")
    
    def show_at_position(self, global_pos: QPoint):
        """Show info widget at top-left of chart area"""
        try:
            # Position widget at absolute top-left of chart area
            if self.parent():
                # Get the parent widget's (chart area) geometry
                parent_rect = self.parent().geometry()
                parent_global_pos = self.parent().mapToGlobal(parent_rect.topLeft())
                
                # Position at top-left corner of chart with small margin
                x = parent_global_pos.x() + 10
                y = parent_global_pos.y() + 10
                
                # Keep within screen bounds
                screen = self.screen()
                if screen:
                    screen_rect = screen.geometry()
                    widget_width = self.width()
                    widget_height = self.height()
                    
                    if x + widget_width > screen_rect.right():
                        x = screen_rect.right() - widget_width - 10
                    if y + widget_height > screen_rect.bottom():
                        y = screen_rect.bottom() - widget_height - 10
                
                self.move(x, y)
            else:
                # Fallback positioning - use global position with offset
                self.move(global_pos.x() + 20, global_pos.y() + 20)
            
            # Show widget
            self.show()
            self.raise_()
            
        except Exception as e:
            print(f"ERROR: Failed to show crosshair info: {e}")
    
    def hide_widget(self):
        """Hide the info widget"""
        self.hide()


def test_crosshair_widgets():
    """Test crosshair overlay and info widgets"""
    print("TESTING CROSSHAIR WIDGETS - STEP 5")
    print("="*50)
    
    from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QPushButton, QHBoxLayout
    import sys
    
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    
    try:
        # Create test data
        np.random.seed(42)
        num_bars = 500
        base_price = 1.2000
        volatility = 0.001
        
        price_changes = np.random.normal(0, volatility, num_bars)
        prices = np.cumsum(price_changes) + base_price
        
        ohlcv_data = {
            'open': prices.copy(),
            'high': prices + np.random.exponential(volatility/4, num_bars),
            'low': prices - np.random.exponential(volatility/4, num_bars),
            'close': prices + np.random.normal(0, volatility/2, num_bars)
        }
        
        # Ensure proper OHLC relationships
        ohlcv_data['high'] = np.maximum(ohlcv_data['high'], ohlcv_data['close'])
        ohlcv_data['low'] = np.minimum(ohlcv_data['low'], ohlcv_data['close'])
        
        # Create main window
        main_window = QMainWindow()
        central_widget = QWidget()
        main_window.setCentralWidget(central_widget)
        
        layout = QVBoxLayout(central_widget)
        
        # Create chart area (simulated)
        chart_area = QWidget()
        chart_area.setMinimumSize(800, 400)
        chart_area.setStyleSheet("background-color: #1e1e1e; border: 1px solid #444444;")
        
        # Add crosshair overlay
        crosshair = CrosshairOverlay(chart_area)
        crosshair.setGeometry(0, 0, 800, 400)
        
        # Set chart bounds (simulate viewport)
        crosshair.set_chart_bounds(0, 499, ohlcv_data['low'].min(), ohlcv_data['high'].max())
        crosshair.set_widget_size(800, 400)
        
        layout.addWidget(chart_area)
        
        # Create crosshair info widget
        crosshair_info = CrosshairInfoWidget(main_window)
        crosshair_info.load_ohlcv_data(ohlcv_data)
        
        # Connect crosshair to info widget
        def on_position_changed(x_val, y_val):
            crosshair_info.update_position(x_val, y_val)
            # Position info widget
            global_pos = main_window.mapToGlobal(main_window.rect().center())
            crosshair_info.show_at_position(global_pos)
        
        crosshair.position_changed.connect(on_position_changed)
        
        # Control buttons
        controls_layout = QHBoxLayout()
        
        toggle_btn = QPushButton("Toggle Crosshair")
        toggle_btn.clicked.connect(lambda: crosshair.enable_crosshair(not crosshair.enabled))
        controls_layout.addWidget(toggle_btn)
        
        lock_btn = QPushButton("Toggle Lock")
        lock_btn.clicked.connect(crosshair.toggle_lock)
        controls_layout.addWidget(lock_btn)
        
        test_btn = QPushButton("Test Position")
        def test_position():
            test_x = np.random.randint(50, 450)
            test_y = np.random.uniform(ohlcv_data['low'].min(), ohlcv_data['high'].max())
            crosshair.set_position(test_x, test_y)
            crosshair.lock_crosshair(True)
        
        test_btn.clicked.connect(test_position)
        controls_layout.addWidget(test_btn)
        
        hide_btn = QPushButton("Hide Info")
        hide_btn.clicked.connect(crosshair_info.hide_widget)
        controls_layout.addWidget(hide_btn)
        
        layout.addLayout(controls_layout)
        
        # Instructions
        instructions = QLabel("""
        CROSSHAIR WIDGET TEST - STEP 5
        
        Features:
        • Move mouse over chart area for crosshair
        • Click to lock/unlock crosshair position
        • Info widget shows precise coordinates
        • Nearest bar OHLC data display
        • Distance from close price statistics
        • Color-coded price changes
        
        Controls:
        • Toggle Crosshair - Enable/disable display
        • Toggle Lock - Lock/unlock current position
        • Test Position - Set random locked position
        • Hide Info - Hide information widget
        
        Close window to complete test.
        """)
        instructions.setStyleSheet("color: white; background-color: #333; padding: 10px; border: 1px solid #666; font-size: 9pt;")
        layout.addWidget(instructions)
        
        # Style main window
        main_window.setStyleSheet("""
            QMainWindow {
                background-color: #2b2b2b;
            }
            QPushButton {
                background-color: #404040;
                color: white;
                border: 1px solid #606060;
                padding: 8px 16px;
                font-weight: bold;
                border-radius: 3px;
            }
            QPushButton:hover {
                background-color: #505050;
            }
            QPushButton:pressed {
                background-color: #606060;
            }
        """)
        
        # Show test window
        main_window.setWindowTitle("Crosshair Widgets Test - Step 5")
        main_window.resize(1000, 800)
        main_window.show()
        
        print("\nStep 5 Test Features:")
        print("• Interactive crosshair with mouse tracking")
        print("• Click-to-lock functionality") 
        print("• Precise coordinate display")
        print("• Nearest bar OHLC information")
        print("• Distance and percentage calculations")
        print("• Professional overlay styling")
        print("• Real-time position updates")
        
        print("\nSUCCESS: Crosshair widgets test ready")
        return True
        
    except Exception as e:
        print(f"ERROR: Crosshair widgets test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_crosshair_widgets()
    
    if success:
        print("\n" + "="*50)
        print("CROSSHAIR WIDGETS - STEP 5 SUCCESS!")
        print("="*50)
        print("+ Interactive crosshair overlay with mouse tracking")
        print("+ Click-to-lock crosshair functionality")
        print("+ Precise axis data points display")
        print("+ Nearest bar OHLC information widget")
        print("+ Distance and percentage statistics")
        print("+ Professional styling with color coding")
        print("+ Real-time coordinate conversion")
        print("\nREADY FOR INTEGRATION INTO STEP 5 DASHBOARD")
    else:
        print("\nCrosshair widgets need additional work")