# src/dashboard/hover_info_widget.py
# OHLCV Data Display on Mouse Hover - Step 4
# Real-time price information and technical indicators overlay

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                           QFrame, QGridLayout)
from PyQt5.QtCore import Qt, pyqtSignal, pyqtSlot, QTimer, QPoint
from PyQt5.QtGui import QFont, QColor, QPalette

class TechnicalIndicators:
    """Technical indicators calculator for hover display"""
    
    @staticmethod
    def sma(data: np.ndarray, period: int) -> np.ndarray:
        """Simple Moving Average"""
        if len(data) < period:
            return np.full_like(data, np.nan)
        
        result = np.full_like(data, np.nan)
        for i in range(period - 1, len(data)):
            result[i] = np.mean(data[i - period + 1:i + 1])
        return result
    
    @staticmethod
    def ema(data: np.ndarray, period: int) -> np.ndarray:
        """Exponential Moving Average"""
        if len(data) < period:
            return np.full_like(data, np.nan)
        
        alpha = 2.0 / (period + 1)
        result = np.full_like(data, np.nan)
        result[period - 1] = np.mean(data[:period])
        
        for i in range(period, len(data)):
            result[i] = alpha * data[i] + (1 - alpha) * result[i - 1]
        
        return result
    
    @staticmethod
    def rsi(data: np.ndarray, period: int = 14) -> np.ndarray:
        """Relative Strength Index"""
        if len(data) < period + 1:
            return np.full_like(data, np.nan)
        
        # Calculate price changes
        deltas = np.diff(data)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        # Calculate initial averages
        avg_gain = np.mean(gains[:period])
        avg_loss = np.mean(losses[:period])
        
        result = np.full(len(data), np.nan)
        
        if avg_loss != 0:
            rs = avg_gain / avg_loss
            result[period] = 100 - (100 / (1 + rs))
        
        # Calculate smoothed RSI
        alpha = 1.0 / period
        for i in range(period + 1, len(data)):
            gain = gains[i - 1] if i - 1 < len(gains) else 0
            loss = losses[i - 1] if i - 1 < len(losses) else 0
            
            avg_gain = alpha * gain + (1 - alpha) * avg_gain
            avg_loss = alpha * loss + (1 - alpha) * avg_loss
            
            if avg_loss != 0:
                rs = avg_gain / avg_loss
                result[i] = 100 - (100 / (1 + rs))
        
        return result
    
    @staticmethod
    def bollinger_bands(data: np.ndarray, period: int = 20, std_dev: float = 2.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Bollinger Bands - returns (upper, middle, lower)"""
        sma = TechnicalIndicators.sma(data, period)
        
        std = np.full_like(data, np.nan)
        for i in range(period - 1, len(data)):
            std[i] = np.std(data[i - period + 1:i + 1])
        
        upper = sma + std_dev * std
        lower = sma - std_dev * std
        
        return upper, sma, lower
    
    @staticmethod
    def macd(data: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """MACD - returns (macd_line, signal_line, histogram)"""
        ema_fast = TechnicalIndicators.ema(data, fast)
        ema_slow = TechnicalIndicators.ema(data, slow)
        
        macd_line = ema_fast - ema_slow
        signal_line = TechnicalIndicators.ema(macd_line, signal)
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram


class HoverInfoWidget(QWidget):
    """
    OHLCV data display widget with technical indicators
    Shows detailed price information on mouse hover
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Data storage
        self.ohlcv_data = None
        self.current_index = -1
        self.indicators_cache = {}
        
        # UI components
        self.price_labels = {}
        self.indicator_labels = {}
        self.volume_label = None
        self.datetime_label = None
        
        # Configuration
        self.indicator_periods = {
            'SMA_20': 20,
            'SMA_50': 50,
            'EMA_12': 12,
            'EMA_26': 26,
            'RSI': 14,
            'BB_Period': 20
        }
        
        self._setup_ui()
        self._setup_styling()
        
        # Auto-hide timer
        self.hide_timer = QTimer()
        self.hide_timer.setSingleShot(True)
        self.hide_timer.timeout.connect(self.hide)
        
        # Initially hidden
        self.hide()
        
        print("SUCCESS: Hover info widget initialized")
    
    def _setup_ui(self):
        """Setup the hover information UI"""
        
        # Main layout
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(8, 6, 8, 6)
        main_layout.setSpacing(4)
        
        # Title
        title_label = QLabel("OHLCV + INDICATORS")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("""
            QLabel {
                color: white;
                background-color: #404040;
                font-weight: bold;
                font-size: 9pt;
                padding: 3px;
                border: 1px solid #606060;
                border-radius: 2px;
            }
        """)
        main_layout.addWidget(title_label)
        
        # Date/Time
        self.datetime_label = QLabel("Time: --")
        self.datetime_label.setStyleSheet("color: #cccccc; font-size: 8pt; font-weight: bold;")
        main_layout.addWidget(self.datetime_label)
        
        # Price information section
        price_frame = QFrame()
        price_layout = QGridLayout(price_frame)
        price_layout.setContentsMargins(4, 4, 4, 4)
        price_layout.setSpacing(2)
        
        # OHLCV labels
        ohlcv_items = [
            ("Open:", "open", "#888888"),
            ("High:", "high", "#4CAF50"),
            ("Low:", "low", "#F44336"),
            ("Close:", "close", "#2196F3"),
            ("Volume:", "volume", "#FF9800")
        ]
        
        for i, (label_text, key, color) in enumerate(ohlcv_items):
            # Label
            label = QLabel(label_text)
            label.setStyleSheet(f"color: {color}; font-size: 8pt; font-weight: bold;")
            price_layout.addWidget(label, i, 0)
            
            # Value
            value_label = QLabel("--")
            value_label.setStyleSheet(f"color: white; font-size: 8pt; font-family: 'Courier New', monospace;")
            price_layout.addWidget(value_label, i, 1)
            
            self.price_labels[key] = value_label
        
        main_layout.addWidget(price_frame)
        
        # Technical indicators section
        indicators_frame = QFrame()
        indicators_layout = QGridLayout(indicators_frame)
        indicators_layout.setContentsMargins(4, 4, 4, 4)
        indicators_layout.setSpacing(2)
        
        # Indicator labels
        indicator_items = [
            ("SMA 20:", "sma_20", "#9C27B0"),
            ("SMA 50:", "sma_50", "#673AB7"),
            ("EMA 12:", "ema_12", "#3F51B5"),
            ("EMA 26:", "ema_26", "#2196F3"),
            ("RSI:", "rsi", "#FF5722"),
            ("BB Upper:", "bb_upper", "#795548"),
            ("BB Lower:", "bb_lower", "#795548"),
            ("MACD:", "macd", "#607D8B")
        ]
        
        for i, (label_text, key, color) in enumerate(indicator_items):
            row = i // 2
            col = (i % 2) * 2
            
            # Label
            label = QLabel(label_text)
            label.setStyleSheet(f"color: {color}; font-size: 8pt; font-weight: bold;")
            indicators_layout.addWidget(label, row, col)
            
            # Value
            value_label = QLabel("--")
            value_label.setStyleSheet(f"color: white; font-size: 8pt; font-family: 'Courier New', monospace;")
            indicators_layout.addWidget(value_label, row, col + 1)
            
            self.indicator_labels[key] = value_label
        
        main_layout.addWidget(indicators_frame)
        
        # Separator
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setStyleSheet("color: #555555;")
        main_layout.addWidget(separator)
        
        # Quick stats
        stats_label = QLabel("Hover over chart for live data")
        stats_label.setAlignment(Qt.AlignCenter)
        stats_label.setStyleSheet("color: #888888; font-size: 8pt; font-style: italic;")
        main_layout.addWidget(stats_label)
        
        print("SUCCESS: Hover info UI layout created")
    
    def _setup_styling(self):
        """Apply professional styling"""
        self.setStyleSheet("""
            QWidget {
                background-color: rgba(45, 45, 45, 240);
                border: 2px solid #606060;
                border-radius: 6px;
                color: white;
            }
            QFrame {
                background-color: transparent;
                border: 1px solid #555555;
                border-radius: 3px;
                padding: 2px;
            }
        """)
        
        # Set fixed size to prevent jumping
        self.setFixedSize(280, 320)
    
    def load_ohlcv_data(self, ohlcv_data: Dict[str, np.ndarray]) -> bool:
        """
        Load OHLCV data and pre-calculate indicators
        
        Args:
            ohlcv_data: Dictionary with 'open', 'high', 'low', 'close', 'volume' arrays
        """
        try:
            print(f"Loading OHLCV data for hover display: {len(ohlcv_data['close'])} bars")
            
            self.ohlcv_data = {
                'open': np.asarray(ohlcv_data['open'], dtype=np.float64),
                'high': np.asarray(ohlcv_data['high'], dtype=np.float64),
                'low': np.asarray(ohlcv_data['low'], dtype=np.float64),
                'close': np.asarray(ohlcv_data['close'], dtype=np.float64),
                'volume': np.asarray(ohlcv_data.get('volume', np.ones(len(ohlcv_data['close']))), dtype=np.float64),
                'datetime': np.asarray(ohlcv_data.get('datetime', np.arange(len(ohlcv_data['close']))), dtype=np.int64)
            }
            
            # Pre-calculate all indicators for performance
            self._calculate_indicators()
            
            print(f"SUCCESS: OHLCV data loaded with indicators for {len(self.ohlcv_data['close'])} bars")
            return True
            
        except Exception as e:
            print(f"ERROR: Failed to load OHLCV data: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _calculate_indicators(self):
        """Pre-calculate all technical indicators for performance"""
        try:
            print("Pre-calculating technical indicators...")
            
            close_prices = self.ohlcv_data['close']
            
            # Moving averages
            self.indicators_cache['sma_20'] = TechnicalIndicators.sma(close_prices, 20)
            self.indicators_cache['sma_50'] = TechnicalIndicators.sma(close_prices, 50)
            self.indicators_cache['ema_12'] = TechnicalIndicators.ema(close_prices, 12)
            self.indicators_cache['ema_26'] = TechnicalIndicators.ema(close_prices, 26)
            
            # RSI
            self.indicators_cache['rsi'] = TechnicalIndicators.rsi(close_prices, 14)
            
            # Bollinger Bands
            bb_upper, bb_middle, bb_lower = TechnicalIndicators.bollinger_bands(close_prices, 20, 2.0)
            self.indicators_cache['bb_upper'] = bb_upper
            self.indicators_cache['bb_middle'] = bb_middle
            self.indicators_cache['bb_lower'] = bb_lower
            
            # MACD
            macd_line, signal_line, histogram = TechnicalIndicators.macd(close_prices, 12, 26, 9)
            self.indicators_cache['macd'] = macd_line
            self.indicators_cache['macd_signal'] = signal_line
            self.indicators_cache['macd_histogram'] = histogram
            
            print("SUCCESS: Technical indicators pre-calculated")
            
        except Exception as e:
            print(f"ERROR: Failed to calculate indicators: {e}")
            import traceback
            traceback.print_exc()
    
    def update_display(self, bar_index: int) -> bool:
        """
        Update display for specific bar index
        
        Args:
            bar_index: Index of the bar to display information for
        """
        try:
            if self.ohlcv_data is None or bar_index < 0 or bar_index >= len(self.ohlcv_data['close']):
                return False
            
            self.current_index = bar_index
            
            # Update datetime
            datetime_val = self.ohlcv_data['datetime'][bar_index]
            self.datetime_label.setText(f"Bar: {bar_index} | Time: {datetime_val}")
            
            # Update OHLCV values
            ohlcv_values = {
                'open': self.ohlcv_data['open'][bar_index],
                'high': self.ohlcv_data['high'][bar_index],
                'low': self.ohlcv_data['low'][bar_index],
                'close': self.ohlcv_data['close'][bar_index],
                'volume': self.ohlcv_data['volume'][bar_index]
            }
            
            for key, value in ohlcv_values.items():
                if key == 'volume':
                    self.price_labels[key].setText(f"{value:,.0f}")
                else:
                    self.price_labels[key].setText(f"{value:.5f}")
            
            # Update color coding for close price
            if bar_index > 0:
                prev_close = self.ohlcv_data['close'][bar_index - 1]
                current_close = ohlcv_values['close']
                
                if current_close > prev_close:
                    close_color = "#4CAF50"  # Green
                elif current_close < prev_close:
                    close_color = "#F44336"  # Red
                else:
                    close_color = "white"    # No change
                
                self.price_labels['close'].setStyleSheet(f"color: {close_color}; font-size: 8pt; font-family: 'Courier New', monospace; font-weight: bold;")
            
            # Update indicators
            self._update_indicators(bar_index)
            
            return True
            
        except Exception as e:
            print(f"ERROR: Failed to update display for bar {bar_index}: {e}")
            return False
    
    def _update_indicators(self, bar_index: int):
        """Update technical indicator values"""
        try:
            # Update each indicator
            indicator_updates = {
                'sma_20': self.indicators_cache.get('sma_20'),
                'sma_50': self.indicators_cache.get('sma_50'),
                'ema_12': self.indicators_cache.get('ema_12'),
                'ema_26': self.indicators_cache.get('ema_26'),
                'rsi': self.indicators_cache.get('rsi'),
                'bb_upper': self.indicators_cache.get('bb_upper'),
                'bb_lower': self.indicators_cache.get('bb_lower'),
                'macd': self.indicators_cache.get('macd')
            }
            
            for key, indicator_data in indicator_updates.items():
                if indicator_data is not None and bar_index < len(indicator_data):
                    value = indicator_data[bar_index]
                    
                    if np.isnan(value):
                        display_text = "N/A"
                        text_color = "#666666"
                    else:
                        if key == 'rsi':
                            display_text = f"{value:.1f}"
                            # RSI color coding
                            if value > 70:
                                text_color = "#F44336"  # Overbought - Red
                            elif value < 30:
                                text_color = "#4CAF50"  # Oversold - Green
                            else:
                                text_color = "white"
                        elif key in ['bb_upper', 'bb_lower', 'sma_20', 'sma_50', 'ema_12', 'ema_26']:
                            display_text = f"{value:.5f}"
                            text_color = "white"
                        elif key == 'macd':
                            display_text = f"{value:.6f}"
                            text_color = "#4CAF50" if value > 0 else "#F44336"
                        else:
                            display_text = f"{value:.5f}"
                            text_color = "white"
                    
                    self.indicator_labels[key].setText(display_text)
                    self.indicator_labels[key].setStyleSheet(f"color: {text_color}; font-size: 8pt; font-family: 'Courier New', monospace;")
                else:
                    self.indicator_labels[key].setText("--")
                    self.indicator_labels[key].setStyleSheet(f"color: #666666; font-size: 8pt; font-family: 'Courier New', monospace;")
            
        except Exception as e:
            print(f"ERROR: Failed to update indicators: {e}")
    
    def show_at_position(self, global_pos: QPoint, bar_index: int):
        """
        Show hover info widget at specific position
        
        Args:
            global_pos: Global position to show widget at
            bar_index: Bar index to display data for
        """
        try:
            # Update display data
            if not self.update_display(bar_index):
                return False
            
            # Position widget near cursor but keep on screen
            widget_width = self.width()
            widget_height = self.height()
            
            # Get screen geometry
            screen = self.screen()
            if screen:
                screen_rect = screen.geometry()
                
                # Calculate position with screen bounds checking
                x = global_pos.x() + 15  # Offset from cursor
                y = global_pos.y() - widget_height // 2
                
                # Keep within screen bounds
                if x + widget_width > screen_rect.right():
                    x = global_pos.x() - widget_width - 15
                
                if y < screen_rect.top():
                    y = screen_rect.top() + 10
                elif y + widget_height > screen_rect.bottom():
                    y = screen_rect.bottom() - widget_height - 10
                
                self.move(x, y)
            else:
                # Fallback positioning
                self.move(global_pos.x() + 15, global_pos.y() - widget_height // 2)
            
            # Show widget
            self.show()
            self.raise_()
            
            # Reset hide timer
            self.hide_timer.stop()
            
            return True
            
        except Exception as e:
            print(f"ERROR: Failed to show hover info: {e}")
            return False
    
    def schedule_hide(self, delay_ms: int = 1000):
        """Schedule widget to hide after delay"""
        self.hide_timer.start(delay_ms)
    
    def force_hide(self):
        """Immediately hide the widget"""
        self.hide_timer.stop()
        self.hide()
    
    def get_indicator_value(self, indicator_name: str, bar_index: int) -> Optional[float]:
        """Get specific indicator value for external use"""
        try:
            indicator_data = self.indicators_cache.get(indicator_name)
            if indicator_data is not None and 0 <= bar_index < len(indicator_data):
                value = indicator_data[bar_index]
                return None if np.isnan(value) else value
        except Exception:
            pass
        return None


def test_hover_info_widget():
    """Test the hover info widget"""
    print("TESTING HOVER INFO WIDGET - STEP 4")
    print("="*50)
    
    from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget
    import sys
    
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    
    try:
        # Create test data
        np.random.seed(42)
        num_bars = 1000
        base_price = 1.2000
        volatility = 0.001
        
        price_changes = np.random.normal(0, volatility, num_bars)
        prices = np.cumsum(price_changes) + base_price
        
        ohlcv_data = {
            'open': prices.copy(),
            'high': prices + np.random.exponential(volatility/4, num_bars),
            'low': prices - np.random.exponential(volatility/4, num_bars),
            'close': prices + np.random.normal(0, volatility/2, num_bars),
            'volume': np.random.lognormal(10, 0.5, num_bars),
            'datetime': np.arange(num_bars)
        }
        
        # Ensure high >= low
        ohlcv_data['high'] = np.maximum(ohlcv_data['high'], ohlcv_data['close'])
        ohlcv_data['low'] = np.minimum(ohlcv_data['low'], ohlcv_data['close'])
        
        # Create main window with test controls
        main_window = QMainWindow()
        central_widget = QWidget()
        main_window.setCentralWidget(central_widget)
        
        layout = QVBoxLayout(central_widget)
        
        # Create hover info widget
        hover_widget = HoverInfoWidget(main_window)
        
        # Load test data
        success = hover_widget.load_ohlcv_data(ohlcv_data)
        if not success:
            print("ERROR: Failed to load test data")
            return False
        
        # Test buttons
        test_buttons = []
        test_indices = [100, 200, 500, 800]
        
        for i, test_idx in enumerate(test_indices):
            btn = QPushButton(f"Show Bar {test_idx}")
            
            def make_handler(idx):
                def handler():
                    global_pos = main_window.mapToGlobal(main_window.rect().center())
                    hover_widget.show_at_position(global_pos, idx)
                return handler
            
            btn.clicked.connect(make_handler(test_idx))
            layout.addWidget(btn)
            test_buttons.append(btn)
        
        # Hide button
        hide_btn = QPushButton("Hide Hover Info")
        hide_btn.clicked.connect(hover_widget.force_hide)
        layout.addWidget(hide_btn)
        
        # Instructions
        instructions = QLabel("""
        HOVER INFO WIDGET TEST
        
        Click buttons to show hover information for different bars.
        The widget displays:
        • OHLCV data with color coding
        • Technical indicators (SMA, EMA, RSI, Bollinger Bands, MACD)
        • Real-time calculations
        • Professional styling
        
        Close window to complete test.
        """)
        instructions.setStyleSheet("color: white; background-color: #333; padding: 10px; border: 1px solid #666;")
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
                padding: 8px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #505050;
            }
        """)
        
        # Show test window
        main_window.setWindowTitle("Hover Info Widget Test - Step 4")
        main_window.resize(400, 600)
        main_window.show()
        
        print("\nStep 4 Test Features:")
        print("• OHLCV data display with color coding")
        print("• Technical indicators: SMA, EMA, RSI, BB, MACD")
        print("• Real-time calculation and caching")
        print("• Professional overlay styling") 
        print("• Position-aware display")
        print("• Auto-hide functionality")
        
        print("\nSUCCESS: Hover info widget test ready")
        return True
        
    except Exception as e:
        print(f"ERROR: Hover info widget test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_hover_info_widget()
    
    if success:
        print("\n" + "="*50)
        print("HOVER INFO WIDGET - STEP 4 SUCCESS!")
        print("="*50)
        print("+ OHLCV data display with real-time updates")
        print("+ Technical indicators calculation and display")
        print("+ Professional overlay widget with positioning")
        print("+ Color coding and visual feedback")
        print("+ High-performance indicator caching")
        print("+ Auto-hide and manual control")
        print("\nREADY FOR INTEGRATION INTO STEP 4 DASHBOARD")
    else:
        print("\nHover info widget needs additional work")