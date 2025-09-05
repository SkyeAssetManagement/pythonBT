"""
Step 2: Fast ES dashboard with real 1m data and proper thin candlesticks
Focus: Load ES data quickly and display with proper thin candlesticks (fix blob issue)
"""

import sys
import time
import numpy as np
import pandas as pd
import pyqtgraph as pg
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt
from pathlib import Path

# Configure PyQtGraph for maximum performance
pg.setConfigOptions(
    antialias=False,
    useOpenGL=True,
    imageAxisOrder='row-major',
    leftButtonPan=True,
    foreground='k',
    background='w'
)

class FastCandlestickItem(pg.GraphicsObject):
    """Ultra-fast candlestick rendering with proper thin shapes"""
    
    def __init__(self, ohlc_data, visible_range=None):
        super().__init__()
        self.full_data = ohlc_data
        self.visible_range = visible_range or (0, len(ohlc_data))
        self.picture = None
        self._generate_picture()
    
    def set_visible_range(self, start_idx, end_idx):
        """Update visible range for efficient rendering"""
        self.visible_range = (max(0, start_idx), min(len(self.full_data), end_idx))
        self._generate_picture()
        self.update()
    
    def _generate_picture(self):
        """Generate candlesticks for visible range only"""
        start_time = time.time()
        
        start_idx, end_idx = self.visible_range
        visible_data = self.full_data[start_idx:end_idx]
        
        if len(visible_data) == 0:
            return
        
        self.picture = QtGui.QPicture()
        painter = QtGui.QPainter(self.picture)
        painter.setRenderHint(QtGui.QPainter.Antialiasing, False)
        
        # CRITICAL: Calculate appropriate candle width based on visible bars
        visible_bars = len(visible_data)
        if visible_bars <= 50:
            candle_width = 0.8    # Wide for zoomed view
        elif visible_bars <= 200:
            candle_width = 0.6    # Medium for normal view  
        else:
            candle_width = 0.4    # Narrow for overview
        
        # PERFORMANCE: Pre-calculate all coordinates
        coordinates = self._calculate_coordinates(visible_data, start_idx, candle_width)
        
        # PERFORMANCE: Batch draw by color
        self._batch_draw_candlesticks(painter, coordinates)
        
        painter.end()
        
        render_time = time.time() - start_time
        print(f"   Rendered {len(visible_data)} candles in {render_time:.4f}s (width: {candle_width})")
    
    def _calculate_coordinates(self, data, start_idx, width):
        """Pre-calculate all drawing coordinates for performance"""
        n = len(data)
        
        # Separate up and down candles for batch drawing
        up_bodies = []
        down_bodies = []
        wicks = []
        
        for i, (x, o, h, l, c) in enumerate(data):
            # Adjust X coordinate for visible range
            screen_x = start_idx + i
            
            # Skip invalid data
            if not all(np.isfinite([o, h, l, c])) or h <= 0 or l <= 0:
                continue
            
            # Wick coordinates (thin vertical line)
            wicks.append((screen_x, l, screen_x, h))
            
            # Body coordinates
            body_bottom = min(o, c)
            body_height = abs(c - o)
            
            # Ensure minimum height for doji candles
            if body_height < (h - l) * 0.01:
                body_height = (h - l) * 0.01
            
            body_rect = (
                screen_x - width/2,  # x
                body_bottom,         # y
                width,               # width (THIN!)
                body_height          # height
            )
            
            # Separate by candle type
            if c >= o:  # Up candle
                up_bodies.append(body_rect)
            else:  # Down candle
                down_bodies.append(body_rect)
        
        return {
            'up_bodies': up_bodies,
            'down_bodies': down_bodies,
            'wicks': wicks
        }
    
    def _batch_draw_candlesticks(self, painter, coords):
        """Batch draw candlesticks for maximum performance"""
        
        # 1. Draw all wicks first (thin black lines)
        if coords['wicks']:
            painter.setPen(QtGui.QPen(QtCore.Qt.black, 1))
            for x1, y1, x2, y2 in coords['wicks']:
                painter.drawLine(QtCore.QLineF(x1, y1, x2, y2))
        
        # 2. Draw up candles (white fill, black border)
        if coords['up_bodies']:
            painter.setBrush(QtGui.QBrush(QtCore.Qt.white))
            painter.setPen(QtGui.QPen(QtCore.Qt.black, 1))
            for x, y, width, height in coords['up_bodies']:
                painter.drawRect(QtCore.QRectF(x, y, width, height))
        
        # 3. Draw down candles (red fill, black border)
        if coords['down_bodies']:
            painter.setBrush(QtGui.QBrush(QtCore.Qt.red))
            painter.setPen(QtGui.QPen(QtCore.Qt.black, 1))
            for x, y, width, height in coords['down_bodies']:
                painter.drawRect(QtCore.QRectF(x, y, width, height))
    
    def paint(self, painter, option, widget):
        if self.picture:
            painter.drawPicture(0, 0, self.picture)
    
    def boundingRect(self):
        if not self.full_data:
            return QtCore.QRectF()
        
        start_idx, end_idx = self.visible_range
        visible_data = self.full_data[start_idx:end_idx]
        
        if not visible_data:
            return QtCore.QRectF()
        
        min_price = min(row[3] for row in visible_data)  # low
        max_price = max(row[2] for row in visible_data)  # high
        
        return QtCore.QRectF(
            start_idx, min_price,
            end_idx - start_idx, max_price - min_price
        )

class FastESDashboard(QtWidgets.QMainWindow):
    """Fast ES 1m dashboard with viewport-based rendering"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Step 2: Fast ES Dashboard - Thin Candlesticks")
        self.setGeometry(100, 100, 1400, 900)
        
        # Data storage
        self.full_data = None
        self.candle_item = None
        
        # Create UI
        self._setup_ui()
        
        # Load ES data immediately
        self.load_es_data()
    
    def _setup_ui(self):
        """Setup user interface"""
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        layout = QtWidgets.QVBoxLayout(central_widget)
        
        # Info panel
        info_panel = QtWidgets.QHBoxLayout()
        self.info_label = QtWidgets.QLabel("Loading ES data...")
        self.performance_label = QtWidgets.QLabel("Performance: --")
        info_panel.addWidget(self.info_label)
        info_panel.addStretch()
        info_panel.addWidget(self.performance_label)
        layout.addLayout(info_panel)
        
        # Chart
        self.chart = pg.PlotWidget()
        self.chart.setBackground('w')
        self.chart.showGrid(x=True, y=True, alpha=0.3)
        self.chart.setLabel('left', 'ES Price', color='black', size='12pt')
        self.chart.setLabel('bottom', 'Time (Bar Index)', color='black', size='12pt')
        
        # Enable keyboard focus for controls
        self.chart.setFocusPolicy(Qt.StrongFocus)
        
        layout.addWidget(self.chart)
        
        # Controls info
        controls_label = QtWidgets.QLabel(
            "Controls: 1=Last 100 bars | 2=Last 1000 bars | 3=Last 10000 bars | "
            "4=Full dataset | Arrow Keys=Pan | +/-=Zoom | R=Reset | Q=Quit"
        )
        controls_label.setStyleSheet("background-color: #f0f0f0; padding: 5px;")
        layout.addWidget(controls_label)
        
        # Connect range change for dynamic rendering
        self.chart.sigRangeChanged.connect(self.on_range_changed)
    
    def load_es_data(self):
        """Load ES 1m data with performance measurement"""
        print("\n=== LOADING ES 1M DATA ===")
        start_time = time.time()
        
        # ES data file path
        es_file = Path(__file__).parent.parent.parent / "dataRaw" / "1m" / "ES" / "Current" / "ES-NONE-1m-EST-NoPad.csv"
        
        if not es_file.exists():
            self.info_label.setText(f"ES data file not found: {es_file}")
            return
        
        # Load CSV with pandas
        df = pd.read_csv(es_file)
        load_time = time.time() - start_time
        print(f"Loaded {len(df):,} bars in {load_time:.3f}s")
        
        # Convert to OHLC format
        convert_start = time.time()
        ohlc_data = []
        for i, row in df.iterrows():
            ohlc_data.append([
                i,                    # x coordinate (bar index)
                float(row['Open']),   # open
                float(row['High']),   # high  
                float(row['Low']),    # low
                float(row['Close'])   # close
            ])
        
        self.full_data = ohlc_data
        convert_time = time.time() - convert_start
        total_time = time.time() - start_time
        
        print(f"Converted to OHLC in {convert_time:.3f}s")
        print(f"Total load time: {total_time:.3f}s")
        print(f"Date range: {df.iloc[0]['Date']} to {df.iloc[-1]['Date']}")
        print(f"Price range: ${df['Low'].min():.2f} - ${df['High'].max():.2f}")
        
        # Update UI
        self.info_label.setText(f"ES 1m Data: {len(self.full_data):,} bars loaded in {total_time:.3f}s")
        
        # Show last 1000 bars initially for fast display
        self.show_recent_data(1000)
    
    def show_recent_data(self, num_bars):
        """Show the most recent N bars"""
        if not self.full_data:
            return
        
        print(f"\n=== SHOWING LAST {num_bars} BARS ===")
        start_time = time.time()
        
        # Clear existing chart
        self.chart.clear()
        
        # Calculate range
        total_bars = len(self.full_data)
        start_idx = max(0, total_bars - num_bars)
        end_idx = total_bars
        
        # Create candlestick item with limited range
        self.candle_item = FastCandlestickItem(self.full_data, (start_idx, end_idx))
        self.chart.addItem(self.candle_item)
        
        # Set view range
        visible_data = self.full_data[start_idx:end_idx]
        if visible_data:
            min_price = min(row[3] for row in visible_data)  # lows
            max_price = max(row[2] for row in visible_data)  # highs
            price_padding = (max_price - min_price) * 0.05
            
            self.chart.setXRange(start_idx, end_idx - 1, padding=0)
            self.chart.setYRange(min_price - price_padding, max_price + price_padding, padding=0)
        
        display_time = time.time() - start_time
        print(f"Displayed {num_bars} bars in {display_time:.4f}s")
        
        self.performance_label.setText(f"Display: {display_time:.4f}s | Bars: {num_bars:,}")
    
    def show_full_dataset(self):
        """Show entire dataset"""
        if not self.full_data:
            return
        
        print(f"\n=== SHOWING FULL DATASET ({len(self.full_data):,} bars) ===")
        start_time = time.time()
        
        self.chart.clear()
        
        # For full dataset, start with overview (every 10th bar for speed)
        sample_rate = max(1, len(self.full_data) // 10000)  # Limit to ~10k bars for performance
        if sample_rate > 1:
            print(f"Sampling every {sample_rate} bars for performance")
            sampled_data = self.full_data[::sample_rate]
            # Adjust x coordinates
            for i, bar in enumerate(sampled_data):
                bar[0] = i * sample_rate
        else:
            sampled_data = self.full_data
        
        self.candle_item = FastCandlestickItem(sampled_data)
        self.chart.addItem(self.candle_item)
        
        # Set view range for full data
        if sampled_data:
            min_price = min(row[3] for row in sampled_data)
            max_price = max(row[2] for row in sampled_data)
            price_padding = (max_price - min_price) * 0.05
            
            self.chart.setXRange(0, len(self.full_data) - 1, padding=0)
            self.chart.setYRange(min_price - price_padding, max_price + price_padding, padding=0)
        
        display_time = time.time() - start_time
        print(f"Displayed full dataset in {display_time:.4f}s (sample rate: {sample_rate})")
        
        self.performance_label.setText(f"Full dataset: {display_time:.4f}s | Sample: 1:{sample_rate}")
    
    def on_range_changed(self):
        """Handle range changes for dynamic rendering"""
        if not self.candle_item or not self.full_data:
            return
        
        # Get current view range
        view_range = self.chart.viewRange()
        x_range = view_range[0]
        
        start_idx = max(0, int(x_range[0] - 100))  # Add buffer
        end_idx = min(len(self.full_data), int(x_range[1] + 100))
        
        # Update candlestick visible range if significantly different
        current_range = self.candle_item.visible_range
        if abs(start_idx - current_range[0]) > 50 or abs(end_idx - current_range[1]) > 50:
            self.candle_item.set_visible_range(start_idx, end_idx)
    
    def keyPressEvent(self, event):
        """Handle keyboard controls"""
        key = event.key()
        
        if key == Qt.Key_1:
            self.show_recent_data(100)
        elif key == Qt.Key_2:
            self.show_recent_data(1000)
        elif key == Qt.Key_3:
            self.show_recent_data(10000)
        elif key == Qt.Key_4:
            self.show_full_dataset()
        elif key == Qt.Key_Q:
            self.close()
        elif key == Qt.Key_R:
            if self.full_data:
                self.chart.autoRange()
        # Pan and zoom
        elif key == Qt.Key_Left:
            self.pan_chart(-0.1)
        elif key == Qt.Key_Right:
            self.pan_chart(0.1)
        elif key == Qt.Key_Plus:
            self.zoom_chart(0.8)
        elif key == Qt.Key_Minus:
            self.zoom_chart(1.25)
        else:
            super().keyPressEvent(event)
    
    def pan_chart(self, factor):
        """Pan chart by factor of current range"""
        x_range = self.chart.viewRange()[0]
        span = x_range[1] - x_range[0]
        shift = span * factor
        self.chart.setXRange(x_range[0] + shift, x_range[1] + shift, padding=0)
    
    def zoom_chart(self, factor):
        """Zoom chart by factor"""
        x_range = self.chart.viewRange()[0]
        center = (x_range[0] + x_range[1]) / 2
        span = (x_range[1] - x_range[0]) * factor
        self.chart.setXRange(center - span/2, center + span/2, padding=0)

def main():
    """Main function"""
    print("="*70)
    print("STEP 2: FAST ES DASHBOARD WITH THIN CANDLESTICKS")
    print("="*70)
    print("Loading real ES 1m data and fixing the fat blob candlesticks...")
    print("="*70)
    
    app = QtWidgets.QApplication(sys.argv)
    
    dashboard = FastESDashboard()
    dashboard.show()
    
    return app.exec_()

if __name__ == "__main__":
    main()