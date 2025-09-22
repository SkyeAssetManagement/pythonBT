#!/usr/bin/env python3
"""
PyQtGraph Range Bars - FINAL PRODUCTION VERSION
================================================
High-performance candlestick chart for range bar visualization
with multi-monitor support, auto-scaling, and enhanced labeling

Features:
- Auto Y-axis scaling on zoom in/out
- Dynamic data loading on pan/zoom
- Multi-monitor DPI awareness
- Enhanced time axis with HH:MM:SS and date boundaries
- Large, readable axis text (3x size)
- Hover display with timestamp including seconds
- AUX1/AUX2 fields displayed as ATR and Range
"""

import sys
import numpy as np
import pandas as pd
import pyqtgraph as pg
from PyQt5 import QtWidgets, QtCore, QtGui
from pathlib import Path
import time
from datetime import datetime

# Trade visualization imports
from trade_panel import TradeListPanel
from simple_white_x_trades import TradeVisualization  # Simple white X marks
from trade_data import TradeCollection

# Performance settings
pg.setConfigOptions(antialias=True)
pg.setConfigOptions(useOpenGL=False)  # Disabled to avoid overflow warnings
pg.setConfigOptions(enableExperimental=False)

# Enable high DPI scaling for multi-monitor support
QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True)

class BoldDateAxisItem(pg.AxisItem):
    """Custom axis item for bold date boundaries"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.date_labels = {}  # Store which labels are date boundaries
        
    def setDateLabels(self, date_labels):
        """Mark which labels should have bold dates"""
        self.date_labels = date_labels
        
    def generateDrawSpecs(self, p):
        """Override to use bold font for date boundaries"""
        specs = super().generateDrawSpecs(p)
        
        if specs is not None:
            axisSpec, tickSpecs, textSpecs = specs
            
            # Modify text specs for date boundaries
            for i, (rect, flags, text) in enumerate(textSpecs):
                if text in self.date_labels:
                    # This is a date boundary - use bold font for the date part
                    textSpecs[i] = (rect, flags | QtCore.Qt.TextWordWrap, text)
            
            return axisSpec, tickSpecs, textSpecs
        return specs

class RangeBarChartFinal(QtWidgets.QMainWindow):
    """Final production version of range bar chart"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Range Bar Chart - FINAL VERSION")
        
        # Get screen info
        screen = QtWidgets.QApplication.primaryScreen()
        screen_rect = screen.availableGeometry()
        
        # Set initial size
        width = int(screen_rect.width() * 0.8)
        height = int(screen_rect.height() * 0.8)
        self.setGeometry(
            int(screen_rect.width() * 0.1),
            int(screen_rect.height() * 0.1),
            width, height
        )
        
        # Data management
        self.full_data = None
        self.total_bars = 0
        self.current_x_range = None
        self.last_screen_dpi = None
        self.is_rendering = False  # Prevent recursive rendering
        
        # UI components
        self.plot_widget = None
        self.candle_item = None
        
        # Trade visualization components
        self.trade_panel = None
        self.trade_visualization = None
        self.trade_arrows_scatter = None
        self.trade_dots_scatter = None
        self.current_trades = TradeCollection([])  # Empty initially

        # Indicator overlays
        self.indicator_lines = {}  # Store indicator line plot items
        self.indicator_data = {}  # Store full indicator data arrays

        # Setup
        self.setup_ui()
        # Only load data if not overridden by subclass
        if not hasattr(self, 'config'):
            self.load_data()
        
        # Screen change monitoring
        self.screen_check_timer = QtCore.QTimer()
        self.screen_check_timer.timeout.connect(self.check_screen_change)
        self.screen_check_timer.start(500)
        
    def setup_ui(self):
        """Setup UI with enhanced formatting and trade panel"""
        # Main horizontal layout with chart and trade panel
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QtWidgets.QHBoxLayout(central_widget)
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.setSpacing(10)
        
        # Chart area (left side) - takes 80% of width
        chart_widget = QtWidgets.QWidget()
        chart_layout = QtWidgets.QVBoxLayout(chart_widget)
        chart_layout.setContentsMargins(0, 0, 0, 0)
        layout = chart_layout  # For compatibility with existing code
        
        # Top data display with larger font
        self.hover_label = QtWidgets.QLabel("Hover over chart for data")
        font = QtGui.QFont("Consolas", 14, QtGui.QFont.Bold)  # Larger font
        font.setStyleHint(QtGui.QFont.Monospace)
        self.hover_label.setFont(font)
        self.hover_label.setStyleSheet("""
            QLabel { 
                color: #00ff00; 
                background: #1a1a1a; 
                padding: 12px;
                border: 2px solid #333;
                border-radius: 5px;
            }
        """)
        self.hover_label.setMinimumHeight(80)
        layout.addWidget(self.hover_label)
        
        # Main plot widget with custom axis
        self.custom_axis = BoldDateAxisItem(orientation='bottom')
        self.plot_widget = pg.PlotWidget(axisItems={'bottom': self.custom_axis})
        self.plot_widget.setBackground('#0a0a0a')
        self.plot_widget.showGrid(x=True, y=True, alpha=0.2)
        self.plot_widget.setMouseEnabled(x=True, y=True)
        
        # Set larger axis fonts (3x size)
        axis_font = QtGui.QFont("Arial", 12, QtGui.QFont.Bold)
        
        # Configure Y-axis
        self.plot_widget.getAxis('left').setLabel('Price ($)', **{'font-size': '14pt', 'color': '#ffffff'})
        self.plot_widget.getAxis('left').setTickFont(axis_font)
        self.plot_widget.getAxis('left').setPen(pg.mkPen(color='w', width=2))
        
        # Configure X-axis - will customize with time labels
        self.plot_widget.getAxis('bottom').setLabel('Time', **{'font-size': '14pt', 'color': '#ffffff'})
        self.plot_widget.getAxis('bottom').setTickFont(axis_font)
        self.plot_widget.getAxis('bottom').setPen(pg.mkPen(color='w', width=2))
        
        # Store date boundary positions for special formatting
        self.date_boundary_positions = []
        
        # Set ViewBox limits to prevent overflow
        viewBox = self.plot_widget.getViewBox()
        viewBox.setLimits(
            xMin=-100,  # Allow some padding
            xMax=200000,  # Increased to handle 122,609 bars with padding
            yMin=0,  # Price shouldn't be negative
            yMax=100000,  # Large but finite limit
            minXRange=10,  # Minimum 10 bars visible
            maxXRange=150000,  # Increased to allow viewing all bars at once
            minYRange=1,  # Minimum price range
            maxYRange=50000  # Maximum price range
        )

        # Disable OpenGL to avoid driver issues
        self.plot_widget.useOpenGL(False)
        self.plot_widget.setAntialiasing(True)
        
        layout.addWidget(self.plot_widget)
        
        # Status bar
        self.status_label = QtWidgets.QLabel("Loading...")
        self.status_label.setFont(QtGui.QFont("Consolas", 11))
        self.status_label.setStyleSheet("""
            QLabel { 
                color: #00ffff; 
                background: #0a0a0a; 
                padding: 10px;
                border: 1px solid #333;
            }
        """)
        layout.addWidget(self.status_label)
        
        # Crosshair
        self.crosshair_v = pg.InfiniteLine(
            angle=90, movable=False,
            pen=pg.mkPen('#ffff00', width=1, style=QtCore.Qt.DashLine)
        )
        self.crosshair_h = pg.InfiniteLine(
            angle=0, movable=False,
            pen=pg.mkPen('#ffff00', width=1, style=QtCore.Qt.DashLine)
        )
        self.plot_widget.addItem(self.crosshair_v, ignoreBounds=True)
        self.plot_widget.addItem(self.crosshair_h, ignoreBounds=True)
        
        # Add chart area to main layout
        main_layout.addWidget(chart_widget, 4)  # 80% width
        
        # Create trade panel (right side) - takes 20% of width
        self.trade_panel = TradeListPanel()
        self.trade_panel.setMinimumWidth(400)  # Minimum readable width, increased from 350
        self.trade_panel.setMaximumWidth(500)  # Maximum width to prevent it from getting too wide
        main_layout.addWidget(self.trade_panel, 1)  # 20% width (layout stretch factor)
        
        # Connect trade panel signals
        self.trade_panel.trade_selected.connect(self.jump_to_trade)
        self.trade_panel.source_selector.trades_loaded.connect(self.load_trades)

        # Connect strategy runner for indicators
        if hasattr(self.trade_panel, 'strategy_runner'):
            self.trade_panel.strategy_runner.trades_generated.connect(self.load_trades)
            self.trade_panel.strategy_runner.indicators_calculated.connect(self.add_indicator_overlays)
        
        # Connect signals - use sigRangeChanged to catch ALL changes
        self.plot_widget.sigRangeChanged.connect(self.on_range_changed)
        # Add specific handler for X-axis changes to ensure dynamic loading
        self.plot_widget.sigXRangeChanged.connect(self.on_x_range_changed)
        # Also connect for trade panel auto-sync
        self.plot_widget.sigXRangeChanged.connect(self.on_viewport_changed)
        self.proxy = pg.SignalProxy(
            self.plot_widget.scene().sigMouseMoved,
            rateLimit=60, slot=self.on_mouse_moved
        )
        
    def load_data(self):
        """Load range bar data with AUX fields"""
        print("Loading range bar data...")
        start_time = time.time()
        
        file_path = Path("parquetData/range/ATR30x0.1/ES-DIFF-range-ATR30x0.1-amibroker.parquet")
        
        if file_path.exists():
            df = pd.read_parquet(file_path)
            self.full_data = {
                'timestamp': pd.to_datetime(df['timestamp']),
                'open': df['open'].values.astype(np.float32),
                'high': df['high'].values.astype(np.float32),
                'low': df['low'].values.astype(np.float32),
                'close': df['close'].values.astype(np.float32),
                'volume': df['volume'].values.astype(np.float32) if 'volume' in df else None,
                'aux1': df['AUX1'].values.astype(np.float32) if 'AUX1' in df else None,  # ATR
                'aux2': df['AUX2'].values.astype(np.float32) if 'AUX2' in df else None   # Range multiplier
            }
            self.total_bars = len(self.full_data['open'])
            print(f"Loaded {self.total_bars:,} bars from {file_path.name}")
        else:
            # Test data
            print("Using test data - file not found")
            n = 10000
            base = 4000
            prices = base + np.cumsum(np.random.randn(n) * 2)
            self.full_data = {
                'timestamp': pd.date_range('2024-01-01', periods=n, freq='5min'),
                'open': np.roll(prices, 1).astype(np.float32),
                'high': (prices + np.abs(np.random.randn(n) * 3)).astype(np.float32),
                'low': (prices - np.abs(np.random.randn(n) * 3)).astype(np.float32),
                'close': prices.astype(np.float32),
                'volume': (np.random.uniform(1000, 10000, n)).astype(np.float32),
                'aux1': np.random.uniform(10, 30, n).astype(np.float32),  # Simulated ATR
                'aux2': np.full(n, 0.1, dtype=np.float32)  # Simulated multiplier
            }
            self.full_data['open'][0] = base
            self.total_bars = n
        
        print(f"Data loaded: {self.total_bars:,} bars in {time.time()-start_time:.2f}s")
        
        # Pass timestamps and bar data to trade panel for coordinated trade generation
        if self.trade_panel and self.full_data['timestamp'] is not None:
            self.trade_panel.set_chart_timestamps(self.full_data['timestamp'])
            
            # Also pass bar data for realistic trade pricing (including timestamps)
            bar_data = {
                'timestamp': self.full_data['timestamp'],  # Include timestamps!
                'open': self.full_data['open'],
                'high': self.full_data['high'],
                'low': self.full_data['low'],
                'close': self.full_data['close']
            }
            self.trade_panel.set_bar_data(bar_data)
            print(f"Set chart timestamps and OHLC data in trade panel: {self.full_data['timestamp'].iloc[0]} to {self.full_data['timestamp'].iloc[-1]}")
        
        # Initial render
        self.render_range(0, min(500, self.total_bars))
        
    def format_time_axis(self, start_idx, end_idx, num_bars):
        """Create custom time axis with HH:MM:SS and date boundaries"""
        if num_bars == 0:
            return

        # Debug logging
        print(f"[FORMAT_TIME_AXIS] start={start_idx}, end={end_idx}, num_bars={num_bars}")
        print(f"[FORMAT_TIME_AXIS] Timestamps type: {type(self.full_data['timestamp'])}")

        try:
            # Get timestamps for visible range - convert to list for safer indexing
            timestamps = self.full_data['timestamp'][start_idx:end_idx]
            timestamps_array = timestamps.values if hasattr(timestamps, 'values') else timestamps

            # Debug: Check what timestamps we're working with (commented out)
            # print(f"[FORMAT_TIME_AXIS] Timestamp check:")
            # for i in range(min(3, len(timestamps_array))):
            #     ts = pd.Timestamp(timestamps_array[i])
            #     print(f"  [{i}] {timestamps_array[i]} -> hour={ts.hour}, minute={ts.minute}, second={ts.second}")

            # print(f"[PHASE1-DEBUG] Timestamp slice: first={timestamps_array[0] if len(timestamps_array) > 0 else 'EMPTY'}, last={timestamps_array[-1] if len(timestamps_array) > 0 else 'EMPTY'}")
            print(f"[FORMAT_TIME_AXIS] First timestamp: {timestamps_array[0] if len(timestamps_array) > 0 else 'EMPTY'}")
            print(f"[FORMAT_TIME_AXIS] Timestamp slice length: {len(timestamps_array)}")
            
            # Downsample if needed
            if num_bars > 2000:
                step = num_bars // 1000
                timestamps_array = timestamps_array[::step]
            
            # Create approximately 10 time labels
            num_labels = min(10, len(timestamps_array))
            if num_labels > 0:
                label_indices = np.linspace(0, len(timestamps_array)-1, num_labels, dtype=int)
                
                x_ticks = []
                x_labels = []
                self.date_boundary_positions = []  # Reset date boundaries
                date_labels = {}  # Track which labels are date boundaries
                
                prev_date = None
                for i, idx in enumerate(label_indices):
                    ts = timestamps_array[idx]
                    # Calculate actual x position - must match what we use in render_range
                    # In render_range we use: x = np.arange(start_idx, end_idx)
                    # So x_pos should be the actual bar index
                    actual_bar_idx = start_idx + idx
                    if num_bars > 2000:
                        # When downsampled, we need to account for the step
                        step = num_bars // 1000
                        actual_bar_idx = start_idx + idx * step

                    x_pos = actual_bar_idx
                    
                    # Check for date boundary
                    current_date = pd.Timestamp(ts).date() if not isinstance(ts, pd.Timestamp) else ts.date()
                    if prev_date and current_date != prev_date:
                        # Add date boundary marker with time on top, date below
                        date_str = current_date.strftime('%y-%m-%d')
                        # Format time as h:mm:ss (remove leading zero from hours)
                        time_obj = pd.Timestamp(ts)
                        hour = time_obj.hour
                        time_str = f"{hour}:{time_obj.strftime('%M:%S')}"
                        # Date boundary: show time on first line, date on second
                        label = f"{time_str}\n{date_str}"
                        x_labels.append(label)
                        self.date_boundary_positions.append(x_pos)
                        date_labels[label] = True  # Mark as date boundary
                    else:
                        # Regular time format h:mm:ss (no leading zero on hours)
                        time_obj = pd.Timestamp(ts)
                        hour = time_obj.hour
                        time_str = f"{hour}:{time_obj.strftime('%M:%S')}"
                        x_labels.append(time_str)
                    
                    x_ticks.append(x_pos)
                    prev_date = current_date

                    # Debug output - show actual time components
                    if i < 3:  # Only first few to avoid spam
                        ts_obj = pd.Timestamp(ts)
                        print(f"[FORMAT_TIME_AXIS] Label {i}: x_pos={x_pos}, timestamp={ts}, hour={ts_obj.hour}, min={ts_obj.minute}, sec={ts_obj.second}, label={x_labels[-1]}")

                # Update custom axis with date labels
                if hasattr(self, 'custom_axis'):
                    self.custom_axis.setDateLabels(date_labels)

                # Set custom ticks
                print(f"[FORMAT_TIME_AXIS] Setting {len(x_ticks)} ticks")
                self.plot_widget.getAxis('bottom').setTicks([list(zip(x_ticks, x_labels))])
        except Exception as e:
            # If time axis formatting fails, don't crash the whole render
            print(f"Warning: Time axis formatting error: {e}")
            pass
        
    def render_range(self, start_idx, end_idx, update_x_range=True):
        """Render range with enhanced features"""
        if self.is_rendering:
            return

        self.is_rendering = True
        render_start = time.time()

        try:
            # Bounds checking
            start_idx = max(0, int(start_idx))
            end_idx = min(self.total_bars, int(end_idx))

            if start_idx >= end_idx:
                return

            num_bars = end_idx - start_idx

            # Clear previous (but preserve indicators)
            # Remove only candle item and trades, not indicators
            if hasattr(self, 'candle_item') and self.candle_item:
                self.plot_widget.removeItem(self.candle_item)
                self.candle_item = None

            # Note: Trade items are handled separately in load_trades()
            # Crosshairs stay persistent (no need to re-add)

            # Get data slice
            x = np.arange(start_idx, end_idx)
            opens = self.full_data['open'][start_idx:end_idx]
            highs = self.full_data['high'][start_idx:end_idx]
            lows = self.full_data['low'][start_idx:end_idx]
            closes = self.full_data['close'][start_idx:end_idx]

            # Verify we have data
            if len(opens) == 0:
                return

            # Store original num_bars for axis formatting
            original_num_bars = num_bars

            # Downsample if needed
            if num_bars > 2000:
                step = num_bars // 1000
                x = x[::step]
                opens = opens[::step]
                highs = highs[::step]
                lows = lows[::step]
                closes = closes[::step]
                num_bars = len(x)

            # Create candlesticks with view range info for adaptive spacing
            view_range = (start_idx, end_idx, original_num_bars)
            self.candle_item = EnhancedCandlestickItem(x, opens, highs, lows, closes, view_range)
            self.plot_widget.addItem(self.candle_item)

            # Format time axis
            self.format_time_axis(start_idx, end_idx, original_num_bars)

            # Calculate Y range with padding
            y_min = lows.min()
            y_max = highs.max()

            # Also consider indicator values in Y range if they exist
            if hasattr(self, 'indicator_data') and self.indicator_data:
                for name, values in self.indicator_data.items():
                    if len(values) > start_idx:
                        visible_end = min(end_idx, len(values))
                        visible_values = values[start_idx:visible_end]
                        if len(visible_values) > 0:
                            ind_min = np.nanmin(visible_values)
                            ind_max = np.nanmax(visible_values)
                            if not np.isnan(ind_min):
                                y_min = min(y_min, ind_min)
                            if not np.isnan(ind_max):
                                y_max = max(y_max, ind_max)

            y_range = y_max - y_min
            y_padding = y_range * 0.05

            # Set ranges
            if update_x_range:
                self.plot_widget.setXRange(start_idx, end_idx, padding=0)

            self.plot_widget.setYRange(y_min - y_padding, y_max + y_padding, padding=0)

            # Store current range
            self.current_x_range = (start_idx, end_idx)

            # Re-add trade visualization if it exists
            if hasattr(self, 'trade_arrows_scatter') and self.trade_arrows_scatter:
                if self.trade_arrows_scatter not in self.plot_widget.items():
                    self.plot_widget.addItem(self.trade_arrows_scatter)
            if hasattr(self, 'trade_dots_scatter') and self.trade_dots_scatter:
                if self.trade_dots_scatter not in self.plot_widget.items():
                    self.plot_widget.addItem(self.trade_dots_scatter)

            # Render visible portion of indicators
            self.render_visible_indicators()

            # Update status
            render_time = time.time() - render_start
            current_screen = QtWidgets.QApplication.screenAt(self.geometry().center())
            if current_screen:
                dpi = current_screen.logicalDotsPerInch()
                self.status_label.setText(
                    f"Bars: {start_idx:,}-{end_idx:,} of {self.total_bars:,} | "
                    f"Rendered: {num_bars} in {render_time*1000:.1f}ms | "
                    f"Price Range: ${y_min:.2f}-${y_max:.2f} | "
                    f"Screen: {current_screen.name()} ({dpi} DPI)"
                )
        
        finally:
            # Always reset rendering flag
            self.is_rendering = False
        
    def on_x_range_changed(self, viewbox, range):
        """Handle X-axis range changes for dynamic loading"""
        if self.full_data is None:
            return

        # Skip if we're already rendering to prevent recursion
        if self.is_rendering:
            return

        new_start = max(0, int(range[0]))
        new_end = min(self.total_bars, int(range[1]))

        # Only render if the range actually changed significantly
        if hasattr(self, 'current_x_range'):
            old_start, old_end = self.current_x_range
            # Check if change is significant (more than 1 bar)
            if abs(new_start - old_start) < 1 and abs(new_end - old_end) < 1:
                return

        # Always re-render for the new range to ensure continuous data loading
        self.render_range(new_start, new_end, update_x_range=False)

    def on_range_changed(self, widget, ranges):
        """Handle any range changes - both X and Y axis"""
        if self.full_data is None or self.is_rendering:
            return

        # Get X range from the ranges tuple
        x_range = ranges[0]
        new_start = max(0, int(x_range[0]))
        new_end = min(self.total_bars, int(x_range[1]))

        # Check if X range actually changed
        if not hasattr(self, 'current_x_range') or self.current_x_range != (new_start, new_end):
            # Always re-render to ensure Y-axis auto-scales based on visible data
            self.render_range(new_start, new_end, update_x_range=False)
            
    def on_mouse_moved(self, evt):
        """Handle mouse hover with enhanced data display"""
        pos = evt[0]
        if self.plot_widget.sceneBoundingRect().contains(pos):
            mouse_point = self.plot_widget.plotItem.vb.mapSceneToView(pos)
            
            self.crosshair_v.setPos(mouse_point.x())
            self.crosshair_h.setPos(mouse_point.y())
            
            bar_idx = int(mouse_point.x())
            
            if 0 <= bar_idx < self.total_bars:
                # Get all data for this bar
                timestamp = self.full_data['timestamp'][bar_idx]
                o = self.full_data['open'][bar_idx]
                h = self.full_data['high'][bar_idx]
                l = self.full_data['low'][bar_idx]
                c = self.full_data['close'][bar_idx]
                v = self.full_data['volume'][bar_idx] if self.full_data['volume'] is not None else 0
                
                # Get AUX fields
                atr = self.full_data['aux1'][bar_idx] if self.full_data['aux1'] is not None else 0
                range_mult = self.full_data['aux2'][bar_idx] if self.full_data['aux2'] is not None else 0
                
                # Format timestamp with seconds - convert numpy.datetime64 to datetime
                if hasattr(timestamp, 'astype'):
                    # Convert numpy.datetime64 to datetime
                    timestamp_dt = pd.to_datetime(timestamp).to_pydatetime()
                    time_str = timestamp_dt.strftime('%Y-%m-%d %H:%M:%S')
                else:
                    # Already a datetime object
                    time_str = timestamp.strftime('%Y-%m-%d %H:%M:%S')
                
                # Calculate range value
                range_value = atr * range_mult if atr > 0 and range_mult > 0 else 0
                
                # Check for trades near the mouse position
                hover_text = f"Bar {bar_idx:,} | {time_str}\n" \
                           f"OHLC: ${o:.2f} / ${h:.2f} / ${l:.2f} / ${c:.2f} | Vol: {v:,.0f}\n" \
                           f"ATR: {atr:.2f} | Range: {atr:.2f} x {range_mult:.1f} = {range_value:.2f}"
                
                # Check if there are trades at this bar or nearby
                if self.trade_visualization and len(self.current_trades) > 0:
                    nearby_trade = self.trade_visualization.get_trade_at_point(
                        mouse_point.x(), mouse_point.y(), tolerance=15.0
                    )
                    if nearby_trade:
                        trade_id = getattr(nearby_trade, 'trade_id', None)
                        size = getattr(nearby_trade, 'size', 1)
                        strategy = getattr(nearby_trade, 'strategy', None)
                        pnl_value = getattr(nearby_trade, 'pnl', None)
                        id_text = f" #{trade_id}" if trade_id is not None else ""
                        hover_text += (
                            f"\n\n🎯 TRADE: {nearby_trade.trade_type}{id_text}\n"
                            f"Price: ${nearby_trade.price:.2f} | Size: {size}\n"
                            f"Strategy: {strategy or 'N/A'}"
                        )
                        if isinstance(pnl_value, (int, float)):
                            hover_text += f" | P&L: ${pnl_value:.2f}"
                
                # Update hover label with all information
                self.hover_label.setText(hover_text)
            else:
                self.hover_label.setText(f"Bar {bar_idx:,} (out of range)")
                
    def check_screen_change(self):
        """Monitor for screen changes"""
        current_screen = QtWidgets.QApplication.screenAt(self.geometry().center())
        
        if current_screen:
            current_dpi = current_screen.logicalDotsPerInch()
            
            if self.last_screen_dpi != current_dpi:
                self.last_screen_dpi = current_dpi
                # Force redraw on screen change
                if self.current_x_range:
                    self.render_range(self.current_x_range[0], self.current_x_range[1])
                    
    def moveEvent(self, event):
        """Handle window move events"""
        super().moveEvent(event)
        self.check_screen_change()
        
    def resizeEvent(self, event):
        """Handle window resize events"""
        super().resizeEvent(event)
        if self.current_x_range:
            QtCore.QTimer.singleShot(100, lambda: self.render_range(
                self.current_x_range[0], self.current_x_range[1]
            ))
    
    def load_trades(self, trades: TradeCollection):
        """Load trades and add visualization to chart using optimized ScatterPlotItem"""
        self.current_trades = trades
        
        if len(trades) == 0:
            return
        
        # Create bar data for arrow positioning
        bar_data = {
            'high': self.full_data['high'],
            'low': self.full_data['low']
        }
        
        # Remove existing trade graphics
        if self.trade_arrows_scatter:
            self.plot_widget.removeItem(self.trade_arrows_scatter)
        if self.trade_dots_scatter:
            self.plot_widget.removeItem(self.trade_dots_scatter)
        
        # Create new trade visualization using ScatterPlotItem
        self.trade_visualization = TradeVisualization(trades, bar_data)
        self.trade_arrows_scatter = self.trade_visualization.create_arrow_scatter()
        self.trade_dots_scatter = self.trade_visualization.create_dots_scatter()

        # Add to chart with proper z-ordering (trades on top)
        if self.trade_arrows_scatter:
            self.trade_arrows_scatter.setZValue(1000)  # Ensure trades render on top
            self.plot_widget.addItem(self.trade_arrows_scatter)
        if self.trade_dots_scatter:
            self.trade_dots_scatter.setZValue(1001)  # Dots even higher if they exist
            self.plot_widget.addItem(self.trade_dots_scatter)
        
        print(f"Loaded {len(trades)} trades to chart using optimized ScatterPlotItem")
        
        # Also populate the trade list panel if available
        if self.trade_panel is not None:
            try:
                self.trade_panel.load_trades(trades)
            except Exception as e:
                print(f"Warning: could not load trades into panel: {e}")

    def add_indicator_overlays(self, indicators: dict):
        """Add indicator lines to the chart - store full data but render incrementally"""
        # Store full indicator data
        self.indicator_data = indicators

        # Clear existing indicator line items
        for name, line in self.indicator_lines.items():
            self.plot_widget.removeItem(line)
        self.indicator_lines.clear()

        print(f"Stored {len(indicators)} indicators for incremental rendering")

        # Render visible portion if we have a current range
        if hasattr(self, 'current_x_range'):
            self.render_visible_indicators()

    def render_visible_indicators(self):
        """Render only the visible portion of indicators for performance"""
        if not hasattr(self, 'indicator_data') or not self.indicator_data:
            return

        if not hasattr(self, 'current_x_range'):
            return

        start_idx, end_idx = self.current_x_range
        colors = ['#00ff00', '#ff00ff', '#00ffff', '#ffff00', '#ff8800']
        color_idx = 0

        # Clear previous indicator lines
        for name, line in self.indicator_lines.items():
            if line in self.plot_widget.items():
                self.plot_widget.removeItem(line)
        self.indicator_lines.clear()

        # Render visible portion of each indicator
        for name, values in self.indicator_data.items():
            # Ensure we have enough data up to the visible range
            if len(values) > 0:
                # Get visible portion - ensure we show data even if it starts before start_idx
                visible_start = max(0, start_idx)
                visible_end = min(end_idx, len(values))

                if visible_end > visible_start:
                    visible_values = values[visible_start:visible_end]
                    visible_x = np.arange(visible_start, visible_end)

                    # Skip if no valid data
                    if len(visible_values) == 0 or np.all(np.isnan(visible_values)):
                        continue

                    # Downsample if needed for performance
                    if len(visible_values) > 2000:
                        step = len(visible_values) // 1000
                        visible_values = visible_values[::step]
                        visible_x = visible_x[::step]

                    # Create line plot for visible portion only
                    pen = pg.mkPen(colors[color_idx % len(colors)], width=2)  # Width 2 for better visibility
                    line = self.plot_widget.plot(visible_x, visible_values, pen=pen, name=name)
                    line.setZValue(500)  # Set z-value above candles but below trades

                    # Store reference
                    self.indicator_lines[name] = line
                    color_idx += 1

    def jump_to_trade(self, trade):
        """Jump to specific trade with 250 bars on each side"""
        print(f"[JUMP_TO_TRADE] Called for trade: {trade.trade_type}, bar_index={trade.bar_index}")
        print(f"[JUMP_TO_TRADE] Current total_bars: {self.total_bars}")

        target_bar = trade.bar_index

        # Calculate viewport range (250 bars on each side)
        start_bar = max(0, target_bar - 250)
        end_bar = min(self.total_bars - 1, target_bar + 250)

        print(f"[JUMP_TO_TRADE] Calculated range: start={start_bar}, end={end_bar}, target={target_bar}")

        # First render the range to ensure data is loaded and X marks are visible
        print(f"[JUMP_TO_TRADE] Calling render_range({start_bar}, {end_bar})")
        self.render_range(start_bar, end_bar)

        # Then set the viewport (render_range already sets X range, but ensure it's exact)
        print(f"[JUMP_TO_TRADE] Setting X range to ({start_bar}, {end_bar})")
        self.plot_widget.setXRange(start_bar, end_bar, padding=0)

        # Verify the range was set
        view_range = self.plot_widget.viewRange()
        print(f"[JUMP_TO_TRADE] After setXRange, actual view range: X={view_range[0]}, Y={view_range[1]}")

        print(f"[JUMP_TO_TRADE] Completed jump to trade: {trade.trade_type} at bar {target_bar}")
    
    def on_viewport_changed(self, view_box, range_obj):
        """Handle viewport changes for trade panel auto-sync"""
        if self.trade_panel and self.current_trades:
            x_range = range_obj
            start_bar = int(max(0, x_range[0]))
            end_bar = int(min(self.total_bars - 1, x_range[1]))
            
            # Sync trade panel to first visible trade
            self.trade_panel.scroll_to_first_visible_trade(start_bar, end_bar)


class EnhancedCandlestickItem(pg.GraphicsObject):
    """Enhanced candlestick graphics with adaptive spacing"""
    
    def __init__(self, x, opens, highs, lows, closes, view_range=None):
        pg.GraphicsObject.__init__(self)
        self.data = (x, opens, highs, lows, closes)
        self.view_range = view_range  # Pass view range for adaptive spacing
        self.generatePicture()
        
    def generatePicture(self):
        self.picture = QtGui.QPicture()
        painter = QtGui.QPainter(self.picture)
        painter.setRenderHint(QtGui.QPainter.Antialiasing, False)
        
        x_orig, opens, highs, lows, closes = self.data
        
        # Enhanced colors for better visibility
        # Adjust pen width based on zoom level too
        if len(x_orig) > 1000:
            pen_width = 1  # Thinner lines when many bars visible
        else:
            pen_width = 2  # Normal width when zoomed in
            
        green_pen = pg.mkPen('#00ff88', width=pen_width)
        red_pen = pg.mkPen('#ff4444', width=pen_width)
        green_brush = pg.mkBrush('#00ff88')
        red_brush = pg.mkBrush('#ff4444')
        
        # Adaptive gap spacing based on number of visible bars
        x = x_orig.copy()
        
        if len(x) > 1:
            num_bars = len(x)
            
            # Determine bar width based on zoom level (number of bars visible)
            # Much more extreme gaps when zoomed out
            if num_bars > 1000:  # Very zoomed out - many bars
                # Ultra-thin bars with massive gaps (approximately 5x normal gap)
                # Normal gap is 30% (width=0.7), so 5x would be 150% gap
                # This means bar should be about 40% of total space: 100/(100+150) = 0.4 of space
                # But we want bar to be smaller portion, so: width = 0.05
                width = 0.05  # 5% width, 95% gap (about 5x the normal 30% gap)
            elif num_bars > 800:
                width = 0.12  # 12% width, 88% gap
            elif num_bars > 600:
                width = 0.15  # 15% width, 85% gap
            elif num_bars > 400:
                width = 0.2  # 20% width, 80% gap
            elif num_bars > 300:
                width = 0.25  # 25% width, 75% gap
            elif num_bars > 200:
                width = 0.3  # 30% width, 70% gap
            elif num_bars > 150:
                width = 0.35  # 35% width, 65% gap
            elif num_bars > 100:
                width = 0.4  # 40% width, 60% gap
            elif num_bars > 75:
                width = 0.45  # 45% width, 55% gap
            elif num_bars > 50:
                width = 0.5  # 50% width, 50% gap
            elif num_bars > 30:
                width = 0.55  # 55% width, 45% gap
            elif num_bars > 20:
                width = 0.6  # 60% width, 40% gap
            else:  # Zoomed in - few bars
                width = 0.7  # 70% width, 30% gap (normal candlesticks)
        else:
            width = 0.6
        
        for i in range(len(x)):
            xi = x[i]
            o, h, l, c = opens[i], highs[i], lows[i], closes[i]
            
            if c >= o:
                painter.setPen(green_pen)
                painter.setBrush(green_brush)
            else:
                painter.setPen(red_pen)
                painter.setBrush(red_brush)
            
            # Draw wick
            painter.drawLine(QtCore.QPointF(xi, l), QtCore.QPointF(xi, h))
            
            # Draw body
            body_height = abs(c - o)
            if body_height > 0:
                painter.drawRect(QtCore.QRectF(
                    xi - width/2, min(o, c),
                    width, body_height
                ))
        
        painter.end()
        
    def paint(self, painter, *args):
        painter.drawPicture(0, 0, self.picture)
        
    def boundingRect(self):
        return QtCore.QRectF(self.picture.boundingRect())


def main():
    """Main entry point"""
    app = QtWidgets.QApplication(sys.argv)
    
    # Enable high DPI support
    app.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)
    app.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps)
    
    # Dark theme
    app.setStyle('Fusion')
    dark_palette = QtGui.QPalette()
    dark_palette.setColor(QtGui.QPalette.Window, QtGui.QColor(20, 20, 20))
    dark_palette.setColor(QtGui.QPalette.WindowText, QtCore.Qt.white)
    dark_palette.setColor(QtGui.QPalette.Base, QtGui.QColor(30, 30, 30))
    dark_palette.setColor(QtGui.QPalette.Text, QtCore.Qt.white)
    app.setPalette(dark_palette)
    
    print("\n" + "="*70)
    print("RANGE BAR CHART - FINAL PRODUCTION VERSION")
    print("="*70)
    print("Features:")
    print("- 3x larger axis text for better readability")
    print("- Time axis with HH:MM:SS format and date boundaries")
    print("- Hover display with full timestamp including seconds")
    print("- ATR and Range (AUX1 x AUX2) fields displayed")
    print("- Multi-monitor support with DPI awareness")
    print("- Auto Y-axis scaling on zoom in/out")
    print("- Dynamic data loading for 122K+ bars")
    print("\nControls:")
    print("- Drag: Pan the chart")
    print("- Wheel: Zoom in/out")
    print("- Hover: See complete bar data")
    print("="*70 + "\n")
    
    # Create and show chart
    chart = RangeBarChartFinal()
    chart.show()
    
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()