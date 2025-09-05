"""
Modular Trading Dashboard - Clean, professional architecture
Main dashboard implementation using modular components.
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Qt OpenGL configuration
os.environ['QT_OPENGL'] = 'desktop'
from PyQt5.QtCore import QCoreApplication, Qt
QCoreApplication.setAttribute(Qt.AA_DontCheckOpenGLContextThreadAffinity, True)
QCoreApplication.setAttribute(Qt.AA_UseDesktopOpenGL, True)

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional

from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QSplitter, QApplication, QTabWidget
)
from PyQt5.QtCore import pyqtSlot, pyqtSignal

# Import modular components
from dashboard.modular_chart_manager import ModularChartManager
from dashboard.trade_list_widget import TradeListContainer
from dashboard.equity_curve_widget import EquityCurveWidget
from dashboard.hover_info_widget import HoverInfoWidget
from dashboard.crosshair_widget import CrosshairOverlay, CrosshairInfoWidget
from dashboard.indicators_panel import VBTIndicatorsPanel
from dashboard.time_axis_widget import TimeAxisWidget

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ModularTradingDashboard(QMainWindow):
    """
    Professional trading dashboard with clean, modular architecture.
    """
    
    def __init__(self):
        super().__init__()
        
        logger.info("="*70)
        logger.info("MODULAR TRADING DASHBOARD - Professional Architecture")
        logger.info("="*70)
        
        # Core components
        self.chart_manager = None
        self.trade_list = None
        self.equity_curve = None
        self.hover_info = None
        self.crosshair = None
        self.indicators_panel = None
        self.time_axis = None
        
        # Data storage
        self.ohlcv_data = None
        self.trades_data = []
        
        # Initialize UI
        self._init_ui()
        
        # Connect signals
        self._connect_signals()
        
        logger.info("Dashboard initialization complete")
    
    def _init_ui(self):
        """Initialize the user interface."""
        self.setWindowTitle("Modular Trading Dashboard")
        self.setGeometry(100, 100, 1920, 1080)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(0)
        main_layout.setContentsMargins(5, 5, 5, 5)
        
        # Create main splitter (horizontal)
        main_splitter = QSplitter(Qt.Horizontal)
        
        # Left panel - Chart and related widgets
        left_panel = self._create_left_panel()
        
        # Right panel - Trade list and indicators
        right_panel = self._create_right_panel()
        
        # Add panels to splitter
        main_splitter.addWidget(left_panel)
        main_splitter.addWidget(right_panel)
        main_splitter.setSizes([1400, 520])
        
        # Add to main layout
        main_layout.addWidget(main_splitter)
        
        logger.info("UI initialized successfully")
    
    def _create_left_panel(self) -> QWidget:
        """Create the left panel with chart and related widgets."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setSpacing(2)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Create chart manager
        self.chart_manager = ModularChartManager(width=1400, height=700)
        
        # Chart container
        chart_container = QWidget()
        chart_layout = QVBoxLayout(chart_container)
        chart_layout.setContentsMargins(0, 0, 0, 0)
        
        # Add VisPy canvas
        chart_layout.addWidget(self.chart_manager.get_canvas())
        
        # Time axis
        self.time_axis = TimeAxisWidget()
        chart_layout.addWidget(self.time_axis)
        
        # Vertical splitter for chart and equity curve
        v_splitter = QSplitter(Qt.Vertical)
        
        # Add chart container
        v_splitter.addWidget(chart_container)
        
        # Equity curve
        self.equity_curve = EquityCurveWidget()
        v_splitter.addWidget(self.equity_curve)
        
        v_splitter.setSizes([700, 300])
        
        # Add to layout
        layout.addWidget(v_splitter)
        
        # Overlays
        self._create_overlays(chart_container)
        
        return panel
    
    def _create_right_panel(self) -> QWidget:
        """Create the right panel with trade list and indicators."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setSpacing(2)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Tab widget
        tab_widget = QTabWidget()
        
        # Trade list
        self.trade_list = TradeListContainer()
        tab_widget.addTab(self.trade_list, "Trades")
        
        # Indicators panel
        self.indicators_panel = VBTIndicatorsPanel()
        tab_widget.addTab(self.indicators_panel, "Indicators")
        
        layout.addWidget(tab_widget)
        
        return panel
    
    def _create_overlays(self, parent: QWidget):
        """Create overlay widgets for the chart."""
        # Hover info
        self.hover_info = HoverInfoWidget(parent)
        self.hover_info.move(10, 10)
        
        # Crosshair
        self.crosshair = CrosshairOverlay(parent)
        self.crosshair_info = CrosshairInfoWidget(parent)
        self.crosshair_info.move(10, 60)
    
    def _connect_signals(self):
        """Connect all component signals."""
        if not self.chart_manager:
            return
        
        # Viewport changes
        self.chart_manager.viewport_changed.connect(self._on_viewport_changed)
        
        # Trade navigation
        if self.trade_list:
            self.trade_list.trade_selected.connect(self._on_trade_selected)
        
        # Hover updates
        self.chart_manager.hover_update.connect(self._on_hover_update)
        
        # Crosshair updates
        self.chart_manager.crosshair_update.connect(self._on_crosshair_update)
        
        logger.info("Signals connected")
    
    # ==================== Data Loading ====================
    
    def load_data(self, ohlcv_data: Dict[str, np.ndarray], 
                  trades_csv_path: Optional[str] = None) -> bool:
        """
        Load OHLCV data and optionally trades into the dashboard.
        
        Args:
            ohlcv_data: Dictionary with OHLCV arrays
            trades_csv_path: Optional path to trades CSV file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Store data
            self.ohlcv_data = ohlcv_data
            
            # Load into chart
            if not self.chart_manager.load_ohlcv_data(ohlcv_data):
                logger.error("Failed to load OHLCV data into chart")
                return False
            
            # Load into other components
            if self.time_axis and 'datetime' in ohlcv_data:
                if hasattr(self.time_axis, 'set_datetime_data'):
                    self.time_axis.set_datetime_data(ohlcv_data.get('datetime'))
                elif hasattr(self.time_axis, 'load_datetime_data'):
                    self.time_axis.load_datetime_data(ohlcv_data.get('datetime'))
            
            if self.hover_info:
                if hasattr(self.hover_info, 'set_ohlcv_data'):
                    self.hover_info.set_ohlcv_data(ohlcv_data)
                elif hasattr(self.hover_info, 'load_ohlcv_data'):
                    self.hover_info.load_ohlcv_data(ohlcv_data)
            
            if self.indicators_panel:
                if hasattr(self.indicators_panel, 'load_data'):
                    self.indicators_panel.load_data(ohlcv_data)
            
            # Load trades if provided
            if trades_csv_path:
                self.load_trades(trades_csv_path)
            
            logger.info(f"Loaded {len(ohlcv_data['open'])} bars of data")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            return False
    
    def load_trades(self, csv_path: str) -> bool:
        """
        Load trades from a CSV file.
        
        Args:
            csv_path: Path to trades CSV file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Read CSV
            trades_df = pd.read_csv(csv_path)
            
            # Convert to list of dicts
            self.trades_data = trades_df.to_dict('records')
            
            # Load into chart
            self.chart_manager.load_trades(self.trades_data)
            
            # Load into trade list
            if self.trade_list:
                self.trade_list.load_trades(csv_path)
            
            # Load into equity curve
            if self.equity_curve:
                # Calculate equity curve from trades
                equity_data = self._calculate_equity_curve(self.trades_data)
                self.equity_curve.load_data(equity_data)
            
            logger.info(f"Loaded {len(self.trades_data)} trades")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load trades: {e}")
            return False
    
    def _calculate_equity_curve(self, trades: List[Dict]) -> Dict[str, np.ndarray]:
        """Calculate equity curve from trades."""
        if not trades:
            return {}
        
        # Simple equity calculation
        initial_capital = 100000
        equity = [initial_capital]
        
        for trade in trades:
            pnl = trade.get('pnl', 0)
            equity.append(equity[-1] + pnl)
        
        return {
            'equity': np.array(equity[1:]),  # Skip initial capital
            'datetime': np.array([t['exit_time'] for t in trades])
        }
    
    # ==================== Event Handlers ====================
    
    @pyqtSlot(int, int)
    def _on_viewport_changed(self, start: int, end: int):
        """Handle viewport change events."""
        # Update time axis
        if self.time_axis:
            if hasattr(self.time_axis, 'update_viewport'):
                self.time_axis.update_viewport(start, end)
            elif hasattr(self.time_axis, 'sync_viewport'):
                self.time_axis.sync_viewport(start, end)
        
        # Update equity curve viewport
        if self.equity_curve:
            if hasattr(self.equity_curve, 'set_viewport'):
                self.equity_curve.set_viewport(start, end)
            elif hasattr(self.equity_curve, 'sync_viewport'):
                self.equity_curve.sync_viewport(start, end)
        
        logger.debug(f"Viewport changed to [{start}:{end}]")
    
    def _on_trade_selected(self, trade_data, index):
        """Handle trade selection from trade list."""
        # Extract trade_id from trade_data
        if isinstance(trade_data, dict):
            trade_id = trade_data.get('trade_id', '')
        else:
            trade_id = str(trade_data)
        
        if trade_id and self.chart_manager:
            self.chart_manager.navigate_to_trade(trade_id)
            logger.info(f"Navigating to trade: {trade_id}")
    
    @pyqtSlot(int, dict)
    def _on_hover_update(self, bar_index: int, data: Dict):
        """Handle hover updates."""
        if self.hover_info:
            # HoverInfoWidget.update_display only takes bar_index
            self.hover_info.update_display(bar_index)
    
    @pyqtSlot(int, float)
    def _on_crosshair_update(self, bar_index: int, price: float):
        """Handle crosshair updates."""
        if self.crosshair:
            # Update crosshair position
            pass  # Implementation depends on crosshair widget
        
        if self.crosshair_info:
            # Update crosshair info
            if self.ohlcv_data and 0 <= bar_index < len(self.ohlcv_data['open']):
                data = {
                    'index': bar_index,
                    'price': price,
                    'datetime': self.ohlcv_data.get('datetime', [None])[bar_index]
                }
                # Update crosshair info widget
                pass  # Implementation depends on widget


def main():
    """Main entry point for the modular trading dashboard."""
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    
    # Create and show dashboard
    dashboard = ModularTradingDashboard()
    
    # Example: Load test data
    # Create synthetic OHLCV data for testing
    n_bars = 1000
    base_time = 1609459200_000_000_000  # 2021-01-01 in nanoseconds
    datetime_array = np.arange(n_bars, dtype=np.int64) * 60_000_000_000 + base_time
    
    prices = []
    price = 100.0
    for i in range(n_bars):
        price *= (1 + np.random.randn() * 0.002)
        prices.append(price)
    prices = np.array(prices)
    
    ohlcv_data = {
        'datetime': datetime_array,
        'open': prices * (1 + np.random.randn(n_bars) * 0.001),
        'high': prices * (1 + np.abs(np.random.randn(n_bars)) * 0.003),
        'low': prices * (1 - np.abs(np.random.randn(n_bars)) * 0.003),
        'close': prices,
        'volume': np.random.randint(1000, 10000, n_bars)
    }
    
    # Load data
    dashboard.load_data(ohlcv_data)
    
    # Show dashboard
    dashboard.show()
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()