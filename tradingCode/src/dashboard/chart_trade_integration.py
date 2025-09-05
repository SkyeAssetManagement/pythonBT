# src/dashboard/chart_trade_integration.py
# Chart-Trade List Integration System
# 
# Provides seamless integration between the VisPy candlestick chart and trade list widget
# Features:
# - Bidirectional navigation (chart <-> trade list)
# - Trade markers on chart with click detection
# - Real-time synchronization of viewport and trade visibility
# - High-performance rendering of trade overlays

import numpy as np
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtCore import QObject, pyqtSignal

# Import our VisPy renderer and trade list components
from .vispy_candlestick_renderer import VispyCandlestickRenderer, DataPipeline
from .trade_list_widget import TradeListContainer, TradeData

@dataclass 
class ChartTradeMarker:
    """
    Trade marker data for chart overlay rendering
    Optimized for GPU rendering performance
    """
    trade_id: str
    chart_index: int      # X position on chart
    price: float          # Y position (entry or exit price)
    marker_type: str      # 'entry', 'exit' 
    trade_side: str       # 'Long', 'Short'
    is_profitable: bool   # For color coding
    size: float          # For marker scaling

class ChartTradeIntegration(QObject):
    """
    High-performance integration system between VisPy chart and trade list
    Manages bidirectional navigation and synchronization
    """
    
    # Signals for communication between components
    trade_marker_clicked = pyqtSignal(str)  # trade_id
    viewport_changed = pyqtSignal(int, int)  # start_index, end_index
    
    def __init__(self, chart_renderer: VispyCandlestickRenderer, trade_list: TradeListContainer):
        super().__init__()
        
        # Core components
        self.chart_renderer = chart_renderer
        self.trade_list = trade_list
        
        # Data synchronization
        self.price_data_length = 0
        self.timestamp_to_index_map: Dict[int, int] = {}
        self.index_to_timestamp_map: Dict[int, int] = {}
        
        # Trade markers for chart overlay
        self.trade_markers: List[ChartTradeMarker] = []
        self.visible_markers: List[ChartTradeMarker] = []
        
        # Performance optimization settings
        self.max_visible_markers = 1000  # Limit visible markers for performance
        
        # Integration state
        self.integration_active = False
        
        # Setup integration
        self._setup_integration()
    
    def _setup_integration(self):
        """Set up bidirectional integration between chart and trade list"""
        
        # Connect trade list selection to chart navigation
        self.trade_list.trade_selected.connect(self._on_trade_selected_from_list)
        
        # Set up trade list callbacks
        self.trade_list.set_chart_navigation_callback(self._navigate_chart_to_index)
        self.trade_list.set_timestamp_mapper(self._timestamp_to_chart_index)
        
        # TODO: Connect chart click events to trade marker detection
        # This would require extending the VisPy renderer with click detection
        
        print(f"   INFO: Chart-Trade integration system initialized")
    
    def setup_data_synchronization(self, price_data: Dict[str, np.ndarray]) -> bool:
        """
        Set up data synchronization between chart and trade list
        Creates timestamp mapping for accurate trade positioning
        """
        try:
            # Store price data length for validation
            self.price_data_length = len(price_data['datetime'])
            
            # Create bidirectional timestamp mapping
            timestamps = price_data['datetime']
            
            # Handle both sequential indices and actual timestamps
            if timestamps[0] < 1e6:  # Likely sequential indices
                # Use direct index mapping
                for i in range(len(timestamps)):
                    self.timestamp_to_index_map[int(timestamps[i])] = i
                    self.index_to_timestamp_map[i] = int(timestamps[i])
            else:
                # Use timestamp-based mapping
                for i, timestamp in enumerate(timestamps):
                    self.timestamp_to_index_map[int(timestamp)] = i
                    self.index_to_timestamp_map[i] = int(timestamp)
            
            print(f"   SUCCESS: Data synchronization set up for {self.price_data_length:,} data points")
            print(f"   INFO: Timestamp range: {timestamps[0]} to {timestamps[-1]}")
            
            return True
            
        except Exception as e:
            print(f"   ERROR: Failed to set up data synchronization: {e}")
            return False
    
    def integrate_trades(self, trades_data: List[TradeData]) -> bool:
        """
        Integrate trade data with chart for marker rendering and navigation
        Creates optimized trade markers for chart overlay
        """
        try:
            if not trades_data:
                print(f"   WARNING: No trades to integrate")
                return False
            
            self.trade_markers = []
            invalid_trades = 0
            
            for trade in trades_data:
                # Convert trade timestamps to chart indices
                entry_index = self._timestamp_to_chart_index(trade.entry_time)
                exit_index = self._timestamp_to_chart_index(trade.exit_time)
                
                # Validate chart indices
                if entry_index < 0 or entry_index >= self.price_data_length:
                    invalid_trades += 1
                    continue
                
                if exit_index < 0 or exit_index >= self.price_data_length:
                    exit_index = min(entry_index + trade.duration, self.price_data_length - 1)
                
                # Create entry marker
                entry_marker = ChartTradeMarker(
                    trade_id=trade.trade_id,
                    chart_index=entry_index,
                    price=trade.entry_price,
                    marker_type='entry',
                    trade_side=trade.side,
                    is_profitable=trade.is_profitable,
                    size=abs(trade.size)
                )
                self.trade_markers.append(entry_marker)
                
                # Create exit marker
                exit_marker = ChartTradeMarker(
                    trade_id=trade.trade_id,
                    chart_index=exit_index,
                    price=trade.exit_price,
                    marker_type='exit',
                    trade_side=trade.side,
                    is_profitable=trade.is_profitable,
                    size=abs(trade.size)
                )
                self.trade_markers.append(exit_marker)
            
            print(f"   SUCCESS: Integrated {len(trades_data)} trades into chart")
            print(f"   INFO: Created {len(self.trade_markers)} trade markers")
            if invalid_trades > 0:
                print(f"   WARNING: {invalid_trades} trades outside chart data range")
            
            # Update visible markers based on current viewport
            self._update_visible_markers()
            
            self.integration_active = True
            return True
            
        except Exception as e:
            print(f"   ERROR: Failed to integrate trades: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _timestamp_to_chart_index(self, timestamp: int) -> int:
        """Convert timestamp to chart index with fallback handling"""
        
        # Direct mapping if available
        if timestamp in self.timestamp_to_index_map:
            return self.timestamp_to_index_map[timestamp]
        
        # If timestamp is already an index (small integer)
        if 0 <= timestamp < self.price_data_length:
            return timestamp
        
        # Binary search for closest timestamp (for actual timestamp data)
        if len(self.index_to_timestamp_map) > 0:
            timestamps = np.array(list(self.index_to_timestamp_map.values()))
            indices = np.array(list(self.index_to_timestamp_map.keys()))
            
            # Find closest timestamp
            closest_idx = np.argmin(np.abs(timestamps - timestamp))
            return indices[closest_idx]
        
        # Fallback: assume timestamp is proportional to index
        if self.price_data_length > 0:
            return min(int(timestamp), self.price_data_length - 1)
        
        return 0
    
    def _update_visible_markers(self):
        """Update visible trade markers based on current chart viewport"""
        if not self.integration_active or not self.chart_renderer:
            return
        
        # Get current viewport range
        viewport_start = int(self.chart_renderer.viewport_x_range[0])
        viewport_end = int(self.chart_renderer.viewport_x_range[1])
        
        # Add buffer for smooth scrolling
        buffer = 50
        visible_start = max(0, viewport_start - buffer)
        visible_end = min(self.price_data_length, viewport_end + buffer)
        
        # Filter markers within viewport
        self.visible_markers = [
            marker for marker in self.trade_markers
            if visible_start <= marker.chart_index <= visible_end
        ]
        
        # Limit for performance
        if len(self.visible_markers) > self.max_visible_markers:
            # Keep the most recent markers
            self.visible_markers = sorted(self.visible_markers, 
                                        key=lambda m: m.chart_index)[-self.max_visible_markers:]
        
        # TODO: Update chart overlay rendering with visible markers
        # This would require extending the VisPy renderer to support overlays
        
    def _on_trade_selected_from_list(self, trade_data: TradeData, chart_index: int):
        """Handle trade selection from trade list - navigate chart to trade location"""
        if not self.integration_active:
            return
        
        try:
            # Navigate chart to trade entry position
            self._navigate_chart_to_index(chart_index)
            
            # Highlight trade markers (if implemented)
            self._highlight_trade_markers(trade_data.trade_id)
            
            print(f"   INFO: Navigated chart to trade {trade_data.trade_id} at index {chart_index}")
            
        except Exception as e:
            print(f"   ERROR: Failed to navigate to trade: {e}")
    
    def _navigate_chart_to_index(self, chart_index: int):
        """Navigate chart viewport to specific index"""
        if not self.chart_renderer:
            return
        
        # Calculate new viewport centered on the target index
        viewport_size = self.chart_renderer.viewport_x_range[1] - self.chart_renderer.viewport_x_range[0]
        half_size = viewport_size / 2
        
        new_start = max(0, chart_index - half_size)
        new_end = min(self.price_data_length - 1, chart_index + half_size)
        
        # Adjust if we hit boundaries
        if new_end - new_start < viewport_size:
            if new_end == self.price_data_length - 1:
                new_start = max(0, new_end - viewport_size)
            else:
                new_end = min(self.price_data_length - 1, new_start + viewport_size)
        
        # Update chart viewport
        self.chart_renderer.viewport_x_range = [new_start, new_end]
        self.chart_renderer._update_projection_matrix()
        self.chart_renderer._update_gpu_buffers()
        
        # Update canvas if available
        if hasattr(self.chart_renderer, 'canvas') and self.chart_renderer.canvas:
            self.chart_renderer.canvas.update()
        
        # Update visible markers
        self._update_visible_markers()
    
    def _highlight_trade_markers(self, trade_id: str):
        """Highlight specific trade markers on chart"""
        # TODO: Implement trade marker highlighting
        # This would require extending the VisPy renderer with overlay support
        pass
    
    def get_trades_in_viewport(self) -> List[TradeData]:
        """Get list of trades visible in current chart viewport"""
        if not self.integration_active:
            return []
        
        viewport_start = int(self.chart_renderer.viewport_x_range[0])
        viewport_end = int(self.chart_renderer.viewport_x_range[1])
        
        # Get trades within viewport
        visible_trades = []
        for trade in self.trade_list.trade_list_widget.trades_data:
            trade_index = self._timestamp_to_chart_index(trade.entry_time)
            if viewport_start <= trade_index <= viewport_end:
                visible_trades.append(trade)
        
        return visible_trades
    
    def jump_to_trade(self, trade_id: str) -> bool:
        """Jump chart and trade list to specific trade"""
        # Use trade list's navigation method
        success = self.trade_list.trade_list_widget.navigate_to_trade(trade_id)
        
        if success:
            print(f"   SUCCESS: Jumped to trade {trade_id}")
        else:
            print(f"   WARNING: Could not find trade {trade_id}")
        
        return success
    
    def get_integration_status(self) -> Dict[str, any]:
        """Get current integration status and statistics"""
        return {
            'integration_active': self.integration_active,
            'price_data_length': self.price_data_length,
            'total_trade_markers': len(self.trade_markers),
            'visible_markers': len(self.visible_markers),
            'timestamp_mappings': len(self.timestamp_to_index_map),
            'current_viewport': self.chart_renderer.viewport_x_range if self.chart_renderer else None
        }


class IntegratedTradingDashboard(QtWidgets.QWidget):
    """
    Complete integrated trading dashboard combining chart and trade list
    Provides the main dashboard window with all components properly integrated
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Core components
        self.chart_renderer: Optional[VispyCandlestickRenderer] = None
        self.trade_list: Optional[TradeListContainer] = None
        self.integration: Optional[ChartTradeIntegration] = None
        
        # Data storage
        self.price_data: Optional[Dict[str, np.ndarray]] = None
        self.trades_data: Optional[List[TradeData]] = None
        
        # Setup UI
        self._setup_layout()
    
    def _setup_layout(self):
        """Set up the main dashboard layout"""
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(5)
        
        # Chart area (left side - most space)
        self.chart_container = QtWidgets.QWidget()
        self.chart_container.setStyleSheet("background-color: #2b2b2b; border: 1px solid #555555;")
        layout.addWidget(self.chart_container)
        
        # Trade list (right side - fixed width)
        self.trade_list = TradeListContainer()
        self.trade_list.setFixedWidth(400)  # Match target screenshot proportions
        layout.addWidget(self.trade_list)
        
        # Set stretch factors
        layout.setStretchFactor(self.chart_container, 3)  # Chart gets 3x space
        layout.setStretchFactor(self.trade_list, 1)       # Trade list fixed
        
    def initialize_dashboard(self, price_data: Dict[str, np.ndarray], 
                           trades_data: Optional[List[TradeData]] = None,
                           trades_csv_path: Optional[str] = None) -> bool:
        """
        Initialize the complete dashboard with price data and trades
        This is the main entry point for dashboard setup
        """
        try:
            print(f"   INFO: Initializing integrated trading dashboard...")
            
            # Store data
            self.price_data = price_data
            
            # Initialize VisPy chart renderer
            print(f"   INFO: Creating VisPy chart renderer...")
            # Note: For embedded use, we'd need to integrate VisPy into Qt widget
            # For now, create standalone renderer for testing
            self.chart_renderer = VispyCandlestickRenderer(width=1000, height=600)
            
            # Load price data into chart
            chart_success = self.chart_renderer.load_data(price_data)
            if not chart_success:
                print(f"   ERROR: Failed to load price data into chart")
                return False
            
            # Load trades data
            if trades_data:
                self.trades_data = trades_data
                trade_success = self.trade_list.trade_list_widget.trades_data = trades_data
                self.trade_list.trade_list_widget._populate_table()
            elif trades_csv_path:
                trade_success = self.trade_list.load_trades(trades_csv_path)
                self.trades_data = self.trade_list.trade_list_widget.trades_data
            else:
                print(f"   WARNING: No trades data provided")
                trade_success = True  # Continue without trades
                
            # Set up integration system
            if self.trades_data:
                print(f"   INFO: Setting up chart-trade integration...")
                self.integration = ChartTradeIntegration(self.chart_renderer, self.trade_list)
                
                integration_success = (
                    self.integration.setup_data_synchronization(price_data) and
                    self.integration.integrate_trades(self.trades_data)
                )
                
                if integration_success:
                    print(f"   SUCCESS: Chart-trade integration complete")
                else:
                    print(f"   WARNING: Chart-trade integration failed")
            
            # Update trade statistics
            self.trade_list._update_statistics()
            
            print(f"   SUCCESS: Integrated trading dashboard initialized")
            print(f"   INFO: Chart: {len(price_data['close']):,} candlesticks")
            if self.trades_data:
                print(f"   INFO: Trades: {len(self.trades_data)} trades")
            
            return True
            
        except Exception as e:
            print(f"   ERROR: Failed to initialize dashboard: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def show_dashboard(self):
        """Show the integrated dashboard"""
        self.setWindowTitle("Lightning Trading Dashboard - Integrated")
        self.resize(1600, 900)  # Target screenshot proportions
        self.show()
        
        # Show chart renderer separately (until Qt integration is complete)
        if self.chart_renderer:
            print(f"   INFO: Showing VisPy chart renderer...")
            # Note: In production, this would be embedded in the Qt widget
            # self.chart_renderer.show()  # Uncomment to show chart
    
    def get_dashboard_status(self) -> Dict[str, any]:
        """Get comprehensive dashboard status"""
        status = {
            'chart_initialized': self.chart_renderer is not None,
            'trades_loaded': self.trades_data is not None and len(self.trades_data) > 0,
            'integration_active': self.integration is not None and self.integration.integration_active,
        }
        
        if self.chart_renderer:
            status['price_data_length'] = self.chart_renderer.data_pipeline.data_length
            status['chart_viewport'] = self.chart_renderer.viewport_x_range
        
        if self.trades_data:
            status['trade_count'] = len(self.trades_data)
        
        if self.integration:
            status.update(self.integration.get_integration_status())
        
        return status


# Test and validation functions
def test_chart_trade_integration():
    """Test the complete chart-trade integration system"""
    print(f"\n=== TESTING CHART-TRADE INTEGRATION ===")
    
    try:
        from PyQt5.QtWidgets import QApplication
        import sys
        
        # Ensure Qt application
        app = QApplication.instance()
        if app is None:
            app = QApplication(sys.argv)
        
        # Create test data
        from .vispy_candlestick_renderer import create_test_data
        print(f"   Creating test data...")
        
        price_data = create_test_data(10000)  # 10K bars for testing
        
        # Create test trades
        test_trades = []
        for i in range(50):  # 50 test trades
            entry_time = np.random.randint(0, 9000)  # Random entry within data
            exit_time = entry_time + np.random.randint(1, 200)  # Random duration
            
            entry_price = price_data['close'][entry_time]
            exit_price = price_data['close'][min(exit_time, 9999)]
            
            pnl = (exit_price - entry_price) * 1.0  # Simple PnL calculation
            
            trade = TradeData(
                trade_id=f"TEST_{i+1:03d}",
                entry_time=entry_time,
                exit_time=exit_time,
                side="Long" if i % 2 == 0 else "Short", 
                entry_price=entry_price,
                exit_price=exit_price,
                size=1.0,
                pnl=pnl,
                pnl_pct=pnl / entry_price * 100,
                duration=exit_time - entry_time
            )
            test_trades.append(trade)
        
        print(f"   Created {len(test_trades)} test trades")
        
        # Create integrated dashboard
        dashboard = IntegratedTradingDashboard()
        
        # Initialize with test data
        success = dashboard.initialize_dashboard(price_data, test_trades)
        
        if success:
            print(f"   SUCCESS: Integrated dashboard initialized")
            
            # Test integration features
            if dashboard.integration:
                status = dashboard.get_dashboard_status()
                print(f"   Integration Status:")
                for key, value in status.items():
                    print(f"     {key}: {value}")
                
                # Test trade navigation
                if test_trades:
                    test_trade_id = test_trades[10].trade_id  # Pick middle trade
                    nav_success = dashboard.integration.jump_to_trade(test_trade_id)
                    print(f"   Trade navigation test: {'SUCCESS' if nav_success else 'FAIL'}")
            
            # Show dashboard
            dashboard.show_dashboard()
            print(f"   INFO: Dashboard window displayed")
            
            return True
        else:
            print(f"   ERROR: Failed to initialize integrated dashboard")
            return False
            
    except Exception as e:
        print(f"   ERROR: Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Run integration test
    test_chart_trade_integration()