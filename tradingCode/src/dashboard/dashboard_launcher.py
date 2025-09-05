# src/dashboard/dashboard_launcher.py
# Main Dashboard Launcher for Integration with main.py
# 
# Provides the entry point for launching the interactive VisPy dashboard
# Integrates seamlessly with the existing trading system

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
import sys
from pathlib import Path

# Import our dashboard components
from .vispy_candlestick_renderer import VispyCandlestickRenderer
from .trade_list_widget import TradeListContainer, TradeData  
from .chart_trade_integration import IntegratedTradingDashboard

def launch_interactive_dashboard(price_data: Dict[str, np.ndarray], 
                                trade_data: Optional[pd.DataFrame] = None,
                                portfolio_data: Optional[Dict] = None,
                                show_chart: bool = True,
                                show_trade_list: bool = True) -> bool:
    """
    Launch the complete interactive trading dashboard
    This is the main entry point called from main.py
    
    Args:
        price_data: OHLCV data from the trading system
        trade_data: Trade list DataFrame from VectorBT
        portfolio_data: Portfolio data including equity curve
        show_chart: Whether to show the VisPy chart
        show_trade_list: Whether to show the trade list panel
        
    Returns:
        True if dashboard launched successfully
    """
    
    print(f"\n=== LAUNCHING INTERACTIVE TRADING DASHBOARD ===")
    
    try:
        from PyQt5.QtWidgets import QApplication, QHBoxLayout, QWidget
        from PyQt5.QtCore import Qt
        
        # Ensure Qt application exists
        app = QApplication.instance()
        if app is None:
            app = QApplication(sys.argv)
        
        # Create main dashboard window
        main_window = QWidget()
        main_window.setWindowTitle("Lightning Trading Dashboard - Interactive")
        main_window.setWindowFlags(Qt.Window)
        main_window.resize(1800, 1000)  # Large window for full experience
        
        # Create horizontal layout
        layout = QHBoxLayout(main_window)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)
        
        # Launch VisPy chart in separate window if requested
        chart_renderer = None
        if show_chart:
            print(f"   Creating high-performance VisPy chart...")
            
            # Create VisPy renderer with optimal size
            chart_renderer = VispyCandlestickRenderer(width=1400, height=900)
            
            # Load price data
            chart_success = chart_renderer.load_data(price_data)
            
            if chart_success:
                print(f"   SUCCESS: VisPy chart loaded with {len(price_data['close']):,} candlesticks")
                print(f"   INFO: Chart will open in separate window")
            else:
                print(f"   ERROR: Failed to load chart data")
                return False
        
        # Create trade list panel if requested
        trade_list_container = None
        if show_trade_list:
            print(f"   Creating trade list panel...")
            
            trade_list_container = TradeListContainer()
            trade_list_container.setMinimumWidth(500)
            trade_list_container.setMaximumWidth(600)
            
            # Load trade data if provided
            if trade_data is not None and not trade_data.empty:
                trade_success = trade_list_container.load_trades(trade_data)
                
                if trade_success:
                    num_trades = len(trade_list_container.trade_list_widget.trades_data)
                    print(f"   SUCCESS: Trade list loaded with {num_trades:,} trades")
                else:
                    print(f"   WARNING: Failed to load trade data")
            else:
                print(f"   INFO: No trade data provided - trade list will be empty")
        
        # Create integrated dashboard if both components are available
        if chart_renderer and trade_list_container and trade_data is not None:
            print(f"   Setting up chart-trade integration...")
            
            try:
                from .chart_trade_integration import ChartTradeIntegration
                
                integration = ChartTradeIntegration(chart_renderer, trade_list_container)
                
                # Set up data synchronization
                sync_success = integration.setup_data_synchronization(price_data)
                
                if sync_success and trade_list_container.trade_list_widget.trades_data:
                    # Integrate trades with chart
                    integrate_success = integration.integrate_trades(
                        trade_list_container.trade_list_widget.trades_data
                    )
                    
                    if integrate_success:
                        print(f"   SUCCESS: Chart-trade integration active")
                        
                        # Connect trade selection to chart navigation
                        def on_trade_selected(trade_data, chart_index):
                            print(f"   Navigating to trade {trade_data.trade_id} at chart index {chart_index}")
                            # The integration system handles the actual navigation
                        
                        trade_list_container.trade_selected.connect(on_trade_selected)
                    else:
                        print(f"   WARNING: Chart-trade integration failed")
                else:
                    print(f"   WARNING: Could not set up data synchronization")
                    
            except Exception as e:
                print(f"   WARNING: Integration setup failed: {e}")
        
        # Set up main window layout
        if trade_list_container:
            layout.addWidget(trade_list_container)
        
        # Add instructions panel
        instructions = create_instructions_panel()
        if not trade_list_container:  # Only show instructions if no trade list
            layout.addWidget(instructions)
        
        # Show main window
        main_window.show()
        main_window.raise_()
        main_window.activateWindow()
        
        # Show VisPy chart in separate window
        if chart_renderer:
            print(f"\n   LAUNCHING VISPY CHART...")
            print(f"   Controls:")
            print(f"     - Mouse wheel: Zoom in/out")
            print(f"     - Left mouse drag: Pan around")
            print(f"     - 'R' key: Reset to recent view")
            print(f"     - 'S' key: Take screenshot")
            print(f"     - 'Q' key: Quit chart")
            
            # Start VisPy chart (this will block until chart is closed)
            def show_chart():
                chart_renderer.show()
            
            # Run chart in background thread so main window stays responsive
            import threading
            chart_thread = threading.Thread(target=show_chart, daemon=True)
            chart_thread.start()
        
        print(f"\n[OK] INTERACTIVE DASHBOARD LAUNCHED!")
        print(f"   [CHART] Chart: VisPy high-performance renderer")
        if trade_list_container:
            print(f"   📋 Trade list: {len(trade_list_container.trade_list_widget.trades_data) if hasattr(trade_list_container.trade_list_widget, 'trades_data') else 0} trades")
        print(f"   🖱️  Click on trades to navigate chart")
        print(f"   🔄 Pan and zoom for detailed analysis")
        print(f"\n   Close this window when finished analyzing results")
        
        # Run Qt event loop (blocks until window closed)
        app.exec_()
        
        print(f"   Dashboard closed by user")
        return True
        
    except Exception as e:
        print(f"   ERROR: Dashboard launch failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def create_instructions_panel():
    """Create instructions panel for the dashboard"""
    from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel
    from PyQt5.QtCore import Qt
    
    panel = QWidget()
    panel.setMaximumWidth(400)
    
    layout = QVBoxLayout(panel)
    layout.setContentsMargins(20, 20, 20, 20)
    
    title = QLabel("Lightning Trading Dashboard")
    title.setStyleSheet("""
        QLabel {
            color: white;
            font-size: 18pt;
            font-weight: bold;
            padding: 10px;
        }
    """)
    title.setAlignment(Qt.AlignCenter)
    layout.addWidget(title)
    
    instructions = QLabel("""
    [LAUNCH] High-Performance Trading Dashboard
    
    [CHART] VisPy Chart Features:
    • 7M+ candlesticks support
    • GPU-accelerated rendering
    • Smooth pan and zoom
    • Viewport optimization
    
    📋 Trade List Features:
    • VectorBT integration
    • Clickable navigation
    • Real-time sync
    • Performance statistics
    
    [TOOLS] Controls:
    • Mouse wheel: Zoom
    • Drag: Pan chart
    • Click trades: Navigate
    • 'S' key: Screenshot
    • 'R' key: Reset view
    
    [GROWTH] The chart window will open
    separately for optimal performance
    """)
    
    instructions.setStyleSheet("""
        QLabel {
            color: #cccccc;
            font-size: 10pt;
            line-height: 1.4;
            padding: 20px;
            background-color: #333333;
            border: 1px solid #555555;
            border-radius: 5px;
        }
    """)
    instructions.setWordWrap(True)
    layout.addWidget(instructions)
    
    panel.setStyleSheet("background-color: #2b2b2b;")
    
    return panel

def launch_dashboard_from_main_results(results_dir: str) -> bool:
    """
    Launch dashboard using results from main.py backtest
    Automatically loads data from the results directory
    """
    
    print(f"\n=== LAUNCHING DASHBOARD FROM RESULTS ===")
    print(f"Results directory: {results_dir}")
    
    try:
        from pathlib import Path
        import pandas as pd
        
        results_path = Path(results_dir)
        
        # Look for trade list CSV
        tradelist_csv = results_path / "tradelist.csv"
        equity_csv = results_path / "equity_curve.csv"
        
        if not tradelist_csv.exists():
            print(f"   ERROR: No tradelist.csv found in {results_dir}")
            print(f"   INFO: Run a backtest first to generate trade data")
            return False
        
        # Load trade data
        print(f"   Loading trade data from {tradelist_csv}")
        trade_data = pd.read_csv(tradelist_csv)
        
        print(f"   SUCCESS: Loaded {len(trade_data)} trades")
        
        # For price data, we'd need to recreate it or save it from main.py
        # For now, create synthetic data matching the trade timeframe
        print(f"   Creating price data for chart...")
        
        # Estimate data length from trade timestamps
        if 'EntryTime' in trade_data.columns:
            max_time = max(trade_data['EntryTime'].max(), trade_data.get('ExitTime', trade_data['EntryTime']).max())
            data_length = int(max_time) + 1000  # Add buffer
        else:
            data_length = 10000  # Default
        
        # Create synthetic price data matching trade timeframe
        from .vispy_candlestick_renderer import create_test_data
        price_data = create_test_data(data_length)
        
        print(f"   Created {data_length:,} bars of price data")
        
        # Load portfolio data if available
        portfolio_data = None
        if equity_csv.exists():
            print(f"   Loading equity curve from {equity_csv}")
            equity_df = pd.read_csv(equity_csv)
            portfolio_data = {
                'equity_curve': equity_df['equity'].values if 'equity' in equity_df.columns else None
            }
        
        # Launch interactive dashboard
        return launch_interactive_dashboard(
            price_data=price_data,
            trade_data=trade_data, 
            portfolio_data=portfolio_data
        )
        
    except Exception as e:
        print(f"   ERROR: Failed to launch dashboard from results: {e}")
        import traceback
        traceback.print_exc()
        return False

# Test function to demonstrate the dashboard
def test_interactive_dashboard():
    """Test the interactive dashboard with sample data"""
    
    print(f"=== INTERACTIVE DASHBOARD TEST ===")
    
    # Create realistic test data
    from .vispy_candlestick_renderer import create_test_data
    
    print(f"Creating test data...")
    price_data = create_test_data(50000)  # 50K bars for good interactivity
    
    # Create test trades
    test_trades = []
    for i in range(200):  # 200 test trades
        entry_time = np.random.randint(0, 45000)
        exit_time = entry_time + np.random.randint(10, 500)
        
        entry_price = price_data['close'][entry_time]
        exit_price = price_data['close'][min(exit_time, 49999)]
        
        pnl = (exit_price - entry_price) * 1.0
        
        test_trades.append({
            'EntryTime': entry_time,
            'ExitTime': exit_time,
            'Direction': 'Long' if i % 3 != 0 else 'Short',
            'Avg Entry Price': entry_price,
            'Avg Exit Price': exit_price, 
            'Size': np.random.uniform(0.5, 2.0),
            'PnL': pnl
        })
    
    trade_df = pd.DataFrame(test_trades)
    
    print(f"Launching interactive dashboard...")
    return launch_interactive_dashboard(price_data, trade_df)

if __name__ == "__main__":
    # Run interactive test
    test_interactive_dashboard()