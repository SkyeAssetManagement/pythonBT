#!/usr/bin/env python3
"""
Integration tests for the Unified Trading GUI
Tests the integration between OMtree and ABtoPython components
"""

import pytest
import tkinter as tk
from tkinter import ttk
import pandas as pd
import numpy as np
from datetime import datetime
import sys
import os
import threading
import time
from unittest.mock import MagicMock, patch

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import GUI components
from unified_gui import UnifiedTradingGUI
from src.feature_flags import get_feature_flags
from src.trading.data.trade_data import TradeData, TradeCollection


class TestUnifiedGUI:
    """Test Unified Trading GUI integration"""
    
    @pytest.fixture
    def setup_flags(self):
        """Enable necessary feature flags for testing"""
        flags = get_feature_flags()
        flags.enable('use_unified_gui')
        flags.enable('show_trade_visualization_tab')
        flags.enable('show_pyqtgraph_charts')
        flags.enable('show_vectorbt_import')
        flags.enable('unified_data_pipeline')
        return flags
    
    @pytest.fixture
    def gui_app(self, setup_flags):
        """Create GUI application for testing"""
        app = UnifiedTradingGUI()
        yield app
        app.quit()
        app.destroy()
    
    def test_gui_creation(self, gui_app):
        """Test that GUI is created successfully"""
        assert gui_app is not None
        assert isinstance(gui_app, tk.Tk)
        assert gui_app.title() == "Unified Trading System - OMtree + Trade Visualization"
    
    def test_notebook_tabs(self, gui_app):
        """Test that all expected tabs are created"""
        notebook = gui_app.notebook
        assert isinstance(notebook, ttk.Notebook)
        
        # Get tab names
        tab_count = notebook.index('end')
        tab_names = [notebook.tab(i, 'text') for i in range(tab_count)]
        
        # Check for expected tabs
        expected_tabs = ['Configuration', 'Walk-Forward', 'Performance']
        for tab in expected_tabs:
            assert tab in tab_names
        
        # Check for feature-flagged tabs
        flags = gui_app.flags
        if flags.is_enabled('show_trade_visualization_tab'):
            assert 'Trade Visualization' in tab_names
        if flags.is_enabled('show_pyqtgraph_charts'):
            assert 'Advanced Charts' in tab_names
        if flags.is_enabled('show_vectorbt_import'):
            assert 'VectorBT Import' in tab_names
    
    def test_configuration_tab_elements(self, gui_app):
        """Test Configuration tab has required elements"""
        # Check for CSV path entry
        assert hasattr(gui_app, 'csv_path_var')
        assert gui_app.csv_path_var.get() == 'data/sample_trading_data.csv'
        
        # Check for buttons
        assert hasattr(gui_app, 'apply_config_button')
    
    def test_trade_visualization_tab(self, gui_app):
        """Test Trade Visualization tab functionality"""
        if not gui_app.flags.is_enabled('show_trade_visualization_tab'):
            pytest.skip("Trade visualization tab not enabled")
        
        # Check trade tree exists
        assert hasattr(gui_app, 'trade_tree')
        assert isinstance(gui_app.trade_tree, ttk.Treeview)
        
        # Check columns
        columns = gui_app.trade_tree['columns']
        expected_columns = ['ID', 'Time', 'Type', 'Price', 'Size', 'PnL']
        for col in expected_columns:
            assert col in columns
    
    def test_vectorbt_import_tab(self, gui_app):
        """Test VectorBT Import tab functionality"""
        if not gui_app.flags.is_enabled('show_vectorbt_import'):
            pytest.skip("VectorBT import tab not enabled")
        
        # Check import type selector
        assert hasattr(gui_app, 'import_type_var')
        assert gui_app.import_type_var.get() == 'trades'
        
        # Check status text area
        assert hasattr(gui_app, 'vbt_status_text')
    
    def test_load_trades_functionality(self, gui_app):
        """Test loading trades into the GUI"""
        # Create sample trades
        trades = [
            TradeData(
                trade_id=1,
                timestamp=pd.Timestamp('2024-01-01 09:00:00'),
                bar_index=100,
                trade_type='BUY',
                price=100.0,
                size=10.0,
                pnl=0.0,
                symbol='TEST'
            ),
            TradeData(
                trade_id=2,
                timestamp=pd.Timestamp('2024-01-01 10:00:00'),
                bar_index=101,
                trade_type='SELL',
                price=101.0,
                size=10.0,
                pnl=10.0,
                symbol='TEST'
            )
        ]
        
        gui_app.trade_collection = TradeCollection(trades)
        
        # Update trade display
        gui_app.update_trade_display()
        
        # Check trades are displayed
        children = gui_app.trade_tree.get_children()
        assert len(children) == 2
        
        # Check first trade details
        first_item = gui_app.trade_tree.item(children[0])
        values = first_item['values']
        assert values[0] == 1  # ID
        assert values[2] == 'BUY'  # Type
        assert values[3] == 100.0  # Price
    
    def test_status_bar_updates(self, gui_app):
        """Test status bar functionality"""
        assert hasattr(gui_app, 'status_label')
        
        # Test status update
        test_message = "Test status message"
        gui_app.update_status(test_message)
        assert gui_app.status_label.cget('text') == test_message
    
    @patch('tkinter.filedialog.askopenfilename')
    def test_load_configuration_dialog(self, mock_dialog, gui_app):
        """Test configuration loading dialog"""
        mock_dialog.return_value = 'test_config.ini'
        
        gui_app.load_configuration()
        mock_dialog.assert_called_once()
        
        # Check status was updated
        status = gui_app.status_label.cget('text')
        assert 'Loading configuration' in status
    
    @patch('tkinter.filedialog.asksaveasfilename')
    def test_save_configuration_dialog(self, mock_dialog, gui_app):
        """Test configuration saving dialog"""
        mock_dialog.return_value = 'test_config.ini'
        
        gui_app.save_configuration()
        mock_dialog.assert_called_once()
        
        # Check status was updated
        status = gui_app.status_label.cget('text')
        assert 'Saving configuration' in status
    
    def test_menu_structure(self, gui_app):
        """Test menu bar structure"""
        # Get menu bar
        menubar = gui_app.menubar
        
        # Check File menu exists
        assert hasattr(gui_app, 'file_menu')
        
        # Check Tools menu exists  
        assert hasattr(gui_app, 'tools_menu')
        
        # Check Help menu exists
        assert hasattr(gui_app, 'help_menu')
    
    @patch('tkinter.messagebox.showinfo')
    def test_about_dialog(self, mock_msgbox, gui_app):
        """Test about dialog"""
        gui_app.show_about()
        mock_msgbox.assert_called_once()
        
        # Check message content
        args = mock_msgbox.call_args[0]
        assert args[0] == "About"
        assert "Unified Trading System" in args[1]
    
    def test_feature_flag_integration(self, gui_app):
        """Test feature flag system integration"""
        flags = gui_app.flags
        
        # Test disabling a feature
        flags.disable('show_trade_visualization_tab')
        
        # Create new GUI instance
        new_app = UnifiedTradingGUI()
        
        # Check tab is not created
        tab_count = new_app.notebook.index('end')
        tab_names = [new_app.notebook.tab(i, 'text') for i in range(tab_count)]
        
        # Trade Visualization should not be present
        assert 'Trade Visualization' not in tab_names
        
        # Clean up
        new_app.quit()
        new_app.destroy()
        
        # Re-enable for other tests
        flags.enable('show_trade_visualization_tab')


class TestDataFlowIntegration:
    """Test data flow between components"""
    
    @pytest.fixture
    def setup_environment(self):
        """Setup test environment"""
        flags = get_feature_flags()
        flags.enable('unified_data_pipeline')
        flags.enable('use_new_trade_data')
        return flags
    
    def test_omtree_to_trade_visualization(self, setup_environment):
        """Test data flow from OMtree to trade visualization"""
        from src.trading.data.unified_pipeline import get_data_adapter, OMtreeAdapter
        
        # Create OMtree data
        omtree_df = pd.DataFrame({
            'Date': ['2024-01-01', '2024-01-02'],
            'Time': ['09:00:00', '10:00:00'],
            'Close': [100.0, 101.0],
            'PIR_0-1hr': [0.5, -0.3],
            'Ret_fwd6hr': [0.8, -0.2],
            'Prediction': [0.7, -0.1]
        })
        
        config = {
            'date_column': 'Date',
            'time_column': 'Time',
            'target_column': 'Ret_fwd6hr',
            'feature_columns': ['PIR_0-1hr'],
            'symbol': 'TEST'
        }
        
        # Convert using pipeline
        adapter = get_data_adapter()
        omtree_adapter = OMtreeAdapter(adapter)
        
        # Process data
        processed_df = omtree_adapter.prepare_data_for_model(omtree_df, config)
        
        assert processed_df is not None
        assert len(processed_df) == 2
        assert 'Close' in processed_df.columns
    
    def test_trades_to_gui_display(self, setup_environment):
        """Test trades display in GUI"""
        from src.trading.data.unified_pipeline import get_data_adapter, ABtoPythonAdapter
        
        # Create trades
        trades = [
            TradeData(
                trade_id=1,
                timestamp=pd.Timestamp('2024-01-01 09:00:00'),
                bar_index=100,
                trade_type='BUY',
                price=100.0,
                size=10.0,
                symbol='TEST'
            )
        ]
        collection = TradeCollection(trades)
        
        # Convert for visualization
        adapter = get_data_adapter()
        ab_adapter = ABtoPythonAdapter(adapter)
        viz_data = ab_adapter.export_to_visualization(collection)
        
        assert 'timestamps' in viz_data
        assert 'prices' in viz_data
        assert len(viz_data['timestamps']) == 1
        assert viz_data['prices'][0] == 100.0


class TestPerformanceIntegration:
    """Test performance with larger datasets"""
    
    def test_large_trade_collection(self):
        """Test handling large number of trades"""
        # Create 10000 trades
        trades = []
        for i in range(10000):
            trade = TradeData(
                trade_id=i,
                timestamp=pd.Timestamp('2024-01-01') + pd.Timedelta(minutes=i),
                bar_index=i,
                trade_type='BUY' if i % 2 == 0 else 'SELL',
                price=100 + np.random.randn(),
                size=10.0
            )
            trades.append(trade)
        
        collection = TradeCollection(trades)
        
        # Test collection operations
        assert len(collection) == 10000
        
        # Test get_by_id performance
        start = time.time()
        trade = collection.get_by_id(5000)
        elapsed = time.time() - start
        
        assert trade is not None
        assert trade.trade_id == 5000
        assert elapsed < 0.01  # Should be very fast with dict lookup
    
    def test_pipeline_conversion_performance(self):
        """Test pipeline conversion performance"""
        from src.trading.data.unified_pipeline import get_data_adapter
        
        # Create large DataFrame
        n_rows = 10000
        df = pd.DataFrame({
            'Date': pd.date_range('2024-01-01', periods=n_rows, freq='1min').date,
            'Time': pd.date_range('2024-01-01', periods=n_rows, freq='1min').time,
            'Close': np.random.randn(n_rows) + 100,
            'PIR_0-1hr': np.random.randn(n_rows),
            'Ret_fwd6hr': np.random.randn(n_rows)
        })
        
        config = {
            'date_column': 'Date',
            'time_column': 'Time',
            'target_column': 'Ret_fwd6hr',
            'feature_columns': ['PIR_0-1hr'],
            'symbol': 'TEST'
        }
        
        # Enable feature flag
        flags = get_feature_flags()
        flags.enable('unified_data_pipeline')
        
        # Test conversion performance
        adapter = get_data_adapter()
        
        start = time.time()
        unified = adapter.omtree_to_unified(df, config)
        elapsed = time.time() - start
        
        assert len(unified) == n_rows
        assert elapsed < 5.0  # Should process 10k rows in under 5 seconds


if __name__ == "__main__":
    pytest.main([__file__, "-v"])