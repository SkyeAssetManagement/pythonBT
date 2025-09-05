#!/usr/bin/env python3
"""
Precision Detection Diagnostic Tool
Tests precision detection and application across all dashboard components
"""

import sys
import numpy as np
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from PyQt5 import QtWidgets, QtCore
from src.dashboard.dashboard_manager import DashboardManager
from src.dashboard.chart_widget import TradingChart

def test_precision_detection():
    """Test precision detection with various data types"""
    
    print("=== PRECISION DETECTION DIAGNOSTIC ===")
    
    # Test data samples
    test_cases = {
        "AD_Forex_5dp": np.array([0.65432, 0.65445, 0.65401, 0.65467, 0.65423]),
        "EUR_Forex_5dp": np.array([1.08432, 1.08445, 1.08401, 1.08467, 1.08423]), 
        "Stock_2dp": np.array([150.25, 150.30, 150.15, 150.40]),
        "Crypto_8dp": np.array([0.00012345, 0.00012367, 0.00012323]),
        "Whole_Numbers": np.array([100, 101, 102, 103])
    }
    
    dm = DashboardManager()
    
    for name, data in test_cases.items():
        print(f"\n--- Testing {name} ---")
        print(f"Sample data: {data[:3]}")
        
        detected = dm._detect_precision(data)
        print(f"Detected precision: {detected}")
        
        # Test full update process
        test_price_data = {'close': data}
        dm._update_precision_from_data(test_price_data)
        print(f"Final dashboard precision: {dm.price_precision}")
        
        # Test formatting
        sample_value = data[0]
        formatted = f"{sample_value:.{dm.price_precision}f}"
        print(f"Format test: {sample_value} -> {formatted}")

def test_chart_axes_precision():
    """Test chart axis precision formatting"""
    
    print("\n=== CHART AXES PRECISION TEST ===")
    
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(sys.argv)
    
    chart = TradingChart()
    
    print(f"Initial left axis precision: {chart.left_price_axis.precision}")
    print(f"Initial right axis precision: {chart.right_price_axis.precision}")
    
    # Test setting precision
    chart.set_precision(5)
    
    print(f"After setting 5dp - Left axis: {chart.left_price_axis.precision}")
    print(f"After setting 5dp - Right axis: {chart.right_price_axis.precision}")
    
    # Test tick formatting
    test_values = [0.65432, 0.65467, 0.65401]
    formatted_ticks = chart.left_price_axis.tickStrings(test_values, 1.0, 0.001)
    
    print(f"Test values: {test_values}")
    print(f"Formatted ticks: {formatted_ticks}")
    
    app.quit()

if __name__ == "__main__":
    test_precision_detection()
    test_chart_axes_precision()