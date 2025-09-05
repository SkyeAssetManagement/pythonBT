#!/usr/bin/env python3
"""
Test the import fixes for the dashboard
"""

import sys
import os
from pathlib import Path

# Add src directory to Python path FIRST (like main.py now does)
src_path = Path('.').absolute() / 'src'
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

print('Testing import fixes...')

try:
    print('Step 1: Import dashboard manager')
    from src.dashboard.dashboard_manager import launch_dashboard
    print('SUCCESS')
    
    print('Step 2: Import chart widget with fixed imports')
    from dashboard.chart_widget import TradingChart
    print('SUCCESS')
    
    print('Step 3: Import data structures')
    from dashboard.data_structures import ChartDataBuffer
    print('SUCCESS')
    
    print('Step 4: Test creating chart widget and data')
    import numpy as np
    
    chart = TradingChart()
    print('Chart widget created')
    
    # Create minimal data
    n = 5
    data_buffer = ChartDataBuffer(
        timestamps=np.arange(1609459200000000000, 1609459200000000000 + n * 60 * 1000000000, 60 * 1000000000),
        open=np.ones(n) * 100.0,
        high=np.ones(n) * 101.0,
        low=np.ones(n) * 99.0,
        close=np.ones(n) * 100.5,
        volume=np.ones(n, dtype=np.int64) * 1000
    )
    
    print('Data buffer created')
    
    # This is where the import error typically occurs - test without GUI
    print('Step 5: Testing data setting (this triggers candlestick generation)')
    chart.set_data(data_buffer)
    print('SUCCESS: Data set on chart - no import errors!')
    
    print('\n=== ALL IMPORT FIXES WORKING ===')
    print('The "No module named dashboard" error should be resolved!')
    
except Exception as e:
    print(f'ERROR: {e}')
    import traceback
    traceback.print_exc()