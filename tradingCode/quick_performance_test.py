#!/usr/bin/env python3
"""
Quick performance test - focus on 1K bars performance issue
"""

import sys
import os
import time
from pathlib import Path
import numpy as np

# Add src to path
src_path = Path(__file__).parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

def test_small_dataset_performance():
    """Test 1K bars specifically - should be under 1s"""
    
    print("QUICK PERFORMANCE TEST: 1K bars")
    print("Target: Under 1 second")
    print("=" * 40)
    
    from PyQt5 import QtWidgets
    from src.dashboard.dashboard_manager import get_dashboard_manager
    from src.dashboard.data_structures import ChartDataBuffer
    
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    
    n_bars = 1000
    
    print(f"Testing {n_bars:,} bars...")
    
    # Step 1: Create data (should be instant)
    start_time = time.time()
    timestamps = np.arange(n_bars, dtype=np.int64) * 60 * 1000000000
    prices = 4000 + np.random.randn(n_bars) * 10
    
    chart_data = ChartDataBuffer(
        timestamps=timestamps,
        open=prices + np.random.randn(n_bars) * 0.1,
        high=prices + np.abs(np.random.randn(n_bars)),
        low=prices - np.abs(np.random.randn(n_bars)),
        close=prices,
        volume=np.random.randint(1000, 10000, n_bars)
    )
    data_time = time.time() - start_time
    print(f"1. Data creation: {data_time:.3f}s")
    
    # Step 2: Dashboard creation (should be fast)
    start_time = time.time()
    dashboard = get_dashboard_manager()
    dashboard.initialize_qt_app()
    dashboard.create_main_window()
    create_time = time.time() - start_time
    print(f"2. Dashboard creation: {create_time:.3f}s")
    
    # Step 3: Data loading (SUSPECTED BOTTLENECK)
    start_time = time.time()
    dashboard.main_chart.set_data(chart_data)
    load_time = time.time() - start_time
    print(f"3. Data loading: {load_time:.3f}s")
    
    # Step 4: Show dashboard (SUSPECTED BOTTLENECK)
    start_time = time.time()
    success = dashboard.show()
    show_time = time.time() - start_time
    print(f"4. Dashboard show: {show_time:.3f}s")
    
    # Step 5: Minimal event processing
    start_time = time.time()
    app.processEvents()
    event_time = time.time() - start_time
    print(f"5. Event processing: {event_time:.3f}s")
    
    # Clean up
    if hasattr(dashboard, 'main_window'):
        dashboard.main_window.close()
    
    total_time = data_time + create_time + load_time + show_time + event_time
    
    print(f"\nRESULTS:")
    print(f"  Total time: {total_time:.3f}s")
    print(f"  Performance: {n_bars/total_time:,.0f} bars/sec")
    
    if total_time < 1.0:
        print("  Status: EXCELLENT - Under 1 second!")
    elif total_time < 2.0:
        print("  Status: GOOD - Close to target")
    else:
        print("  Status: SLOW - Needs optimization")
    
    # Identify worst step
    steps = [
        ("Data creation", data_time),
        ("Dashboard creation", create_time),
        ("Data loading", load_time),
        ("Dashboard show", show_time),
        ("Event processing", event_time)
    ]
    
    worst_step = max(steps, key=lambda x: x[1])
    print(f"  Bottleneck: {worst_step[0]} ({worst_step[1]:.3f}s)")
    
    return total_time < 1.0

if __name__ == "__main__":
    success = test_small_dataset_performance()
    if success:
        print("\n[OK] Dashboard performance target achieved!")
    else:
        print("\n[WARNING] Dashboard still too slow - needs more optimization")