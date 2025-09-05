#!/usr/bin/env python3
"""
Test dashboard with 491K bars to isolate large dataset performance issues
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

def test_large_dataset():
    """Test dashboard with simulated 491K bars like real ES data"""
    
    print("LARGE DATASET PERFORMANCE TEST: 491K bars")
    print("Simulating real ES data size")
    print("=" * 50)
    
    from PyQt5 import QtWidgets
    from src.dashboard.dashboard_manager import get_dashboard_manager
    from src.dashboard.data_structures import ChartDataBuffer
    
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    
    n_bars = 491349  # Same as real ES data
    
    print(f"Testing {n_bars:,} bars...")
    
    # Step 1: Create large dataset (should be fast)
    print("1. Creating large dataset...", end=" ", flush=True)
    start_time = time.time()
    
    timestamps = np.arange(n_bars, dtype=np.int64) * 60 * 1000000000
    base_price = 4000
    # Generate realistic price series with small random walk
    price_changes = np.random.randn(n_bars) * 0.5
    prices = base_price + np.cumsum(price_changes)
    
    data_time = time.time() - start_time
    print(f"{data_time:.3f}s")
    
    # Step 2: Create ChartDataBuffer (potential bottleneck)
    print("2. Creating ChartDataBuffer...", end=" ", flush=True)
    start_time = time.time()
    
    chart_data = ChartDataBuffer(
        timestamps=timestamps,
        open=prices + np.random.randn(n_bars) * 0.1,
        high=prices + np.abs(np.random.randn(n_bars)),
        low=prices - np.abs(np.random.randn(n_bars)),
        close=prices,
        volume=np.random.randint(1000, 10000, n_bars).astype(np.float64)
    )
    
    buffer_time = time.time() - start_time
    print(f"{buffer_time:.3f}s")
    
    # Step 3: Dashboard creation 
    print("3. Creating dashboard...", end=" ", flush=True)
    start_time = time.time()
    
    dashboard = get_dashboard_manager()
    dashboard.initialize_qt_app()
    dashboard.create_main_window()
    
    create_time = time.time() - start_time
    print(f"{create_time:.3f}s")
    
    # Step 4: Data loading (SUSPECTED BOTTLENECK)
    print("4. Loading data into dashboard...", end=" ", flush=True)
    start_time = time.time()
    
    dashboard.main_chart.set_data(chart_data)
    
    load_time = time.time() - start_time
    print(f"{load_time:.3f}s")
    
    # Step 5: Show dashboard
    print("5. Showing dashboard...", end=" ", flush=True)
    start_time = time.time()
    
    success = dashboard.show()
    
    show_time = time.time() - start_time
    print(f"{show_time:.3f}s")
    
    # Step 6: Minimal event processing
    print("6. Processing events...", end=" ", flush=True)
    start_time = time.time()
    
    for i in range(3):  # Minimal events
        app.processEvents()
    
    event_time = time.time() - start_time
    print(f"{event_time:.3f}s")
    
    # Clean up
    if hasattr(dashboard, 'main_window'):
        dashboard.main_window.close()
    
    total_time = data_time + buffer_time + create_time + load_time + show_time + event_time
    
    print(f"\nRESULTS for {n_bars:,} bars:")
    print(f"  Data creation: {data_time:.3f}s")
    print(f"  Buffer creation: {buffer_time:.3f}s ({'SLOW' if buffer_time > 2.0 else 'OK'})")
    print(f"  Dashboard create: {create_time:.3f}s ({'SLOW' if create_time > 2.0 else 'OK'})")
    print(f"  Data loading: {load_time:.3f}s ({'SLOW' if load_time > 5.0 else 'OK'})")
    print(f"  Dashboard show: {show_time:.3f}s ({'SLOW' if show_time > 10.0 else 'OK'})")
    print(f"  Event processing: {event_time:.3f}s ({'SLOW' if event_time > 10.0 else 'OK'})")
    print(f"  TOTAL: {total_time:.3f}s")
    print(f"  Performance: {n_bars/total_time:,.0f} bars/sec")
    
    # Find worst bottleneck
    steps = [
        ("Data creation", data_time),
        ("Buffer creation", buffer_time),
        ("Dashboard create", create_time),
        ("Data loading", load_time),
        ("Dashboard show", show_time),
        ("Event processing", event_time)
    ]
    
    worst_step = max(steps, key=lambda x: x[1])
    print(f"  Main bottleneck: {worst_step[0]} ({worst_step[1]:.3f}s)")
    
    if total_time < 10.0:
        print("  Status: GOOD - Under 10 seconds for large dataset")
    elif total_time < 30.0:
        print("  Status: ACCEPTABLE - Under 30 seconds")
    else:
        print("  Status: TOO SLOW - Need optimization")
    
    return total_time

if __name__ == "__main__":
    total_time = test_large_dataset()
    
    print(f"\n{'='*50}")
    if total_time < 10.0:
        print("EXCELLENT: Large dataset performance is good!")
    elif total_time < 30.0:
        print("ACCEPTABLE: Large dataset performance is okay")
    else:
        print("NEEDS WORK: Large dataset performance requires optimization")