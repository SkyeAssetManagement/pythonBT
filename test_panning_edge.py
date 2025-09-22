"""
Test panning to the edge of data
"""

import sys
import os
from PyQt5.QtWidgets import QApplication

sys.path.append('src/trading')
sys.path.append('src/trading/visualization')

def test_panning():
    """Test panning to various positions including the edge"""
    print("="*60)
    print("TESTING PANNING TO EDGE")
    print("="*60)

    # Create app
    app = QApplication(sys.argv)

    # Import chart class
    from pyqtgraph_range_bars_final import RangeBarChartFinal

    # Create chart instance
    print("\nCreating chart...")
    chart = RangeBarChartFinal()

    print(f"\nTotal bars loaded: {chart.total_bars}")

    # Test panning to different positions
    test_positions = [
        (0, 500, "Start"),
        (250, 750, "Near start"),
        (122109, 122609, "End of data - should see bars"),
        (122500, 122609, "Very end"),
        (500, 1000, "Past initial 500"),
        (1000, 1500, "Further out"),
    ]

    for start, end, desc in test_positions:
        print(f"\n{'-'*40}")
        print(f"Testing: {desc} (bars {start}-{end})")
        print(f"{'-'*40}")

        # This should trigger render_range
        chart.render_range(start, end)

        print("Check debug output above:")
        print("- Should see [RENDER_RANGE] with correct indices")
        print("- Should NOT see 'Skipping - already rendering'")
        print("- is_rendering flag should reset to False")

    # Test the on_x_range_changed handler
    print(f"\n{'='*40}")
    print("Testing on_x_range_changed handler")
    print(f"{'='*40}")

    # Simulate range change
    chart.on_x_range_changed(None, (122000, 122609))

    print("\nSUMMARY:")
    print("1. Check that rendering works at all positions")
    print("2. Verify is_rendering flag gets reset properly")
    print("3. Confirm data slices correctly at the edge")

    return chart

if __name__ == "__main__":
    chart = test_panning()
    print("\n\nShowing chart window for manual testing...")
    print("Try panning to the far right edge manually")
    chart.show()
    sys.exit(QApplication.instance().exec_())