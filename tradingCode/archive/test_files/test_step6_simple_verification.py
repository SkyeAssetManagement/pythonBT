# test_step6_simple_verification.py
# Simple verification of Step 6 text sizing fix
# Check that all font sizes are now 8pt in crosshair/hover widgets

import sys
from pathlib import Path

# Add src path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_crosshair_text_sizes():
    """Test crosshair widget text sizing"""
    print("Testing crosshair widget text sizes...")
    
    try:
        from dashboard.crosshair_widget import CrosshairInfoWidget
        from step6_complete_final import create_ultimate_test_data
        
        # Create test data
        ohlcv_data, _ = create_ultimate_test_data()
        
        # Create widget but don't show UI
        widget = CrosshairInfoWidget()
        widget.load_ohlcv_data(ohlcv_data)
        widget._setup_ui()
        
        # Check position labels
        font_issues = []
        for key, label in widget.position_labels.items():
            style = label.styleSheet()
            if "font-size: 8pt" not in style:
                font_issues.append(f"Position {key}: {style}")
            if "Courier New" not in style:
                font_issues.append(f"Position {key}: Missing Courier New")
        
        # Check bar labels
        for key, label in widget.bar_labels.items():
            style = label.styleSheet()
            if "font-size: 8pt" not in style:
                font_issues.append(f"Bar {key}: {style}")
            if "Courier New" not in style:
                font_issues.append(f"Bar {key}: Missing Courier New")
                
        # Check stats labels
        for key, label in widget.stats_labels.items():
            style = label.styleSheet()
            if "font-size: 8pt" not in style:
                font_issues.append(f"Stats {key}: {style}")
            if "Courier New" not in style:
                font_issues.append(f"Stats {key}: Missing Courier New")
        
        if font_issues:
            print("CROSSHAIR WIDGET FONT ISSUES:")
            for issue in font_issues:
                print(f"  FAIL: {issue}")
            return False
        else:
            print("  SUCCESS: All crosshair text is 8pt Courier New")
            return True
            
    except Exception as e:
        print(f"  ERROR: Crosshair test failed: {e}")
        return False

def test_hover_text_sizes():
    """Test hover widget text sizing"""
    print("Testing hover widget text sizes...")
    
    try:
        from dashboard.hover_info_widget import HoverInfoWidget
        from step6_complete_final import create_ultimate_test_data
        
        # Create test data
        ohlcv_data, _ = create_ultimate_test_data()
        
        # Create widget but don't show UI
        widget = HoverInfoWidget()
        widget.load_ohlcv_data(ohlcv_data)
        
        # Check price labels
        font_issues = []
        for key, label in widget.price_labels.items():
            style = label.styleSheet()
            if "font-size: 8pt" not in style:
                font_issues.append(f"Price {key}: {style}")
            if "Courier New" not in style:
                font_issues.append(f"Price {key}: Missing Courier New")
        
        # Check indicator labels (these were 7pt, should now be 8pt)
        for key, label in widget.indicator_labels.items():
            style = label.styleSheet()
            if "font-size: 8pt" not in style:
                font_issues.append(f"Indicator {key}: {style}")
            if "Courier New" not in style:
                font_issues.append(f"Indicator {key}: Missing Courier New")
        
        if font_issues:
            print("HOVER WIDGET FONT ISSUES:")
            for issue in font_issues:
                print(f"  FAIL: {issue}")
            return False
        else:
            print("  SUCCESS: All hover text is 8pt Courier New")
            return True
            
    except Exception as e:
        print(f"  ERROR: Hover test failed: {e}")
        return False

def main():
    """Main verification function"""
    print("STEP 6 TEXT SIZING VERIFICATION")
    print("="*50)
    print("Checking that all crosshair/hover text matches X/Y axis size (8pt)")
    print()
    
    # Run tests
    crosshair_ok = test_crosshair_text_sizes()
    hover_ok = test_hover_text_sizes()
    
    print()
    print("="*50)
    print("VERIFICATION RESULTS:")
    
    if crosshair_ok:
        print("SUCCESS: Crosshair widget text sizing fixed")
    else:
        print("FAIL: Crosshair widget has font size issues")
    
    if hover_ok:
        print("SUCCESS: Hover widget text sizing fixed")
    else:
        print("FAIL: Hover widget has font size issues")
    
    if crosshair_ok and hover_ok:
        print()
        print("SUCCESS: STEP 6 TEXT SIZING COMPLETE!")
        print("All crosshair/hover text now matches X/Y axis values (8pt Courier New)")
        return True
    else:
        print()
        print("FAIL: Step 6 text sizing needs additional work")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)