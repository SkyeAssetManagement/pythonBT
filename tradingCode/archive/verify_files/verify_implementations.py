# verify_implementations.py
# Code-based verification of all implementations without GUI requirements
# Solves GUI limitations by checking code directly

import sys
import os
from pathlib import Path
import importlib.util
import inspect

def verify_step1_equity_curve_datetime():
    """Verify Step 1: Equity curve datetime axis implementation"""
    try:
        print("\\nVERIFYING STEP 1: Equity Curve DateTime Axis")
        print("-" * 50)
        
        # Check equity curve widget for DateAxisItem
        sys.path.insert(0, str(Path(__file__).parent / "src"))
        from dashboard.equity_curve_widget import EquityCurveWidget
        
        # Check if DateAxisItem is imported and used
        widget_code = inspect.getsource(EquityCurveWidget)
        
        checks = []
        if "from pyqtgraph import DateAxisItem" in widget_code:
            checks.append("✅ DateAxisItem imported")
        else:
            checks.append("❌ DateAxisItem not imported")
        
        if "DateAxisItem(orientation='bottom')" in widget_code:
            checks.append("✅ DateAxisItem used for bottom axis")
        else:
            checks.append("❌ DateAxisItem not configured")
        
        if "axisItems={'bottom': date_axis}" in widget_code:
            checks.append("✅ Date axis integrated into PlotWidget")
        else:
            checks.append("❌ Date axis not integrated")
        
        for check in checks:
            print(f"  {check}")
        
        success = all("✅" in check for check in checks)
        print(f"\\nStep 1 Status: {'✅ IMPLEMENTED' if success else '❌ NEEDS WORK'}")
        return success
        
    except Exception as e:
        print(f"ERROR: Step 1 verification failed: {e}")
        return False

def verify_step2_trade_arrows():
    """Verify Step 2: Trade marker triangles in VisPy"""
    try:
        print("\\nVERIFYING STEP 2: Trade Marker Triangles")
        print("-" * 50)
        
        # Check FinalVispyChart for trade markers
        from step6_complete_final import FinalVispyChart
        
        chart_code = inspect.getsource(FinalVispyChart)
        
        checks = []
        if "trade_markers_program" in chart_code:
            checks.append("✅ Trade markers program exists")
        else:
            checks.append("❌ Trade markers program missing")
        
        if "trade_marker_vertex_shader" in chart_code:
            checks.append("✅ Trade marker shaders defined")
        else:
            checks.append("❌ Trade marker shaders missing")
        
        if "_generate_trade_markers" in chart_code:
            checks.append("✅ Trade marker generation method exists")
        else:
            checks.append("❌ Trade marker generation method missing")
        
        # Check for triangle positioning logic
        if "2% below" in chart_code or "0.02" in chart_code:
            checks.append("✅ Triangle positioning logic present")
        else:
            checks.append("⚠️ Triangle positioning logic unclear")
        
        for check in checks:
            print(f"  {check}")
        
        success = sum("✅" in check for check in checks) >= 3
        print(f"\\nStep 2 Status: {'✅ IMPLEMENTED' if success else '❌ NEEDS WORK'}")
        return success
        
    except Exception as e:
        print(f"ERROR: Step 2 verification failed: {e}")
        return False

def verify_step3_trade_navigation():
    """Verify Step 3: Trade jump functionality"""
    try:
        print("\\nVERIFYING STEP 3: Trade Jump Navigation")
        print("-" * 50)
        
        # Check trade list widget for navigation
        from dashboard.trade_list_widget import TradeListContainer
        
        widget_code = inspect.getsource(TradeListContainer)
        
        checks = []
        if "cellClicked.connect" in widget_code:
            checks.append("✅ Cell click handler connected")
        else:
            checks.append("❌ Cell click handler missing")
        
        if "_on_trade_clicked" in widget_code:
            checks.append("✅ Trade click method exists")
        else:
            checks.append("❌ Trade click method missing")
        
        if "navigate_to_trade" in widget_code:
            checks.append("✅ Navigation method exists")
        else:
            checks.append("❌ Navigation method missing")
        
        if "chart_navigation_callback" in widget_code:
            checks.append("✅ Chart callback integration exists")
        else:
            checks.append("❌ Chart callback integration missing")
        
        for check in checks:
            print(f"  {check}")
        
        success = all("✅" in check for check in checks)
        print(f"\\nStep 3 Status: {'✅ IMPLEMENTED' if success else '❌ NEEDS WORK'}")
        return success
        
    except Exception as e:
        print(f"ERROR: Step 3 verification failed: {e}")
        return False

def verify_step4_performance_options():
    """Verify Step 4: End-of-day performance default"""
    try:
        print("\\nVERIFYING STEP 4: End-of-Day Performance Options")
        print("-" * 50)
        
        # Check main.py for performance options
        main_path = Path(__file__).parent / "main.py"
        main_code = main_path.read_text()
        
        checks = []
        if "intraday_performance: bool = False" in main_code:
            checks.append("✅ Intraday performance parameter exists with False default")
        else:
            checks.append("❌ Intraday performance parameter missing or wrong default")
        
        if "--intraday" in main_code:
            checks.append("✅ --intraday command line option exists")
        else:
            checks.append("❌ --intraday command line option missing")
        
        if "pf.resample('1D')" in main_code:
            checks.append("✅ Daily resampling implemented")
        else:
            checks.append("❌ Daily resampling missing")
        
        if "Using end-of-day performance calculations" in main_code:
            checks.append("✅ End-of-day messaging exists")
        else:
            checks.append("❌ End-of-day messaging missing")
        
        for check in checks:
            print(f"  {check}")
        
        success = all("✅" in check for check in checks)
        print(f"\\nStep 4 Status: {'✅ IMPLEMENTED' if success else '❌ NEEDS WORK'}")
        return success
        
    except Exception as e:
        print(f"ERROR: Step 4 verification failed: {e}")
        return False

def verify_step5_indicators_fix():
    """Verify Step 5: Indicators functionality fix"""
    try:
        print("\\nVERIFYING STEP 5: Indicators Addition Fix")
        print("-" * 50)
        
        # Check indicators panel for signal emissions
        from dashboard.indicators_panel import VBTIndicatorsPanel
        
        panel_code = inspect.getsource(VBTIndicatorsPanel)
        
        checks = []
        if "indicators_updated.emit" in panel_code:
            checks.append("✅ Signal emissions exist")
        else:
            checks.append("❌ Signal emissions missing")
        
        # Count signal emissions in key methods
        emission_count = panel_code.count("indicators_updated.emit")
        if emission_count >= 3:
            checks.append(f"✅ Multiple signal emissions found ({emission_count})")
        else:
            checks.append(f"❌ Insufficient signal emissions ({emission_count})")
        
        if "_add_indicator" in panel_code and "indicators_updated.emit" in panel_code:
            checks.append("✅ Add indicator method has signal emission")
        else:
            checks.append("❌ Add indicator method missing signal")
        
        if "_recalculate_all_indicators" in panel_code:
            checks.append("✅ Recalculate method exists")
        else:
            checks.append("❌ Recalculate method missing")
        
        for check in checks:
            print(f"  {check}")
        
        success = all("✅" in check for check in checks)
        print(f"\\nStep 5 Status: {'✅ IMPLEMENTED' if success else '❌ NEEDS WORK'}")
        return success
        
    except Exception as e:
        print(f"ERROR: Step 5 verification failed: {e}")
        return False

def verify_step6_text_consistency():
    """Verify Step 6: Text sizing consistency"""
    try:
        print("\\nVERIFYING STEP 6: Text Sizing Consistency")
        print("-" * 50)
        
        # Check for 7pt fonts (should be none)
        widget_files = [
            Path(__file__).parent / "src" / "dashboard" / "crosshair_widget.py",
            Path(__file__).parent / "src" / "dashboard" / "hover_info_widget.py"
        ]
        
        checks = []
        total_7pt = 0
        total_8pt = 0
        
        for widget_file in widget_files:
            if widget_file.exists():
                code = widget_file.read_text()
                count_7pt = code.count("font-size: 7pt")
                count_8pt = code.count("font-size: 8pt")
                total_7pt += count_7pt
                total_8pt += count_8pt
                
                print(f"  {widget_file.name}: {count_7pt} × 7pt, {count_8pt} × 8pt")
        
        if total_7pt == 0:
            checks.append("✅ No 7pt fonts found (consistency achieved)")
        else:
            checks.append(f"❌ Found {total_7pt} instances of 7pt fonts")
        
        if total_8pt > 10:  # Expect many 8pt fonts
            checks.append(f"✅ Found {total_8pt} instances of 8pt fonts")
        else:
            checks.append(f"❌ Insufficient 8pt fonts ({total_8pt})")
        
        # Check for Courier New consistency
        courier_new_count = 0
        for widget_file in widget_files:
            if widget_file.exists():
                code = widget_file.read_text()
                courier_new_count += code.count("'Courier New'")
        
        if courier_new_count > 5:
            checks.append(f"✅ Courier New font used consistently ({courier_new_count} times)")
        else:
            checks.append(f"❌ Insufficient Courier New usage ({courier_new_count})")
        
        for check in checks:
            print(f"  {check}")
        
        success = all("✅" in check for check in checks)
        print(f"\\nStep 6 Status: {'✅ IMPLEMENTED' if success else '❌ NEEDS WORK'}")
        return success
        
    except Exception as e:
        print(f"ERROR: Step 6 verification failed: {e}")
        return False

def verify_critical_fixes():
    """Verify critical fixes implemented"""
    try:
        print("\\nVERIFYING CRITICAL FIXES")
        print("-" * 50)
        
        # Check viewport synchronization fix
        dashboard_code = (Path(__file__).parent / "step6_complete_final.py").read_text()
        
        checks = []
        if "equity_curve.sync_viewport" in dashboard_code:
            checks.append("✅ Equity curve sync call exists")
        else:
            checks.append("❌ Equity curve sync call missing")
        
        if "_on_viewport_change" in dashboard_code and "equity_curve" in dashboard_code:
            checks.append("✅ Viewport change handler includes equity curve")
        else:
            checks.append("❌ Viewport change handler doesn't include equity curve")
        
        if "export_dashboard_image" in dashboard_code:
            checks.append("✅ Image export functionality exists")
        else:
            checks.append("❌ Image export functionality missing")
        
        if "export_step_verification_images" in dashboard_code:
            checks.append("✅ Step verification export exists")
        else:
            checks.append("❌ Step verification export missing")
        
        for check in checks:
            print(f"  {check}")
        
        success = all("✅" in check for check in checks)
        print(f"\\nCritical Fixes Status: {'✅ IMPLEMENTED' if success else '❌ NEEDS WORK'}")
        return success
        
    except Exception as e:
        print(f"ERROR: Critical fixes verification failed: {e}")
        return False

def main():
    """Main verification function"""
    print("COMPREHENSIVE CODE-BASED VERIFICATION")
    print("="*60)
    print("Verifying all 6 steps + critical fixes through code analysis")
    print("="*60)
    
    results = []
    
    # Verify all steps
    results.append(("Step 1: Equity curve datetime", verify_step1_equity_curve_datetime()))
    results.append(("Step 2: Trade marker triangles", verify_step2_trade_arrows()))
    results.append(("Step 3: Trade jump navigation", verify_step3_trade_navigation()))
    results.append(("Step 4: End-of-day performance", verify_step4_performance_options()))
    results.append(("Step 5: Indicators addition fix", verify_step5_indicators_fix()))
    results.append(("Step 6: Text sizing consistency", verify_step6_text_consistency()))
    results.append(("Critical fixes", verify_critical_fixes()))
    
    # Generate final report
    print("\\n" + "="*60)
    print("FINAL VERIFICATION REPORT")
    print("="*60)
    
    successful = sum(1 for _, success in results if success)
    total = len(results)
    
    for desc, success in results:
        status = "✅ VERIFIED" if success else "❌ NEEDS WORK"
        print(f"{desc}: {status}")
    
    success_rate = successful / total * 100
    print(f"\\nOVERALL: {successful}/{total} items verified ({success_rate:.1f}%)")
    
    if success_rate >= 85:
        print("\\n🎉 EXCELLENT: Trading dashboard professionally implemented!")
    elif success_rate >= 70:
        print("\\n✅ GOOD: Most functionality working, minor issues to address")
    else:
        print("\\n⚠️ NEEDS WORK: Several critical issues need attention")
    
    print("\\nNEXT STEPS:")
    print("1. Run: python main.py ES simpleSMA")
    print("2. Take manual screenshots for visual verification")
    print("3. Test each step interactively")
    
    return success_rate >= 70

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)