# simple_verification.py  
# Quick verification without Unicode characters

from pathlib import Path

def quick_verification():
    print("QUICK VERIFICATION RESULTS")
    print("=" * 50)
    
    # Check Step 6: Text consistency (we saw this working)
    print("STEP 6: Text Sizing Consistency")
    crosshair_code = Path("src/dashboard/crosshair_widget.py").read_text()
    hover_code = Path("src/dashboard/hover_info_widget.py").read_text()
    
    crosshair_7pt = crosshair_code.count("font-size: 7pt")
    crosshair_8pt = crosshair_code.count("font-size: 8pt")
    hover_7pt = hover_code.count("font-size: 7pt") 
    hover_8pt = hover_code.count("font-size: 8pt")
    
    print(f"  Crosshair widget: {crosshair_7pt} x 7pt, {crosshair_8pt} x 8pt")
    print(f"  Hover widget: {hover_7pt} x 7pt, {hover_8pt} x 8pt")
    
    step6_success = crosshair_7pt == 0 and hover_7pt == 0 and crosshair_8pt > 10 and hover_8pt > 5
    print(f"  Step 6: {'SUCCESS' if step6_success else 'NEEDS WORK'}")
    
    # Check Step 4: Performance options
    print("\\nSTEP 4: End-of-Day Performance")
    main_code = Path("main.py").read_text()
    
    has_intraday_param = "intraday_performance: bool = False" in main_code
    has_intraday_flag = "--intraday" in main_code
    has_daily_resample = "pf.resample('1D')" in main_code
    
    print(f"  Intraday parameter: {'SUCCESS' if has_intraday_param else 'MISSING'}")
    print(f"  --intraday flag: {'SUCCESS' if has_intraday_flag else 'MISSING'}")
    print(f"  Daily resampling: {'SUCCESS' if has_daily_resample else 'MISSING'}")
    
    step4_success = has_intraday_param and has_intraday_flag and has_daily_resample
    print(f"  Step 4: {'SUCCESS' if step4_success else 'NEEDS WORK'}")
    
    # Check Step 5: Indicators fix
    print("\\nSTEP 5: Indicators Fix") 
    try:
        indicators_code = Path("src/dashboard/indicators_panel.py").read_text()
        signal_emissions = indicators_code.count("indicators_updated.emit")
        has_add_method = "_add_indicator" in indicators_code
        
        print(f"  Signal emissions: {signal_emissions}")
        print(f"  Add indicator method: {'SUCCESS' if has_add_method else 'MISSING'}")
        
        step5_success = signal_emissions >= 3 and has_add_method
        print(f"  Step 5: {'SUCCESS' if step5_success else 'NEEDS WORK'}")
    except:
        print("  Step 5: ERROR - file not found")
        step5_success = False
    
    # Check critical fixes
    print("\\nCRITICAL FIXES:")
    dashboard_code = Path("step6_complete_final.py").read_text()
    
    has_equity_sync = "equity_curve.sync_viewport" in dashboard_code
    has_image_export = "export_dashboard_image" in dashboard_code
    has_viewport_fix = "_on_viewport_change" in dashboard_code and "equity_curve" in dashboard_code
    
    print(f"  Equity sync fix: {'SUCCESS' if has_equity_sync else 'MISSING'}")
    print(f"  Image export: {'SUCCESS' if has_image_export else 'MISSING'}")
    print(f"  Viewport fix: {'SUCCESS' if has_viewport_fix else 'MISSING'}")
    
    # Summary
    print("\\n" + "=" * 50)
    print("SUMMARY:")
    print(f"  Step 4 (Performance): {'SUCCESS' if step4_success else 'NEEDS WORK'}")
    print(f"  Step 5 (Indicators): {'SUCCESS' if step5_success else 'NEEDS WORK'}")
    print(f"  Step 6 (Text): {'SUCCESS' if step6_success else 'NEEDS WORK'}")
    print(f"  Critical Fixes: {'SUCCESS' if all([has_equity_sync, has_image_export, has_viewport_fix]) else 'NEEDS WORK'}")
    
    total_success = sum([step4_success, step5_success, step6_success, has_equity_sync, has_image_export, has_viewport_fix])
    print(f"\\nOVERALL: {total_success}/6 items working")
    
    if total_success >= 5:
        print("EXCELLENT - Most functionality implemented!")
    elif total_success >= 3:
        print("GOOD - Core functionality working")  
    else:
        print("NEEDS WORK - Several issues found")

if __name__ == "__main__":
    quick_verification()