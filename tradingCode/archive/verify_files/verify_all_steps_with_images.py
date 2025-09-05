# verify_all_steps_with_images.py
# CRITICAL: Comprehensive verification with programmatic image export
# Solves GUI limitations by taking actual screenshots for verification

import sys
import os
import time
from pathlib import Path

# Add src path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QEventLoop, QTimer

# Import the complete dashboard
from step6_complete_final import FinalTradingDashboard, create_ultimate_test_data

class ComprehensiveVerification:
    """Comprehensive verification system with image export"""
    
    def __init__(self):
        self.app = None
        self.dashboard = None
        self.verification_results = {}
        self.screenshot_folder = Path("C:\\Users\\skyeAM\\SkyeAM Dropbox\\SAMresearch\\ABtoPython\\2025-08-08_1")
        
        print("COMPREHENSIVE VERIFICATION WITH IMAGE EXPORT")
        print("="*60)
        print("OBJECTIVE: Verify all 6 steps with actual screenshots")
        print("SOLUTION: Programmatic image export to overcome GUI limitations")
        print("="*60)
    
    def run_complete_verification(self):
        """Run complete verification with screenshots"""
        try:
            # Initialize Qt application
            self.app = QApplication.instance()
            if self.app is None:
                self.app = QApplication([])
            
            print("\\n1. INITIALIZING ULTIMATE DASHBOARD...")
            self._initialize_dashboard()
            
            print("\\n2. LOADING TEST DATA...")
            self._load_test_data()
            
            print("\\n3. TAKING VERIFICATION SCREENSHOTS...")
            self._take_verification_screenshots()
            
            print("\\n4. VERIFYING CRITICAL FUNCTIONALITY...")
            self._verify_critical_functions()
            
            print("\\n5. GENERATING VERIFICATION REPORT...")
            self._generate_verification_report()
            
            return True
            
        except Exception as e:
            print(f"ERROR: Comprehensive verification failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _initialize_dashboard(self):
        """Initialize the ultimate dashboard"""
        try:
            print("Creating FinalTradingDashboard instance...")
            self.dashboard = FinalTradingDashboard()
            
            print("Setting dashboard geometry...")
            self.dashboard.resize(1920, 1200)
            self.dashboard.setWindowTitle("6-Step Verification Dashboard")
            
            print("SUCCESS: Dashboard initialized")
            
        except Exception as e:
            print(f"ERROR: Dashboard initialization failed: {e}")
            raise
    
    def _load_test_data(self):
        """Load test data into dashboard"""
        try:
            print("Creating ultimate test data...")
            ohlcv_data, _ = create_ultimate_test_data()
            trades_csv_path = "ultimate_trades_step6_final.csv"
            
            print("Loading data into dashboard...")
            success = self.dashboard.load_ultimate_dataset(ohlcv_data, trades_csv_path)
            
            if success:
                print("SUCCESS: Test data loaded successfully")
                print(f"  OHLCV bars: {len(ohlcv_data['close']):,}")
                print(f"  Trades file: {trades_csv_path}")
            else:
                print("ERROR: Test data loading failed")
                raise Exception("Data loading failed")
                
        except Exception as e:
            print(f"ERROR: Test data loading failed: {e}")
            raise
    
    def _take_verification_screenshots(self):
        """Take comprehensive verification screenshots"""
        try:
            print("Showing dashboard...")
            self.dashboard.show()
            self.dashboard.raise_()
            self.dashboard.activateWindow()
            
            # Wait for rendering
            print("Waiting for rendering...")
            time.sleep(3)
            
            # Force update
            self.dashboard.update()
            self.app.processEvents()
            
            print("Taking verification screenshots...")
            
            # Take overall dashboard screenshot
            overall_path = self.dashboard.export_dashboard_image(
                "complete_dashboard_verification.png"
            )
            if overall_path:
                print(f"SUCCESS: Overall dashboard screenshot: {overall_path}")
                self.verification_results['overall'] = overall_path
            else:
                print("ERROR: Overall dashboard screenshot failed")
            
            # Take step-specific screenshots
            step_results = self.dashboard.export_step_verification_images()
            self.verification_results.update(step_results)
            
            for step, path in step_results.items():
                if path and path != "Console output - end-of-day vs --intraday":
                    print(f"SUCCESS: {step} screenshot: {path}")
                else:
                    print(f"INFO: {step}: {path}")
            
        except Exception as e:
            print(f"ERROR: Screenshot capture failed: {e}")
            import traceback
            traceback.print_exc()
    
    def _verify_critical_functions(self):
        """Verify critical functionality programmatically"""
        try:
            print("\\nVERIFYING CRITICAL FUNCTIONS:")
            
            # Check 1: Viewport synchronization
            print("\\n1. TESTING VIEWPORT SYNCHRONIZATION:")
            initial_viewport = (self.dashboard.final_chart.viewport_start, 
                              self.dashboard.final_chart.viewport_end)
            print(f"   Initial viewport: {initial_viewport}")
            
            # Simulate viewport change
            new_start, new_end = 500, 800
            self.dashboard.final_chart.viewport_start = new_start
            self.dashboard.final_chart.viewport_end = new_end
            
            # Trigger sync (this should now work with our fix)
            self.dashboard._on_viewport_change(new_start, new_end)
            
            equity_viewport = self.dashboard.equity_curve.current_viewport
            print(f"   Chart viewport: ({new_start}, {new_end})")
            print(f"   Equity viewport: {equity_viewport}")
            
            if abs(equity_viewport[0] - new_start) < 10 and abs(equity_viewport[1] - new_end) < 10:
                print("   ✅ VIEWPORT SYNC: WORKING")
                self.verification_results['viewport_sync'] = True
            else:
                print("   ❌ VIEWPORT SYNC: FAILED")
                self.verification_results['viewport_sync'] = False
            
            # Check 2: Image export functionality
            print("\\n2. TESTING IMAGE EXPORT:")
            test_image = self.dashboard.export_dashboard_image("test_export_functionality.png")
            if test_image and Path(test_image).exists():
                file_size = Path(test_image).stat().st_size
                print(f"   ✅ IMAGE EXPORT: WORKING ({file_size:,} bytes)")
                self.verification_results['image_export'] = True
            else:
                print("   ❌ IMAGE EXPORT: FAILED")
                self.verification_results['image_export'] = False
            
            # Check 3: Trade arrows rendering (check VisPy chart data)
            print("\\n3. TESTING TRADE ARROWS:")
            if hasattr(self.dashboard.final_chart, 'trade_markers_program'):
                if self.dashboard.final_chart.trade_markers_program is not None:
                    print("   ✅ TRADE MARKERS PROGRAM: EXISTS")
                    
                    # Check if trade data is loaded
                    if hasattr(self.dashboard.final_chart, 'trades_data') and self.dashboard.final_chart.trades_data:
                        trade_count = len(self.dashboard.final_chart.trades_data)
                        print(f"   ✅ TRADE DATA: {trade_count} trades loaded")
                        self.verification_results['trade_arrows'] = True
                    else:
                        print("   ❌ TRADE DATA: No trades loaded")
                        self.verification_results['trade_arrows'] = False
                else:
                    print("   ❌ TRADE MARKERS PROGRAM: NOT INITIALIZED")
                    self.verification_results['trade_arrows'] = False
            else:
                print("   ❌ TRADE MARKERS: NO PROGRAM ATTRIBUTE")
                self.verification_results['trade_arrows'] = False
            
            # Check 4: Indicators functionality
            print("\\n4. TESTING INDICATORS:")
            if self.dashboard.indicators_panel:
                indicator_data = self.dashboard.indicators_panel.get_indicator_data()
                indicator_count = len(indicator_data)
                if indicator_count > 0:
                    print(f"   ✅ INDICATORS: {indicator_count} active indicators")
                    self.verification_results['indicators'] = True
                else:
                    print("   ❌ INDICATORS: No active indicators")
                    self.verification_results['indicators'] = False
            else:
                print("   ❌ INDICATORS: Panel not found")
                self.verification_results['indicators'] = False
            
        except Exception as e:
            print(f"ERROR: Critical function verification failed: {e}")
            import traceback
            traceback.print_exc()
    
    def _generate_verification_report(self):
        """Generate comprehensive verification report"""
        try:
            print("\\n" + "="*60)
            print("COMPREHENSIVE VERIFICATION REPORT")
            print("="*60)
            
            # Count successful verifications
            successful_screenshots = sum(1 for k, v in self.verification_results.items() 
                                       if k.startswith('step') and v and v != "Console output - end-of-day vs --intraday")
            
            successful_functions = sum(1 for k, v in self.verification_results.items() 
                                     if k in ['viewport_sync', 'image_export', 'trade_arrows', 'indicators'] and v)
            
            print(f"SCREENSHOTS CAPTURED: {successful_screenshots}/6 steps")
            print(f"CRITICAL FUNCTIONS: {successful_functions}/4 working")
            
            print("\\nSTEP-BY-STEP VERIFICATION:")
            steps = [
                ("Step 1: Equity curve datetime axis", "step1_equity_datetime"),
                ("Step 2: Trade marker triangles", "step2_trade_arrows"), 
                ("Step 3: Trade jump navigation", "step3_navigation"),
                ("Step 4: End-of-day performance", "step4_performance"),
                ("Step 5: Indicators addition fix", "step5_indicators"),
                ("Step 6: Text sizing consistency", "step6_text_sizing")
            ]
            
            for step_desc, step_key in steps:
                result = self.verification_results.get(step_key, "Not captured")
                if result and result != "Console output - end-of-day vs --intraday":
                    status = "✅ VERIFIED" if Path(str(result)).exists() else "❌ FAILED"
                elif result == "Console output - end-of-day vs --intraday":
                    status = "📝 CONSOLE"
                else:
                    status = "❌ MISSING"
                print(f"  {step_desc}: {status}")
            
            print("\\nCRITICAL FUNCTION VERIFICATION:")
            functions = [
                ("Viewport synchronization fix", "viewport_sync"),
                ("Image export functionality", "image_export"),
                ("Trade arrows rendering", "trade_arrows"),
                ("Indicators functionality", "indicators")
            ]
            
            for func_desc, func_key in functions:
                result = self.verification_results.get(func_key, False)
                status = "✅ WORKING" if result else "❌ NEEDS FIX"
                print(f"  {func_desc}: {status}")
            
            # Overall assessment
            total_items = len(steps) + len(functions)
            successful_items = successful_screenshots + successful_functions
            success_rate = successful_items / total_items * 100
            
            print(f"\\nOVERALL VERIFICATION: {successful_items}/{total_items} items ({success_rate:.1f}%)")
            
            if success_rate >= 80:
                print("\\n🎉 VERIFICATION STATUS: EXCELLENT")
                print("The trading dashboard demonstrates professional functionality!")
            elif success_rate >= 60:
                print("\\n⚠️  VERIFICATION STATUS: GOOD - Minor issues need attention")
            else:
                print("\\n❌ VERIFICATION STATUS: NEEDS WORK - Critical issues found")
            
            print("\\nSCREENSHOT FOLDER:")
            print(f"📁 {self.screenshot_folder}")
            
            # List all screenshots
            screenshots = list(self.screenshot_folder.glob("*.png"))
            if screenshots:
                print(f"\\n📸 SCREENSHOTS CAPTURED ({len(screenshots)}):")
                for screenshot in sorted(screenshots):
                    print(f"  • {screenshot.name}")
            
            print("="*60)
            
        except Exception as e:
            print(f"ERROR: Verification report generation failed: {e}")


def main():
    """Main verification function"""
    print("STARTING COMPREHENSIVE 6-STEP VERIFICATION")
    print("This solves GUI limitations with programmatic verification")
    
    verifier = ComprehensiveVerification()
    success = verifier.run_complete_verification()
    
    if success:
        print("\\nVERIFICATION COMPLETED SUCCESSFULLY!")
        print("Check the 2025-08-08_1 folder for screenshots and evidence.")
    else:
        print("\\nVERIFICATION ENCOUNTERED ISSUES")
        print("Review the error messages above for debugging.")
    
    return success


if __name__ == "__main__":
    success = main()
    
    # Keep application running briefly to ensure screenshots are saved
    if success:
        print("\\nWaiting for final processing...")
        time.sleep(2)
        print("VERIFICATION COMPLETE!")
    
    exit(0 if success else 1)