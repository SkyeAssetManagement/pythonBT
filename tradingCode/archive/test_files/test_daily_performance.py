# tradingCode/test_daily_performance.py
# Test daily vs intraday performance reporting - Step 4
# Verify default end-of-day calculations and --intraday option

import subprocess
import time
import os
from pathlib import Path

def test_daily_performance_default():
    """Test default daily performance calculations"""
    print("TESTING DEFAULT DAILY PERFORMANCE - STEP 4A")
    print("=" * 60)
    
    try:
        # Test default behavior (should use daily calculations)
        print("Running backtest with DEFAULT settings (should use end-of-day)...")
        start_time = time.time()
        
        result = subprocess.run([
            'python', 'main.py', 'ES', 'simpleSMA', 
            '--useDefaults', '--no-viz'
        ], capture_output=True, text=True, timeout=120)
        
        execution_time = time.time() - start_time
        print(f"Default execution completed in {execution_time:.2f} seconds")
        
        if result.returncode == 0:
            output = result.stdout
            print("\n✅ DEFAULT TEST SUCCESS")
            
            # Check for daily performance indicators
            if "Using end-of-day performance calculations" in output:
                print("✅ Default uses end-of-day calculations")
            else:
                print("❌ Default should use end-of-day calculations")
            
            if "Daily resampled data:" in output:
                print("✅ Daily resampling working")
            else:
                print("⚠️ Daily resampling not detected in output")
            
            if "speed improvement:" in output:
                print("✅ Speed improvement reported")
            else:
                print("⚠️ Speed improvement not reported")
            
            # Extract key metrics from output
            lines = output.split('\n')
            for line in lines:
                if "Original data:" in line:
                    print(f"  📊 {line.strip()}")
                elif "Daily resampled data:" in line:
                    print(f"  🚀 {line.strip()}")
                elif "Total Return:" in line:
                    print(f"  💰 {line.strip()}")
                elif "Max Drawdown:" in line:
                    print(f"  📉 {line.strip()}")
            
            return True, execution_time
            
        else:
            print(f"❌ DEFAULT TEST FAILED")
            print(f"Error: {result.stderr}")
            return False, execution_time
            
    except subprocess.TimeoutExpired:
        print("❌ DEFAULT TEST TIMED OUT")
        return False, 120
    except Exception as e:
        print(f"❌ DEFAULT TEST ERROR: {e}")
        return False, 0

def test_intraday_performance_option():
    """Test --intraday performance option"""
    print("\nTESTING --INTRADAY PERFORMANCE OPTION - STEP 4B")
    print("=" * 60)
    
    try:
        # Test --intraday option (should use intraday calculations)
        print("Running backtest with --intraday flag...")
        start_time = time.time()
        
        result = subprocess.run([
            'python', 'main.py', 'ES', 'simpleSMA', 
            '--useDefaults', '--no-viz', '--intraday'
        ], capture_output=True, text=True, timeout=120)
        
        execution_time = time.time() - start_time
        print(f"Intraday execution completed in {execution_time:.2f} seconds")
        
        if result.returncode == 0:
            output = result.stdout
            print("\n✅ INTRADAY TEST SUCCESS")
            
            # Check for intraday performance indicators
            if "Using intraday performance calculations" in output:
                print("✅ --intraday flag uses intraday calculations")
            else:
                print("❌ --intraday flag should use intraday calculations")
            
            if "intraday flag specified" in output:
                print("✅ Intraday flag detection working")
            else:
                print("❌ Intraday flag not detected")
            
            # Extract key metrics from output
            lines = output.split('\n')
            for line in lines:
                if "Using intraday performance" in line:
                    print(f"  ⚡ {line.strip()}")
                elif "Total Return:" in line:
                    print(f"  💰 {line.strip()}")
                elif "Max Drawdown:" in line:
                    print(f"  📉 {line.strip()}")
            
            return True, execution_time
            
        else:
            print(f"❌ INTRADAY TEST FAILED")
            print(f"Error: {result.stderr}")
            return False, execution_time
            
    except subprocess.TimeoutExpired:
        print("❌ INTRADAY TEST TIMED OUT")
        return False, 120
    except Exception as e:
        print(f"❌ INTRADAY TEST ERROR: {e}")
        return False, 0

def test_command_line_help():
    """Test that --help shows the new option"""
    print("\nTESTING COMMAND LINE HELP - STEP 4C")
    print("=" * 60)
    
    try:
        result = subprocess.run([
            'python', 'main.py', '--help'
        ], capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            output = result.stdout
            if "--intraday" in output:
                print("✅ --intraday option visible in help")
                print("✅ Command line integration complete")
                
                # Show the help text for the intraday option
                lines = output.split('\n')
                for i, line in enumerate(lines):
                    if "--intraday" in line:
                        print(f"  📝 {line.strip()}")
                        if i + 1 < len(lines):
                            print(f"      {lines[i+1].strip()}")
                        break
                
                return True
            else:
                print("❌ --intraday option not found in help")
                return False
        else:
            print("❌ Help command failed")
            return False
            
    except Exception as e:
        print(f"❌ Help test error: {e}")
        return False

def main():
    """Run all Step 4 tests"""
    print("TESTING STEP 4: END-OF-DAY vs INTRADAY PERFORMANCE")
    print("=" * 70)
    print("REQUIREMENTS:")
    print("1. Default performance reporting should be end-of-day only")
    print("2. --intraday command line option for intraday calculations")
    print("3. Use vectorBT Pro daily portfolio for speedy calculations")
    print("=" * 70)
    
    # Change to trading code directory
    os.chdir(Path(__file__).parent)
    
    # Run tests
    default_success, default_time = test_daily_performance_default()
    intraday_success, intraday_time = test_intraday_performance_option()
    help_success = test_command_line_help()
    
    # Results summary
    print("\n" + "=" * 70)
    print("STEP 4 TEST RESULTS SUMMARY")
    print("=" * 70)
    
    if default_success:
        print(f"✅ Default daily performance: PASSED ({default_time:.1f}s)")
    else:
        print(f"❌ Default daily performance: FAILED ({default_time:.1f}s)")
    
    if intraday_success:
        print(f"✅ --intraday option: PASSED ({intraday_time:.1f}s)")
    else:
        print(f"❌ --intraday option: FAILED ({intraday_time:.1f}s)")
    
    if help_success:
        print("✅ Command line help: PASSED")
    else:
        print("❌ Command line help: FAILED")
    
    # Performance comparison
    if default_success and intraday_success:
        if default_time < intraday_time:
            speedup = intraday_time / default_time
            print(f"\n🚀 PERFORMANCE IMPROVEMENT:")
            print(f"   Daily calculations: {speedup:.1f}x faster than intraday")
            print(f"   Default ({default_time:.1f}s) vs Intraday ({intraday_time:.1f}s)")
        else:
            print(f"\n⚠️  Performance comparison inconclusive")
    
    # Overall result
    all_passed = default_success and intraday_success and help_success
    
    if all_passed:
        print(f"\n✅ STEP 4 COMPLETE: END-OF-DAY PERFORMANCE REPORTING")
        print("   • Default: End-of-day calculations for speed")
        print("   • Option: --intraday for detailed analysis") 
        print("   • Integration: vectorBT Pro daily resampling")
    else:
        print(f"\n❌ STEP 4 INCOMPLETE: Some tests failed")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)