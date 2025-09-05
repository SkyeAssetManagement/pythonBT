#!/usr/bin/env python3
"""
Minimal Optimization Test
Tests just 3 combinations to verify the system works completely
"""

import sys
sys.path.append('src')
sys.path.append('strategies')

from optimization_performance_test import OptimizationPerformanceTester
import logging

# Configure logging for minimal test
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def minimal_test():
    """Run a minimal test with just 3 parameter combinations"""
    
    print("=" * 60)
    print("MINIMAL OPTIMIZATION TEST")
    print("Testing with 3 parameter combinations")
    print("=" * 60)
    
    # Initialize tester
    tester = OptimizationPerformanceTester()
    
    # Load data
    if not tester.load_data():
        print("ERROR: Data loading failed")
        return
    
    # Test just 3 combinations
    test_combinations = [(20, 100), (30, 200), (50, 300)]
    
    print(f"Testing with {len(test_combinations)} combinations:")
    for i, (ma1, ma2) in enumerate(test_combinations):
        print(f"  {i+1}. MA1={ma1}, MA2={ma2}")
    print()
    
    # Test serial mode only
    print("Testing SERIAL mode...")  
    results, exec_time = tester.run_optimization_serial(test_combinations)
    
    print(f"  Time: {exec_time:.2f}s")
    print(f"  Rate: {len(test_combinations)/exec_time:.2f} backtests/second")
    
    # Check results
    successful = [r for r in results if r['Status'] == 'Success']
    failed = [r for r in results if r['Status'] != 'Success']
    
    print(f"  Success: {len(successful)}/{len(test_combinations)}")
    print(f"  Failed: {len(failed)}/{len(test_combinations)}")
    print()
    
    # Show successful results
    if successful:
        print("SUCCESSFUL RESULTS:")
        for result in successful:
            print(f"  MA1={result['MA1']}, MA2={result['MA2']}: Return={result['Total_Return']:.2f}%, Sharpe={result['Sharpe_Ratio']:.2f}, Trades={result['Trade_Count']}")
    
    # Show failed results
    if failed:
        print("FAILED RESULTS:")
        for result in failed:
            print(f"  MA1={result['MA1']}, MA2={result['MA2']}: {result['Status']}")
    
    print()
    if len(successful) >= 1:
        print("SUCCESS: System is working and ready for full optimization!")
        return True
    else:
        print("ERROR: No successful backtests - system needs debugging")
        return False

if __name__ == "__main__":
    success = minimal_test()
    if success:
        print("\nSYSTEM READY FOR FULL 10,000 COMBINATION TEST!")
    else:
        print("\nSYSTEM NEEDS MORE DEBUGGING")