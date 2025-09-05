#!/usr/bin/env python3
"""
Quick Optimization Performance Test
Tests a small subset (100 combinations) to verify the system works
"""

import sys
import os
sys.path.append('src')
sys.path.append('strategies')

from optimization_performance_test import OptimizationPerformanceTester
import logging

# Configure logging for quick test
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def quick_test():
    """Run a quick test with limited parameter combinations"""
    
    print("=" * 60)
    print("QUICK OPTIMIZATION PERFORMANCE TEST")
    print("Testing with limited parameter combinations")
    print("=" * 60)
    
    # Initialize tester
    tester = OptimizationPerformanceTester()
    
    # Load data
    if not tester.load_data():
        print("ERROR: Data loading failed")
        return
    
    # Generate limited parameter set for testing
    # MA1: 20, 30, 40, 50, 60 (5 values)
    # MA2: 100, 200, 300, 400 (4 values)  
    # Total: 20 combinations (only where MA2 > MA1)
    
    test_combinations = []
    for ma1 in [20, 30, 40, 50, 60]:
        for ma2 in [100, 200, 300, 400]:
            if ma2 > ma1:
                test_combinations.append((ma1, ma2))
    
    print(f"Testing with {len(test_combinations)} combinations:")
    for i, (ma1, ma2) in enumerate(test_combinations[:5]):
        print(f"  {i+1}. MA1={ma1}, MA2={ma2}")
    if len(test_combinations) > 5:
        print(f"  ... and {len(test_combinations)-5} more")
    print()
    
    # Test each mode
    test_modes = ['serial', 'parallel_process', 'parallel_thread']
    performance_stats = {}
    
    for mode in test_modes:
        print(f"Testing {mode.upper()} mode...")
        
        if mode == 'serial':
            results, exec_time = tester.run_optimization_serial(test_combinations)
        elif mode == 'parallel_process':
            results, exec_time = tester.run_optimization_parallel_process(test_combinations, max_workers=4)
        elif mode == 'parallel_thread':
            results, exec_time = tester.run_optimization_parallel_thread(test_combinations, max_workers=8)
        
        performance_stats[mode] = {
            'time': exec_time,
            'combinations': len(test_combinations),
            'rate': len(test_combinations) / exec_time,
            'successful': len([r for r in results if r['Status'] == 'Success']),
            'failed': len([r for r in results if r['Status'] != 'Success'])
        }
        
        print(f"  Time: {exec_time:.2f}s")
        print(f"  Rate: {performance_stats[mode]['rate']:.2f} backtests/second")
        print(f"  Success: {performance_stats[mode]['successful']}/{len(test_combinations)}")
        print()
    
    # Performance comparison
    print("PERFORMANCE COMPARISON:")
    serial_time = performance_stats['serial']['time']
    
    for mode, stats in performance_stats.items():
        if mode != 'serial':
            speedup = serial_time / stats['time']
            print(f"  {mode.upper()} vs Serial: {speedup:.2f}x speedup")
    
    # Sample results
    if results:
        print(f"\nSAMPLE RESULTS (first 3):")
        for i, result in enumerate(results[:3]):
            print(f"  {i+1}. MA1={result['MA1']}, MA2={result['MA2']}: Return={result['Total_Return']:.2f}%, Sharpe={result['Sharpe_Ratio']:.2f}")
    
    print(f"\nQuick test complete!")
    print(f"System appears ready for full optimization test with {10000:,} combinations")

if __name__ == "__main__":
    quick_test()