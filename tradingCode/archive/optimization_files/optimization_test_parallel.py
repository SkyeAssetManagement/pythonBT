#!/usr/bin/env python3
"""
Parallel Optimization Performance Test
Tests 100 combinations using only parallel processing (no serial)
Optimized for 16 cores / 32 threads
"""

import sys
sys.path.append('src')
sys.path.append('strategies')

from optimization_performance_test import OptimizationPerformanceTester
import logging

# Configure logging for parallel test
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def parallel_test():
    """Run parallel test with 100 parameter combinations"""
    
    print("=" * 60)
    print("PARALLEL OPTIMIZATION PERFORMANCE TEST")
    print("Testing with 100 combinations - 16 cores / 32 threads")
    print("SKIPPING SERIAL MODE (too slow)")
    print("=" * 60)
    
    # Initialize tester
    tester = OptimizationPerformanceTester()
    
    # Load data
    if not tester.load_data():
        print("ERROR: Data loading failed")
        return
    
    # Generate 100 test combinations
    # MA1: 20, 30, 40, 50, 60, 70, 80, 90, 100, 110 (10 values)
    # MA2: 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100 (10 values)  
    # Total: 100 combinations
    
    test_combinations = []
    for ma1 in range(20, 121, 10):  # 20 to 110, step 10
        for ma2 in range(200, 1101, 100):  # 200 to 1100, step 100
            if ma2 > ma1:
                test_combinations.append((ma1, ma2))
    
    print(f"Testing with {len(test_combinations)} combinations")
    print(f"System: {tester.cpu_count} cores, {tester.memory_gb:.1f}GB RAM")
    print()
    
    # Test both parallel modes
    test_modes = ['parallel_process', 'parallel_thread']
    performance_stats = {}
    
    for mode in test_modes:
        print(f"Testing {mode.upper()} mode...")
        
        if mode == 'parallel_process':
            # Use all 16 cores (physical cores)
            results, exec_time = tester.run_optimization_parallel_process(test_combinations, max_workers=16)
        elif mode == 'parallel_thread':
            # Use all 32 logical threads
            results, exec_time = tester.run_optimization_parallel_thread(test_combinations, max_workers=32)
        
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
    process_time = performance_stats['parallel_process']['time']
    thread_time = performance_stats['parallel_thread']['time']
    
    if process_time < thread_time:
        speedup = thread_time / process_time
        print(f"  PARALLEL_PROCESS is {speedup:.2f}x faster than PARALLEL_THREAD")
        best_mode = "PARALLEL_PROCESS"
        best_time = process_time
        best_rate = performance_stats['parallel_process']['rate']
    else:
        speedup = process_time / thread_time
        print(f"  PARALLEL_THREAD is {speedup:.2f}x faster than PARALLEL_PROCESS")
        best_mode = "PARALLEL_THREAD"
        best_time = thread_time
        best_rate = performance_stats['parallel_thread']['rate']
    
    print(f"  WINNER: {best_mode} - {best_rate:.2f} backtests/second")
    print()
    
    # Estimate full test time
    full_combinations = 10000
    estimated_time = full_combinations / best_rate
    estimated_hours = estimated_time / 3600
    estimated_minutes = estimated_time / 60
    
    print("FULL TEST ESTIMATES:")
    print(f"  10,000 combinations using {best_mode}")
    print(f"  Estimated time: {estimated_time:.0f}s ({estimated_minutes:.1f} minutes / {estimated_hours:.2f} hours)")
    print(f"  Estimated rate: {best_rate:.2f} backtests/second")
    
    # Sample successful results
    successful = [r for r in results if r['Status'] == 'Success']
    if successful:
        print(f"\nSAMPLE SUCCESSFUL RESULTS (first 3):")
        for i, result in enumerate(successful[:3]):
            print(f"  {i+1}. MA1={result['MA1']}, MA2={result['MA2']}: Return={result['Total_Return']:.2f}%, Sharpe={result['Sharpe_Ratio']:.2f}, Trades={result['Trade_Count']}")
    
    print(f"\nParallel test complete! System ready for full 10,000 combination test.")
    return True

if __name__ == "__main__":
    success = parallel_test()
    if success:
        print("\nREADY TO RUN: python optimization_performance_test.py")
        print("(Will use parallel processing only, skipping slow serial mode)")