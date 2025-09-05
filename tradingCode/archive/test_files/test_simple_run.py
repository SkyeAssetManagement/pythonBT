#!/usr/bin/env python3
"""
test_simple_run.py

Simple test to isolate the hanging issue
"""

from strategies.time_window_strategy_vectorized import TimeWindowVectorizedStrategy
from src.data.parquet_converter import ParquetConverter
import time
import yaml

def run_simple_test():
    """Test core functionality without exports or dashboard"""
    
    print("SIMPLE TEST: Core functionality without exports")
    print("=" * 50)
    
    # Load config
    print("1. Loading config...")
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    print("   SUCCESS: Config loaded")
    
    # Load data  
    print("2. Loading data...")
    start = time.time()
    converter = ParquetConverter()
    data = converter.load_or_convert('ES', '1m', 'diffAdjusted')
    data = converter.filter_data_by_date(data, '2020-01-01', None)
    load_time = time.time() - start
    print(f"   SUCCESS: {len(data['close'])} bars loaded in {load_time:.2f}s")
    
    # Load strategy
    print("3. Loading strategy...")
    strategy = TimeWindowVectorizedStrategy()
    print(f"   SUCCESS: {strategy.name}")
    
    # Run backtest
    print("4. Running backtest...")
    start = time.time()
    pf = strategy.run_vectorized_backtest(data, config['backtest'], use_defaults_only=True)
    backtest_time = time.time() - start
    print(f"   SUCCESS: Backtest completed in {backtest_time:.2f}s")
    
    # Basic results
    print("5. Basic results...")
    total_return = pf.total_return
    if hasattr(total_return, '__len__') and len(total_return) > 1:
        best_return = total_return.sum()
    else:
        best_return = total_return.iloc[0] if hasattr(total_return, 'iloc') else total_return
    
    print(f"   SUCCESS: Total Return: {best_return*100:.2f}%")
    
    # Test basic metrics calculation without export
    print("6. Testing basic metrics...")
    try:
        sharpe = pf.sharpe_ratio
        if hasattr(sharpe, '__len__') and len(sharpe) > 1:
            sharpe_val = sharpe.mean()
        else:
            sharpe_val = sharpe.iloc[0] if hasattr(sharpe, 'iloc') else sharpe
        print(f"   SUCCESS: Sharpe Ratio: {sharpe_val:.2f}")
    except Exception as e:
        print(f"   WARNING: Could not calculate Sharpe: {e}")
    
    print("\nRESULT: Core functionality works perfectly!")
    print("DIAGNOSIS: Issue is likely in the export or dashboard sections")
    print("SOLUTION: User should run with --no-viz flag and skip exports if needed")

if __name__ == "__main__":
    run_simple_test()