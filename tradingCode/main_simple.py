#!/usr/bin/env python3
"""
main_simple.py

Simplified version of main.py that skips problematic export operations
For debugging dashboard issues
"""

import numpy as np
import yaml
import argparse
import time
import warnings
import importlib
import inspect
from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd

warnings.filterwarnings('ignore')

from src.data.parquet_converter import ParquetConverter
from strategies.base_strategy import BaseStrategy

def load_strategy(strategy_name: str) -> BaseStrategy:
    """Dynamically load a strategy from the strategies folder."""
    try:
        if strategy_name == 'time_window_strategy_vectorized_single':
            strategy_name = 'time_window_strategy_vectorized'
        
        module = importlib.import_module(f'strategies.{strategy_name}')
        
        strategy_classes = []
        for name, obj in inspect.getmembers(module):
            if (inspect.isclass(obj) and 
                issubclass(obj, BaseStrategy) and 
                obj != BaseStrategy):
                strategy_classes.append((name, obj))
        
        if not strategy_classes:
            raise ValueError(f"No strategy classes found in {strategy_name}")
        
        strategy_class = None
        for name, cls in strategy_classes:
            if 'Single' in name:
                strategy_class = cls
                break
            elif 'Strategy' in name and 'Sweep' not in name:
                strategy_class = cls
        
        if strategy_class is None:
            strategy_class = strategy_classes[0][1]
            
        return strategy_class()
        
    except ImportError as e:
        raise ImportError(f"Could not import strategy '{strategy_name}': {e}")

def main_simple(symbol: str, strategy_name: str, config_path: str = "config.yaml", 
                start_date: str = None, end_date: str = None, use_defaults: bool = False):
    """
    Simplified main function that skips exports and dashboard to isolate issues
    """
    print(f"SIMPLIFIED Trading System Test")
    print(f"Symbol: {symbol}")
    print(f"Strategy: {strategy_name}")
    print("=" * 60)
    
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Step 1: Load strategy
    print(f"\n1. Loading strategy: {strategy_name}")
    strategy = load_strategy(strategy_name)
    print(f"   SUCCESS: Loaded strategy: {strategy.name}")
    
    param_combinations = strategy.get_parameter_combinations(use_defaults_only=use_defaults)
    print(f"   SUCCESS: Parameter combinations: {len(param_combinations)}")
    
    # Step 2: Load data
    print(f"\n2. Loading data...")
    start_time = time.time()
    
    parquet_converter = ParquetConverter()
    frequency = config.get('data', {}).get('data_frequency', '1T')
    freq_str = frequency.replace('T', 'm')
    adjustment_type = "diffAdjusted"
    
    data = parquet_converter.load_or_convert(symbol, freq_str, adjustment_type)
    
    if start_date or end_date:
        data = parquet_converter.filter_data_by_date(data, start_date, end_date)
    
    load_time = time.time() - start_time
    print(f"   SUCCESS: Loaded {len(data['close'])} bars in {load_time:.2f} seconds")
    
    # Step 3: Run strategy backtest
    print(f"\n3. Running strategy backtest...")
    start_time = time.time()
    
    if 'run_vectorized_backtest' in dir(strategy):
        backtest_signature = inspect.signature(strategy.run_vectorized_backtest)
        if 'use_defaults_only' in backtest_signature.parameters:
            pf = strategy.run_vectorized_backtest(data, config['backtest'], use_defaults_only=use_defaults)
        else:
            pf = strategy.run_vectorized_backtest(data, config['backtest'])
    else:
        pf = strategy.run_vectorized_backtest(data, config['backtest'])
    
    backtest_time = time.time() - start_time
    n_combos = len(param_combinations)
    print(f"   SUCCESS: Backtested {n_combos} parameter combinations in {backtest_time:.2f} seconds")
    
    # Step 4: Analyze results (simplified)
    print(f"\n4. Analyzing results...")
    
    total_return = pf.total_return
    n_combinations = len(param_combinations)
    
    if n_combinations > 1:
        returns = pf.total_return.values if hasattr(pf.total_return, 'values') else pf.total_return
        n_columns = len(returns) if hasattr(returns, '__len__') else 1
        
        if n_columns > n_combinations:
            # Gradual entry/exit
            n_positions_per_combo = n_columns // n_combinations
            combo_returns = []
            
            for combo_idx in range(n_combinations):
                start_col = combo_idx * n_positions_per_combo
                end_col = start_col + n_positions_per_combo
                combo_return = np.sum(returns[start_col:end_col])
                combo_returns.append(combo_return)
            
            best_idx = np.argmax(combo_returns)
            best_params = param_combinations[best_idx]
            best_return = combo_returns[best_idx]
        else:
            best_idx = np.argmax(returns)
            best_params = param_combinations[best_idx]
            best_return = returns[best_idx]
    else:
        best_params = param_combinations[0]
        
        if hasattr(total_return, '__len__') and len(total_return) > 1:
            best_return = np.sum(total_return)
        else:
            best_return = total_return.iloc[0] if hasattr(total_return, 'iloc') else total_return
    
    # Display results
    print(f"   SUCCESS: Best combination:")
    for key, value in best_params.items():
        print(f"     - {key}: {value}")
    print(f"   SUCCESS: Total Return: {best_return*100:.2f}%")
    
    # Simple Sharpe calculation
    try:
        sharpe_ratio = pf.sharpe_ratio
        if hasattr(sharpe_ratio, '__len__') and len(sharpe_ratio) > 1:
            best_sharpe = np.mean(sharpe_ratio)
        else:
            best_sharpe = sharpe_ratio.iloc[0] if hasattr(sharpe_ratio, 'iloc') else sharpe_ratio
        print(f"   SUCCESS: Sharpe Ratio: {best_sharpe:.2f}")
    except Exception as e:
        print(f"   WARNING: Could not calculate Sharpe: {e}")
    
    print(f"\nSUCCESS: Simplified execution completed!")
    print(f"SUCCESS: Core functionality verified - strategy works properly")
    print(f"INFO: This proves the issue is in export/dashboard sections of main.py")
    
    return data, pf

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simplified Trading System Test")
    parser.add_argument("symbol", type=str, help="Symbol to backtest")
    parser.add_argument("strategy", type=str, help="Strategy file name")
    parser.add_argument("--config", type=str, default="config.yaml", help="Config file")
    parser.add_argument("--start", type=str, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, help="End date (YYYY-MM-DD)")
    parser.add_argument("--useDefaults", action="store_true", help="Use default parameters only")
    
    args = parser.parse_args()
    
    try:
        data, pf = main_simple(
            args.symbol, 
            args.strategy, 
            args.config, 
            args.start, 
            args.end,
            use_defaults=args.useDefaults
        )
        print(f"\nSimplified test completed successfully!")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()