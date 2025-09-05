"""
Parallel processing module for PermuteAlpha functionality
Speeds up permutation testing by running multiple combinations in parallel
"""

import os
import sys
import pandas as pd
import numpy as np
import tempfile
import subprocess
import json
from multiprocessing import Pool, Queue, Manager, cpu_count
from datetime import datetime
import time
from typing import Dict, List, Tuple, Any
import queue


def run_single_permutation(args: Tuple) -> Dict[str, Any]:
    """
    Run a single permutation combination
    This function is designed to be run in parallel
    Each worker gets its own isolated temporary directory to avoid conflicts
    
    Args:
        args: Tuple containing (ticker, target, hour, direction, features, config_path, output_dir, permutation_id)
    
    Returns:
        Dictionary with results or error information
    """
    ticker, target, hour, direction, features, base_config_path, output_dir, perm_id = args
    
    result = {
        'permutation_id': perm_id,
        'ticker': ticker,
        'target': target,
        'hour': hour,
        'direction': direction,
        'success': False,
        'error': None,
        'metrics': {},
        'output_file': None,
        'timing': 0
    }
    
    start_time = time.time()
    
    # Create isolated temporary directory for this worker
    temp_dir = tempfile.mkdtemp(prefix=f"permute_{perm_id}_")
    
    try:
        # Create temporary config file in isolated directory
        temp_config_path = os.path.join(temp_dir, f"config_{perm_id}.ini")
        
        # Read base config
        import configparser
        config = configparser.ConfigParser(inline_comment_prefixes='#')
        config.read(base_config_path)
        
        # Update config for this permutation
        if ticker != 'ALL':
            config['data']['ticker'] = ticker
        if hour != 'ALL':
            config['data']['hour'] = str(hour)
        config['data']['target'] = target
        config['model']['model_type'] = direction
        
        # Update features - convert list to comma-separated string
        config['data']['features'] = ','.join(features)
        
        # CRITICAL: Set output to isolated directory to avoid conflicts
        config['walkforward']['results_file'] = os.path.join(temp_dir, f'results_{perm_id}.csv')
        
        # Write temporary config
        with open(temp_config_path, 'w') as f:
            config.write(f)
        
        # Keep original directory 
        original_cwd = os.getcwd()
        
        try:
            # Run walk-forward validation - use full paths
            walkforward_script = os.path.join(original_cwd, 'OMtree_walkforward.py')
            
            # Make sure the script exists
            if not os.path.exists(walkforward_script):
                result['error'] = f"Walk-forward script not found: {walkforward_script}"
                return result
            
            cmd = [sys.executable, walkforward_script, temp_config_path]
            
            result_process = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout per permutation
                cwd=original_cwd  # Run from original directory so imports work
            )
            
            if result_process.returncode == 0:
                # Look for results file - it might be in current directory, not temp
                results_csv = os.path.join(original_cwd, 'OMtree_results.csv')
                
                # Also check temp directory
                if not os.path.exists(results_csv):
                    results_csv = os.path.join(temp_dir, f'results_{perm_id}.csv')
                if not os.path.exists(results_csv):
                    results_csv = os.path.join(temp_dir, 'OMtree_results.csv')
                
                # Load results and calculate metrics
                if os.path.exists(results_csv):
                    # Read the walk-forward results
                    wf_results = pd.read_csv(results_csv)
                    
                    # Convert to daily returns format
                    daily_returns = pd.DataFrame()
                    
                    # Handle different column names that might exist
                    if 'date' in wf_results.columns:
                        daily_returns['Date'] = pd.to_datetime(wf_results['date'])
                    elif 'Date' in wf_results.columns:
                        daily_returns['Date'] = pd.to_datetime(wf_results['Date'])
                    else:
                        daily_returns['Date'] = pd.date_range(start='2020-01-01', periods=len(wf_results))
                    
                    # Get return values - handle different possible column names
                    if 'target_value' in wf_results.columns:
                        daily_returns['Return'] = wf_results['target_value']
                    elif 'pnl' in wf_results.columns:
                        daily_returns['Return'] = wf_results['pnl']
                    elif 'return' in wf_results.columns:
                        daily_returns['Return'] = wf_results['return']
                    else:
                        daily_returns['Return'] = 0
                    
                    # Get trade flags
                    if 'prediction' in wf_results.columns:
                        daily_returns['TradeFlag'] = wf_results['prediction'].astype(int)
                    elif 'signal' in wf_results.columns:
                        daily_returns['TradeFlag'] = wf_results['signal'].astype(int)
                    else:
                        daily_returns['TradeFlag'] = (daily_returns['Return'] != 0).astype(int)
                    
                    # Save detailed results to final output directory
                    output_file = f"{ticker}_{direction}_{target}_{hour}.csv"
                    output_path = os.path.join(original_cwd, output_dir, output_file)
                    
                    # For shortonly, flip the returns in the CSV output to show P&L from short perspective
                    if direction == 'shortonly':
                        daily_returns_output = daily_returns.copy()
                        daily_returns_output['Return'] = -daily_returns_output['Return']
                        daily_returns_output.to_csv(output_path, index=False)
                    else:
                        daily_returns.to_csv(output_path, index=False)
                    
                    # Calculate tradestats metrics
                    metrics = calculate_metrics_for_permutation(daily_returns, direction)
                    
                    result['success'] = True
                    result['metrics'] = metrics
                    result['output_file'] = output_file
                else:
                    result['error'] = f"No results file found in {temp_dir}"
            else:
                result['error'] = f"Process failed with code {result_process.returncode}: {result_process.stderr[:200]}"
                
        except Exception as inner_e:
            result['error'] = f"Inner exception: {str(inner_e)}"
        
    except subprocess.TimeoutExpired:
        result['error'] = "Permutation timed out after 5 minutes"
    except Exception as e:
        result['error'] = f"Exception: {str(e)}"
    finally:
        # Clean up temporary directory
        try:
            import shutil
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
        except:
            pass  # Don't fail if cleanup fails
    
    result['timing'] = time.time() - start_time
    return result


def calculate_metrics_for_permutation(daily_returns: pd.DataFrame, direction: str) -> Dict[str, float]:
    """
    Calculate all tradestats metrics for a permutation
    Matches the metrics calculation in OMtree_gui.py
    """
    total_days = len(daily_returns)
    num_trades = daily_returns['TradeFlag'].sum()
    years = total_days / 252.0
    
    # Get trade returns only (convert from percentage to decimal)
    # IMPORTANT: For shortonly, invert returns since shorts profit from negative moves
    if direction == 'shortonly':
        trade_returns = -daily_returns[daily_returns['TradeFlag'] == 1]['Return'] / 100.0
    else:
        trade_returns = daily_returns[daily_returns['TradeFlag'] == 1]['Return'] / 100.0
    
    metrics = {}
    metrics['num_observations'] = total_days
    metrics['years_of_data'] = years
    metrics['num_trades'] = int(num_trades)
    metrics['trade_frequency_pct'] = (num_trades / total_days * 100) if total_days > 0 else 0
    metrics['avg_trades_pa'] = num_trades / years if years > 0 else 0
    metrics['avg_trades_pm'] = metrics['avg_trades_pa'] / 12.0
    
    if len(trade_returns) > 0:
        winning_trades = trade_returns[trade_returns > 0]
        losing_trades = trade_returns[trade_returns < 0]
        
        metrics['win_pct'] = (len(winning_trades) / len(trade_returns)) * 100
        metrics['avg_loss_pct'] = losing_trades.mean() * 100 if len(losing_trades) > 0 else 0
        metrics['avg_profit_pct'] = winning_trades.mean() * 100 if len(winning_trades) > 0 else 0
        metrics['avg_pnl_pct'] = trade_returns.mean() * 100
        
        win_rate = metrics['win_pct'] / 100
        metrics['expectancy'] = (win_rate * metrics['avg_profit_pct']) - ((1 - win_rate) * abs(metrics['avg_loss_pct']))
        metrics['best_day_pct'] = trade_returns.max() * 100
        metrics['worst_day_pct'] = trade_returns.min() * 100
    else:
        metrics.update({
            'win_pct': 0, 'avg_loss_pct': 0, 'avg_profit_pct': 0,
            'avg_pnl_pct': 0, 'expectancy': 0, 'best_day_pct': 0, 'worst_day_pct': 0
        })
    
    # Calculate compound returns for model metrics
    # For shortonly, invert the returns since shorts profit from negative moves
    if direction == 'shortonly':
        daily_returns_decimal = -daily_returns['Return'] / 100.0
    else:
        daily_returns_decimal = daily_returns['Return'] / 100.0
    cumulative_returns = (1 + daily_returns_decimal).cumprod()
    final_value = cumulative_returns.iloc[-1] if len(cumulative_returns) > 0 else 1
    
    if years > 0 and final_value > 0:
        metrics['avg_annual_pct'] = ((final_value ** (1/years)) - 1) * 100
    else:
        metrics['avg_annual_pct'] = 0
    
    # Maximum Drawdown
    running_max = cumulative_returns.expanding().max()
    drawdown_pct = ((cumulative_returns - running_max) / running_max) * 100
    metrics['max_draw_pct'] = abs(drawdown_pct.min()) if len(drawdown_pct) > 0 else 0
    
    # Sharpe Ratio
    if daily_returns_decimal.std() > 0:
        metrics['sharpe'] = (daily_returns_decimal.mean() * 252) / (daily_returns_decimal.std() * np.sqrt(252))
    else:
        metrics['sharpe'] = 0
    
    # Profit to DD Ratio
    if metrics['max_draw_pct'] > 0:
        metrics['profit_dd_ratio'] = metrics['avg_annual_pct'] / metrics['max_draw_pct']
    else:
        metrics['profit_dd_ratio'] = 0 if metrics['avg_annual_pct'] <= 0 else float('inf')
    
    # UPI
    if len(drawdown_pct) > 0:
        rms_dd = np.sqrt(np.mean(drawdown_pct**2))
        metrics['upi'] = metrics['avg_annual_pct'] / rms_dd if rms_dd > 0 else 0
    else:
        metrics['upi'] = 0
    
    return metrics


class ParallelPermuteAlpha:
    """
    Parallel execution engine for PermuteAlpha
    """
    
    def __init__(self, n_workers: int = None, progress_callback=None, log_callback=None):
        """
        Initialize parallel permutation engine
        
        Args:
            n_workers: Number of parallel workers (None = use all CPU cores - 1)
            progress_callback: Function to call with progress updates
            log_callback: Function to call with log messages
        """
        self.n_workers = n_workers or max(1, cpu_count() - 1)
        self.progress_callback = progress_callback
        self.log_callback = log_callback
        self.stop_flag = False
        
    def run_permutations(self, 
                        tickers: List[str], 
                        targets: List[str], 
                        hours: List[str], 
                        directions: List[str],
                        features: List[str],
                        base_config_path: str,
                        output_dir: str,
                        result_callback=None) -> Tuple[List[Dict], List[str]]:
        """
        Run all permutation combinations in parallel
        
        Args:
            result_callback: Optional callback for each completed result
        
        Returns:
            Tuple of (successful_results, failed_permutations)
        """
        import itertools
        from multiprocessing import Pool
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        # Generate all combinations
        combinations = list(itertools.product(tickers, targets, hours, directions))
        total_combinations = len(combinations)
        
        if self.log_callback:
            self.log_callback(f"Starting {total_combinations} permutations with {self.n_workers} workers")
        
        # Prepare arguments for parallel execution
        args_list = []
        for idx, (ticker, target, hour, direction) in enumerate(combinations):
            args = (ticker, target, hour, direction, features, base_config_path, output_dir, idx)
            args_list.append(args)
        
        # Run in parallel with progress tracking
        successful_results = []
        failed_permutations = []
        completed = 0
        
        # Use multiprocessing Pool with imap_unordered for results as they complete
        with Pool(processes=self.n_workers) as pool:
            try:
                # Use imap_unordered to get results as they complete
                for result in pool.imap_unordered(run_single_permutation, args_list):
                    if self.stop_flag:
                        pool.terminate()
                        break
                    
                    completed += 1
                    
                    if result['success']:
                        successful_results.append(result)
                        # Call callback to update GUI immediately
                        if result_callback:
                            result_callback(result)
                    else:
                        failed_msg = f"{result['ticker']}_{result['direction']}_{result['target']}_{result['hour']}: {result['error']}"
                        failed_permutations.append(failed_msg)
                        if self.log_callback:
                            self.log_callback(f"Failed: {failed_msg}")
                    
                    if self.progress_callback:
                        self.progress_callback(completed, total_combinations)
                    
            except KeyboardInterrupt:
                pool.terminate()
                pool.join()
                if self.log_callback:
                    self.log_callback("Interrupted by user")
        
        if self.log_callback:
            self.log_callback(f"Completed: {len(successful_results)} successful, {len(failed_permutations)} failed")
        
        return successful_results, failed_permutations
    
    def stop(self):
        """Stop the parallel execution"""
        self.stop_flag = True


def estimate_time_savings(n_permutations: int, avg_time_per_permutation: float, n_workers: int = None) -> Dict[str, float]:
    """
    Estimate time savings from parallel execution
    
    Returns:
        Dictionary with timing estimates
    """
    if n_workers is None:
        n_workers = max(1, cpu_count() - 1)
    
    sequential_time = n_permutations * avg_time_per_permutation
    
    # Account for overhead and non-perfect parallelization
    overhead_factor = 1.1  # 10% overhead
    parallel_time = (n_permutations * avg_time_per_permutation / n_workers) * overhead_factor
    
    return {
        'sequential_time_seconds': sequential_time,
        'parallel_time_seconds': parallel_time,
        'speedup_factor': sequential_time / parallel_time,
        'time_saved_seconds': sequential_time - parallel_time,
        'n_workers': n_workers
    }


if __name__ == "__main__":
    # Test example
    print(f"CPU cores available: {cpu_count()}")
    print(f"Recommended workers: {max(1, cpu_count() - 1)}")
    
    # Example timing estimate
    estimates = estimate_time_savings(
        n_permutations=100,
        avg_time_per_permutation=5.0,
        n_workers=4
    )
    
    print("\nTime estimates for 100 permutations at 5 seconds each:")
    print(f"Sequential: {estimates['sequential_time_seconds']/60:.1f} minutes")
    print(f"Parallel ({estimates['n_workers']} workers): {estimates['parallel_time_seconds']/60:.1f} minutes")
    print(f"Speedup: {estimates['speedup_factor']:.1f}x faster")
    print(f"Time saved: {estimates['time_saved_seconds']/60:.1f} minutes")