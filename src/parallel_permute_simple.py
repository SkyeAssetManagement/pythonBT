"""
Simplified parallel processing for PermuteAlpha
Uses the existing sequential permutation code but runs multiple in parallel
"""

import os
import sys
import pandas as pd
import numpy as np
import tempfile
import subprocess
import uuid
from multiprocessing import Pool, cpu_count, Lock
import threading
from datetime import datetime
import time
from typing import Dict, List, Tuple, Any


def run_single_permutation_simple(args: Tuple) -> Dict[str, Any]:
    """
    Run a single permutation using the existing GUI code logic
    Each process gets a unique results file to avoid conflicts
    """
    ticker, target, hour, direction, features, base_config_path, output_dir, perm_id = args
    
    # Log the permutation details
    print(f"\n[Worker {perm_id}] Starting: {ticker}_{direction}_{target}_{hour}")
    
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
    
    try:
        import configparser
        import subprocess
        import uuid
        import shutil
        
        # Create a unique identifier for this permutation
        unique_id = f"{ticker}_{direction}_{target}_{hour}_{perm_id}"
        
        # Create a unique working directory for this permutation
        work_dir = tempfile.mkdtemp(prefix=f"perm_{perm_id}_")
        original_dir = os.getcwd()
        print(f"[Worker {perm_id}] Temp directory: {work_dir}")
        
        # Create temporary config in the work directory
        temp_config_path = os.path.join(work_dir, f'config_{unique_id}.ini')
        
        # Read base config
        config = configparser.ConfigParser(inline_comment_prefixes='#')
        config.read(base_config_path)
        
        # Update config for this permutation
        # IMPORTANT: Use ticker_filter and hour_filter, not ticker and hour!
        config['data']['ticker_filter'] = ticker  # Validation code looks for ticker_filter
        config['data']['hour_filter'] = str(hour)  # Validation code looks for hour_filter
        # CRITICAL: Use target_column not target! The validator looks for target_column
        config['data']['target_column'] = target  # Validator uses target_column to select the correct return column
        config['model']['model_type'] = direction
        config['data']['features'] = ','.join(features)
        
        # CRITICAL: Use unique random seed for each worker to avoid identical results!
        unique_seed = 42 + perm_id * 1000  # Each worker gets a different seed
        config['model']['random_seed'] = str(unique_seed)
        print(f"[Worker {perm_id}] Using random_seed: {unique_seed}")
        
        # CRITICAL: Set unique output file for this worker to avoid collisions
        unique_results_file = os.path.join(work_dir, f'results_{perm_id}_{ticker}_{direction}.csv')
        if 'output' not in config:
            config.add_section('output')
        config['output']['results_file'] = unique_results_file
        print(f"[Worker {perm_id}] Output will be written to: {unique_results_file}")
        
        # Update data file path to point to work directory
        if 'data' in config and 'data_file' in config['data']:
            data_filename = os.path.basename(config['data']['data_file'])
            config['data']['data_file'] = os.path.join(work_dir, data_filename)
        
        # Write config to work directory
        with open(temp_config_path, 'w') as f:
            config.write(f)
        
        # Also write as OMtree_config.ini since walkforward expects that name
        omtree_config_path = os.path.join(work_dir, 'OMtree_config.ini')
        with open(omtree_config_path, 'w') as f:
            config.write(f)
        print(f"[Worker {perm_id}] Created config at: {omtree_config_path}")
        
        # Copy necessary data files to work directory
        data_files = ['OMtree_data.csv', 'data.csv']
        for data_file in data_files:
            src_path = os.path.join(original_dir, data_file)
            if os.path.exists(src_path):
                shutil.copy(src_path, os.path.join(work_dir, data_file))
                print(f"[Worker {perm_id}] Copied {data_file} to work directory")
        
        # Create results directory in work_dir to prevent log file collisions
        results_dir = os.path.join(work_dir, 'results')
        os.makedirs(results_dir, exist_ok=True)
        print(f"[Worker {perm_id}] Created results directory: {results_dir}")
        
        # We need to run from work_dir so relative paths like 'results/' work correctly
        
        # Create a local runner script that imports walkforward without path manipulation
        runner_script = os.path.join(work_dir, 'run_walkforward.py')
        runner_code = f'''
import sys
import os
# Add original directory to path BUT AFTER current directory
sys.path.append(r"{original_dir}")
# Now import and run walkforward
from src import OMtree_walkforward
'''
        with open(runner_script, 'w') as f:
            f.write(runner_code)
        print(f"[Worker {perm_id}] Created runner script: {runner_script}")
        
        # Set up environment - work_dir MUST come before original_dir in Python path
        # so that OMtree_config.ini in work_dir is found first
        env = os.environ.copy()
        if 'PYTHONPATH' in env:
            env['PYTHONPATH'] = f"{work_dir}{os.pathsep}{original_dir}{os.pathsep}{env['PYTHONPATH']}"
        else:
            env['PYTHONPATH'] = f"{work_dir}{os.pathsep}{original_dir}"
        
        # Set flag to skip debug CSV generation during permutation runs
        env['PERMUTATION_RUN'] = '1'
        
        # Run our local runner script instead of the original walkforward
        cmd = [sys.executable, runner_script]
        
        # Run the process from work_dir so relative paths work correctly
        print(f"[Worker {perm_id}] Running walkforward with config:")
        print(f"  - Ticker: {ticker}, Target: {target}, Hour: {hour}, Direction: {direction}")
        print(f"  - Config file: {omtree_config_path}")
        print(f"  - Expected output: {unique_results_file}")
        print(f"  - Working directory: {work_dir}")
        print(f"[Worker {perm_id}] Command: {' '.join(cmd)}")
        process_result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
            cwd=work_dir,  # Run from work_dir so each worker has isolated file writes
            env=env
        )
        print(f"[Worker {perm_id}] Process returned code: {process_result.returncode}")
        
        # Debug output
        if process_result.returncode != 0:
            error_msg = f"Process failed with code {process_result.returncode}\nSTDOUT: {process_result.stdout[:500]}\nSTDERR: {process_result.stderr[:500]}"
            print(f"[Worker {perm_id}] ERROR: {error_msg}")
            result['error'] = error_msg
            result['timing'] = time.time() - start_time
            return result
            
        if process_result.returncode == 0:
            # Look for the unique results file we configured
            results_csv = unique_results_file
            print(f"[Worker {perm_id}] Looking for results at: {results_csv}")
            
            if os.path.exists(results_csv):
                # Load and process results
                print(f"[Worker {perm_id}] Found results file: {results_csv}")
                df = pd.read_csv(results_csv)
                print(f"[Worker {perm_id}] Loaded {len(df)} rows from results")
                
                # Aggregate to daily level (matching GUI code)
                if 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date'])
                    df.set_index('date', inplace=True)
                
                # Create daily returns with index from df
                daily_returns = pd.DataFrame(index=df.index)
                
                if 'prediction' in df.columns and 'target_value' in df.columns:
                    # Only include days with trades
                    daily_returns['Return'] = df['target_value'].copy()
                    daily_returns['TradeFlag'] = df['prediction'].astype(int)
                else:
                    daily_returns['Return'] = 0
                    daily_returns['TradeFlag'] = 0
                
                # Load raw data and join NoneTradePrice values
                try:
                    raw_data_path = config['data']['csv_file']
                    if os.path.exists(raw_data_path):
                        print(f"[Worker {perm_id}] Loading NoneTradePrice from {raw_data_path}")
                        raw_df = pd.read_csv(raw_data_path)
                        print(f"[Worker {perm_id}] Raw data has {len(raw_df)} rows")
                        
                        # Filter raw data to match this permutation
                        if ticker != 'ALL' and 'Ticker' in raw_df.columns:
                            raw_df = raw_df[raw_df['Ticker'] == ticker].copy()
                            print(f"[Worker {perm_id}] After ticker filter: {len(raw_df)} rows")
                        if hour != 'ALL' and 'Hour' in raw_df.columns:
                            raw_df = raw_df[raw_df['Hour'] == int(hour)].copy()
                            print(f"[Worker {perm_id}] After hour filter: {len(raw_df)} rows")
                        
                        # Prepare raw data for joining - combine Date and Time to match walkforward format
                        # The walkforward results have datetime with time, raw data has separate columns
                        if 'Time' in raw_df.columns:
                            # Combine Date and Time columns to create full datetime
                            raw_df['datetime_str'] = raw_df['Date'].astype(str) + ' ' + raw_df['Time'].astype(str)
                            raw_df['date'] = pd.to_datetime(raw_df['datetime_str'])
                        else:
                            raw_df['date'] = pd.to_datetime(raw_df['Date'])
                        
                        raw_df.set_index('date', inplace=True)
                        print(f"[Worker {perm_id}] Raw data date range: {raw_df.index.min()} to {raw_df.index.max()}")
                        print(f"[Worker {perm_id}] Results date range: {daily_returns.index.min()} to {daily_returns.index.max()}")
                        
                        # Join NoneTradePrice using index alignment (both DataFrames have date as index)
                        # Also handle backward compatibility with NoneClose column name
                        if 'NoneTradePrice' in raw_df.columns:
                            # Use index alignment to join - this handles the date matching automatically
                            daily_returns['NoneTradePrice'] = raw_df['NoneTradePrice']
                            
                            # Check how many matched
                            matched_count = daily_returns['NoneTradePrice'].notna().sum()
                            print(f"[Worker {perm_id}] Matched {matched_count} out of {len(daily_returns)} rows")
                            
                            # If no matches, there might be a date format issue
                            if matched_count == 0:
                                print(f"[Worker {perm_id}] WARNING: No dates matched! Check date formats.")
                                # Try to show a few dates from each for debugging
                                print(f"[Worker {perm_id}] Sample result dates: {daily_returns.index[:3].tolist()}")
                                print(f"[Worker {perm_id}] Sample raw dates: {raw_df.index[:3].tolist()}")
                            else:
                                # Fill any NaN values that might result from missing dates
                                daily_returns['NoneTradePrice'] = daily_returns['NoneTradePrice'].ffill()
                                print(f"[Worker {perm_id}] Added NoneTradePrice values - final non-null count: {daily_returns['NoneTradePrice'].notna().sum()}")
                        elif 'NoneClose' in raw_df.columns:
                            # Backward compatibility - if old column name exists, use it but rename
                            print(f"[Worker {perm_id}] Using NoneClose column (legacy name) as NoneTradePrice")
                            daily_returns['NoneTradePrice'] = raw_df['NoneClose']
                            matched_count = daily_returns['NoneTradePrice'].notna().sum()
                            if matched_count > 0:
                                daily_returns['NoneTradePrice'] = daily_returns['NoneTradePrice'].ffill()
                                print(f"[Worker {perm_id}] Added NoneTradePrice values - final non-null count: {daily_returns['NoneTradePrice'].notna().sum()}")
                        else:
                            print(f"[Worker {perm_id}] Warning: NoneTradePrice or NoneClose column not found in raw data")
                except Exception as e:
                    print(f"[Worker {perm_id}] Warning: Could not add NoneTradePrice: {e}")
                    import traceback
                    traceback.print_exc()
                
                # Save output file without perm_id suffix
                output_file = f"{ticker}_{direction}_{target}_{hour}.csv"
                
                # Ensure output directory exists
                full_output_dir = os.path.join(original_dir, output_dir)
                if not os.path.exists(full_output_dir):
                    os.makedirs(full_output_dir)
                    print(f"[Worker {perm_id}] Created output directory: {full_output_dir}")
                
                output_path = os.path.join(full_output_dir, output_file)
                print(f"[Worker {perm_id}] Saving {ticker}/{direction}/{target}/{hour} to: {output_path}")
                
                # Add metadata to CSV to verify correct association
                daily_returns['_perm_id'] = perm_id
                daily_returns['_ticker'] = ticker
                daily_returns['_direction'] = direction
                daily_returns['_target'] = target
                daily_returns['_hour'] = hour
                
                # For shortonly, flip the returns in the CSV output to show P&L from short perspective
                if direction == 'shortonly':
                    daily_returns_output = daily_returns.copy()
                    daily_returns_output['Return'] = -daily_returns_output['Return']
                    daily_returns_output.to_csv(output_path)
                else:
                    daily_returns.to_csv(output_path)
                print(f"[Worker {perm_id}] Successfully saved {len(daily_returns)} rows with metadata")
                
                # Calculate metrics (matching GUI code exactly)
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
                
                # Model metrics - ONLY compound returns when we have trades
                # When TradeFlag == 0, return should be 0 (no position)
                trade_adjusted_returns = daily_returns.copy()
                trade_adjusted_returns.loc[trade_adjusted_returns['TradeFlag'] == 0, 'Return'] = 0
                
                # For shortonly, invert the returns since shorts profit from negative moves
                if direction == 'shortonly':
                    daily_returns_decimal = -trade_adjusted_returns['Return'] / 100.0
                else:
                    daily_returns_decimal = trade_adjusted_returns['Return'] / 100.0
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
                
                # Sharpe - using trade-adjusted returns (0 when not in trade)
                if daily_returns_decimal.std() > 0:
                    metrics['sharpe'] = (daily_returns_decimal.mean() * 252) / (daily_returns_decimal.std() * np.sqrt(252))
                else:
                    metrics['sharpe'] = 0
                
                # Profit/DD
                if metrics['max_draw_pct'] > 0:
                    metrics['profit_dd_ratio'] = metrics['avg_annual_pct'] / metrics['max_draw_pct']
                else:
                    metrics['profit_dd_ratio'] = 0
                
                # UPI
                if len(drawdown_pct) > 0:
                    rms_dd = np.sqrt(np.mean(drawdown_pct**2))
                    metrics['upi'] = metrics['avg_annual_pct'] / rms_dd if rms_dd > 0 else 0
                else:
                    metrics['upi'] = 0
                
                result['success'] = True
                result['metrics'] = metrics
                result['output_file'] = output_file
                print(f"[Worker {perm_id}] SUCCESS - Metrics: Sharpe={metrics.get('sharpe', 0):.2f}, Annual%={metrics.get('avg_annual_pct', 0):.2f}")
            else:
                error_msg = f"No results file found. Checked: {results_csv}"
                print(f"[Worker {perm_id}] ERROR: {error_msg}")
                result['error'] = error_msg
        
        # Clean up work directory
        try:
            if os.path.exists(work_dir):
                print(f"[Worker {perm_id}] Cleaning up temp dir: {work_dir}")
                shutil.rmtree(work_dir)
        except Exception as cleanup_error:
            print(f"[Worker {perm_id}] Warning: Could not clean up {work_dir}: {cleanup_error}")
            
    except Exception as e:
        error_msg = str(e)
        print(f"[Worker {perm_id}] EXCEPTION: {error_msg}")
        import traceback
        print(f"[Worker {perm_id}] Traceback: {traceback.format_exc()}")
        result['error'] = error_msg
    
    result['timing'] = time.time() - start_time
    return result


class SimpleParallelPermute:
    """Simplified parallel permutation runner"""
    
    def __init__(self, n_workers=None):
        self.n_workers = n_workers or max(1, cpu_count() - 1)
        self.pool = None
        self.should_stop = False
        
    def stop(self):
        """Signal to stop processing"""
        self.should_stop = True
        if self.pool:
            self.pool.terminate()
            self.pool.join()
            
    def run(self, combinations, features, config_file, output_dir, 
            progress_callback=None, result_callback=None, check_stop=None):
        """Run permutations in parallel
        
        Args:
            check_stop: Optional callback that returns True if processing should stop
        """
        
        # Prepare arguments
        args_list = []
        for idx, (ticker, target, hour, direction) in enumerate(combinations):
            args = (ticker, target, hour, direction, features, config_file, output_dir, idx)
            args_list.append(args)
        
        results = []
        failed = []
        self.should_stop = False
        
        try:
            self.pool = Pool(processes=self.n_workers)
            # Use imap_unordered for results as they complete
            iterator = self.pool.imap_unordered(run_single_permutation_simple, args_list)
            
            for i, result in enumerate(iterator):
                # Check if we should stop
                if self.should_stop or (check_stop and check_stop()):
                    print("Permutation processing stopped by user")
                    self.pool.terminate()
                    break
                    
                if result['success']:
                    results.append(result)
                    if result_callback:
                        result_callback(result)
                else:
                    failed.append(f"{result['ticker']}_{result['direction']}_{result['target']}_{result['hour']}: {result['error']}")
                
                if progress_callback:
                    progress_callback(i + 1, len(args_list))
        
        finally:
            if self.pool:
                self.pool.close()
                self.pool.join()
                self.pool = None
        
        return results, failed