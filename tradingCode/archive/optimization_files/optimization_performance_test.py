#!/usr/bin/env python3
"""
Optimization Performance Test
Tests parallel processing and multithreading performance of the backtesting engine

Test Parameters:
- Simple MA Crossover System
- Data: 1 Jan 2000 to 31 Dec 2024 (AD diff data)
- MA1: 10 to 1000, step 10 (100 values)
- MA2: 100 to 10000, step 100 (100 values)
- Total: 10,000 optimization combinations

Metrics: Total Return, Win/Loss Ratio, Sharpe, Trade Count, Average Trade, Max Drawdown
Performance: Load time, optimization time, save time with parallel processing analysis
"""

import sys
import os
import time
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import logging
from datetime import datetime, timedelta
import psutil
import warnings
from tqdm import tqdm

warnings.filterwarnings('ignore')

# Add project paths
sys.path.append('src')
sys.path.append('strategies')

from src.data.parquet_loader import ParquetLoader
from src.data.parquet_converter import ParquetConverter
from src.data.array_validator import ArrayValidator
from src.backtest.vbt_engine import VectorBTEngine
from strategies.simpleSMA import SimpleSMAStrategy

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('optimization_performance.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class OptimizationPerformanceTester:
    """High-performance optimization testing with comprehensive metrics"""
    
    def __init__(self):
        self.start_date = "2000-01-01"
        self.end_date = "2024-12-31"
        self.symbol = "GC"  # Gold futures
        self.data = None
        self.results = []
        
        # Performance tracking
        self.timing_results = {
            'data_load_time': 0,
            'optimization_time': 0,
            'results_save_time': 0,
            'total_time': 0
        }
        
        # System info
        self.cpu_count = mp.cpu_count()
        self.memory_gb = psutil.virtual_memory().total / (1024**3)
        
        logger.info(f"Initialized OptimizationPerformanceTester")
        logger.info(f"System: {self.cpu_count} CPUs, {self.memory_gb:.1f}GB RAM")
        logger.info(f"Date range: {self.start_date} to {self.end_date}")
    
    def load_data(self) -> bool:
        """Load GC data for the specified date range with performance timing"""
        
        logger.info("=== DATA LOADING PHASE ===")
        start_time = time.time()
        
        try:
            # Try ParquetConverter first (most optimized)
            logger.info("Attempting optimized Parquet data loading...")
            parquet_converter = ParquetConverter()
            
            # Load full dataset first, then filter (optimal for parquet)
            self.data = parquet_converter.load_or_convert(self.symbol, "1m", "diffAdjusted")
            
            if self.data:
                # Apply date filtering
                self.data = parquet_converter.filter_data_by_date(
                    self.data, self.start_date, self.end_date
                )
                
                if self.data and len(self.data['close']) > 0:
                    load_time = time.time() - start_time
                    self.timing_results['data_load_time'] = load_time
                    
                    logger.info(f"SUCCESS: Loaded {len(self.data['close']):,} bars in {load_time:.2f}s")
                    logger.info(f"Data range: {pd.to_datetime(self.data['datetime_ns'][0]).strftime('%Y-%m-%d')} to {pd.to_datetime(self.data['datetime_ns'][-1]).strftime('%Y-%m-%d')}")
                    logger.info(f"Price range: ${np.min(self.data['low']):.2f} - ${np.max(self.data['high']):.2f}")
                    
                    return True
            
            # Fallback to legacy parquet loader
            logger.info("Trying legacy parquet loader...")
            parquet_root = Path(__file__).parent.parent / "parquet_data"
            if parquet_root.exists():
                parquet_loader = ParquetLoader(str(parquet_root))
                self.data = parquet_loader.load_symbol_data(self.symbol)
                
                if self.data and len(self.data['close']) > 0:
                    load_time = time.time() - start_time
                    self.timing_results['data_load_time'] = load_time
                    
                    logger.info(f"SUCCESS: Loaded {len(self.data['close']):,} bars from legacy loader in {load_time:.2f}s")
                    return True
            
            logger.error("No data sources available")
            return False
            
        except Exception as e:
            logger.error(f"Data loading failed: {e}")
            return False
    
    def generate_optimization_parameters(self) -> List[Tuple[int, int]]:
        """Generate all optimization parameter combinations"""
        
        logger.info("=== PARAMETER GENERATION ===")
        
        # MA1: 10 to 1000, step 10 (100 values)
        ma1_values = list(range(10, 1001, 10))
        
        # MA2: 100 to 10000, step 100 (100 values)  
        ma2_values = list(range(100, 10001, 100))
        
        # Generate all combinations
        param_combinations = []
        for ma1 in ma1_values:
            for ma2 in ma2_values:
                if ma2 > ma1:  # Only valid combinations where MA2 > MA1
                    param_combinations.append((ma1, ma2))
        
        logger.info(f"Generated {len(param_combinations):,} parameter combinations")
        logger.info(f"MA1 range: {min(ma1_values)} to {max(ma1_values)} (step 10)")
        logger.info(f"MA2 range: {min(ma2_values)} to {max(ma2_values)} (step 100)")
        logger.info(f"Valid combinations (MA2 > MA1): {len(param_combinations):,}")
        
        return param_combinations
    
    def run_single_backtest(self, params: Tuple[int, int]) -> Dict[str, Any]:
        """Run a single backtest with given parameters using direct VectorBT approach"""
        
        ma1, ma2 = params
        
        try:
            import vectorbtpro as vbt
            
            # Create strategy with these parameters
            strategy = SimpleSMAStrategy()
            
            # Generate signals using the strategy with custom parameters
            entries, exits = strategy._generate_signals_for_params(
                self.data, 
                {'fast_period': ma1, 'slow_period': ma2}
            )
            close_prices = self.data['close']
            
            # Create portfolio using VectorBT directly (bypass config file requirement)
            portfolio = vbt.Portfolio.from_signals(
                close=close_prices,
                entries=entries,
                exits=exits,
                size=np.inf,  # Use all available cash
                init_cash=100000,
                direction='both',
                freq='1min'
            )
            
            # Calculate metrics
            total_return = (portfolio.value.iloc[-1] / portfolio.value.iloc[0] - 1) * 100
            
            # Trade analysis
            trades = portfolio.trades.records_readable
            trade_count = len(trades) if len(trades) > 0 else 0
            
            if trade_count > 0:
                winning_trades = len(trades[trades['PnL'] > 0])
                losing_trades = len(trades[trades['PnL'] <= 0])
                win_loss_ratio = winning_trades / max(losing_trades, 1)
                avg_trade = trades['PnL'].mean()
            else:
                win_loss_ratio = 0
                avg_trade = 0
            
            # Sharpe ratio (annualized)
            returns = portfolio.returns
            if len(returns) > 0 and np.std(returns) > 0:
                sharpe = np.sqrt(252) * np.mean(returns) / np.std(returns)
            else:
                sharpe = 0
            
            # Max drawdown
            try:
                max_dd = portfolio.drawdown.max() * 100  # Convert to percentage
            except:
                max_dd = 0
            
            return {
                'MA1': ma1,
                'MA2': ma2,
                'Total_Return': total_return,
                'Win_Loss_Ratio': win_loss_ratio,
                'Sharpe_Ratio': sharpe,
                'Trade_Count': trade_count,
                'Avg_Trade': avg_trade,
                'Max_Drawdown': max_dd,
                'Status': 'Success'
            }
            
        except Exception as e:
            logger.warning(f"Backtest failed for MA1={ma1}, MA2={ma2}: {e}")
            return {
                'MA1': ma1,
                'MA2': ma2,
                'Total_Return': 0,
                'Win_Loss_Ratio': 0,
                'Sharpe_Ratio': 0,
                'Trade_Count': 0,
                'Avg_Trade': 0,
                'Max_Drawdown': 0,
                'Status': f'Failed: {str(e)[:50]}'
            }
    
    def run_optimization_serial(self, param_combinations: List[Tuple[int, int]]) -> List[Dict]:
        """Run optimization in serial mode (baseline)"""
        
        logger.info("=== SERIAL OPTIMIZATION ===")
        start_time = time.time()
        
        results = []
        total_combinations = len(param_combinations)
        
        for i, params in enumerate(param_combinations):
            if i % 100 == 0:  # Progress every 100 iterations
                elapsed = time.time() - start_time
                rate = i / elapsed if elapsed > 0 else 0
                eta = (total_combinations - i) / rate if rate > 0 else 0
                logger.info(f"Serial progress: {i:,}/{total_combinations:,} ({i/total_combinations*100:.1f}%) - Rate: {rate:.1f}/s - ETA: {eta/60:.1f}min")
            
            result = self.run_single_backtest(params)
            results.append(result)
        
        serial_time = time.time() - start_time
        logger.info(f"Serial optimization completed in {serial_time:.2f}s")
        logger.info(f"Rate: {len(param_combinations)/serial_time:.2f} backtests/second")
        
        return results, serial_time
    
    def run_optimization_parallel_process(self, param_combinations: List[Tuple[int, int]], max_workers: int = None) -> Tuple[List[Dict], float]:
        """Run optimization using ProcessPoolExecutor with progress bar"""
        
        if max_workers is None:
            max_workers = self.cpu_count  # Use all 16 cores (32 logical threads)
        
        logger.info(f"=== PARALLEL OPTIMIZATION (PROCESSES) - {max_workers} workers ===")
        start_time = time.time()
        
        results = []
        total_combinations = len(param_combinations)
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all jobs
            future_to_params = {
                executor.submit(self.run_single_backtest, params): params 
                for params in param_combinations
            }
            
            # Create progress bar
            with tqdm(total=total_combinations, desc="Process Optimization", 
                     unit="backtests", ncols=100, 
                     bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]') as pbar:
                
                # Collect results with progress bar
                for future in as_completed(future_to_params):
                    try:
                        result = future.result(timeout=60)  # 60 second timeout per backtest
                        results.append(result)
                        
                    except Exception as e:
                        params = future_to_params[future]
                        logger.warning(f"Parallel backtest failed for {params}: {e}")
                        results.append({
                            'MA1': params[0], 'MA2': params[1],
                            'Total_Return': 0, 'Win_Loss_Ratio': 0, 'Sharpe_Ratio': 0,
                            'Trade_Count': 0, 'Avg_Trade': 0, 'Max_Drawdown': 0,
                            'Status': f'Timeout/Error: {str(e)[:30]}'
                        })
                    
                    # Update progress bar
                    pbar.update(1)
                    
                    # Update description with current rate and ETA
                    elapsed = time.time() - start_time
                    current_rate = pbar.n / elapsed if elapsed > 0 else 0
                    pbar.set_description(f"Process Opt ({current_rate:.1f}/s)")
        
        parallel_time = time.time() - start_time
        logger.info(f"Parallel optimization completed in {parallel_time:.2f}s")
        logger.info(f"Rate: {len(param_combinations)/parallel_time:.2f} backtests/second")
        
        return results, parallel_time
    
    def run_optimization_parallel_thread(self, param_combinations: List[Tuple[int, int]], max_workers: int = None) -> Tuple[List[Dict], float]:
        """Run optimization using ThreadPoolExecutor with progress bar"""
        
        if max_workers is None:
            max_workers = self.cpu_count * 2  # Use all 32 logical threads (optimal for VectorBT)
        
        logger.info(f"=== PARALLEL OPTIMIZATION (THREADS) - {max_workers} workers ===")
        start_time = time.time()
        
        results = []
        total_combinations = len(param_combinations)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all jobs
            future_to_params = {
                executor.submit(self.run_single_backtest, params): params 
                for params in param_combinations
            }
            
            # Create progress bar
            with tqdm(total=total_combinations, desc="Thread Optimization", 
                     unit="backtests", ncols=100,
                     bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]') as pbar:
                
                # Collect results with progress bar
                for future in as_completed(future_to_params):
                    try:
                        result = future.result(timeout=60)  # 60 second timeout per backtest
                        results.append(result)
                        
                    except Exception as e:
                        params = future_to_params[future]
                        logger.warning(f"Thread backtest failed for {params}: {e}")
                        results.append({
                            'MA1': params[0], 'MA2': params[1],
                            'Total_Return': 0, 'Win_Loss_Ratio': 0, 'Sharpe_Ratio': 0,
                            'Trade_Count': 0, 'Avg_Trade': 0, 'Max_Drawdown': 0,
                            'Status': f'Timeout/Error: {str(e)[:30]}'
                        })
                    
                    # Update progress bar
                    pbar.update(1)
                    
                    # Update description with current rate and ETA
                    elapsed = time.time() - start_time
                    current_rate = pbar.n / elapsed if elapsed > 0 else 0
                    pbar.set_description(f"Thread Opt ({current_rate:.1f}/s)")
        
        thread_time = time.time() - start_time
        logger.info(f"Thread optimization completed in {thread_time:.2f}s")
        logger.info(f"Rate: {len(param_combinations)/thread_time:.2f} backtests/second")
        
        return results, thread_time
    
    def save_results(self, results: List[Dict], performance_stats: Dict) -> str:
        """Save optimization results and performance stats to files"""
        
        logger.info("=== SAVING RESULTS ===")
        start_time = time.time()
        
        try:
            # Create results DataFrame
            results_df = pd.DataFrame(results)
            
            # Add timestamp and system info
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save main results
            results_filename = f"optimization_results_{timestamp}.csv"
            results_df.to_csv(results_filename, index=False)
            
            # Create performance summary
            performance_df = pd.DataFrame([performance_stats])
            performance_filename = f"optimization_performance_{timestamp}.csv"
            performance_df.to_csv(performance_filename, index=False)
            
            # Create detailed analysis
            analysis = self.analyze_results(results_df, performance_stats)
            analysis_filename = f"optimization_analysis_{timestamp}.txt"
            with open(analysis_filename, 'w') as f:
                f.write(analysis)
            
            save_time = time.time() - start_time
            self.timing_results['results_save_time'] = save_time
            
            logger.info(f"Results saved in {save_time:.2f}s:")
            logger.info(f"  - Results: {results_filename}")
            logger.info(f"  - Performance: {performance_filename}")
            logger.info(f"  - Analysis: {analysis_filename}")
            
            return results_filename
            
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
            return ""
    
    def analyze_results(self, results_df: pd.DataFrame, performance_stats: Dict) -> str:
        """Generate comprehensive analysis of optimization results"""
        
        analysis = []
        analysis.append("=" * 80)
        analysis.append("OPTIMIZATION PERFORMANCE ANALYSIS")
        analysis.append("=" * 80)
        analysis.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        analysis.append("")
        
        # System Information
        analysis.append("SYSTEM CONFIGURATION:")
        analysis.append(f"  CPUs: {self.cpu_count}")
        analysis.append(f"  Memory: {self.memory_gb:.1f} GB")
        analysis.append(f"  Data Period: {self.start_date} to {self.end_date}")
        analysis.append(f"  Total Bars: {len(self.data['close']):,}")
        analysis.append("")
        
        # Performance Comparison
        analysis.append("PERFORMANCE COMPARISON:")
        for method, stats in performance_stats.items():
            if 'time' in stats:
                rate = stats['combinations'] / stats['time']
                analysis.append(f"  {method.upper()}:")
                analysis.append(f"    Time: {stats['time']:.2f}s")
                analysis.append(f"    Rate: {rate:.2f} backtests/second")
                analysis.append(f"    Combinations: {stats['combinations']:,}")
        analysis.append("")
        
        # Speed Improvements
        if 'serial' in performance_stats and 'parallel_process' in performance_stats:
            serial_time = performance_stats['serial']['time']
            parallel_time = performance_stats['parallel_process']['time']
            speedup = serial_time / parallel_time
            analysis.append(f"PARALLEL PROCESSING SPEEDUP:")
            analysis.append(f"  Process Pool Speedup: {speedup:.2f}x")
            
        if 'serial' in performance_stats and 'parallel_thread' in performance_stats:
            thread_time = performance_stats['parallel_thread']['time']
            thread_speedup = serial_time / thread_time
            analysis.append(f"  Thread Pool Speedup: {thread_speedup:.2f}x")
        analysis.append("")
        
        # Results Analysis
        analysis.append("OPTIMIZATION RESULTS ANALYSIS:")
        analysis.append(f"  Total Combinations: {len(results_df):,}")
        analysis.append(f"  Successful: {len(results_df[results_df['Status'] == 'Success']):,}")
        analysis.append(f"  Failed: {len(results_df[results_df['Status'] != 'Success']):,}")
        analysis.append("")
        
        # Best Results
        if len(results_df[results_df['Status'] == 'Success']) > 0:
            successful_df = results_df[results_df['Status'] == 'Success']
            
            # Best by Total Return
            best_return = successful_df.loc[successful_df['Total_Return'].idxmax()]
            analysis.append("BEST RESULTS BY TOTAL RETURN:")
            analysis.append(f"  MA1: {best_return['MA1']}, MA2: {best_return['MA2']}")
            analysis.append(f"  Total Return: {best_return['Total_Return']:.2f}%")
            analysis.append(f"  Sharpe Ratio: {best_return['Sharpe_Ratio']:.2f}")
            analysis.append(f"  Max Drawdown: {best_return['Max_Drawdown']:.2f}%")
            analysis.append(f"  Trade Count: {best_return['Trade_Count']}")
            analysis.append("")
            
            # Best by Sharpe Ratio
            best_sharpe = successful_df.loc[successful_df['Sharpe_Ratio'].idxmax()]
            analysis.append("BEST RESULTS BY SHARPE RATIO:")
            analysis.append(f"  MA1: {best_sharpe['MA1']}, MA2: {best_sharpe['MA2']}")
            analysis.append(f"  Sharpe Ratio: {best_sharpe['Sharpe_Ratio']:.2f}")
            analysis.append(f"  Total Return: {best_sharpe['Total_Return']:.2f}%")
            analysis.append(f"  Max Drawdown: {best_sharpe['Max_Drawdown']:.2f}%")
            analysis.append(f"  Trade Count: {best_sharpe['Trade_Count']}")
            analysis.append("")
            
            # Summary Statistics
            analysis.append("SUMMARY STATISTICS:")
            analysis.append(f"  Avg Total Return: {successful_df['Total_Return'].mean():.2f}%")
            analysis.append(f"  Std Total Return: {successful_df['Total_Return'].std():.2f}%")
            analysis.append(f"  Avg Sharpe Ratio: {successful_df['Sharpe_Ratio'].mean():.2f}")
            analysis.append(f"  Avg Trade Count: {successful_df['Trade_Count'].mean():.0f}")
            analysis.append(f"  Avg Max Drawdown: {successful_df['Max_Drawdown'].mean():.2f}%")
        
        return "\n".join(analysis)
    
    def run_full_optimization_test(self, test_modes: List[str] = None, skip_serial: bool = False) -> Dict:
        """Run the complete optimization performance test"""
        
        if test_modes is None:
            if skip_serial:
                test_modes = ['parallel_process', 'parallel_thread']
                logger.info("SKIPPING SERIAL MODE - Using only parallel processing")
            else:
                test_modes = ['serial', 'parallel_process', 'parallel_thread']
        
        logger.info("=" * 80)
        logger.info("STARTING OPTIMIZATION PERFORMANCE TEST")  
        logger.info(f"Hardware: {self.cpu_count} cores, {self.memory_gb:.1f}GB RAM")
        logger.info(f"Test modes: {test_modes}")
        logger.info("=" * 80)
        
        total_start_time = time.time()
        
        # Step 1: Load Data
        if not self.load_data():
            logger.error("Data loading failed - cannot continue")
            return {}
        
        # Step 2: Generate Parameters
        param_combinations = self.generate_optimization_parameters()
        
        # For testing, you might want to limit combinations
        # Uncomment next line to test with fewer combinations
        # param_combinations = param_combinations[:100]  # Test with first 100 combinations
        
        logger.info(f"Testing with {len(param_combinations):,} combinations")
        
        # Step 3: Run Different Optimization Methods
        performance_stats = {}
        all_results = {}
        
        for mode in test_modes:
            logger.info(f"\n{'='*60}")
            logger.info(f"TESTING MODE: {mode.upper()}")
            logger.info(f"{'='*60}")
            
            if mode == 'serial':
                results, exec_time = self.run_optimization_serial(param_combinations)
                
            elif mode == 'parallel_process':
                results, exec_time = self.run_optimization_parallel_process(param_combinations)
                
            elif mode == 'parallel_thread':
                results, exec_time = self.run_optimization_parallel_thread(param_combinations)
            
            else:
                logger.warning(f"Unknown test mode: {mode}")
                continue
            
            # Store results
            all_results[mode] = results
            performance_stats[mode] = {
                'time': exec_time,
                'combinations': len(param_combinations),
                'rate': len(param_combinations) / exec_time,
                'mode': mode
            }
        
        # Step 4: Save Results
        if all_results:
            # Use the first available results for saving
            first_mode = list(all_results.keys())[0]
            results_filename = self.save_results(all_results[first_mode], performance_stats)
        
        # Final timing
        total_time = time.time() - total_start_time
        self.timing_results['total_time'] = total_time
        
        # Final Report
        logger.info("\n" + "=" * 80)
        logger.info("OPTIMIZATION PERFORMANCE TEST COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Total execution time: {total_time:.2f}s ({total_time/60:.1f} minutes)")
        logger.info(f"Data loading: {self.timing_results['data_load_time']:.2f}s")
        logger.info(f"Results saving: {self.timing_results['results_save_time']:.2f}s")
        
        for mode, stats in performance_stats.items():
            logger.info(f"{mode.upper()}: {stats['time']:.2f}s ({stats['rate']:.2f} backtests/s)")
        
        return {
            'performance_stats': performance_stats,
            'timing_results': self.timing_results,
            'total_combinations': len(param_combinations),
            'data_bars': len(self.data['close']),
            'results_file': results_filename if 'results_filename' in locals() else ""
        }

def main():
    """Main execution function"""
    
    # Initialize tester
    tester = OptimizationPerformanceTester()
    
    # Skip serial mode for full test (too slow with 10,000 combinations)
    # Use only parallel processing with all 32 cores
    
    # Run the test (skip_serial=True for fast parallel-only testing)
    final_results = tester.run_full_optimization_test(skip_serial=True)
    
    print("\n" + "=" * 80)
    print("OPTIMIZATION PERFORMANCE TEST RESULTS")
    print("=" * 80)
    
    if final_results:
        print(f"Total combinations tested: {final_results['total_combinations']:,}")
        print(f"Data bars processed: {final_results['data_bars']:,}")
        print(f"Total execution time: {final_results['timing_results']['total_time']:.2f}s")
        print()
        
        for mode, stats in final_results['performance_stats'].items():
            print(f"{mode.upper()}:")
            print(f"  Time: {stats['time']:.2f}s")
            print(f"  Rate: {stats['rate']:.2f} backtests/second") 
            print()
        
        if final_results['results_file']:
            print(f"Results saved to: {final_results['results_file']}")
    
    print("Test complete!")

if __name__ == "__main__":
    main()