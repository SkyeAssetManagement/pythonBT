#!/usr/bin/env python3
"""
VectorBT Pro Proper Multi-Dimensional Optimization
Uses vectorized parameter sweeps instead of manual iteration loops

Test Parameters:
- MA1: 10 to 1000, step 10 (100 values)  
- MA2: 100 to 10000, step 100 (100 values)
- Total: 10,000 optimization combinations (vectorized)

Saves: Total Return, Win/Loss Ratio, Sharpe, Trade Count, Average Trade, Max Drawdown
"""

import sys
import os
import time
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any
import logging
from datetime import datetime, timedelta
import psutil
import warnings
from tqdm import tqdm
import vectorbtpro as vbt

warnings.filterwarnings('ignore')

# Add project paths
sys.path.append('src')

from src.data.parquet_converter import ParquetConverter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('vectorbt_optimization.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class VectorBTProperOptimizer:
    """Proper VectorBT Pro multi-dimensional optimization without manual loops"""
    
    def __init__(self):
        self.start_date = "2000-01-01"
        self.end_date = "2024-12-31"
        self.symbol = "GC"  # Gold futures
        self.data = None
        
        # System info
        self.cpu_count = psutil.cpu_count()
        self.memory_gb = psutil.virtual_memory().total / (1024**3)
        
        logger.info(f"Initialized VectorBT Proper Optimizer")
        logger.info(f"System: {self.cpu_count} cores, {self.memory_gb:.1f}GB RAM")
        logger.info(f"Date range: {self.start_date} to {self.end_date}")
    
    def load_data(self) -> bool:
        """Load GC data for the specified date range"""
        
        logger.info("=== DATA LOADING PHASE ===")
        start_time = time.time()
        
        try:
            # Load using ParquetConverter (most optimized)
            logger.info("Loading data using ParquetConverter...")
            parquet_converter = ParquetConverter()
            
            # Load full dataset first, then filter
            self.data = parquet_converter.load_or_convert(self.symbol, "1m", "diffAdjusted")
            
            if self.data:
                # Apply date filtering
                self.data = parquet_converter.filter_data_by_date(
                    self.data, self.start_date, self.end_date
                )
                
                if self.data and len(self.data['close']) > 0:
                    load_time = time.time() - start_time
                    
                    logger.info(f"SUCCESS: Loaded {len(self.data['close']):,} bars in {load_time:.2f}s")
                    logger.info(f"Data range: {pd.to_datetime(self.data['datetime_ns'][0]).strftime('%Y-%m-%d')} to {pd.to_datetime(self.data['datetime_ns'][-1]).strftime('%Y-%m-%d')}")
                    logger.info(f"Price range: ${np.min(self.data['low']):.2f} - ${np.max(self.data['high']):.2f}")
                    
                    return True
            
            logger.error("Data loading failed")
            return False
            
        except Exception as e:
            logger.error(f"Data loading failed: {e}")
            return False
    
    def run_vectorized_optimization_test(self, small_test: bool = True) -> Dict:
        """Run vectorized optimization using VectorBT Pro's proper multi-dimensional approach"""
        
        logger.info("=== VECTORIZED OPTIMIZATION TEST ===")
        
        if small_test:
            # Small test: MA1 (10,20,30,40,50) vs MA2 (100,200,300,400,500) = 25 combinations
            ma1_range = np.arange(10, 51, 10)  # [10, 20, 30, 40, 50]
            ma2_range = np.arange(100, 501, 100)  # [100, 200, 300, 400, 500]
            logger.info(f"SMALL TEST: MA1 range {ma1_range}, MA2 range {ma2_range}")
            logger.info(f"Total combinations: {len(ma1_range) * len(ma2_range)}")
        else:
            # Full test: MA1 (10 to 1000, step 10) vs MA2 (100 to 10000, step 100)
            ma1_range = np.arange(10, 1001, 10)  # 100 values
            ma2_range = np.arange(100, 10001, 100)  # 100 values  
            logger.info(f"FULL TEST: MA1 range 10-1000 (step 10), MA2 range 100-10000 (step 100)")
            logger.info(f"Total combinations: {len(ma1_range) * len(ma2_range):,}")
        
        # Convert to pandas Series for VectorBT
        close_prices = pd.Series(self.data['close'], index=pd.to_datetime(self.data['datetime_ns']))
        
        start_time = time.time()
        
        # === PROPER VECTORBT PRO APPROACH ===
        logger.info("Using VectorBT Pro run_combs for true vectorization...")
        ma_start = time.time()
        
        # Create parameter grid for valid combinations only
        valid_combinations = []
        param_combinations = []
        
        for ma1_window in ma1_range:
            for ma2_window in ma2_range:
                if ma2_window > ma1_window:  # Only valid combinations
                    valid_combinations.append([ma1_window, ma2_window])
                    param_combinations.append((ma1_window, ma2_window))
        
        # Convert to numpy array for VectorBT
        param_grid = np.array(valid_combinations)
        
        logger.info(f"Parameter grid shape: {param_grid.shape}")
        logger.info(f"Valid combinations: {len(param_combinations)}")
        
        # Use VectorBT Pro's run_combs for true vectorization
        with tqdm(desc="Vectorized MA Calculation", total=1) as pbar:
            # Calculate fast MAs for all parameter combinations
            fast_ma = vbt.MA.run(
                close_prices, 
                window=param_grid[:, 0],  # All fast MA periods
                per_column=True
            ).ma
            
            # Calculate slow MAs for all parameter combinations  
            slow_ma = vbt.MA.run(
                close_prices,
                window=param_grid[:, 1],  # All slow MA periods
                per_column=True
            ).ma
            
            pbar.update(1)
        
        ma_time = time.time() - ma_start
        logger.info(f"Vectorized MA calculation completed in {ma_time:.2f}s")
        
        # === VECTORIZED SIGNAL GENERATION ===
        logger.info("Generating crossover signals (vectorized)...")
        signal_start = time.time()
        
        with tqdm(desc="Signal Generation", total=1) as pbar:
            # Generate entry/exit signals vectorized across all combinations
            entries = fast_ma.vbt.crossed_above(slow_ma)
            exits = fast_ma.vbt.crossed_below(slow_ma)
            pbar.update(1)
        
        entries_df = entries
        exits_df = exits
        
        signal_time = time.time() - signal_start
        logger.info(f"Signal generation completed in {signal_time:.2f}s")
        logger.info(f"Valid combinations: {len(valid_combinations):,}")
        
        # === VECTORIZED PORTFOLIO BACKTESTING ===
        logger.info("Running vectorized portfolio backtesting...")
        backtest_start = time.time()
        
        # Run portfolio backtest on ALL combinations simultaneously
        with tqdm(desc="Portfolio Backtesting", total=1) as pbar:
            pf = vbt.Portfolio.from_signals(
                close=close_prices,
                entries=entries_df,
                exits=exits_df,
                size=np.inf,  # Use all available cash
                init_cash=100000,
                fees=0.001,  # 0.1% fees
                freq='1min'
            )
            pbar.update(1)
        
        backtest_time = time.time() - backtest_start
        logger.info(f"Portfolio backtesting completed in {backtest_time:.2f}s")
        
        # === EXTRACT RESULTS ===
        logger.info("Extracting results...")
        results_start = time.time()
        
        results_list = []
        
        with tqdm(desc="Extracting Metrics", total=len(param_combinations)) as pbar:
            for i, (ma1, ma2) in enumerate(param_combinations):
                
                try:
                    # Extract metrics for this combination (column i)
                    if hasattr(pf, 'iloc'):
                        col_pf = pf.iloc[:, i] if pf.shape[1] > 1 else pf
                    else:
                        col_pf = pf
                    
                    # Basic metrics
                    total_return = col_pf.total_return * 100  # Convert to percentage
                    sharpe_ratio = col_pf.sharpe_ratio
                    max_drawdown = col_pf.max_drawdown * 100  # Convert to percentage
                    
                    # Trade metrics
                    trades = col_pf.trades.records_readable if hasattr(col_pf.trades, 'records_readable') else pd.DataFrame()
                    trade_count = len(trades)
                    
                    if trade_count > 0:
                        winning_trades = len(trades[trades['PnL'] > 0])
                        losing_trades = len(trades[trades['PnL'] <= 0])
                        win_loss_ratio = winning_trades / max(losing_trades, 1)
                        avg_trade = trades['PnL'].mean()
                    else:
                        win_loss_ratio = 0
                        avg_trade = 0
                    
                    results_list.append({
                        'MA1': ma1,
                        'MA2': ma2,
                        'Total_Return': float(total_return) if not pd.isna(total_return) else 0,
                        'Win_Loss_Ratio': float(win_loss_ratio),
                        'Sharpe_Ratio': float(sharpe_ratio) if not pd.isna(sharpe_ratio) else 0,
                        'Trade_Count': int(trade_count),
                        'Avg_Trade': float(avg_trade),
                        'Max_Drawdown': float(max_drawdown) if not pd.isna(max_drawdown) else 0,
                        'Status': 'Success'
                    })
                    
                except Exception as e:
                    logger.warning(f"Failed to extract results for MA1={ma1}, MA2={ma2}: {e}")
                    results_list.append({
                        'MA1': ma1,
                        'MA2': ma2,
                        'Total_Return': 0,
                        'Win_Loss_Ratio': 0,
                        'Sharpe_Ratio': 0,
                        'Trade_Count': 0,
                        'Avg_Trade': 0,
                        'Max_Drawdown': 0,
                        'Status': f'Failed: {str(e)[:50]}'
                    })
                
                pbar.update(1)
        
        results_time = time.time() - results_start
        total_time = time.time() - start_time
        
        logger.info(f"Results extraction completed in {results_time:.2f}s")
        logger.info(f"TOTAL VECTORIZED OPTIMIZATION TIME: {total_time:.2f}s")
        
        # Create results DataFrame
        results_df = pd.DataFrame(results_list)
        
        # Performance stats
        performance_stats = {
            'total_time': total_time,
            'ma_calculation_time': ma_time,
            'signal_generation_time': signal_time,
            'backtesting_time': backtest_time,
            'results_extraction_time': results_time,
            'combinations_tested': len(valid_combinations),
            'rate_per_second': len(valid_combinations) / total_time,
            'data_bars': len(self.data['close'])
        }
        
        return {
            'results_df': results_df,
            'performance_stats': performance_stats,  
            'valid_combinations': len(valid_combinations)
        }
    
    def save_results(self, results_dict: Dict) -> str:
        """Save vectorized optimization results"""
        
        logger.info("=== SAVING RESULTS ===")
        start_time = time.time()
        
        try:
            results_df = results_dict['results_df']
            performance_stats = results_dict['performance_stats']
            
            # Add timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save main results
            results_filename = f"vectorbt_optimization_results_{timestamp}.csv"
            results_df.to_csv(results_filename, index=False)
            
            # Save performance stats
            perf_df = pd.DataFrame([performance_stats])
            perf_filename = f"vectorbt_optimization_performance_{timestamp}.csv"
            perf_df.to_csv(perf_filename, index=False)
            
            # Create analysis summary
            analysis = self.create_analysis_summary(results_df, performance_stats)
            analysis_filename = f"vectorbt_optimization_analysis_{timestamp}.txt"
            with open(analysis_filename, 'w') as f:
                f.write(analysis)
            
            save_time = time.time() - start_time
            
            logger.info(f"Results saved in {save_time:.2f}s:")
            logger.info(f"  - Results: {results_filename}")
            logger.info(f"  - Performance: {perf_filename}")
            logger.info(f"  - Analysis: {analysis_filename}")
            
            return results_filename
            
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
            return ""
    
    def create_analysis_summary(self, results_df: pd.DataFrame, performance_stats: Dict) -> str:
        """Create comprehensive analysis summary"""
        
        analysis = []
        analysis.append("=" * 80)
        analysis.append("VECTORBT PRO OPTIMIZATION ANALYSIS")
        analysis.append("=" * 80)
        analysis.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        analysis.append("")
        
        # System Information
        analysis.append("SYSTEM CONFIGURATION:")
        analysis.append(f"  CPUs: {self.cpu_count}")
        analysis.append(f"  Memory: {self.memory_gb:.1f} GB")
        analysis.append(f"  Data Period: {self.start_date} to {self.end_date}")
        analysis.append(f"  Total Bars: {performance_stats['data_bars']:,}")
        analysis.append("")
        
        # Performance Stats
        analysis.append("VECTORIZED OPTIMIZATION PERFORMANCE:")
        analysis.append(f"  Total Time: {performance_stats['total_time']:.2f}s ({performance_stats['total_time']/60:.1f} minutes)")
        analysis.append(f"  MA Calculation: {performance_stats['ma_calculation_time']:.2f}s")
        analysis.append(f"  Signal Generation: {performance_stats['signal_generation_time']:.2f}s")
        analysis.append(f"  Portfolio Backtesting: {performance_stats['backtesting_time']:.2f}s")
        analysis.append(f"  Results Extraction: {performance_stats['results_extraction_time']:.2f}s")
        analysis.append(f"  Rate: {performance_stats['rate_per_second']:.2f} backtests/second")
        analysis.append(f"  Combinations: {performance_stats['combinations_tested']:,}")
        analysis.append("")
        
        # Results Analysis
        successful = results_df[results_df['Status'] == 'Success']
        analysis.append("RESULTS ANALYSIS:")
        analysis.append(f"  Total Combinations: {len(results_df):,}")
        analysis.append(f"  Successful: {len(successful):,}")
        analysis.append(f"  Failed: {len(results_df) - len(successful):,}")
        analysis.append("")
        
        if len(successful) > 0:
            # Best Results
            best_return = successful.loc[successful['Total_Return'].idxmax()]
            analysis.append("BEST RESULTS BY TOTAL RETURN:")
            analysis.append(f"  MA1: {best_return['MA1']}, MA2: {best_return['MA2']}")
            analysis.append(f"  Total Return: {best_return['Total_Return']:.2f}%")
            analysis.append(f"  Sharpe Ratio: {best_return['Sharpe_Ratio']:.2f}")
            analysis.append(f"  Max Drawdown: {best_return['Max_Drawdown']:.2f}%")
            analysis.append(f"  Trade Count: {best_return['Trade_Count']}")
            analysis.append("")
            
            # Best by Sharpe
            best_sharpe = successful.loc[successful['Sharpe_Ratio'].idxmax()]
            analysis.append("BEST RESULTS BY SHARPE RATIO:")
            analysis.append(f"  MA1: {best_sharpe['MA1']}, MA2: {best_sharpe['MA2']}")
            analysis.append(f"  Sharpe Ratio: {best_sharpe['Sharpe_Ratio']:.2f}")
            analysis.append(f"  Total Return: {best_sharpe['Total_Return']:.2f}%")
            analysis.append(f"  Max Drawdown: {best_sharpe['Max_Drawdown']:.2f}%")
            analysis.append(f"  Trade Count: {best_sharpe['Trade_Count']}")
            analysis.append("")
            
            # Summary Statistics
            analysis.append("SUMMARY STATISTICS:")
            analysis.append(f"  Avg Total Return: {successful['Total_Return'].mean():.2f}%")
            analysis.append(f"  Std Total Return: {successful['Total_Return'].std():.2f}%")
            analysis.append(f"  Avg Sharpe Ratio: {successful['Sharpe_Ratio'].mean():.2f}")
            analysis.append(f"  Avg Trade Count: {successful['Trade_Count'].mean():.0f}")
            analysis.append(f"  Avg Max Drawdown: {successful['Max_Drawdown'].mean():.2f}%")
        
        return "\n".join(analysis)

def main():
    """Main execution function"""
    
    print("=" * 80)
    print("VECTORBT PRO PROPER MULTI-DIMENSIONAL OPTIMIZATION")
    print("Using vectorized parameter sweeps (no manual loops)")
    print("=" * 80)
    
    # Initialize optimizer
    optimizer = VectorBTProperOptimizer()
    
    # Load data
    if not optimizer.load_data():
        print("ERROR: Data loading failed")
        return
    
    # Ask user for test size
    test_choice = input("\nChoose test size:\n1. Small test (25 combinations)\n2. Full test (10,000 combinations)\nEnter choice (1 or 2): ").strip()
    
    small_test = (test_choice == "1")
    
    if small_test:
        print("\nRunning SMALL TEST (25 combinations)...")
    else:
        print("\nRunning FULL TEST (10,000 combinations)...")
        print("This may take several minutes...")
    
    # Run optimization
    results = optimizer.run_vectorized_optimization_test(small_test=small_test)
    
    if results:
        # Save results
        results_file = optimizer.save_results(results)
        
        # Final summary
        perf = results['performance_stats']
        print("\n" + "=" * 80)
        print("VECTORBT OPTIMIZATION COMPLETE")
        print("=" * 80)
        print(f"Total time: {perf['total_time']:.2f}s ({perf['total_time']/60:.1f} minutes)")
        print(f"Combinations tested: {perf['combinations_tested']:,}")
        print(f"Rate: {perf['rate_per_second']:.2f} backtests/second")
        print(f"Data bars: {perf['data_bars']:,}")
        
        if results_file:
            print(f"Results saved to: {results_file}")
        
        # Show sample results
        results_df = results['results_df']
        successful = results_df[results_df['Status'] == 'Success']
        
        if len(successful) > 0:
            print(f"\nSample successful results (top 3 by return):")
            top_results = successful.nlargest(3, 'Total_Return')
            for _, row in top_results.iterrows():
                print(f"  MA1={row['MA1']}, MA2={row['MA2']}: Return={row['Total_Return']:.2f}%, Sharpe={row['Sharpe_Ratio']:.2f}, Trades={row['Trade_Count']}")
        
        print("\nVectorized optimization complete!")
    
    else:
        print("ERROR: Optimization failed")

if __name__ == "__main__":
    main()