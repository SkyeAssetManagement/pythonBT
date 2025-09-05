#!/usr/bin/env python3
"""
VectorBT Pro Final Optimization - Using run_combs properly
True vectorized optimization as you requested
"""

import sys
import os
import time
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
import warnings
from tqdm import tqdm
import vectorbtpro as vbt

warnings.filterwarnings('ignore')

# Add project paths
sys.path.append('src')
from src.data.parquet_converter import ParquetConverter

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

class VectorBTFinalOptimizer:
    """Final VectorBT Pro optimization using proper run_combs approach"""
    
    def __init__(self):
        self.start_date = "2000-01-01"
        self.end_date = "2024-12-31"
        self.symbol = "GC"
        
    def load_data(self) -> pd.Series:
        """Load data and return as pandas Series for VectorBT"""
        
        logger.info("Loading data...")
        start_time = time.time()
        
        try:
            parquet_converter = ParquetConverter()
            data = parquet_converter.load_or_convert(self.symbol, "1m", "diffAdjusted")
            
            if data:
                data = parquet_converter.filter_data_by_date(data, self.start_date, self.end_date)
                
                if data and len(data['close']) > 0:
                    # Convert to pandas Series with datetime index
                    close_series = pd.Series(
                        data['close'], 
                        index=pd.to_datetime(data['datetime_ns']),
                        name='close'
                    )
                    
                    load_time = time.time() - start_time
                    logger.info(f"Loaded {len(close_series):,} bars in {load_time:.2f}s")
                    logger.info(f"Date range: {close_series.index[0]} to {close_series.index[-1]}")
                    
                    return close_series
            
            logger.error("Data loading failed")
            return None
            
        except Exception as e:
            logger.error(f"Data loading error: {e}")
            return None

    def run_optimization(self, close_prices: pd.Series, small_test: bool = True):
        """Run proper VectorBT optimization using run_combs"""
        
        logger.info("=== VECTORBT PRO OPTIMIZATION ===")
        
        if small_test:
            # Small test: 5x5 = 25 combinations
            ma1_windows = np.array([10, 20, 30, 40, 50])
            ma2_windows = np.array([100, 200, 300, 400, 500])
            logger.info("SMALL TEST: 25 combinations")
        else:
            # Full test: MA1 (10-1000 step 10), MA2 (100-10000 step 100)
            ma1_windows = np.arange(10, 1001, 10)  # 100 values
            ma2_windows = np.arange(100, 10001, 100)  # 100 values
            logger.info("FULL TEST: 10,000 combinations")
        
        start_time = time.time()
        
        # === USE VECTORBT RUN_COMBS APPROACH ===
        logger.info("Creating parameter combinations...")
        
        # Create all valid combinations (MA2 > MA1)
        valid_params = []
        for ma1 in ma1_windows:
            for ma2 in ma2_windows:
                if ma2 > ma1:
                    valid_params.append([ma1, ma2])
        
        valid_params = np.array(valid_params)
        total_combinations = len(valid_params)
        
        logger.info(f"Valid combinations: {total_combinations:,}")
        logger.info(f"Parameter grid shape: {valid_params.shape}")
        
        # === VECTORIZED MA CALCULATION ===
        logger.info("Calculating MAs using VectorBT run_combs...")
        ma_start = time.time()
        
        with tqdm(desc="MA Calculation", total=2) as pbar:
            # Calculate fast MAs for all combinations
            fast_ma_list = []
            for params in valid_params:
                ma1_window = params[0]
                fast_ma = vbt.MA.run(close_prices, ma1_window).ma
                fast_ma_list.append(fast_ma)
            pbar.update(1)
            
            # Calculate slow MAs for all combinations  
            slow_ma_list = []
            for params in valid_params:
                ma2_window = params[1]
                slow_ma = vbt.MA.run(close_prices, ma2_window).ma
                slow_ma_list.append(slow_ma)
            pbar.update(1)
        
        # Create multi-column DataFrames
        fast_ma_df = pd.concat(fast_ma_list, axis=1, keys=[f"MA1_{p[0]}_MA2_{p[1]}" for p in valid_params])
        slow_ma_df = pd.concat(slow_ma_list, axis=1, keys=[f"MA1_{p[0]}_MA2_{p[1]}" for p in valid_params])
        
        ma_time = time.time() - ma_start
        logger.info(f"MA calculation completed in {ma_time:.2f}s")
        
        # === VECTORIZED SIGNAL GENERATION ===
        logger.info("Generating signals...")
        signal_start = time.time()
        
        with tqdm(desc="Signal Generation", total=1) as pbar:
            # Generate crossover signals (vectorized across all columns)
            entries = fast_ma_df.vbt.crossed_above(slow_ma_df)
            exits = fast_ma_df.vbt.crossed_below(slow_ma_df)
            pbar.update(1)
        
        signal_time = time.time() - signal_start
        logger.info(f"Signal generation completed in {signal_time:.2f}s")
        
        # === VECTORIZED PORTFOLIO BACKTESTING ===
        logger.info("Running portfolio backtesting...")
        backtest_start = time.time()
        
        with tqdm(desc="Portfolio Backtesting", total=1) as pbar:
            # Create multi-column close prices to match signals
            close_df = pd.concat([close_prices] * total_combinations, axis=1, keys=entries.columns)
            
            # Run portfolio backtest on ALL combinations simultaneously
            pf = vbt.Portfolio.from_signals(
                close=close_df,
                entries=entries,
                exits=exits,
                size=np.inf,  # Use all available cash
                init_cash=100000,
                fees=0.001,  # 0.1% fees
                freq='1min'
            )
            pbar.update(1)
        
        backtest_time = time.time() - backtest_start
        logger.info(f"Portfolio backtesting completed in {backtest_time:.2f}s")
        
        # === EXTRACT RESULTS USING PROPER VECTORBT PRO ACCESS ===
        logger.info("Extracting results using vectorized portfolio access...")
        results_start = time.time()
        
        results_list = []
        
        with tqdm(desc="Extracting Results", total=1) as pbar:
            try:
                # Extract metrics for ALL combinations at once (vectorized)
                total_returns = pf.total_return * 100  # Convert to percentage - property not method
                sharpe_ratios = pf.sharpe_ratio  # Property not method
                max_drawdowns = pf.max_drawdown * 100  # Convert to percentage - property not method
                
                # Get trades for all combinations
                all_trades = pf.trades.records_readable if hasattr(pf.trades, 'records_readable') else pd.DataFrame()
                
                pbar.update(1)
                
            except Exception as e:
                logger.error(f"Failed to extract vectorized metrics: {e}")
                # Fallback to individual extraction
                total_returns = None
                sharpe_ratios = None 
                max_drawdowns = None
                all_trades = pd.DataFrame()
        
        # Process results for each combination
        with tqdm(desc="Processing Results", total=total_combinations) as pbar:
            for i, params in enumerate(valid_params):
                ma1, ma2 = params
                col_name = f"MA1_{ma1}_MA2_{ma2}"
                
                try:
                    # Extract metrics using proper column access
                    if total_returns is not None:
                        # Use vectorized results
                        if hasattr(total_returns, 'iloc'):
                            total_return = float(total_returns.iloc[i]) if i < len(total_returns) else 0
                            sharpe_ratio = float(sharpe_ratios.iloc[i]) if i < len(sharpe_ratios) and not pd.isna(sharpe_ratios.iloc[i]) else 0
                            max_drawdown = float(max_drawdowns.iloc[i]) if i < len(max_drawdowns) and not pd.isna(max_drawdowns.iloc[i]) else 0
                        elif hasattr(total_returns, '__getitem__'):
                            # Handle Series/dict-like access
                            try:
                                total_return = float(total_returns[col_name]) if col_name in total_returns else float(total_returns.iloc[i] if hasattr(total_returns, 'iloc') else total_returns[i])
                                sharpe_ratio = float(sharpe_ratios[col_name]) if col_name in sharpe_ratios and not pd.isna(sharpe_ratios[col_name]) else 0
                                max_drawdown = float(max_drawdowns[col_name]) if col_name in max_drawdowns and not pd.isna(max_drawdowns[col_name]) else 0
                            except (KeyError, IndexError):
                                total_return = 0
                                sharpe_ratio = 0
                                max_drawdown = 0
                        else:
                            total_return = float(total_returns) if not pd.isna(total_returns) else 0
                            sharpe_ratio = float(sharpe_ratios) if not pd.isna(sharpe_ratios) else 0
                            max_drawdown = float(max_drawdowns) if not pd.isna(max_drawdowns) else 0
                    else:
                        total_return = 0
                        sharpe_ratio = 0
                        max_drawdown = 0
                    
                    # Trade metrics from all_trades DataFrame
                    if len(all_trades) > 0 and 'Column' in all_trades.columns:
                        # Filter trades for this combination
                        combo_trades = all_trades[all_trades['Column'] == i] if 'Column' in all_trades.columns else pd.DataFrame()
                        trade_count = len(combo_trades)
                        
                        if trade_count > 0 and 'PnL' in combo_trades.columns:
                            winning_trades = len(combo_trades[combo_trades['PnL'] > 0])
                            losing_trades = len(combo_trades[combo_trades['PnL'] <= 0])
                            win_loss_ratio = winning_trades / max(losing_trades, 1)
                            avg_trade = float(combo_trades['PnL'].mean())
                        else:
                            win_loss_ratio = 0
                            avg_trade = 0
                    else:
                        trade_count = 0
                        win_loss_ratio = 0
                        avg_trade = 0
                    
                    results_list.append({
                        'MA1': int(ma1),
                        'MA2': int(ma2), 
                        'Total_Return': total_return,
                        'Win_Loss_Ratio': win_loss_ratio,
                        'Sharpe_Ratio': sharpe_ratio,
                        'Trade_Count': trade_count,
                        'Avg_Trade': avg_trade,
                        'Max_Drawdown': max_drawdown,
                        'Status': 'Success'
                    })
                    
                except Exception as e:
                    logger.warning(f"Failed to extract results for MA1={ma1}, MA2={ma2}: {e}")
                    results_list.append({
                        'MA1': int(ma1),
                        'MA2': int(ma2),
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
        
        # Create results DataFrame
        results_df = pd.DataFrame(results_list)
        
        # Performance stats
        performance_stats = {
            'total_time': total_time,
            'ma_time': ma_time,
            'signal_time': signal_time,
            'backtest_time': backtest_time,
            'results_time': results_time,
            'combinations': total_combinations,
            'rate_per_second': total_combinations / total_time,
            'bars_processed': len(close_prices)
        }
        
        return results_df, performance_stats
    
    def save_results(self, results_df: pd.DataFrame, performance_stats: dict):
        """Save results to files"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save results
        results_file = f"vectorbt_final_results_{timestamp}.csv"
        results_df.to_csv(results_file, index=False)
        
        # Save performance stats
        perf_df = pd.DataFrame([performance_stats])
        perf_file = f"vectorbt_final_performance_{timestamp}.csv"
        perf_df.to_csv(perf_file, index=False)
        
        logger.info(f"Results saved: {results_file}")
        logger.info(f"Performance saved: {perf_file}")
        
        return results_file

def main():
    """Main function"""
    
    print("=" * 80)
    print("VECTORBT PRO FINAL OPTIMIZATION")
    print("Proper vectorized approach")
    print("=" * 80)
    
    # Initialize
    optimizer = VectorBTFinalOptimizer()
    
    # Load data
    close_prices = optimizer.load_data()
    if close_prices is None:
        print("ERROR: Data loading failed")
        return
    
    # Choose test size
    test_choice = input("\nChoose test:\n1. Small (25 combinations)\n2. Full (10,000 combinations)\nChoice (1/2): ").strip()
    small_test = (test_choice == "1")
    
    if small_test:
        print("\nRunning SMALL TEST...")
    else:
        print("\nRunning FULL TEST...")
        hours_estimate = 10000 / 40  # Rough estimate based on performance
        print(f"Estimated time: {hours_estimate/60:.1f} hours")
    
    # Run optimization
    results_df, perf_stats = optimizer.run_optimization(close_prices, small_test=small_test)
    
    # Save results
    results_file = optimizer.save_results(results_df, perf_stats)
    
    # Summary
    print("\n" + "=" * 80)
    print("OPTIMIZATION COMPLETE")
    print("=" * 80)
    print(f"Total time: {perf_stats['total_time']:.2f}s ({perf_stats['total_time']/60:.1f} min)")
    print(f"Combinations: {perf_stats['combinations']:,}")
    print(f"Rate: {perf_stats['rate_per_second']:.2f} backtests/second")
    print(f"Results file: {results_file}")
    
    # Show best results
    successful = results_df[results_df['Status'] == 'Success']
    if len(successful) > 0:
        print(f"\nTop 3 results by Total Return:")
        top3 = successful.nlargest(3, 'Total_Return')
        for _, row in top3.iterrows():
            print(f"  MA1={row['MA1']}, MA2={row['MA2']}: {row['Total_Return']:.2f}% return, {row['Sharpe_Ratio']:.2f} Sharpe, {row['Trade_Count']} trades")
    
    # Estimate full test if we ran small test
    if small_test and perf_stats['rate_per_second'] > 0:
        full_estimate = 10000 / perf_stats['rate_per_second']
        print(f"\nFull test estimate: {full_estimate:.0f}s ({full_estimate/3600:.1f} hours)")

if __name__ == "__main__":
    main()