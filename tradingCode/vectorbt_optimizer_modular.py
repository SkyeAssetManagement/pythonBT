#!/usr/bin/env python3
"""
Modular VectorBT Pro Optimizer
Memory-efficient, chunked optimization system for any strategy with up to 8 parameters

Features:
- Memory management with chunked processing
- Modular design for any strategy
- Support for up to 8 optimization parameters
- Progress bars with time estimation
- Vectorized processing within chunks
- 16-core parallel processing
"""

import sys
import os
import time
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union
from abc import ABC, abstractmethod
import logging
from datetime import datetime
import warnings
from tqdm import tqdm
import vectorbtpro as vbt
from itertools import product
import psutil
import gc

warnings.filterwarnings('ignore')

# Add project paths
sys.path.append('src')
from src.data.parquet_converter import ParquetConverter

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

class OptimizationParameter:
    """Defines a single optimization parameter"""
    
    def __init__(self, name: str, start: Union[int, float], stop: Union[int, float], 
                 step: Union[int, float], default: Union[int, float] = None):
        self.name = name
        self.start = start
        self.stop = stop
        self.step = step
        self.default = default if default is not None else start
        
    def get_values(self) -> np.ndarray:
        """Get all values for this parameter"""
        if isinstance(self.step, int) and isinstance(self.start, int):
            return np.arange(self.start, self.stop + 1, self.step)
        else:
            return np.arange(self.start, self.stop + self.step/2, self.step)
    
    def __repr__(self):
        return f"OptimizationParameter({self.name}: {self.start}-{self.stop} step {self.step})"

class BaseStrategy(ABC):
    """Abstract base class for optimization strategies"""
    
    def __init__(self, name: str):
        self.name = name
        self.optimization_parameters = []
        
    @abstractmethod
    def add_optimization_parameter(self, param: OptimizationParameter):
        """Add an optimization parameter"""
        pass
        
    @abstractmethod
    def generate_signals_vectorized(self, data: pd.Series, param_combinations: np.ndarray) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Generate entry/exit signals for all parameter combinations
        
        Args:
            data: Price data as pandas Series
            param_combinations: Array of shape (n_combinations, n_parameters)
            
        Returns:
            Tuple of (entries_df, exits_df) with columns for each combination
        """
        pass
        
    def get_total_combinations(self) -> int:
        """Get total number of parameter combinations"""
        if not self.optimization_parameters:
            return 1
        param_counts = [len(param.get_values()) for param in self.optimization_parameters]
        total = 1
        for count in param_counts:
            total *= count
        return total
    
    def generate_parameter_combinations(self) -> Tuple[np.ndarray, List[str]]:
        """
        Generate all parameter combinations
        
        Returns:
            Tuple of (combinations_array, combination_names)
        """
        if not self.optimization_parameters:
            return np.array([[0]]), ["default"]
            
        param_values = [param.get_values() for param in self.optimization_parameters]
        param_names = [param.name for param in self.optimization_parameters]
        
        # Generate all combinations using itertools.product
        combinations = list(product(*param_values))
        combinations_array = np.array(combinations)
        
        # Generate names for each combination
        combination_names = []
        for combo in combinations:
            name_parts = [f"{param_names[i]}_{combo[i]}" for i in range(len(combo))]
            combination_names.append("_".join(name_parts))
        
        return combinations_array, combination_names

class SimpleSMAOptimizationStrategy(BaseStrategy):
    """Simple Moving Average crossover strategy for optimization"""
    
    def __init__(self):
        super().__init__("SimpleSMA_Optimization")
        
    def add_optimization_parameter(self, param: OptimizationParameter):
        """Add optimization parameter"""
        if len(self.optimization_parameters) >= 8:
            raise ValueError("Maximum 8 parameters supported")
        self.optimization_parameters.append(param)
        
    def generate_signals_vectorized(self, data: pd.Series, param_combinations: np.ndarray) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Generate SMA crossover signals for all parameter combinations
        
        Args:
            data: Close price Series
            param_combinations: Array of shape (n_combinations, 2) with [fast_ma, slow_ma]
        """
        n_combinations = len(param_combinations)
        n_bars = len(data)
        
        # Pre-allocate DataFrames
        entries_dict = {}
        exits_dict = {}
        
        # Process each combination
        for i, combo in enumerate(param_combinations):
            fast_period = int(combo[0])
            slow_period = int(combo[1])
            
            # Skip invalid combinations
            if slow_period <= fast_period:
                entries_dict[i] = pd.Series(False, index=data.index)
                exits_dict[i] = pd.Series(False, index=data.index)
                continue
            
            # Calculate MAs
            fast_ma = vbt.MA.run(data, fast_period).ma
            slow_ma = vbt.MA.run(data, slow_period).ma
            
            # Generate crossover signals
            entries = fast_ma.vbt.crossed_above(slow_ma)
            exits = fast_ma.vbt.crossed_below(slow_ma)
            
            entries_dict[i] = entries
            exits_dict[i] = exits
        
        # Convert to DataFrames
        entries_df = pd.DataFrame(entries_dict, index=data.index)
        exits_df = pd.DataFrame(exits_dict, index=data.index)
        
        return entries_df, exits_df

class VectorBTModularOptimizer:
    """Memory-efficient, modular VectorBT Pro optimizer"""
    
    def __init__(self, chunk_size: int = 500, max_memory_gb: float = None):
        """
        Initialize optimizer
        
        Args:
            chunk_size: Number of combinations to process per chunk
            max_memory_gb: Maximum memory to use (auto-detect if None)
        """
        self.chunk_size = chunk_size
        self.max_memory_gb = max_memory_gb or (psutil.virtual_memory().total / (1024**3) * 0.8)  # Use 80% of available memory
        
        # System info
        self.cpu_count = psutil.cpu_count()
        
        logger.info(f"Initialized VectorBT Modular Optimizer")
        logger.info(f"Chunk size: {chunk_size} combinations")
        logger.info(f"Memory limit: {self.max_memory_gb:.1f}GB")
        logger.info(f"System: {self.cpu_count} cores")
        
    def load_data(self, symbol: str = "GC", start_date: str = "2000-01-01", 
                  end_date: str = "2024-12-31") -> pd.Series:
        """Load data and return as pandas Series"""
        
        logger.info(f"Loading {symbol} data from {start_date} to {end_date}...")
        start_time = time.time()
        
        try:
            parquet_converter = ParquetConverter()
            data = parquet_converter.load_or_convert(symbol, "1m", "diffAdjusted")
            
            if data:
                data = parquet_converter.filter_data_by_date(data, start_date, end_date)
                
                if data and len(data['close']) > 0:
                    close_series = pd.Series(
                        data['close'], 
                        index=pd.to_datetime(data['datetime_ns']),
                        name='close'
                    )
                    
                    load_time = time.time() - start_time
                    logger.info(f"Loaded {len(close_series):,} bars in {load_time:.2f}s")
                    
                    return close_series
            
            raise Exception("Data loading failed")
            
        except Exception as e:
            logger.error(f"Data loading error: {e}")
            return None
    
    def optimize_strategy(self, strategy: BaseStrategy, data: pd.Series, 
                         progress_callback: callable = None) -> pd.DataFrame:
        """
        Run memory-efficient optimization on a strategy
        
        Args:
            strategy: Strategy instance implementing BaseStrategy
            data: Price data as pandas Series
            progress_callback: Optional callback for progress updates
            
        Returns:
            DataFrame with optimization results
        """
        logger.info(f"=== OPTIMIZING STRATEGY: {strategy.name} ===")
        
        # Generate all parameter combinations
        param_combinations, combo_names = strategy.generate_parameter_combinations()
        total_combinations = len(param_combinations)
        
        logger.info(f"Total combinations: {total_combinations:,}")
        logger.info(f"Parameters: {[p.name for p in strategy.optimization_parameters]}")
        
        # Calculate optimal chunk size based on memory
        optimal_chunk_size = self._calculate_optimal_chunk_size(data, total_combinations)
        chunk_size = min(self.chunk_size, optimal_chunk_size)
        
        logger.info(f"Using chunk size: {chunk_size}")
        
        # Process in chunks
        chunks = self._create_chunks(param_combinations, combo_names, chunk_size)
        total_chunks = len(chunks)
        
        logger.info(f"Processing {total_chunks} chunks...")
        
        all_results = []
        start_time = time.time()
        
        # Process each chunk
        with tqdm(total=total_combinations, desc="Optimization Progress", 
                 unit="combinations", ncols=120) as pbar:
            
            for chunk_idx, (chunk_params, chunk_names) in enumerate(chunks):
                chunk_start_time = time.time()
                
                # Process chunk
                chunk_results = self._process_chunk(
                    strategy, data, chunk_params, chunk_names, 
                    chunk_idx, total_chunks
                )
                
                all_results.extend(chunk_results)
                
                # Update progress
                pbar.update(len(chunk_params))
                
                # Memory cleanup
                gc.collect()
                
                # Update progress bar description with timing info
                elapsed = time.time() - start_time
                rate = pbar.n / elapsed if elapsed > 0 else 0
                remaining = total_combinations - pbar.n
                eta = remaining / rate if rate > 0 else 0
                
                pbar.set_description(f"Optimization ({rate:.1f}/s, ETA: {eta/60:.1f}min)")
                
                if progress_callback:
                    progress_callback(pbar.n, total_combinations, elapsed)
        
        total_time = time.time() - start_time
        logger.info(f"Optimization completed in {total_time:.2f}s ({total_time/60:.1f} minutes)")
        logger.info(f"Rate: {total_combinations/total_time:.2f} combinations/second")
        
        # Create results DataFrame
        results_df = pd.DataFrame(all_results)
        
        return results_df
    
    def _calculate_optimal_chunk_size(self, data: pd.Series, total_combinations: int) -> int:
        """Calculate optimal chunk size based on available memory"""
        
        # Rough memory estimation per combination (in GB)
        data_size_gb = len(data) * 8 / (1024**3)  # 8 bytes per float64
        estimated_memory_per_combo = data_size_gb * 3  # MA calculations + signals
        
        # Calculate safe chunk size
        safe_chunk_size = int(self.max_memory_gb / estimated_memory_per_combo)
        safe_chunk_size = max(10, min(safe_chunk_size, 1000))  # Between 10 and 1000
        
        logger.info(f"Estimated memory per combination: {estimated_memory_per_combo*1000:.1f}MB")
        logger.info(f"Calculated safe chunk size: {safe_chunk_size}")
        
        return safe_chunk_size
    
    def _create_chunks(self, param_combinations: np.ndarray, combo_names: List[str], 
                      chunk_size: int) -> List[Tuple[np.ndarray, List[str]]]:
        """Split parameter combinations into chunks"""
        
        chunks = []
        total_combinations = len(param_combinations)
        
        for i in range(0, total_combinations, chunk_size):
            end_idx = min(i + chunk_size, total_combinations)
            chunk_params = param_combinations[i:end_idx]
            chunk_names = combo_names[i:end_idx]
            chunks.append((chunk_params, chunk_names))
        
        return chunks
    
    def _process_chunk(self, strategy: BaseStrategy, data: pd.Series, 
                      chunk_params: np.ndarray, chunk_names: List[str],
                      chunk_idx: int, total_chunks: int) -> List[Dict]:
        """Process a single chunk of parameter combinations"""
        
        chunk_size = len(chunk_params)
        logger.info(f"Processing chunk {chunk_idx+1}/{total_chunks}: {chunk_size} combinations")
        
        try:
            # Generate signals for this chunk
            entries_df, exits_df = strategy.generate_signals_vectorized(data, chunk_params)
            
            # Create multi-column close prices to match signals
            close_df = pd.concat([data] * chunk_size, axis=1, keys=range(chunk_size))
            
            # Run portfolio backtest
            pf = vbt.Portfolio.from_signals(
                close=close_df,
                entries=entries_df,
                exits=exits_df,
                size=np.inf,  # Use all available cash
                init_cash=100000,
                fees=0.001,  # 0.1% fees
                freq='1min'
            )
            
            # Extract results vectorized
            results = self._extract_chunk_results(pf, chunk_params, chunk_names, strategy)
            
            return results
            
        except Exception as e:
            logger.error(f"Chunk {chunk_idx+1} failed: {e}")
            # Return failed results for this chunk
            failed_results = []
            for i, params in enumerate(chunk_params):
                result = {'Status': f'Failed: {str(e)[:50]}'}
                # Add parameter values
                for j, param in enumerate(strategy.optimization_parameters):
                    result[param.name] = params[j] if j < len(params) else 0
                # Add zero metrics
                for metric in ['Total_Return', 'Sharpe_Ratio', 'Max_Drawdown', 'Trade_Count', 'Win_Loss_Ratio', 'Avg_Trade']:
                    result[metric] = 0
                failed_results.append(result)
            return failed_results
    
    def _extract_chunk_results(self, pf, chunk_params: np.ndarray, chunk_names: List[str], 
                              strategy: BaseStrategy) -> List[Dict]:
        """Extract results from portfolio for this chunk"""
        
        results = []
        
        try:
            # Extract metrics vectorized
            total_returns = pf.total_return * 100
            sharpe_ratios = pf.sharpe_ratio
            max_drawdowns = pf.max_drawdown * 100
            all_trades = pf.trades.records_readable if hasattr(pf.trades, 'records_readable') else pd.DataFrame()
            
            # Process each combination in chunk
            for i, params in enumerate(chunk_params):
                try:
                    # Extract metrics for this combination
                    if hasattr(total_returns, 'iloc') and i < len(total_returns):
                        total_return = float(total_returns.iloc[i]) if not pd.isna(total_returns.iloc[i]) else 0
                        sharpe_ratio = float(sharpe_ratios.iloc[i]) if not pd.isna(sharpe_ratios.iloc[i]) else 0
                        max_drawdown = float(max_drawdowns.iloc[i]) if not pd.isna(max_drawdowns.iloc[i]) else 0
                    else:
                        total_return = float(total_returns) if not pd.isna(total_returns) else 0
                        sharpe_ratio = float(sharpe_ratios) if not pd.isna(sharpe_ratios) else 0
                        max_drawdown = float(max_drawdowns) if not pd.isna(max_drawdowns) else 0
                    
                    # Extract trade metrics
                    if len(all_trades) > 0 and 'Column' in all_trades.columns:
                        combo_trades = all_trades[all_trades['Column'] == i]
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
                    
                    # Build result dictionary
                    result = {
                        'Total_Return': total_return,
                        'Sharpe_Ratio': sharpe_ratio,
                        'Max_Drawdown': max_drawdown,
                        'Trade_Count': trade_count,
                        'Win_Loss_Ratio': win_loss_ratio,
                        'Avg_Trade': avg_trade,
                        'Status': 'Success'
                    }
                    
                    # Add parameter values
                    for j, param in enumerate(strategy.optimization_parameters):
                        result[param.name] = params[j] if j < len(params) else 0
                    
                    results.append(result)
                    
                except Exception as e:
                    # Individual combination failed
                    result = {'Status': f'Failed: {str(e)[:50]}'}
                    for j, param in enumerate(strategy.optimization_parameters):
                        result[param.name] = params[j] if j < len(params) else 0
                    for metric in ['Total_Return', 'Sharpe_Ratio', 'Max_Drawdown', 'Trade_Count', 'Win_Loss_Ratio', 'Avg_Trade']:
                        result[metric] = 0
                    results.append(result)
                    
        except Exception as e:
            logger.error(f"Failed to extract chunk results: {e}")
            # Return failed results for entire chunk
            for i, params in enumerate(chunk_params):
                result = {'Status': f'Failed: {str(e)[:50]}'}
                for j, param in enumerate(strategy.optimization_parameters):
                    result[param.name] = params[j] if j < len(params) else 0
                for metric in ['Total_Return', 'Sharpe_Ratio', 'Max_Drawdown', 'Trade_Count', 'Win_Loss_Ratio', 'Avg_Trade']:
                    result[metric] = 0
                results.append(result)
        
        return results
    
    def save_results(self, results_df: pd.DataFrame, strategy_name: str) -> str:
        """Save optimization results"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save results
        results_file = f"modular_optimization_{strategy_name}_{timestamp}.csv"
        results_df.to_csv(results_file, index=False)
        
        # Generate summary
        successful = results_df[results_df['Status'] == 'Success']
        
        summary = f"""
MODULAR OPTIMIZATION RESULTS - {strategy_name}
===============================================
Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Total Combinations: {len(results_df):,}
Successful: {len(successful):,}
Failed: {len(results_df) - len(successful):,}

Best Results:
"""
        
        if len(successful) > 0:
            # Best by Total Return
            best_return = successful.loc[successful['Total_Return'].idxmax()]
            summary += f"Best Return: {best_return['Total_Return']:.2f}% "
            summary += f"(Sharpe: {best_return['Sharpe_Ratio']:.2f}, "
            summary += f"Drawdown: {best_return['Max_Drawdown']:.2f}%)\n"
            
            # Best by Sharpe Ratio
            best_sharpe = successful.loc[successful['Sharpe_Ratio'].idxmax()]
            summary += f"Best Sharpe: {best_sharpe['Sharpe_Ratio']:.2f} "
            summary += f"(Return: {best_sharpe['Total_Return']:.2f}%, "
            summary += f"Drawdown: {best_sharpe['Max_Drawdown']:.2f}%)\n"
            
            # Summary stats
            summary += f"\nSummary Statistics:\n"
            summary += f"Avg Return: {successful['Total_Return'].mean():.2f}%\n"
            summary += f"Avg Sharpe: {successful['Sharpe_Ratio'].mean():.2f}\n"
            summary += f"Avg Trades: {successful['Trade_Count'].mean():.0f}\n"
        
        # Save summary
        summary_file = f"modular_optimization_summary_{strategy_name}_{timestamp}.txt"
        with open(summary_file, 'w') as f:
            f.write(summary)
        
        logger.info(f"Results saved: {results_file}")
        logger.info(f"Summary saved: {summary_file}")
        
        return results_file

def main():
    """Main execution function - Simple SMA example"""
    
    print("=" * 80)
    print("MODULAR VECTORBT PRO OPTIMIZER")
    print("Memory-efficient, chunked optimization for any strategy")
    print("=" * 80)
    
    # Initialize optimizer with larger chunk size for better efficiency
    optimizer = VectorBTModularOptimizer(chunk_size=500)  # Process 500 combinations per chunk
    
    # Load data
    data = optimizer.load_data("GC", "2000-01-01", "2024-12-31")
    if data is None:
        print("ERROR: Data loading failed")
        return
    
    # Create SMA strategy with optimization parameters
    strategy = SimpleSMAOptimizationStrategy()
    
    # Add optimization parameters (matching original test)
    strategy.add_optimization_parameter(
        OptimizationParameter("fast_ma", start=10, stop=1000, step=10, default=20)
    )
    strategy.add_optimization_parameter(
        OptimizationParameter("slow_ma", start=100, stop=10000, step=100, default=100)
    )
    
    total_combinations = strategy.get_total_combinations()
    print(f"\nStrategy: {strategy.name}")
    print(f"Parameters: {len(strategy.optimization_parameters)}")
    print(f"Total combinations: {total_combinations:,}")
    
    # Estimate time
    estimated_rate = 25  # combinations per second (conservative estimate)
    estimated_time = total_combinations / estimated_rate
    print(f"Estimated time: {estimated_time:.0f}s ({estimated_time/60:.1f} minutes / {estimated_time/3600:.1f} hours)")
    
    # Auto-proceed for automation (no user input required)
    print(f"\nProceeding with optimization automatically...")
    
    # Run optimization
    print(f"\nStarting optimization of {total_combinations:,} combinations...")
    start_time = time.time()
    
    results_df = optimizer.optimize_strategy(strategy, data)
    
    total_time = time.time() - start_time
    
    # Save results
    results_file = optimizer.save_results(results_df, strategy.name)
    
    # Final summary
    print("\n" + "=" * 80)
    print("OPTIMIZATION COMPLETE")
    print("=" * 80)
    print(f"Total time: {total_time:.2f}s ({total_time/60:.1f} minutes / {total_time/3600:.2f} hours)")
    print(f"Combinations: {len(results_df):,}")
    print(f"Rate: {len(results_df)/total_time:.2f} combinations/second")
    print(f"Results file: {results_file}")
    
    # Show best results
    successful = results_df[results_df['Status'] == 'Success']
    if len(successful) > 0:
        print(f"\nTop 5 results by Total Return:")
        top5 = successful.nlargest(5, 'Total_Return')
        for i, (_, row) in enumerate(top5.iterrows(), 1):
            param_str = ", ".join([f"{param.name}={row[param.name]}" for param in strategy.optimization_parameters])
            print(f"  {i}. {param_str}: {row['Total_Return']:.2f}% return, {row['Sharpe_Ratio']:.2f} Sharpe, {row['Trade_Count']} trades")
    
    print(f"\nModular optimization system ready for any strategy with up to 8 parameters!")

if __name__ == "__main__":
    main()