"""
Performance testing for phased trading implementation
Tests scalability with 1, 2, and 20 years of data
"""

import numpy as np
import pandas as pd
import vectorbtpro as vbt
from datetime import datetime, timedelta
import time
from pathlib import Path
import sys
import gc

sys.path.insert(0, str(Path(__file__).parent))

from phased_trading_correct import PhasedTradingCorrect


class OptimizedPhasedTrading:
    """
    Optimized phased trading implementation using pure array operations.
    Avoids loops and uses vectorized operations for all calculations.
    """
    
    def __init__(self, n_phases: int = 5, phase_distribution: str = "equal"):
        """Initialize optimized phased trading."""
        self.n_phases = n_phases
        self.phase_distribution = phase_distribution
        self.phase_weights = self._calculate_phase_weights()
        
    def _calculate_phase_weights(self) -> np.ndarray:
        """Calculate the weights for each phase."""
        if self.phase_distribution == "equal":
            weights = np.ones(self.n_phases) / self.n_phases
        elif self.phase_distribution == "front_loaded":
            weights = np.linspace(2, 1, self.n_phases)
            weights = weights / weights.sum()
        elif self.phase_distribution == "back_loaded":
            weights = np.linspace(1, 2, self.n_phases)
            weights = weights / weights.sum()
        else:
            raise ValueError(f"Unknown distribution: {self.phase_distribution}")
        return weights
    
    def create_phased_signals_optimized(self, 
                                       master_entries: np.ndarray,
                                       master_exits: np.ndarray,
                                       data: dict) -> dict:
        """
        Optimized version using convolution for signal spreading.
        This avoids loops and uses pure array operations.
        """
        n_bars = len(master_entries)
        
        # Create convolution kernel for spreading signals
        kernel = self.phase_weights
        
        # Use convolution to spread signals across multiple bars
        # This is much faster than looping through signal indices
        phased_entries = np.convolve(master_entries.astype(float), kernel, mode='same')
        phased_exits = np.convolve(master_exits.astype(float), kernel, mode='same')
        
        # Convert back to boolean where signal exists
        phased_entries_bool = phased_entries > 0
        phased_exits_bool = phased_exits > 0
        
        # For size arrays, we use the convolution result directly
        entry_sizes = phased_entries
        exit_sizes = phased_exits
        
        # Calculate HLC3 prices using vectorized operations
        hlc3_prices = (data['high'] + data['low'] + data['close']) / 3.0
        
        # Expected prices are just HLC3 where we have signals
        expected_entry_prices = np.where(phased_entries_bool, hlc3_prices, np.nan)
        expected_exit_prices = np.where(phased_exits_bool, hlc3_prices, np.nan)
        
        return {
            'phased_entries': phased_entries_bool,
            'phased_exits': phased_exits_bool,
            'entry_sizes': entry_sizes,
            'exit_sizes': exit_sizes,
            'expected_entry_prices': expected_entry_prices,
            'expected_exit_prices': expected_exit_prices,
            'hlc3_prices': hlc3_prices
        }
    
    def run_phased_backtest_optimized(self,
                                     data: dict,
                                     master_entries: np.ndarray,
                                     master_exits: np.ndarray,
                                     config: dict) -> vbt.Portfolio:
        """
        Optimized backtest using vectorized operations throughout.
        """
        # Create phased signals using optimized method
        phased_results = self.create_phased_signals_optimized(master_entries, master_exits, data)
        
        # Use HLC3 prices for execution
        execution_prices = phased_results['hlc3_prices']
        
        # Position sizing
        position_size = config['backtest']['position_size']
        
        # Create size arrays efficiently
        entry_mask = phased_results['phased_entries']
        exit_mask = phased_results['phased_exits']
        
        # Vectorized size calculation
        custom_sizes = np.where(entry_mask, 
                               phased_results['entry_sizes'] * position_size,
                               0)
        
        # For exits, use percent sizing
        size_type_array = np.full(len(data['close']), vbt.pf_enums.SizeType.Value)
        size_type_array[exit_mask] = vbt.pf_enums.SizeType.Percent
        
        # Exit sizes as percentages
        exit_sizes_pct = phased_results['exit_sizes'] * 100
        
        # Combine sizes
        final_sizes = np.where(exit_mask, exit_sizes_pct, custom_sizes)
        
        # Run VectorBT backtest
        pf = vbt.Portfolio.from_signals(
            close=execution_prices,
            entries=phased_results['phased_entries'],
            exits=phased_results['phased_exits'],
            size=final_sizes,
            size_type=size_type_array,
            init_cash=config['backtest']['initial_cash'],
            direction=config['backtest']['direction'],
            fees=config['backtest'].get('fees', 0),
            fixed_fees=config['backtest'].get('fixed_fees', 0),
            slippage=config['backtest'].get('slippage', 0),
            freq=config['backtest'].get('freq', '1T')
        )
        
        return pf


def generate_synthetic_data(n_years: int, bars_per_day: int = 390) -> dict:
    """
    Generate synthetic OHLC data for testing.
    Uses realistic parameters but synthetic data for consistent testing.
    """
    n_days = n_years * 252  # Trading days per year
    n_bars = n_days * bars_per_day
    
    print(f"Generating {n_years} year(s) of data: {n_bars:,} bars")
    
    # Generate realistic price movement
    np.random.seed(42)
    
    # Random walk for price
    returns = np.random.normal(0.0001, 0.001, n_bars)
    close = 100 * np.exp(np.cumsum(returns))
    
    # Generate OHLC from close
    noise = np.random.uniform(0.998, 1.002, n_bars)
    open_prices = close * noise
    
    high = np.maximum(open_prices, close) * np.random.uniform(1.0001, 1.002, n_bars)
    low = np.minimum(open_prices, close) * np.random.uniform(0.998, 0.9999, n_bars)
    
    volume = np.random.uniform(1000, 10000, n_bars)
    
    return {
        'open': open_prices,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    }


def generate_signals(n_bars: int, n_signals: int = 100) -> tuple:
    """
    Generate evenly spaced signals for testing.
    """
    # Space signals evenly through the data
    signal_spacing = n_bars // (n_signals * 2)
    
    entries = np.zeros(n_bars, dtype=bool)
    exits = np.zeros(n_bars, dtype=bool)
    
    for i in range(n_signals):
        entry_idx = i * signal_spacing * 2
        exit_idx = entry_idx + signal_spacing
        
        if entry_idx < n_bars:
            entries[entry_idx] = True
        if exit_idx < n_bars:
            exits[exit_idx] = True
    
    return entries, exits


def run_performance_test():
    """
    Run performance tests with different data sizes.
    """
    print("="*80)
    print("PHASED TRADING PERFORMANCE TEST")
    print("="*80)
    print(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    # Configuration
    config = {
        'backtest': {
            'initial_cash': 100000,
            'position_size': 10000,
            'position_size_type': 'value',
            'direction': 'longonly',
            'fees': 0.0002,
            'fixed_fees': 1.0,
            'slippage': 0.0001,
            'freq': '1T'
        }
    }
    
    # Test with different data sizes
    test_years = [1, 2, 20]
    results = []
    
    # Test both implementations
    implementations = [
        ("Original", PhasedTradingCorrect),
        ("Optimized", OptimizedPhasedTrading)
    ]
    
    for impl_name, impl_class in implementations:
        print(f"\n### Testing {impl_name} Implementation")
        print("-"*60)
        
        impl_results = []
        
        for n_years in test_years:
            # Generate data
            print(f"\n## {n_years} Year(s) Test")
            
            start_gen = time.time()
            data = generate_synthetic_data(n_years)
            gen_time = time.time() - start_gen
            print(f"Data generation: {gen_time:.2f}s")
            
            # Generate signals (100 trades per year)
            n_signals = n_years * 100
            entries, exits = generate_signals(len(data['close']), n_signals)
            print(f"Signals: {entries.sum()} entries, {exits.sum()} exits")
            
            # Create phased trader
            trader = impl_class(n_phases=5, phase_distribution="equal")
            
            # Time the signal creation
            start_signal = time.time()
            if impl_name == "Optimized":
                phased_results = trader.create_phased_signals_optimized(entries, exits, data)
            else:
                phased_results = trader.create_phased_signals(entries, exits, data)
            signal_time = time.time() - start_signal
            
            print(f"Signal creation: {signal_time:.3f}s")
            print(f"Phased entries: {phased_results['phased_entries'].sum()}")
            print(f"Phased exits: {phased_results['phased_exits'].sum()}")
            
            # Time the backtest
            start_backtest = time.time()
            if impl_name == "Optimized":
                pf = trader.run_phased_backtest_optimized(data, entries, exits, config)
            else:
                pf, _ = trader.run_phased_backtest(data, entries, exits, config)
            backtest_time = time.time() - start_backtest
            
            print(f"Backtest execution: {backtest_time:.3f}s")
            print(f"Total time: {signal_time + backtest_time:.3f}s")
            
            # Calculate metrics
            n_bars = len(data['close'])
            bars_per_second = n_bars / (signal_time + backtest_time)
            
            impl_results.append({
                'implementation': impl_name,
                'years': n_years,
                'n_bars': n_bars,
                'n_signals': n_signals,
                'signal_time': signal_time,
                'backtest_time': backtest_time,
                'total_time': signal_time + backtest_time,
                'bars_per_second': bars_per_second,
                'return': float(pf.total_return),
                'n_trades': len(pf.trades.records)
            })
            
            # Clean up memory
            del pf
            del data
            gc.collect()
        
        results.extend(impl_results)
    
    # Analyze scaling
    print("\n" + "="*80)
    print("PERFORMANCE SCALING ANALYSIS")
    print("="*80)
    
    # Create DataFrame for analysis
    results_df = pd.DataFrame(results)
    
    for impl_name in ["Original", "Optimized"]:
        impl_data = results_df[results_df['implementation'] == impl_name]
        
        print(f"\n### {impl_name} Implementation")
        print("-"*60)
        
        # Check scaling
        times = impl_data['total_time'].values
        years = impl_data['years'].values
        
        # Calculate scaling factor
        time_1yr = times[0]
        time_2yr = times[1] if len(times) > 1 else 0
        time_20yr = times[2] if len(times) > 2 else 0
        
        if time_2yr > 0:
            scaling_2yr = time_2yr / time_1yr
            print(f"2yr vs 1yr scaling: {scaling_2yr:.2f}x (ideal: 2.0x)")
        
        if time_20yr > 0:
            scaling_20yr = time_20yr / time_1yr
            print(f"20yr vs 1yr scaling: {scaling_20yr:.2f}x (ideal: 20.0x)")
            
            # Check if scaling is sublinear (good) or superlinear (bad)
            if scaling_20yr < 20:
                print("✓ GOOD: Sublinear scaling detected (faster than linear)")
            elif scaling_20yr > 25:
                print("✗ BAD: Superlinear scaling detected (slower than linear)")
            else:
                print("~ OK: Approximately linear scaling")
        
        # Show performance table
        print("\nPerformance Summary:")
        print(impl_data[['years', 'n_bars', 'total_time', 'bars_per_second']].to_string(index=False))
    
    # Save results
    results_df.to_csv("phased_performance_results.csv", index=False)
    print("\n" + "="*80)
    print("Results saved to phased_performance_results.csv")
    
    # Compare implementations
    if len(results_df['implementation'].unique()) > 1:
        print("\n### Implementation Comparison")
        print("-"*60)
        
        for n_years in test_years:
            year_data = results_df[results_df['years'] == n_years]
            if len(year_data) > 1:
                orig_time = year_data[year_data['implementation'] == 'Original']['total_time'].values[0]
                opt_time = year_data[year_data['implementation'] == 'Optimized']['total_time'].values[0]
                speedup = orig_time / opt_time
                print(f"{n_years} year(s): {speedup:.2f}x speedup")
    
    return results_df


if __name__ == "__main__":
    # Run performance tests
    results = run_performance_test()
    
    # Check if optimization is needed
    opt_data = results[results['implementation'] == 'Optimized']
    if len(opt_data) > 0:
        times = opt_data['total_time'].values
        if len(times) >= 3:
            scaling_20yr = times[2] / times[0]
            if scaling_20yr > 25:
                print("\n" + "!"*60)
                print("WARNING: Performance scaling is not optimal!")
                print("Further optimization needed for large datasets")
                print("!"*60)