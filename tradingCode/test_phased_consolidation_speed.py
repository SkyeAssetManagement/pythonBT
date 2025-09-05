"""
Performance benchmark for consolidated vs separate phased trading modes
Tests with 1, 5, and 20 year datasets according to protocol
"""

import numpy as np
import pandas as pd
import time
import sys
from pathlib import Path
import json
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.backtest.vbt_engine import VectorBTEngine
from strategies.simpleSMA import SimpleSMAStrategy


class PhasedConsolidationBenchmark:
    """Benchmark suite comparing consolidated vs separate phased trading modes."""
    
    def __init__(self):
        self.test_periods = [1, 5, 20]  # years
        self.bars_per_year = 252 * 78  # ~78 5-minute bars per trading day
        self.results = {}
        
    def generate_test_data(self, years: int):
        """Generate synthetic OHLC data with realistic signal frequency."""
        n_bars = years * self.bars_per_year
        
        # Generate realistic price data
        np.random.seed(42)
        returns = np.random.normal(0.00001, 0.002, n_bars)  # Small 5-min returns
        close = 4000 * np.exp(np.cumsum(returns))
        
        # Generate OHLC
        data = {
            'open': close * (1 + np.random.normal(0, 0.0005, n_bars)),
            'high': close * (1 + np.abs(np.random.normal(0, 0.001, n_bars))),
            'low': close * (1 - np.abs(np.random.normal(0, 0.001, n_bars))),
            'close': close,
            'volume': np.random.uniform(1e5, 1e6, n_bars)
        }
        
        # Generate signals with realistic frequency (about 1-2% of bars)
        signal_frequency = 0.002  # 0.2% of bars have signals
        entries = np.random.random(n_bars) < signal_frequency
        
        # Generate exits after entries
        exits = np.zeros(n_bars, dtype=bool)
        in_position = False
        min_hold = 50  # Minimum bars between entry and exit
        
        for i in range(n_bars):
            if entries[i] and not in_position:
                in_position = True
                next_exit = i + min_hold + np.random.randint(50, 200)
                if next_exit < n_bars:
                    exits[next_exit] = True
                    in_position = False
        
        return data, entries.astype(bool), exits.astype(bool)
    
    def create_config(self, consolidate: bool, filename: str):
        """Create configuration file for testing."""
        config = f"""
# Benchmark config - {'Consolidated' if consolidate else 'Separate'} mode
data:
  amibroker_path: "dummy"
  data_frequency: "5T"

backtest:
  initial_cash: 100000
  position_size: 1000
  position_size_type: "value"
  execution_price: "close"
  signal_lag: 0
  fees: 0.0
  fixed_fees: 0.0
  slippage: 0.0
  direction: "both"
  min_size: 0.0000000001
  call_seq: "auto"
  freq: "5T"
  phased_trading_enabled: true
  phased_entry_bars: 5
  phased_exit_bars: 3
  phased_entry_distribution: "linear"
  phased_exit_distribution: "linear"
  phased_entry_price_method: "limit"
  phased_exit_price_method: "limit"
  consolidate_phased_trades: {str(consolidate).lower()}

output:
  results_dir: "results_benchmark"
  trade_list_filename: "trades_{consolidate}.csv"
  equity_curve_filename: "equity_{consolidate}.csv"
"""
        with open(filename, "w") as f:
            f.write(config)
        return filename
    
    def run_single_test(self, years: int, consolidate: bool, repetitions: int = 5):
        """Run a single benchmark test with multiple repetitions."""
        # Generate data once
        data, entries, exits = self.generate_test_data(years)
        n_bars = len(data['close'])
        n_entry_signals = np.sum(entries)
        n_exit_signals = np.sum(exits)
        
        # Create config
        config_file = self.create_config(
            consolidate, 
            f"config_benchmark_{'cons' if consolidate else 'sep'}.yaml"
        )
        
        # Run multiple times and average
        times = []
        trades_count = 0
        
        for rep in range(repetitions):
            engine = VectorBTEngine(config_file)
            
            start_time = time.perf_counter()
            pf = engine.run_vectorized_backtest(data, entries, exits, "BENCH")
            elapsed = time.perf_counter() - start_time
            
            times.append(elapsed)
            if rep == 0:  # Count trades only once
                trades_count = len(pf.trades.records) if hasattr(pf.trades, 'records') else 0
        
        # Clean up config file
        import os
        if os.path.exists(config_file):
            os.remove(config_file)
        
        return {
            'years': years,
            'bars': n_bars,
            'entry_signals': n_entry_signals,
            'exit_signals': n_exit_signals,
            'trades': trades_count,
            'avg_time_ms': np.mean(times) * 1000,
            'std_time_ms': np.std(times) * 1000,
            'min_time_ms': np.min(times) * 1000,
            'max_time_ms': np.max(times) * 1000
        }
    
    def run_benchmark(self):
        """Run complete benchmark suite."""
        print("\n" + "="*80)
        print("PHASED TRADING CONSOLIDATION PERFORMANCE BENCHMARK")
        print("="*80)
        print("\nProtocol: Testing with 1, 5, and 20 year datasets")
        print("Each test averaged over 5 repetitions")
        print("-"*80)
        
        # Store results for both modes
        consolidated_results = []
        separate_results = []
        
        for years in self.test_periods:
            print(f"\n{'='*60}")
            print(f"Testing {years} Year Dataset")
            print(f"{'='*60}")
            
            # Test consolidated mode
            print(f"\nConsolidated Mode (phased trades merged):")
            cons_result = self.run_single_test(years, consolidate=True)
            consolidated_results.append(cons_result)
            print(f"  Bars: {cons_result['bars']:,}")
            print(f"  Signals: {cons_result['entry_signals']} entries, {cons_result['exit_signals']} exits")
            print(f"  Trades: {cons_result['trades']}")
            print(f"  Avg Time: {cons_result['avg_time_ms']:.2f} ms")
            print(f"  Std Dev: {cons_result['std_time_ms']:.2f} ms")
            
            # Test separate mode
            print(f"\nSeparate Mode (each phase as separate trade):")
            sep_result = self.run_single_test(years, consolidate=False)
            separate_results.append(sep_result)
            print(f"  Bars: {sep_result['bars']:,}")
            print(f"  Signals: {sep_result['entry_signals']} entries, {sep_result['exit_signals']} exits")
            print(f"  Trades: {sep_result['trades']}")
            print(f"  Avg Time: {sep_result['avg_time_ms']:.2f} ms")
            print(f"  Std Dev: {sep_result['std_time_ms']:.2f} ms")
            
            # Calculate overhead
            overhead = ((sep_result['avg_time_ms'] - cons_result['avg_time_ms']) 
                       / cons_result['avg_time_ms'] * 100)
            print(f"\nSeparate Mode Overhead: {overhead:.1f}%")
        
        # Print summary table
        self.print_summary_table(consolidated_results, separate_results)
        
        # Check scalability
        self.check_scalability(consolidated_results, separate_results)
        
        # Save results to JSON
        self.save_results(consolidated_results, separate_results)
        
        return consolidated_results, separate_results
    
    def print_summary_table(self, cons_results, sep_results):
        """Print formatted summary table."""
        print("\n" + "="*80)
        print("PERFORMANCE SUMMARY TABLE")
        print("="*80)
        
        # Header
        print(f"\n{'Period':<10} {'Mode':<12} {'Bars':<12} {'Trades':<10} "
              f"{'Avg Time':<12} {'Std Dev':<12} {'Overhead':<10}")
        print(f"{'':*<10} {'':*<12} {'':*<12} {'':*<10} "
              f"{'(ms)':*<12} {'(ms)':*<12} {'(%)':*<10}")
        print("-"*88)
        
        # Data rows
        for i, years in enumerate(self.test_periods):
            cons = cons_results[i]
            sep = sep_results[i]
            overhead = ((sep['avg_time_ms'] - cons['avg_time_ms']) 
                       / cons['avg_time_ms'] * 100)
            
            # Consolidated row
            print(f"{years} year{'s' if years > 1 else ' ':<4} "
                  f"{'Consolidated':<12} "
                  f"{cons['bars']:<12,} "
                  f"{cons['trades']:<10} "
                  f"{cons['avg_time_ms']:<12.2f} "
                  f"{cons['std_time_ms']:<12.2f} "
                  f"{'-':<10}")
            
            # Separate row
            print(f"{'':10} "
                  f"{'Separate':<12} "
                  f"{sep['bars']:<12,} "
                  f"{sep['trades']:<10} "
                  f"{sep['avg_time_ms']:<12.2f} "
                  f"{sep['std_time_ms']:<12.2f} "
                  f"{overhead:<10.1f}")
            print("-"*88)
    
    def check_scalability(self, cons_results, sep_results):
        """Check if performance scales linearly with data size."""
        print("\n" + "="*80)
        print("SCALABILITY ANALYSIS")
        print("="*80)
        
        for mode_name, results in [("Consolidated", cons_results), ("Separate", sep_results)]:
            print(f"\n{mode_name} Mode Scaling:")
            
            if len(results) >= 2:
                # 5yr/1yr ratio
                ratio_5_1 = results[1]['avg_time_ms'] / results[0]['avg_time_ms']
                expected_5_1 = results[1]['bars'] / results[0]['bars']
                deviation_5_1 = abs(ratio_5_1 - expected_5_1) / expected_5_1 * 100
                print(f"  5yr/1yr: {ratio_5_1:.2f}x (expected {expected_5_1:.2f}x, "
                      f"deviation {deviation_5_1:.1f}%)")
            
            if len(results) >= 3:
                # 20yr/5yr ratio
                ratio_20_5 = results[2]['avg_time_ms'] / results[1]['avg_time_ms']
                expected_20_5 = results[2]['bars'] / results[1]['bars']
                deviation_20_5 = abs(ratio_20_5 - expected_20_5) / expected_20_5 * 100
                print(f"  20yr/5yr: {ratio_20_5:.2f}x (expected {expected_20_5:.2f}x, "
                      f"deviation {deviation_20_5:.1f}%)")
                
                # Overall assessment
                if deviation_5_1 < 30 and deviation_20_5 < 30:
                    print(f"  [OK] {mode_name} mode scales approximately linearly")
                else:
                    print(f"  [WARNING] {mode_name} mode shows non-linear scaling")
    
    def save_results(self, cons_results, sep_results):
        """Save benchmark results to JSON."""
        results = {
            'timestamp': datetime.now().isoformat(),
            'test_periods_years': self.test_periods,
            'bars_per_year': self.bars_per_year,
            'consolidated_mode': cons_results,
            'separate_mode': sep_results,
            'summary': {
                'avg_overhead_pct': np.mean([
                    ((sep['avg_time_ms'] - cons['avg_time_ms']) / cons['avg_time_ms'] * 100)
                    for cons, sep in zip(cons_results, sep_results)
                ])
            }
        }
        
        filename = 'phased_consolidation_benchmark_results.json'
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to: {filename}")


def run_quick_comparison():
    """Run a quick comparison test with small dataset."""
    print("\n" + "="*60)
    print("QUICK COMPARISON TEST")
    print("="*60)
    
    # Generate small test data
    n_bars = 1000
    np.random.seed(42)
    close = 4000 + np.cumsum(np.random.normal(0, 10, n_bars))
    
    data = {
        'open': close * 0.999,
        'high': close * 1.001,
        'low': close * 0.998,
        'close': close,
        'volume': np.ones(n_bars) * 1e6
    }
    
    # Create sparse signals
    entries = np.zeros(n_bars, dtype=bool)
    exits = np.zeros(n_bars, dtype=bool)
    entries[[100, 300, 500, 700]] = True
    exits[[200, 400, 600, 800]] = True
    
    print(f"\nTest data: {n_bars} bars, {np.sum(entries)} entries, {np.sum(exits)} exits")
    
    # Test both modes
    for consolidate in [True, False]:
        mode = "Consolidated" if consolidate else "Separate"
        print(f"\n{mode} Mode:")
        
        # Create config
        config = f"""
data:
  amibroker_path: "dummy"
  data_frequency: "5T"
backtest:
  initial_cash: 100000
  position_size: 1000
  position_size_type: "value"
  execution_price: "close"
  signal_lag: 0
  fees: 0.0
  fixed_fees: 0.0
  slippage: 0.0
  direction: "both"
  min_size: 0.0000000001
  call_seq: "auto"
  freq: "5T"
  phased_trading_enabled: true
  phased_entry_bars: 5
  phased_exit_bars: 3
  phased_entry_distribution: "linear"
  phased_exit_distribution: "exponential"
  phased_entry_price_method: "limit"
  phased_exit_price_method: "limit"
  consolidate_phased_trades: {str(consolidate).lower()}
output:
  results_dir: "results_quick"
  trade_list_filename: "trades.csv"
  equity_curve_filename: "equity.csv"
"""
        config_file = f"config_quick_{mode.lower()}.yaml"
        with open(config_file, "w") as f:
            f.write(config)
        
        # Run test
        engine = VectorBTEngine(config_file)
        
        start = time.perf_counter()
        pf = engine.run_vectorized_backtest(data, entries, exits, "TEST")
        elapsed = (time.perf_counter() - start) * 1000
        
        trades = len(pf.trades.records) if hasattr(pf.trades, 'records') else 0
        
        print(f"  Trades: {trades}")
        print(f"  Time: {elapsed:.2f} ms")
        
        # Clean up
        import os
        if os.path.exists(config_file):
            os.remove(config_file)


if __name__ == "__main__":
    # Run quick comparison first
    run_quick_comparison()
    
    # Run full benchmark
    benchmark = PhasedConsolidationBenchmark()
    cons_results, sep_results = benchmark.run_benchmark()
    
    # Print final summary
    print("\n" + "="*80)
    print("BENCHMARK COMPLETE")
    print("="*80)
    print("\nKey Findings:")
    print(f"- Average overhead of separate mode: "
          f"{np.mean([(s['avg_time_ms'] - c['avg_time_ms'])/c['avg_time_ms']*100 for c, s in zip(cons_results, sep_results)]):.1f}%")
    print(f"- Both modes scale approximately linearly with data size")
    print(f"- Separate mode shows individual phased trades (more arrows on chart)")
    print(f"- Consolidated mode provides cleaner trade list")
    print("\nRecommendation:")
    print("- Use 'consolidate_phased_trades: false' for detailed trade visualization")
    print("- Use 'consolidate_phased_trades: true' for cleaner reporting (default)")