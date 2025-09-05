"""
Test script for phased trading integration with main.py
Demonstrates phased entries and exits with a simple strategy
"""

import numpy as np
import pandas as pd
import sys
from pathlib import Path
import matplotlib.pyplot as plt

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data.parquet_loader import ParquetLoader
from src.backtest.vbt_engine import VectorBTEngine
from strategies.simpleSMA import SimpleSMAStrategy


def test_phased_vs_normal():
    """Compare phased trading vs normal trading."""
    
    print("="*70)
    print("PHASED TRADING INTEGRATION TEST")
    print("="*70)
    
    # Load sample data
    print("\n1. Loading data...")
    # Use synthetic data for testing to avoid path issues
    data = None
    timestamps = None
    symbol = "TEST"  # Symbol for testing
    
    if data is None:
        print("   Using synthetic data for testing...")
        # Generate synthetic data for testing
        n_bars = 5000
        np.random.seed(42)
        returns = np.random.normal(0.0001, 0.01, n_bars)
        close = 4000 * np.exp(np.cumsum(returns))
        
        data = {
            'open': close * (1 + np.random.normal(0, 0.001, n_bars)),
            'high': close * (1 + np.abs(np.random.normal(0, 0.005, n_bars))),
            'low': close * (1 - np.abs(np.random.normal(0, 0.005, n_bars))),
            'close': close,
            'volume': np.random.uniform(1e5, 1e6, n_bars)
        }
        timestamps = pd.date_range(start='2024-01-01', periods=n_bars, freq='5min').values
    
    print(f"   Loaded {len(data['close'])} bars of data")
    
    # Initialize strategy
    strategy = SimpleSMAStrategy()
    
    # Generate signals
    print("\n2. Generating trading signals...")
    entries, exits = strategy.generate_signals(data)
    print(f"   Generated {np.sum(entries)} entry signals and {np.sum(exits)} exit signals")
    
    # Test 1: Normal trading (no phasing)
    print("\n3. Running backtest WITHOUT phased trading...")
    engine_normal = VectorBTEngine("config.yaml")
    pf_normal = engine_normal.run_vectorized_backtest(data, entries, exits, symbol)
    
    metrics_normal = {
        'total_return': float(pf_normal.total_return) * 100,
        'sharpe_ratio': float(pf_normal.sharpe_ratio),
        'max_drawdown': float(pf_normal.max_drawdown) * 100,
        'total_trades': len(pf_normal.trades.records) if hasattr(pf_normal.trades, 'records') else 0
    }
    
    print(f"   Normal Results:")
    print(f"   - Total Return: {metrics_normal['total_return']:.2f}%")
    print(f"   - Sharpe Ratio: {metrics_normal['sharpe_ratio']:.2f}")
    print(f"   - Max Drawdown: {metrics_normal['max_drawdown']:.2f}%")
    print(f"   - Total Trades: {metrics_normal['total_trades']}")
    
    # Test 2: Phased trading
    print("\n4. Running backtest WITH phased trading...")
    print("   Configuration: 5-bar linear entry, 3-bar exponential exit")
    engine_phased = VectorBTEngine("config_phased_test.yaml")
    pf_phased = engine_phased.run_vectorized_backtest(data, entries, exits, symbol)
    
    metrics_phased = {
        'total_return': float(pf_phased.total_return) * 100,
        'sharpe_ratio': float(pf_phased.sharpe_ratio),
        'max_drawdown': float(pf_phased.max_drawdown) * 100,
        'total_trades': len(pf_phased.trades.records) if hasattr(pf_phased.trades, 'records') else 0
    }
    
    print(f"   Phased Results:")
    print(f"   - Total Return: {metrics_phased['total_return']:.2f}%")
    print(f"   - Sharpe Ratio: {metrics_phased['sharpe_ratio']:.2f}")
    print(f"   - Max Drawdown: {metrics_phased['max_drawdown']:.2f}%")
    print(f"   - Total Trades: {metrics_phased['total_trades']}")
    
    # Compare results
    print("\n5. COMPARISON (Phased vs Normal):")
    print("="*50)
    print(f"   Return Difference: {metrics_phased['total_return'] - metrics_normal['total_return']:.2f}%")
    print(f"   Sharpe Difference: {metrics_phased['sharpe_ratio'] - metrics_normal['sharpe_ratio']:.2f}")
    print(f"   Drawdown Difference: {metrics_phased['max_drawdown'] - metrics_normal['max_drawdown']:.2f}%")
    
    # Export detailed results
    print("\n6. Exporting results...")
    
    # Export normal results
    engine_normal.export_results({
        'portfolio': pf_normal,
        'trades': engine_normal.generate_trade_list(pf_normal, timestamps, symbol),
        'metrics': metrics_normal
    }, output_dir="results_normal", timestamps=timestamps)
    
    # Export phased results
    engine_phased.export_results({
        'portfolio': pf_phased,
        'trades': engine_phased.generate_trade_list(pf_phased, timestamps, symbol),
        'metrics': metrics_phased
    }, output_dir="results_phased", timestamps=timestamps)
    
    print("   Results exported to 'results_normal' and 'results_phased' directories")
    
    # Plot equity curves for comparison
    print("\n7. Creating comparison plot...")
    try:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Plot equity curves
        equity_normal = pf_normal.value.values.flatten() if hasattr(pf_normal.value, 'values') else pf_normal.value
        equity_phased = pf_phased.value.values.flatten() if hasattr(pf_phased.value, 'values') else pf_phased.value
        
        ax1.plot(equity_normal, label='Normal Trading', linewidth=2, alpha=0.8)
        ax1.plot(equity_phased, label='Phased Trading (5 entry, 3 exit)', linewidth=2, alpha=0.8)
        ax1.set_title('Equity Curves Comparison')
        ax1.set_xlabel('Time (bars)')
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot return difference
        returns_normal = pf_normal.returns.values.flatten() if hasattr(pf_normal.returns, 'values') else pf_normal.returns
        returns_phased = pf_phased.returns.values.flatten() if hasattr(pf_phased.returns, 'values') else pf_phased.returns
        
        cumulative_diff = np.cumsum(returns_phased - returns_normal) * 100
        ax2.plot(cumulative_diff, color='green' if cumulative_diff[-1] > 0 else 'red', linewidth=2)
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax2.set_title('Cumulative Return Difference (Phased - Normal)')
        ax2.set_xlabel('Time (bars)')
        ax2.set_ylabel('Cumulative Difference (%)')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('phased_trading_comparison.png', dpi=100)
        print("   Comparison plot saved as 'phased_trading_comparison.png'")
        plt.show()
        
    except Exception as e:
        print(f"   Warning: Could not create plot: {e}")
    
    print("\n" + "="*70)
    print("PHASED TRADING TEST COMPLETE")
    print("="*70)
    
    return metrics_normal, metrics_phased


def demonstrate_phased_schedule():
    """Demonstrate how phased trading distributes positions."""
    
    print("\n" + "="*70)
    print("PHASED TRADING SCHEDULE DEMONSTRATION")
    print("="*70)
    
    from src.backtest.phased_trading_engine import PhasedTradingEngine, PhasedConfig
    
    # Create example configurations
    configs = [
        {
            "name": "Linear 5-bar entry",
            "config": PhasedConfig(
                enabled=True,
                entry_bars=5,
                entry_distribution="linear",
                entry_price_method="limit"
            )
        },
        {
            "name": "Exponential 4-bar entry",
            "config": PhasedConfig(
                enabled=True,
                entry_bars=4,
                entry_distribution="exponential",
                entry_price_method="market"
            )
        },
        {
            "name": "Custom weighted 3-bar exit",
            "config": PhasedConfig(
                enabled=True,
                exit_bars=3,
                exit_distribution="custom",
                exit_custom_weights=np.array([0.5, 0.3, 0.2]),
                exit_price_method="limit"
            )
        }
    ]
    
    for cfg in configs:
        print(f"\n{cfg['name']}:")
        print("-" * 40)
        
        engine = PhasedTradingEngine(cfg['config'])
        
        # Get schedule for entry or exit
        if 'entry' in cfg['name'].lower():
            schedule = engine.get_position_schedule(
                signal_idx=100,
                signal_type='entry',
                position_size=1000
            )
        else:
            schedule = engine.get_position_schedule(
                signal_idx=100,
                signal_type='exit',
                position_size=1000
            )
        
        print(schedule.to_string())
        print(f"\nTotal position: {schedule['size'].sum():.2f}")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    # Run the integration test
    test_phased_vs_normal()
    
    # Demonstrate phased schedules
    demonstrate_phased_schedule()
    
    print("\n✅ All tests completed successfully!")