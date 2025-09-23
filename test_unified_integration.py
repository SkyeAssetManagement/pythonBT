"""
Quick test to verify unified engine integration with actual lag
"""

import sys
import os
sys.path.insert(0, 'src')
sys.path.insert(0, 'src/trading')
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
import yaml

from core.standalone_execution import ExecutionConfig
from strategies.strategy_wrapper import StrategyFactory

def test_unified_with_config():
    """Test that unified engine uses config.yaml settings"""

    # Load config to verify settings
    config_path = "C:\\code\\PythonBT\\tradingCode\\config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    print("=" * 80)
    print("UNIFIED ENGINE INTEGRATION TEST")
    print("=" * 80)
    print(f"\nConfig settings:")
    print(f"  use_unified_engine: {config.get('use_unified_engine')}")
    print(f"  signal_lag: {config['backtest']['signal_lag']}")
    print(f"  buy_execution_formula: {config['backtest']['buy_execution_formula']}")
    print(f"  sell_execution_formula: {config['backtest']['sell_execution_formula']}")

    # Create sample data
    dates = pd.date_range('2024-01-01', periods=100, freq='5min')
    prices = 100 + np.cumsum(np.random.randn(100) * 0.5)

    df = pd.DataFrame({
        'DateTime': dates,
        'Open': prices + np.random.randn(100) * 0.1,
        'High': prices + np.abs(np.random.randn(100) * 0.3),
        'Low': prices - np.abs(np.random.randn(100) * 0.3),
        'Close': prices,
        'Volume': np.random.randint(1000, 10000, 100)
    })

    # Ensure High is highest and Low is lowest
    df['High'] = df[['Open', 'High', 'Low', 'Close']].max(axis=1)
    df['Low'] = df[['Open', 'High', 'Low', 'Close']].min(axis=1)

    # Load execution config from yaml
    exec_config = ExecutionConfig.from_yaml(config_path)

    # Create strategy with unified execution
    strategy = StrategyFactory.create_sma_crossover(
        fast_period=5,
        slow_period=10,
        execution_config=exec_config
    )

    print(f"\nStrategy created with:")
    print(f"  Name: {strategy.metadata.name}")
    print(f"  Execution lag: {exec_config.signal_lag} bars")

    # Execute trades
    trades = strategy.execute_trades(df)

    print(f"\nGenerated {len(trades)} trades")

    # Show execution details for first 5 trades
    if len(trades) > 0:
        print("\nDetailed trade execution (showing lag):")
        print("-" * 60)
        for trade in trades[:5]:
            print(f"Trade {trade.trade_id}: {trade.trade_type}")
            print(f"  Signal generated at bar: {trade.signal_bar}")
            print(f"  Trade executed at bar: {trade.execution_bar}")
            print(f"  LAG: {trade.lag} bars")
            print(f"  Execution price: ${trade.price:.2f}")

            # Show the formula calculation
            bar_data = df.iloc[trade.execution_bar]
            H, L, C = bar_data['High'], bar_data['Low'], bar_data['Close']
            formula_price = (H + L + C) / 3
            print(f"  Formula (H+L+C)/3 = ({H:.2f}+{L:.2f}+{C:.2f})/3 = ${formula_price:.2f}")

            if trade.pnl_percent is not None:
                print(f"  P&L: {trade.pnl_percent:.2f}%")
            print()

    # Verify lag is working
    if len(trades) > 0:
        lag_values = [t.lag for t in trades if t.lag is not None]
        if lag_values:
            avg_lag = sum(lag_values) / len(lag_values)
            print(f"Average execution lag: {avg_lag:.1f} bars")
            if avg_lag == exec_config.signal_lag:
                print("[OK] Lag matches config setting!")
            else:
                print(f"[ERROR] Lag mismatch! Expected {exec_config.signal_lag}, got {avg_lag}")

if __name__ == "__main__":
    test_unified_with_config()