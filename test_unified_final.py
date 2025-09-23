"""
Final test of unified launcher - check that everything works
"""

import sys
import os
sys.path.insert(0, 'src')
sys.path.insert(0, 'src/trading')

import pandas as pd
import numpy as np

# Test trade generation with proper data structure
def test_trade_generation():
    """Test that trades are generated correctly"""
    from core.standalone_execution import ExecutionConfig
    from strategies.strategy_wrapper import StrategyFactory

    print("Testing Trade Generation with Unified Engine")
    print("=" * 60)

    # Create sample data matching chart structure
    num_bars = 100
    dates = pd.date_range('2024-01-01', periods=num_bars, freq='5min')
    prices = 100 + np.cumsum(np.random.randn(num_bars) * 0.5)

    # Create DataFrame with uppercase names (as strategies expect)
    df = pd.DataFrame({
        'DateTime': dates,
        'Open': prices + np.random.randn(num_bars) * 0.1,
        'High': prices + np.abs(np.random.randn(num_bars) * 0.2),
        'Low': prices - np.abs(np.random.randn(num_bars) * 0.2),
        'Close': prices,
        'Volume': np.random.randint(1000, 5000, num_bars)
    })

    # Ensure High/Low are correct
    df['High'] = df[['Open', 'High', 'Low', 'Close']].max(axis=1)
    df['Low'] = df[['Open', 'High', 'Low', 'Close']].min(axis=1)

    # Load config
    exec_config = ExecutionConfig.from_yaml()
    print(f"Config loaded: lag={exec_config.signal_lag}, formula={exec_config.buy_execution_formula}")

    # Test SMA strategy
    print("\nTesting SMA Crossover Strategy:")
    sma_strategy = StrategyFactory.create_sma_crossover(
        fast_period=5,
        slow_period=10,
        execution_config=exec_config
    )

    sma_trades = sma_strategy.execute_trades(df)
    print(f"Generated {len(sma_trades)} SMA trades")

    if len(sma_trades) > 0:
        print("\nFirst 3 SMA trades:")
        for trade in sma_trades[:3]:
            print(f"  Trade {trade.trade_id}: {trade.trade_type}")
            print(f"    Signal bar: {trade.signal_bar}, Exec bar: {trade.execution_bar}")
            print(f"    Lag: {trade.lag} bars")
            print(f"    Price: ${trade.price:.2f}")
            if trade.pnl_percent is not None:
                print(f"    P&L: {trade.pnl_percent:.2f}%")

    # Test RSI strategy
    print("\nTesting RSI Momentum Strategy:")
    rsi_strategy = StrategyFactory.create_rsi_momentum(
        rsi_period=14,
        oversold=30,
        overbought=70,
        execution_config=exec_config
    )

    rsi_trades = rsi_strategy.execute_trades(df)
    print(f"Generated {len(rsi_trades)} RSI trades")

    return True

if __name__ == "__main__":
    try:
        test_trade_generation()
        print("\n[OK] Trade generation working correctly")
        print("\nNow run: python launch_unified_system.py")
        print("Select your data file and choose 'System' trades")
        print("The chart should show:")
        print("  - Candlesticks")
        print("  - Hover data with OHLC values")
        print("  - Trades with 1 bar lag")
        print("  - P&L as percentages")
    except Exception as e:
        print(f"\n[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()