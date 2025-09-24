"""
Comprehensive test suite for the phased entry system
Tests core functionality, risk management, and performance
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from trading.core.phased_entry import PhasedEntryConfig, PhasedEntry, TradePhase, PhasedPositionTracker
from trading.core.phased_execution_engine import PhasedExecutionEngine, PhasedExecutionConfig
from trading.strategies.phased_strategy_base import PhasedTradingStrategy


class TestPhasedStrategy(PhasedTradingStrategy):
    """Simple test strategy for phased entry testing"""

    def __init__(self, name="TestPhasedStrategy"):
        super().__init__(name)

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """Generate simple moving average crossover signals for testing"""
        if len(df) < 50:
            return pd.Series(0, index=df.index)

        # Simple SMA crossover for testing
        sma_short = df['Close'].rolling(window=10).mean()
        sma_long = df['Close'].rolling(window=20).mean()

        signals = pd.Series(0, index=df.index)

        # Long when short MA > long MA
        signals[sma_short > sma_long] = 1
        # Short when short MA < long MA
        signals[sma_short < sma_long] = -1

        # Clean signals (only change when crossing)
        signals = signals.diff().fillna(0)
        signals = signals.cumsum().clip(-1, 1)

        return signals


def create_test_data(bars=1000, start_price=100.0, trend_strength=0.001):
    """Create synthetic price data for testing"""
    dates = pd.date_range(start='2023-01-01', periods=bars, freq='1H')

    # Generate trending price data with some volatility
    np.random.seed(42)  # For reproducible tests
    returns = np.random.normal(trend_strength, 0.02, bars)
    prices = [start_price]

    for i in range(1, bars):
        price = prices[-1] * (1 + returns[i])
        prices.append(max(price, 1.0))  # Prevent negative prices

    # Create OHLC data
    data = {
        'DateTime': dates,
        'Open': prices,
        'High': [p * (1 + abs(np.random.normal(0, 0.005))) for p in prices],
        'Low': [p * (1 - abs(np.random.normal(0, 0.005))) for p in prices],
        'Close': prices,
        'Volume': np.random.randint(1000, 10000, bars)
    }

    df = pd.DataFrame(data)
    df['High'] = np.maximum(df['High'], df[['Open', 'Close']].max(axis=1))
    df['Low'] = np.minimum(df['Low'], df[['Open', 'Close']].min(axis=1))

    return df


def test_phased_entry_config():
    """Test phased entry configuration loading"""
    print("=== Testing Phased Entry Configuration ===")

    # Test default config
    config = PhasedEntryConfig()
    assert not config.enabled, "Default config should have phased entries disabled"
    assert config.max_phases == 3, "Default max phases should be 3"
    assert config.initial_size_percent == 33.33, "Default initial size should be 33.33%"

    # Test custom config
    config = PhasedEntryConfig(
        enabled=True,
        max_phases=4,
        initial_size_percent=25.0,
        phase_trigger_value=2.5
    )
    assert config.enabled, "Custom config should have phased entries enabled"
    assert config.max_phases == 4, "Custom max phases should be 4"
    assert config.phase_trigger_value == 2.5, "Custom trigger value should be 2.5"

    print("✓ Configuration tests passed")


def test_phased_entry_creation():
    """Test phased entry object creation and phase setup"""
    print("=== Testing Phased Entry Creation ===")

    config = PhasedEntryConfig(
        enabled=True,
        max_phases=3,
        initial_size_percent=40.0,
        phase_trigger_value=2.0
    )

    # Test long entry
    entry = PhasedEntry(
        config=config,
        initial_signal_bar=10,
        initial_price=100.0,
        is_long=True,
        total_target_size=1000.0
    )

    assert len(entry.phases) == 3, "Should create 3 phases"
    assert entry.phases[0].size == 400.0, f"First phase size should be 400.0, got {entry.phases[0].size}"
    assert entry.phases[0].trigger_price == 100.0, "First phase trigger should be entry price"

    # Check phase triggers are increasing for long positions
    assert entry.phases[1].trigger_price > entry.phases[0].trigger_price, "Phase 2 trigger should be higher"
    assert entry.phases[2].trigger_price > entry.phases[1].trigger_price, "Phase 3 trigger should be highest"

    # Test short entry
    short_entry = PhasedEntry(
        config=config,
        initial_signal_bar=10,
        initial_price=100.0,
        is_long=False,
        total_target_size=1000.0
    )

    # Check phase triggers are decreasing for short positions
    assert short_entry.phases[1].trigger_price < short_entry.phases[0].trigger_price, "Short phase 2 trigger should be lower"

    print("✓ Phased entry creation tests passed")


def test_phase_triggers():
    """Test phase trigger logic"""
    print("=== Testing Phase Trigger Logic ===")

    config = PhasedEntryConfig(
        enabled=True,
        max_phases=3,
        initial_size_percent=40.0,
        phase_trigger_value=2.0,
        require_profit=True
    )

    # Create long entry
    entry = PhasedEntry(
        config=config,
        initial_signal_bar=10,
        initial_price=100.0,
        is_long=True,
        total_target_size=1000.0
    )

    # Execute first phase
    entry.execute_phase(entry.phases[0], 100.0, 10)

    # Test trigger with profitable move
    triggered = entry.check_phase_triggers(102.5, 15)  # 2.5% move up
    assert len(triggered) == 1, f"Should trigger 1 phase at 102.5, got {len(triggered)}"

    # Test adverse move limit
    entry_with_limit = PhasedEntry(
        config=PhasedEntryConfig(enabled=True, max_adverse_move=1.0),
        initial_signal_bar=10,
        initial_price=100.0,
        is_long=True,
        total_target_size=1000.0
    )
    entry_with_limit.execute_phase(entry_with_limit.phases[0], 100.0, 10)

    triggered = entry_with_limit.check_phase_triggers(98.5, 15)  # 1.5% adverse move
    assert len(triggered) == 0, "Should not trigger phases with large adverse move"
    assert entry_with_limit.stop_scaling, "Should stop scaling after adverse move limit"

    print("✓ Phase trigger tests passed")


def test_execution_engine():
    """Test the phased execution engine"""
    print("=== Testing Phased Execution Engine ===")

    # Create test data
    df = create_test_data(100, start_price=100.0)

    # Create simple signals (long at bar 10, exit at bar 20)
    signals = pd.Series(0, index=range(len(df)))
    signals.iloc[10] = 1  # Enter long
    signals.iloc[20] = 0  # Exit

    # Test with phased entries disabled
    config = PhasedExecutionConfig()
    config.phased_config.enabled = False
    engine = PhasedExecutionEngine(config)

    trades_single = engine.execute_signals_with_phases(signals, df, "TEST")
    print(f"Single entry trades: {len(trades_single)}")

    # Test with phased entries enabled
    config.phased_config.enabled = True
    config.phased_config.max_phases = 3
    config.phased_config.phase_trigger_value = 1.0  # 1% trigger
    engine_phased = PhasedExecutionEngine(config)

    trades_phased = engine_phased.execute_signals_with_phases(signals, df, "TEST")
    print(f"Phased entry trades: {len(trades_phased)}")

    # Should have more trades with phased entries (if price moves favorably)
    # At minimum should have entry and exit trades
    assert len(trades_phased) >= 2, "Should have at least entry and exit trades"

    # Check for phase information in trades
    phased_trades = [t for t in trades_phased if t.get('is_phased_entry') or t.get('is_phased_exit')]
    print(f"Trades with phase info: {len(phased_trades)}")

    print("✓ Execution engine tests passed")


def test_strategy_integration():
    """Test strategy integration with phased entries"""
    print("=== Testing Strategy Integration ===")

    # Create test strategy with phased entries enabled
    strategy = TestPhasedStrategy()
    strategy.phased_config.enabled = True
    strategy.phased_config.max_phases = 3
    strategy.phased_config.phase_trigger_value = 1.5

    # Create test data with clear trend
    df = create_test_data(200, start_price=100.0, trend_strength=0.005)

    # Run backtest
    trades, performance = strategy.run_backtest_with_phases(df, "TEST_SYMBOL")

    print(f"Total trades generated: {len(trades.trades)}")
    print(f"Phased entry statistics: {performance}")

    # Verify we got some trades
    assert len(trades.trades) > 0, "Should generate some trades"

    # Check for phased trades
    phased_trades = [t for t in trades.trades if hasattr(t, 'is_phased') and t.is_phased]
    print(f"Phased trades: {len(phased_trades)}")

    print("✓ Strategy integration tests passed")


def test_risk_management():
    """Test risk management features"""
    print("=== Testing Risk Management ===")

    config = PhasedEntryConfig(
        enabled=True,
        max_phases=5,
        max_adverse_move=2.0,
        require_profit=True,
        time_limit_bars=10
    )

    entry = PhasedEntry(
        config=config,
        initial_signal_bar=0,
        initial_price=100.0,
        is_long=True,
        total_target_size=1000.0
    )

    # Execute first phase
    entry.execute_phase(entry.phases[0], 100.0, 0)

    # Test time limit
    triggered = entry.check_phase_triggers(102.0, 15)  # Beyond time limit
    assert len(triggered) == 0, "Should not trigger phases beyond time limit"
    assert entry.stop_scaling, "Should stop scaling after time limit"

    # Test adverse move in new entry
    entry2 = PhasedEntry(config, 0, 100.0, True, 1000.0)
    entry2.execute_phase(entry2.phases[0], 100.0, 0)

    triggered = entry2.check_phase_triggers(97.0, 5)  # 3% adverse move
    assert len(triggered) == 0, "Should not trigger with large adverse move"

    print("✓ Risk management tests passed")


def test_pnl_calculations():
    """Test P&L calculations for phased entries"""
    print("=== Testing P&L Calculations ===")

    config = PhasedEntryConfig(enabled=True, max_phases=2, initial_size_percent=50.0)

    entry = PhasedEntry(
        config=config,
        initial_signal_bar=0,
        initial_price=100.0,
        is_long=True,
        total_target_size=1000.0
    )

    # Execute both phases at different prices
    entry.execute_phase(entry.phases[0], 100.0, 0)  # 500 shares at $100
    entry.execute_phase(entry.phases[1], 105.0, 5)  # 500 shares at $105

    # Test average entry price calculation
    avg_price = entry.get_average_entry_price()
    expected_avg = (500 * 100.0 + 500 * 105.0) / 1000.0  # $102.50
    assert abs(avg_price - expected_avg) < 0.01, f"Average price should be ${expected_avg}, got ${avg_price}"

    # Test unrealized P&L at $110
    unrealized_pnl = entry.calculate_unrealized_pnl(110.0)
    expected_pnl = (110.0 - 100.0) * 500 + (110.0 - 105.0) * 500  # $5000 + $2500 = $7500
    assert abs(unrealized_pnl - expected_pnl) < 0.01, f"Unrealized P&L should be ${expected_pnl}, got ${unrealized_pnl}"

    # Test phase-specific P&L
    phase1_pnl = entry.calculate_phase_pnl(entry.phases[0], 110.0)
    expected_phase1 = (110.0 - 100.0) * 500  # $5000
    assert abs(phase1_pnl - expected_phase1) < 0.01, f"Phase 1 P&L should be ${expected_phase1}, got ${phase1_pnl}"

    print("✓ P&L calculation tests passed")


def run_performance_comparison():
    """Compare performance between single and phased entries"""
    print("=== Performance Comparison ===")

    # Create trending test data
    df = create_test_data(500, start_price=100.0, trend_strength=0.003)

    # Test single entry strategy
    single_strategy = TestPhasedStrategy("SingleEntry")
    single_strategy.phased_config.enabled = False
    single_trades, single_perf = single_strategy.run_backtest_with_phases(df, "TEST")

    # Test phased entry strategy
    phased_strategy = TestPhasedStrategy("PhasedEntry")
    phased_strategy.phased_config.enabled = True
    phased_strategy.phased_config.max_phases = 3
    phased_strategy.phased_config.phase_trigger_value = 1.0
    phased_trades, phased_perf = phased_strategy.run_backtest_with_phases(df, "TEST")

    print(f"\nSingle Entry Results:")
    print(f"  Trades: {len(single_trades.trades)}")
    single_pnl = sum(t.pnl or 0 for t in single_trades.trades)
    print(f"  Total P&L: ${single_pnl:.2f}")

    print(f"\nPhased Entry Results:")
    print(f"  Trades: {len(phased_trades.trades)}")
    phased_pnl = sum(t.pnl or 0 for t in phased_trades.trades)
    print(f"  Total P&L: ${phased_pnl:.2f}")
    print(f"  Phased Statistics: {phased_perf}")

    # Show phase breakdown if available
    phased_trade_records = [t for t in phased_trades.trades if hasattr(t, 'is_phased') and t.is_phased]
    if phased_trade_records:
        print(f"  Phased trade records: {len(phased_trade_records)}")

    print("✓ Performance comparison completed")


def main():
    """Run all tests"""
    print("Starting Phased Entry System Tests\n")

    try:
        test_phased_entry_config()
        test_phased_entry_creation()
        test_phase_triggers()
        test_execution_engine()
        test_strategy_integration()
        test_risk_management()
        test_pnl_calculations()
        run_performance_comparison()

        print(f"\n{'='*50}")
        print("🎉 ALL TESTS PASSED!")
        print("Phased entry system is working correctly.")
        print(f"{'='*50}")

        return True

    except Exception as e:
        print(f"\n{'='*50}")
        print(f"❌ TEST FAILED: {e}")
        print(f"{'='*50}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)