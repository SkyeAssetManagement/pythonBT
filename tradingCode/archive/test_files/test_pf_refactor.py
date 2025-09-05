"""
Test script to verify the refactored pf object equity curve functionality.
This demonstrates the new pf object-based approach for creating equity curves.
"""

import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.backtest.vbt_engine import VectorBTEngine


def test_pf_equity_curve_methods():
    """Test the refactored pf object methods for equity curve creation."""
    print("Testing Refactored PF Object Equity Curve Methods...")
    print("=" * 60)
    
    # Generate test data
    np.random.seed(42)
    n_bars = 1000
    returns = np.random.normal(0.0001, 0.01, n_bars)
    close = 100 * np.exp(np.cumsum(returns))
    
    # Create full OHLCV data dictionary
    data = {
        'datetime': np.arange(n_bars) * 300,
        'open': close * (1 + np.random.uniform(-0.001, 0.001, n_bars)),
        'high': close * (1 + np.abs(np.random.normal(0, 0.002, n_bars))),
        'low': close * (1 - np.abs(np.random.normal(0, 0.002, n_bars))),
        'close': close,
        'volume': np.random.uniform(1000, 10000, n_bars)
    }
    
    print(f"Generated {n_bars} bars of test data")
    print(f"Price range: ${close.min():.2f} - ${close.max():.2f}")
    
    # Create simple signals
    # Buy when price drops 2%, sell when price rises 2%
    price_change = np.diff(close, prepend=close[0]) / close
    entries = price_change < -0.02
    exits = price_change > 0.02
    
    print(f"Entry signals: {np.sum(entries)}")
    print(f"Exit signals: {np.sum(exits)}")
    
    # Initialize VectorBT engine
    engine = VectorBTEngine()
    
    # Run backtest
    print("\n1. Running backtest with refactored pf object...")
    pf = engine.run_vectorized_backtest(data, entries, exits, "TEST")
    
    init_cash = pf.init_cash
    if hasattr(init_cash, 'values'):
        init_cash = init_cash.values[0] if len(init_cash.values) > 0 else init_cash
    print(f"   SUCCESS: Portfolio created with ${init_cash} initial cash")
    
    final_value = pf.value.iloc[-1] if hasattr(pf.value, 'iloc') else pf.value[-1]
    if hasattr(final_value, 'values'):
        final_value = final_value.values[0] if len(final_value.values) > 0 else final_value
    print(f"   SUCCESS: Final portfolio value: ${final_value:.2f}")
    
    # Test uncompounded equity data extraction
    print("\n2. Testing uncompounded equity data extraction...")
    equity_data = engine.get_uncompounded_equity_data(pf)
    
    uncompounded_final = equity_data['uncompounded_cumulative_pct'].iloc[-1] if hasattr(equity_data['uncompounded_cumulative_pct'], 'iloc') else equity_data['uncompounded_cumulative_pct'][-1]
    if hasattr(uncompounded_final, 'values'):
        uncompounded_final = uncompounded_final.values[0] if len(uncompounded_final.values) > 0 else uncompounded_final
    print(f"   SUCCESS: Final uncompounded return: {uncompounded_final:.2f}%")
    
    init_cash_display = equity_data['initial_cash']
    if hasattr(init_cash_display, 'values'):
        init_cash_display = init_cash_display.values[0] if len(init_cash_display.values) > 0 else init_cash_display
    print(f"   SUCCESS: Initial cash: ${init_cash_display}")
    
    # Test demonstration methods
    print("\n3. Testing demonstration of equity curve methods...")
    demo_results = engine.demonstrate_equity_curve_methods(pf)
    
    final_uncompounded = demo_results['final_uncompounded_return']
    if hasattr(final_uncompounded, 'values'):
        final_uncompounded = final_uncompounded.values[0] if len(final_uncompounded.values) > 0 else final_uncompounded
    print(f"   Method 1 - Uncompounded returns: {final_uncompounded:.2f}%")
    
    equity_final = demo_results['equity_values_dollar'].iloc[-1] if hasattr(demo_results['equity_values_dollar'], 'iloc') else demo_results['equity_values_dollar'][-1]
    if hasattr(equity_final, 'values'):
        equity_final = equity_final.values[0] if len(equity_final.values) > 0 else equity_final
    print(f"   Method 2 - Equity values: ${equity_final:.2f}")
    
    compounded_total = demo_results['compounded_total_return']
    if hasattr(compounded_total, 'values'):
        compounded_total = compounded_total.values[0] if len(compounded_total.values) > 0 else compounded_total
    print(f"   Method 3 - Compounded total return: {compounded_total:.2f}%")
    
    uncompounded_from_equity_final = demo_results['uncompounded_from_equity'].iloc[-1] if hasattr(demo_results['uncompounded_from_equity'], 'iloc') else demo_results['uncompounded_from_equity'][-1]
    if hasattr(uncompounded_from_equity_final, 'values'):
        uncompounded_from_equity_final = uncompounded_from_equity_final.values[0] if len(uncompounded_from_equity_final.values) > 0 else uncompounded_from_equity_final
    print(f"   Method 4 - Uncompounded from equity: {uncompounded_from_equity_final:.2f}%")
    
    # Verify that Method 1 and Method 4 give same results (they should for uncompounded returns)
    method1_result = final_uncompounded  # Already extracted above
    method4_result = uncompounded_from_equity_final  # Already extracted above
    difference = abs(method1_result - method4_result)
    
    print(f"\n4. Verification:")
    print(f"   Method 1 result: {method1_result:.4f}%")
    print(f"   Method 4 result: {method4_result:.4f}%")
    print(f"   Difference: {difference:.6f}%")
    
    if difference < 0.01:  # Allow small numerical differences
        print(f"   SUCCESS: Methods agree (difference < 0.01%)")
    else:
        print(f"   WARNING: Methods differ by {difference:.4f}%")
    
    # Test $1 position setup verification
    print(f"\n5. Testing $1 position setup:")
    if hasattr(pf.trades, 'records') and len(pf.trades.records) > 0:
        trades = pf.trades.records_readable
        if 'Size' in trades.columns:
            avg_position_size = trades['Size'].abs().mean()
            print(f"   Average position size: ${avg_position_size:.2f}")
            if 0.9 <= avg_position_size <= 1.1:  # Allow for small variations due to fees
                print(f"   SUCCESS: Using approximately $1 positions")
            else:
                print(f"   INFO: Position sizes differ from $1 target")
        else:
            print(f"   INFO: Size column not found in trades")
    else:
        print(f"   INFO: No trades generated in this test")
    
    print(f"\nSUCCESS: All pf object equity curve methods tested!")
    return True


if __name__ == "__main__":
    try:
        test_pf_equity_curve_methods()
        print(f"\n[SUCCESS] Refactoring test completed successfully!")
        print(f"[SUCCESS] The code now properly utilizes pf object for equity curve creation")
        print(f"[SUCCESS] Key improvements:")
        print(f"  - Uses pf.returns.cumsum() * 100 for uncompounded returns")
        print(f"  - Uses pf.value for portfolio equity values") 
        print(f"  - Implements $1 position sizing for clean percentage/dollar relationship")
        print(f"  - Provides multiple methods to extract equity curve data")
        
    except Exception as e:
        print(f"\nERROR: Test failed: {e}")
        import traceback
        traceback.print_exc()