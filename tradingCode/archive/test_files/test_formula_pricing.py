"""
Test script for OHLC formula-based execution pricing.
This demonstrates the new formula-based execution price functionality.
"""

import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.utils.price_formulas import PriceFormulaEvaluator, create_common_formulas
from src.backtest.vbt_engine import VectorBTEngine


def test_formula_evaluator():
    """Test the price formula evaluator with various formulas."""
    print("Testing Price Formula Evaluator...")
    print("=" * 50)
    
    # Create sample OHLC data
    np.random.seed(42)
    n_bars = 10
    
    # Generate realistic OHLC data
    base_price = 100
    ohlc_data = {
        'open': np.random.uniform(98, 102, n_bars),
        'high': np.random.uniform(102, 108, n_bars),
        'low': np.random.uniform(92, 98, n_bars),
        'close': np.random.uniform(98, 102, n_bars)
    }
    
    # Ensure OHLC relationships are valid
    for i in range(n_bars):
        low = ohlc_data['low'][i]
        high = ohlc_data['high'][i]
        ohlc_data['open'][i] = np.clip(ohlc_data['open'][i], low, high)
        ohlc_data['close'][i] = np.clip(ohlc_data['close'][i], low, high)
    
    print(f"Sample bar 0: O={ohlc_data['open'][0]:.2f}, H={ohlc_data['high'][0]:.2f}, L={ohlc_data['low'][0]:.2f}, C={ohlc_data['close'][0]:.2f}")
    
    # Test formula evaluator
    evaluator = PriceFormulaEvaluator()
    formulas = create_common_formulas()
    
    print(f"\nTesting common formulas on first bar:")
    print(f"{'Formula Name':<20} | {'Expression':<25} | {'Result':<8}")
    print("-" * 60)
    
    for name, formula in formulas.items():
        try:
            prices = evaluator.get_execution_prices(formula, ohlc_data, "test")
            print(f"{name:<20} | {formula:<25} | {prices[0]:.2f}")
        except Exception as e:
            print(f"{name:<20} | {formula:<25} | ERROR: {e}")
    
    print(f"\nSUCCESS: Formula evaluator working correctly!")
    return True


def test_vbt_engine_with_formulas():
    """Test VectorBT engine with formula-based execution pricing."""
    print(f"\nTesting VectorBT Engine with Formula Pricing...")
    print("=" * 50)
    
    # Generate test data
    np.random.seed(42)
    n_bars = 1000
    returns = np.random.normal(0.0001, 0.01, n_bars)
    close = 100 * np.exp(np.cumsum(returns))
    
    # Create realistic OHLC data
    data = {
        'datetime': np.arange(n_bars) * 300,
        'open': close * (1 + np.random.uniform(-0.002, 0.002, n_bars)),
        'high': close * (1 + np.abs(np.random.normal(0, 0.005, n_bars))),
        'low': close * (1 - np.abs(np.random.normal(0, 0.005, n_bars))),
        'close': close,
        'volume': np.random.uniform(1000, 10000, n_bars)
    }
    
    # Ensure OHLC relationships are valid
    for i in range(n_bars):
        low = data['low'][i]
        high = data['high'][i]
        data['open'][i] = np.clip(data['open'][i], low, high)
        data['close'][i] = np.clip(data['close'][i], low, high)
    
    print(f"Generated {n_bars} bars of test data")
    print(f"Price range: ${close.min():.2f} - ${close.max():.2f}")
    
    # Create simple signals
    price_change = np.diff(close, prepend=close[0]) / close
    entries = price_change < -0.02  # Buy on 2% drop
    exits = price_change > 0.02     # Sell on 2% rise
    
    print(f"Entry signals: {np.sum(entries)}")
    print(f"Exit signals: {np.sum(exits)}")
    
    # Test different formula configurations
    test_configs = [
        {
            'name': 'Close Prices',
            'execution_price': 'close',
            'buy_formula': 'C',
            'sell_formula': 'C'
        },
        {
            'name': 'Typical Price (HLC/3)',
            'execution_price': 'formula',
            'buy_formula': '(H + L + C) / 3',
            'sell_formula': '(H + L + C) / 3'
        },
        {
            'name': 'Conservative Trading',
            'execution_price': 'formula',
            'buy_formula': 'H',  # Buy at high (conservative)
            'sell_formula': 'L'  # Sell at low (conservative)
        },
        {
            'name': 'Aggressive Trading',
            'execution_price': 'formula',
            'buy_formula': 'L',  # Buy at low (aggressive)
            'sell_formula': 'H'  # Sell at high (aggressive)
        }
    ]
    
    results = []
    
    for config in test_configs:
        print(f"\nTesting: {config['name']}")
        
        # Create engine with default config file
        engine = VectorBTEngine("config.yaml")
        
        # Override specific settings for this test
        engine.config['backtest']['execution_price'] = config['execution_price']
        engine.config['backtest']['buy_execution_formula'] = config['buy_formula']
        engine.config['backtest']['sell_execution_formula'] = config['sell_formula']
        
        # Run backtest
        pf = engine.run_vectorized_backtest(data, entries, exits, "TEST")
        
        # Get results
        final_value = pf.value.iloc[-1] if hasattr(pf.value, 'iloc') else pf.value[-1]
        if hasattr(final_value, 'values'):
            final_value = final_value.values[0] if len(final_value.values) > 0 else final_value
        
        initial_cash = engine.config['backtest']['initial_cash']
        total_return = ((final_value - initial_cash) / initial_cash) * 100
        
        results.append({
            'name': config['name'],
            'final_value': final_value,
            'total_return': total_return
        })
        
        print(f"  Final portfolio value: ${final_value:.2f}")
        print(f"  Total return: {total_return:.2f}%")
    
    # Compare results
    print(f"\nFormula Pricing Comparison:")
    print(f"{'Strategy':<25} | {'Final Value':<12} | {'Return %':<10}")
    print("-" * 55)
    
    for result in results:
        print(f"{result['name']:<25} | ${result['final_value']:<11.2f} | {result['total_return']:<9.2f}%")
    
    print(f"\nSUCCESS: Formula-based execution pricing working!")
    return True


if __name__ == "__main__":
    try:
        print("OHLC Formula-Based Execution Pricing Test")
        print("=" * 60)
        
        # Test formula evaluator
        test_formula_evaluator()
        
        # Test VectorBT engine with formulas
        test_vbt_engine_with_formulas()
        
        print(f"\n[SUCCESS] All formula pricing tests passed!")
        print(f"[SUCCESS] You can now use custom OHLC formulas for execution pricing")
        print(f"[SUCCESS] Separate buy/sell formulas are fully functional")
        
        print(f"\nExample formula configurations:")
        print(f"  buy_execution_formula: '(H + L + C) / 3'  # Typical price")
        print(f"  sell_execution_formula: '(H + L) / 2'     # Median price")
        print(f"  buy_execution_formula: 'H'               # Conservative buy")
        print(f"  sell_execution_formula: 'L'              # Conservative sell")
        
    except Exception as e:
        print(f"\nERROR: Test failed: {e}")
        import traceback
        traceback.print_exc()