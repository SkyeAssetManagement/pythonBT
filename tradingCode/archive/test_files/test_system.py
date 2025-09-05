"""
Test script for the modular trading system.
This demonstrates how to use the new strategy-based architecture.
"""

import numpy as np
from strategies.simpleSMA import SimpleSMAStrategy, SimpleSMAParameterSweep


def test_simple_sma():
    """Test the SimpleSMA strategy with synthetic data."""
    print("Testing SimpleSMA Strategy...")
    
    # Generate test data
    np.random.seed(42)
    n_bars = 1000
    returns = np.random.normal(0.0001, 0.01, n_bars)
    close = 100 * np.exp(np.cumsum(returns))
    
    data = {
        'datetime': np.arange(n_bars) * 300,
        'open': close * (1 + np.random.uniform(-0.001, 0.001, n_bars)),
        'high': close * (1 + np.abs(np.random.normal(0, 0.002, n_bars))),
        'low': close * (1 - np.abs(np.random.normal(0, 0.002, n_bars))),
        'close': close,
        'volume': np.random.uniform(1000, 10000, n_bars)
    }
    
    # Test strategy
    strategy = SimpleSMAStrategy()
    print(f"Strategy name: {strategy.name}")
    
    # Generate signals
    entries, exits = strategy.generate_signals(data)
    print(f"Entry signals: {np.sum(entries)}")
    print(f"Exit signals: {np.sum(exits)}")
    
    # Test parameter combinations
    param_combinations = strategy.get_parameter_combinations()
    print(f"Parameter combinations: {len(param_combinations)}")
    print(f"Parameters: {param_combinations}")
    
    print("SUCCESS: SimpleSMA test passed!\n")


def test_parameter_sweep():
    """Test the parameter sweep strategy (simplified for now)."""
    print("Testing SimpleSMA Parameter Sweep...")
    
    # Generate test data
    np.random.seed(42)
    n_bars = 1000
    returns = np.random.normal(0.0001, 0.01, n_bars)
    close = 100 * np.exp(np.cumsum(returns))
    
    data = {
        'datetime': np.arange(n_bars) * 300,
        'open': close * (1 + np.random.uniform(-0.001, 0.001, n_bars)),
        'high': close * (1 + np.abs(np.random.normal(0, 0.002, n_bars))),
        'low': close * (1 - np.abs(np.random.normal(0, 0.002, n_bars))),
        'close': close,
        'volume': np.random.uniform(1000, 10000, n_bars)
    }
    
    # Test parameter sweep strategy
    strategy = SimpleSMAParameterSweep()
    print(f"Strategy name: {strategy.name}")
    
    # Test parameter combinations
    param_combinations = strategy.get_parameter_combinations()
    print(f"Parameter combinations: {len(param_combinations)}")
    print(f"First 5 combinations: {param_combinations[:5]}")
    
    # For now, just test individual parameter combinations
    print("Testing individual parameter combinations...")
    
    config = {
        'initial_cash': 100000,
        'commission': 0.001,
        'slippage': 0.0005
    }
    
    # Test first 3 combinations individually
    test_combinations = param_combinations[:3]
    results = []
    
    for i, params in enumerate(test_combinations):
        print(f"Testing combination {i+1}: {params}")
        
        # Create single-parameter strategy
        class SingleParamStrategy(SimpleSMAStrategy):
            def get_parameter_combinations(self):
                return [params]
            
            def _generate_signals_for_params(self, data, params):
                # Use parent class method
                return super()._generate_signals_for_params(data, params)
        
        single_strategy = SingleParamStrategy()
        pf = single_strategy.run_vectorized_backtest(data, config)
        
        total_return = pf.total_return()
        if hasattr(total_return, 'values'):
            total_return = total_return.values[0] if len(total_return.values) > 0 else total_return
        
        results.append(total_return)
        print(f"  Result: {total_return*100:.2f}%")
    
    print(f"\nTested {len(results)} combinations successfully!")
    print(f"Best return: {max(results)*100:.2f}%")
    print(f"Worst return: {min(results)*100:.2f}%")
    print(f"Average return: {np.mean(results)*100:.2f}%")
    
    print("SUCCESS: Parameter sweep test passed!\n")


if __name__ == "__main__":
    print("Modular Trading System Tests")
    print("=" * 40)
    
    test_simple_sma()
    test_parameter_sweep()
    
    print("All tests passed! SUCCESS")
    print("\nTo run with real data, use:")
    print("python main.py SYMBOL_NAME simpleSMA")
    print("python main.py SYMBOL_NAME simpleSMA --config config.yaml")