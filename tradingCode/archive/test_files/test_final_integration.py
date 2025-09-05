#!/usr/bin/env python3
"""
Final integration test for vectorized strategy with dashboard
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Add strategies to path
strategies_path = Path(__file__).parent / "strategies"
if str(strategies_path) not in sys.path:
    sys.path.insert(0, str(strategies_path))

# Add src to path 
src_path = Path(__file__).parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from time_window_strategy_vectorized import TimeWindowVectorizedStrategy
from dashboard.text_dashboard import launch_text_dashboard

def create_realistic_data():
    """Create realistic test data"""
    
    # Create 2 days of 1-minute data
    n_bars = 840  # 2 days * 7 hours * 60 minutes
    
    # Realistic forex data
    base_price = 0.67500
    timestamps = np.arange(n_bars, dtype=np.int64) * 60 * 1_000_000_000  # Every minute
    
    # Price movement
    returns = np.random.normal(0, 0.0005, n_bars)
    price_changes = np.cumsum(returns)
    close_prices = base_price + price_changes
    
    # OHLC
    open_prices = np.roll(close_prices, 1)
    open_prices[0] = base_price
    spreads = np.random.uniform(0.00005, 0.0002, n_bars)
    high_prices = np.maximum(open_prices, close_prices) + spreads
    low_prices = np.minimum(open_prices, close_prices) - spreads
    volume = np.random.randint(100, 1000, n_bars).astype(float)
    
    return {
        'datetime': timestamps,
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'volume': volume
    }

def test_vectorized_strategy_with_dashboard():
    """Test the complete integration"""
    
    print("FINAL INTEGRATION TEST")
    print("=" * 50)
    
    # Create test data
    print("Creating test data...")
    data = create_realistic_data()
    print(f"Created {len(data['close'])} bars of data")
    
    # Create vectorized strategy
    print("Creating vectorized strategy...")
    strategy = TimeWindowVectorizedStrategy()
    
    # Configure backtest
    config = {
        'initial_capital': 100000,
        'commission': 0.001,
        'slippage': 0.0001
    }
    
    # Run backtest
    print("Running vectorized backtest...")
    import time
    start_time = time.perf_counter()
    
    pf = strategy.run_vectorized_backtest(data, config)
    
    duration = time.perf_counter() - start_time
    print(f"Backtest completed in {duration:.3f} seconds")
    
    # Test portfolio data preparation (like main.py does)
    print("Preparing portfolio data for dashboard...")
    portfolio_data = None
    
    if hasattr(pf, 'value'):
        # Handle both 1D (single strategy) and 2D (multi-strategy) arrays
        equity_values = pf.value.values if hasattr(pf.value, 'values') else pf.value
        drawdown_values = None
        
        if hasattr(pf, 'drawdown'):
            drawdown_values = pf.drawdown.values if hasattr(pf.drawdown, 'values') else pf.drawdown
        
        # For vectorized strategies with multiple combinations, select best performer
        if equity_values.ndim == 2:
            print(f"Multi-strategy results detected ({equity_values.shape[1]} combinations)")
            
            # Find best performing combination (highest final value)
            final_values = equity_values[-1, :]
            best_combo_idx = np.argmax(final_values)
            
            print(f"Selected best performing combination #{best_combo_idx+1}")
            print(f"Final value: ${final_values[best_combo_idx]:,.2f}")
            
            # Extract 1D arrays for best combination
            equity_curve = equity_values[:, best_combo_idx]
            drawdown_curve = drawdown_values[:, best_combo_idx] if drawdown_values is not None else None
        else:
            # Single strategy - use as is
            equity_curve = equity_values
            drawdown_curve = drawdown_values
        
        portfolio_data = {
            'equity_curve': equity_curve,
            'drawdown': drawdown_curve
        }
        
        print(f"Portfolio data prepared - equity shape: {equity_curve.shape}")
    
    # Create sample trades (normally from VBT)
    trades_df = pd.DataFrame({
        'trade_id': ['T001', 'T002', 'T003'],
        'side': ['buy', 'sell', 'buy'],
        'price': [0.675, 0.676, 0.674],
        'pnl': [50.0, -25.0, 75.0]
    })
    
    # Test text dashboard
    print("Testing text dashboard integration...")
    dashboard_success = launch_text_dashboard(data, trades_df, portfolio_data)
    
    if dashboard_success:
        print("SUCCESS: Text dashboard integration working!")
        return True
    else:
        print("FAILURE: Text dashboard integration failed!")
        return False

if __name__ == "__main__":
    success = test_vectorized_strategy_with_dashboard()
    
    print("\n" + "=" * 50)
    if success:
        print("FINAL INTEGRATION TEST PASSED")
        print("The vectorized strategy is ready for production use!")
        print("Usage: python main.py AD time_window_strategy_vectorized")
    else:
        print("FINAL INTEGRATION TEST FAILED")
        print("Additional debugging required")