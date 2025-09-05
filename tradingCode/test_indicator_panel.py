#!/usr/bin/env python3
"""
Test script for the new indicator panel in Enhanced Plotly Dashboard V2
"""

import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

def test_indicator_panel():
    """Test the new indicator panel functionality."""
    
    print("=" * 60)
    print("TESTING ENHANCED PLOTLY DASHBOARD V2")
    print("New Features:")
    print("- NO automatic indicators on load")
    print("- NO jump-to-trade input box")
    print("- Full VectorBT Pro indicator dropdown")
    print("- Dynamic parameter inputs")
    print("- Add/Clear Last/Clear All buttons")
    print("=" * 60)
    
    # Generate sample OHLCV data
    n_bars = 2000
    dates = pd.date_range(start='2024-01-01', periods=n_bars, freq='5min')
    
    # Generate realistic price data
    np.random.seed(42)
    returns = np.random.randn(n_bars) * 0.002  # 0.2% volatility
    close = 4000 * np.exp(np.cumsum(returns))
    
    # Create OHLCV data
    ohlcv_data = {
        'datetime': dates,
        'open': close * (1 + np.random.randn(n_bars) * 0.001),
        'high': close * (1 + np.abs(np.random.randn(n_bars) * 0.003)),
        'low': close * (1 - np.abs(np.random.randn(n_bars) * 0.003)),
        'close': close,
        'volume': np.random.randint(100, 1000, n_bars) * 1000
    }
    
    # Create sample trades with enhanced data
    n_trades = 20
    trade_indices = sorted(np.random.choice(range(100, n_bars-100), n_trades, replace=False))
    
    trades_data = []
    for i in range(0, len(trade_indices), 2):
        if i+1 < len(trade_indices):
            entry_idx = trade_indices[i]
            exit_idx = trade_indices[i+1]
            entry_price = close[entry_idx]
            exit_price = close[exit_idx]
            
            # Determine direction
            direction = 'Long' if np.random.random() > 0.3 else 'Short'
            
            # Calculate PnL based on direction
            if direction == 'Long':
                pnl = (exit_price - entry_price) * 100  # Position size of 100
            else:
                pnl = (entry_price - exit_price) * 100
            
            # Position size
            pos_size = np.random.choice([50, 100, 150, 200])
            
            trades_data.append({
                'Exit Trade Id': f'T{i//2 + 1}',
                'Direction': direction,
                'Avg Entry Price': entry_price,
                'Avg Exit Price': exit_price,
                'PnL': pnl,
                'Entry Index': entry_idx,
                'Exit Index': exit_idx,
                'Position Size': pos_size,
                'Entry Time': dates[entry_idx],
                'Exit Time': dates[exit_idx]
            })
    
    trades_df = pd.DataFrame(trades_data)
    
    # Save trades to CSV
    trades_csv_path = 'test_trades_v2.csv'
    trades_df.to_csv(trades_csv_path, index=False)
    print(f"\nCreated test data:")
    print(f"- {n_bars} bars of OHLCV data")
    print(f"- {len(trades_df)} test trades")
    
    # Import and launch the V2 dashboard
    try:
        import plotly_dashboard_enhanced_v2 as dashboard_v2
        
        print("\n" + "=" * 60)
        print("LAUNCHING DASHBOARD V2")
        print("=" * 60)
        print("\nFeatures to test:")
        print("\nINDICATOR PANEL:")
        print("1. Select category from dropdown (Price, Momentum, Volatility, Volume, Trend)")
        print("2. Select indicator from dropdown")
        print("3. Adjust parameters that appear")
        print("4. Click 'Add Indicator' to add to chart")
        print("5. Try 'Clear Last' to remove most recent")
        print("6. Try 'Clear All' to remove all indicators")
        print("7. Add multiple indicators and see them stack")
        
        print("\nENHANCED TRADE LIST:")
        print("1. Entry and Exit are now SEPARATE rows")
        print("2. New columns: Type, Direction, DateTime, Price, Pos Size, Entry Trade, PnL, Time in Trade")
        print("3. Horizontal scrollbar to see all columns")
        print("4. Click on any trade row to navigate to it on the chart")
        print("5. Entry rows are highlighted in green, Exit rows in red")
        print("6. Sortable and filterable columns")
        print("7. Trade markers appear on chart (triangles up/down)")
        
        print("\nNOTE: Chart starts with NO indicators (user controlled)")
        print("\nOpening dashboard in browser...")
        
        dashboard = dashboard_v2.launch_enhanced_dashboard_v2(
            ohlcv_data=ohlcv_data,
            trades_csv_path=trades_csv_path,
            symbol="TEST",
            strategy_name="Indicator Panel Test"
        )
        
    except ImportError as e:
        print(f"\nERROR: Could not import dashboard V2: {e}")
        print("Make sure plotly_dashboard_enhanced_v2.py is in the current directory")
        return False
    except Exception as e:
        print(f"\nERROR: Failed to launch dashboard: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    print("\nTEST: Enhanced Plotly Dashboard V2 - Indicator Panel")
    print("This will launch the dashboard in your browser")
    print("Press Ctrl+C to stop the server when done testing\n")
    
    success = test_indicator_panel()
    
    if success:
        print("\n[OK] Dashboard launched successfully!")
    else:
        print("\n[FAILED] Dashboard test failed")