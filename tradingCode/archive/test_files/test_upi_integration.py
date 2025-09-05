#!/usr/bin/env python3
"""
Test UPI integration with VBT engine
"""

import numpy as np
import pandas as pd
from src.backtest.vbt_engine import VectorBTEngine

def test_upi_integration():
    """Test UPI integration with mock portfolio data"""
    
    print("TESTING UPI INTEGRATION WITH VBT ENGINE")
    print("=" * 60)
    
    try:
        # Create a mock VBT engine
        engine = VectorBTEngine.__new__(VectorBTEngine)  # Skip __init__ for testing
        
        # Create mock portfolio object with necessary attributes
        class MockPortfolio:
            def __init__(self):
                self.total_return = 0.15  # 15% return
                self.annualized_return = 0.12  # 12% annualized
                self.sharpe_ratio = 1.5
                self.max_drawdown = -0.08  # 8% drawdown
                
                # Mock trades
                self.trades = MockTrades()
        
        class MockTrades:
            def __init__(self):
                self.records = [1, 2, 3]  # Mock 3 trades
                self.win_rate = 0.6
                self.profit_factor = 1.8
                
                # Mock readable records
                self.records_readable = pd.DataFrame({
                    'Entry Index': [10, 25, 40],
                    'Exit Index': [20, 35, 50], 
                    'Direction': ['Long', 'Long', 'Short']
                })
        
        # Create mock equity curve (100 data points)
        n_points = 100
        base_equity = 100000
        daily_returns = np.random.normal(0.001, 0.015, n_points)  # 0.1% daily with 1.5% vol
        daily_returns[20:30] = np.random.normal(-0.002, 0.01, 10)  # Drawdown period
        
        cumulative_returns = np.cumprod(1 + daily_returns)
        equity_curve = base_equity * cumulative_returns
        
        # Create timestamps (daily for 100 days)
        start_timestamp = 1640995200000000000  # Jan 1, 2022
        timestamps = np.array([start_timestamp + i * 24 * 60 * 60 * 1_000_000_000 
                              for i in range(n_points)])
        
        print(f"Mock data created:") 
        print(f"  - Equity points: {n_points}")
        print(f"  - Start equity: ${equity_curve[0]:,.2f}")
        print(f"  - End equity: ${equity_curve[-1]:,.2f}")
        print(f"  - Total return: {(equity_curve[-1]/equity_curve[0] - 1)*100:.2f}%")
        print()
        
        # Test the calculate_performance_metrics function
        print("Testing calculate_performance_metrics with UPI...")
        mock_pf = MockPortfolio()
        
        metrics = engine.calculate_performance_metrics(mock_pf, equity_curve, timestamps)
        
        print("Calculated metrics:")
        for key, value in sorted(metrics.items()):
            if isinstance(value, (int, float)) and not np.isnan(value):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")
        
        # Check if UPI metrics are present
        upi_metrics = ['UPI_30', 'UPI_50', 'UPI_30_max', 'UPI_50_max', 
                      'UPI_30_adj', 'UPI_50_adj', 'UPI_30_adj_max', 'UPI_50_adj_max']
        
        print("\nUPI Metrics Status:")
        for metric in upi_metrics:
            if metric in metrics:
                value = metrics[metric]
                if not np.isnan(value):
                    print(f"  [OK] {metric}: {value:.4f}")
                else:
                    print(f"  [WARNING] {metric}: NaN")
            else:
                print(f"  [X] {metric}: Missing")
        
        # Test success criteria
        has_upi_30 = 'UPI_30' in metrics and not np.isnan(metrics['UPI_30'])
        has_upi_50 = 'UPI_50' in metrics and not np.isnan(metrics['UPI_50'])
        has_upi_max = 'UPI_30_max' in metrics and not np.isnan(metrics['UPI_30_max'])
        
        return has_upi_30 and has_upi_50 and has_upi_max
        
    except Exception as e:
        print(f"Error during integration test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_upi_integration()
    
    print("\n" + "=" * 60)
    if success:
        print("SUCCESS: UPI Integration Working!")
        print()
        print("[OK] UPI calculator properly integrated with VBT engine")
        print("[OK] UPI_30 and UPI_50 metrics calculated") 
        print("[OK] UPI_max values calculated")
        print("[OK] Ready for performance_summary.csv export")
        print()
        print("Expected columns in performance_summary.csv:")
        print("- UPI_30, UPI_50 (final values)")
        print("- UPI_30_max, UPI_50_max (maximum values over time)")
        print("- UPI_30_adj, UPI_50_adj (adjusted values)")
        print("- UPI_30_adj_max, UPI_50_adj_max (maximum adjusted values)")
    else:
        print("FAILURE: UPI Integration has issues")
        print("Check the errors above and fix integration problems")