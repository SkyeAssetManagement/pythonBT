#!/usr/bin/env python
"""
Test script for performance statistics calculations
Tests the improved performance_stats module with sample data
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.performance_stats import calculate_performance_stats, format_stats_for_display

def create_test_data_simple():
    """Create simple test data with known expected results"""
    print("\n=== TEST 1: Simple Test Data ===")
    print("Creating test data with 10 trades, each with 1% profit")
    
    # Create 10 trades with 1% profit each
    dates = pd.date_range(start='2023-01-01', periods=10, freq='D')
    data = {
        'date': dates,
        'prediction': [1] * 10,  # All trades taken
        'target_value': [0.01] * 10,  # 1% profit each
        'actual_profitable': [1] * 10  # All profitable
    }
    
    df = pd.DataFrame(data)
    
    # Calculate stats
    stats = calculate_performance_stats(df, model_type='longonly')
    
    # Verify calculations
    print("\nKey Results:")
    print(f"Total trades: {stats.get('total_trades', 'N/A')}")
    print(f"Total return: {stats.get('total_return', 'N/A'):.4f} (Expected: ~0.10)")
    print(f"Hit rate: {stats.get('hit_rate', 'N/A'):.1f}% (Expected: 100%)")
    
    # Check portfolio value calculation
    # Starting with 1, after 10 trades of 1% each: 1.01^10 = 1.1046
    expected_final_value = 1.01 ** 10
    print(f"\nCompound return check:")
    print(f"Expected final portfolio value: {expected_final_value:.4f}")
    print(f"Expected total return: {(expected_final_value - 1):.4f}")
    
    return stats

def create_test_data_mixed():
    """Create mixed test data with wins and losses"""
    print("\n=== TEST 2: Mixed Win/Loss Data ===")
    print("Creating test data with alternating 2% wins and -1% losses")
    
    # Create 20 trades: alternating wins and losses
    dates = pd.date_range(start='2023-01-01', periods=20, freq='D')
    returns = []
    profitable = []
    for i in range(20):
        if i % 2 == 0:
            returns.append(0.02)  # 2% win
            profitable.append(1)
        else:
            returns.append(-0.01)  # 1% loss
            profitable.append(0)
    
    data = {
        'date': dates,
        'prediction': [1] * 20,
        'target_value': returns,
        'actual_profitable': profitable
    }
    
    df = pd.DataFrame(data)
    
    # Calculate stats
    stats = calculate_performance_stats(df, model_type='longonly')
    
    print("\nKey Results:")
    print(f"Total trades: {stats.get('total_trades', 'N/A')}")
    print(f"Hit rate: {stats.get('hit_rate', 'N/A'):.1f}% (Expected: 50%)")
    print(f"Average win: {stats.get('avg_win', 'N/A'):.4f} (Expected: 0.02)")
    print(f"Average loss: {stats.get('avg_loss', 'N/A'):.4f} (Expected: -0.01)")
    print(f"Win/Loss ratio: {stats.get('win_loss_ratio', 'N/A'):.2f} (Expected: 2.0)")
    
    # Compound return: (1.02 * 0.99)^10 = 1.0099^10 ≈ 1.1040
    expected_final = (1.02 * 0.99) ** 10
    print(f"\nCompound return check:")
    print(f"Expected final portfolio value: {expected_final:.4f}")
    
    return stats

def create_test_data_drawdown():
    """Create test data with a significant drawdown"""
    print("\n=== TEST 3: Drawdown Test ===")
    print("Creating test data with initial gains followed by losses")
    
    # Create scenario: 5 wins of 2%, then 5 losses of -3%
    dates = pd.date_range(start='2023-01-01', periods=10, freq='D')
    returns = [0.02] * 5 + [-0.03] * 5
    profitable = [1] * 5 + [0] * 5
    
    data = {
        'date': dates,
        'prediction': [1] * 10,
        'target_value': returns,
        'actual_profitable': profitable
    }
    
    df = pd.DataFrame(data)
    
    # Calculate stats
    stats = calculate_performance_stats(df, model_type='longonly')
    
    print("\nKey Results:")
    print(f"Total return: {stats.get('total_return', 'N/A'):.4f}")
    print(f"Max drawdown %: {stats.get('max_drawdown_pct', 'N/A'):.2f}%")
    
    # Manual calculation
    # After 5 wins: 1.02^5 = 1.1041
    # After 5 losses: 1.1041 * 0.97^5 = 1.1041 * 0.8587 = 0.9479
    # Max DD: (0.9479 - 1.1041) / 1.1041 = -14.14%
    peak = 1.02 ** 5
    trough = peak * (0.97 ** 5)
    expected_dd = (trough - peak) / peak * 100
    
    print(f"\nExpected calculations:")
    print(f"Peak value after 5 wins: {peak:.4f}")
    print(f"Trough after 5 losses: {trough:.4f}")
    print(f"Expected max drawdown: {expected_dd:.2f}%")
    
    if stats.get('drawdown_recovery_trades') is not None:
        print(f"Drawdown recovery: {stats['drawdown_recovery_trades']} trades")
    else:
        print(f"Drawdown recovery: Not recovered")
    
    return stats

def create_test_data_annual():
    """Create test data spanning multiple years"""
    print("\n=== TEST 4: Annual Metrics Test ===")
    print("Creating 2 years of daily data with small consistent returns")
    
    # Create 500 trading days over ~2 years
    dates = pd.date_range(start='2022-01-01', periods=500, freq='B')  # Business days
    
    # Small positive bias: 60% wins of 1%, 40% losses of -0.8%
    np.random.seed(42)
    returns = []
    profitable = []
    for _ in range(500):
        if np.random.random() < 0.6:  # 60% win rate
            returns.append(0.01)
            profitable.append(1)
        else:
            returns.append(-0.008)
            profitable.append(0)
    
    data = {
        'date': dates,
        'prediction': [1] * 500,
        'target_value': returns,
        'actual_profitable': profitable
    }
    
    df = pd.DataFrame(data)
    
    # Calculate stats
    stats = calculate_performance_stats(df, model_type='longonly')
    
    print("\nKey Results:")
    print(f"Years of data: {stats.get('years_of_data', 'N/A'):.2f}")
    print(f"Total return: {stats.get('total_return', 'N/A'):.4f}")
    print(f"Annual return %: {stats.get('annual_return_pct', 'N/A'):.2f}%")
    print(f"Sharpe ratio: {stats.get('sharpe_ratio', 'N/A'):.2f}")
    print(f"Calmar ratio: {stats.get('calmar_ratio', 'N/A'):.2f}")
    print(f"Max drawdown %: {stats.get('max_drawdown_pct', 'N/A'):.2f}%")
    
    if 'monthly_win_rate' in stats:
        print(f"\nMonthly statistics:")
        print(f"Best month: {stats.get('best_month', 'N/A'):.4f}")
        print(f"Worst month: {stats.get('worst_month', 'N/A'):.4f}")
        print(f"Monthly win rate: {stats.get('monthly_win_rate', 'N/A'):.1f}%")
    
    return stats

def test_short_model():
    """Test calculations for short-only model"""
    print("\n=== TEST 5: Short Model Test ===")
    print("Testing short model where negative returns are profitable")
    
    dates = pd.date_range(start='2023-01-01', periods=10, freq='D')
    # For shorts, negative target_value means profit
    data = {
        'date': dates,
        'prediction': [1] * 10,
        'target_value': [-0.01] * 5 + [0.01] * 5,  # First 5 are wins, last 5 are losses
        'actual_profitable': [1] * 5 + [0] * 5
    }
    
    df = pd.DataFrame(data)
    
    # Calculate stats for short model
    stats = calculate_performance_stats(df, model_type='shortonly')
    
    print("\nKey Results for Short Model:")
    print(f"Total trades: {stats.get('total_trades', 'N/A')}")
    print(f"Hit rate: {stats.get('hit_rate', 'N/A'):.1f}% (Expected: 50%)")
    print(f"Total return: {stats.get('total_return', 'N/A'):.4f}")
    print("(Negative target values should be inverted for shorts)")
    
    return stats

def main():
    """Run all tests"""
    print("=" * 60)
    print("PERFORMANCE STATISTICS TEST SUITE")
    print("=" * 60)
    
    # Run tests
    stats1 = create_test_data_simple()
    stats2 = create_test_data_mixed()
    stats3 = create_test_data_drawdown()
    stats4 = create_test_data_annual()
    stats5 = test_short_model()
    
    # Print formatted output for one test
    print("\n" + "=" * 60)
    print("FORMATTED OUTPUT EXAMPLE (Annual Metrics Test)")
    print("=" * 60)
    print(format_stats_for_display(stats4))
    
    print("\n" + "=" * 60)
    print("ALL TESTS COMPLETED")
    print("=" * 60)
    print("\nPlease review the results above to ensure calculations are reasonable.")
    print("Key items to verify:")
    print("1. Compound returns match expected calculations")
    print("2. Drawdown percentages are correctly calculated")
    print("3. Annual metrics (CAGR, Sharpe) are reasonable")
    print("4. Short model inverts returns correctly")

if __name__ == "__main__":
    main()