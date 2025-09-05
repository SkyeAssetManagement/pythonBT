#!/usr/bin/env python
"""
Test script to verify GUI fixes for performance stats and charts
Creates sample data to test the consistency between stats and equity curves
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import json
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.performance_stats import calculate_performance_stats

def create_sample_results():
    """Create sample walk-forward results for testing"""
    print("Creating sample walk-forward results...")
    
    # Create 200 days of data with known returns
    dates = pd.date_range(start='2023-01-01', periods=200, freq='B')
    
    # Create a mix of trades and non-trades
    predictions = []
    target_values = []
    actual_profitable = []
    
    np.random.seed(42)
    for i in range(200):
        # 60% chance of making a trade
        if np.random.random() < 0.6:
            predictions.append(1)
            # 55% win rate with 1.5% avg win, -1% avg loss
            if np.random.random() < 0.55:
                target_values.append(np.random.uniform(0.005, 0.025))  # Win: 0.5% to 2.5%
                actual_profitable.append(1)
            else:
                target_values.append(np.random.uniform(-0.015, -0.005))  # Loss: -1.5% to -0.5%
                actual_profitable.append(0)
        else:
            predictions.append(0)  # No trade
            target_values.append(0)
            actual_profitable.append(0)
    
    data = {
        'date': dates,
        'prediction': predictions,
        'target_value': target_values,
        'actual_profitable': actual_profitable
    }
    
    df = pd.DataFrame(data)
    
    # Save as OMtree_results.csv
    df.to_csv('OMtree_results.csv', index=False)
    print(f"Created OMtree_results.csv with {len(df)} rows, {sum(predictions)} trades")
    
    # Calculate and display expected statistics
    trades_only = df[df['prediction'] == 1].copy()
    
    # Calculate compound returns manually
    portfolio_value = 1.0
    for ret in trades_only['target_value'].values:
        portfolio_value *= (1 + ret)
    
    total_return = portfolio_value - 1
    print(f"\nExpected Statistics:")
    print(f"Total trades: {len(trades_only)}")
    print(f"Win rate: {(trades_only['actual_profitable'].sum() / len(trades_only) * 100):.1f}%")
    print(f"Final portfolio value: {portfolio_value:.4f}")
    print(f"Total compound return: {total_return * 100:.2f}%")
    
    # Calculate using our stats module
    stats = calculate_performance_stats(trades_only, model_type='longonly')
    print(f"\nCalculated Statistics:")
    print(f"Total return: {stats.get('total_return', 0) * 100:.2f}%")
    print(f"Hit rate: {stats.get('hit_rate', 0):.1f}%")
    print(f"Max drawdown: {stats.get('max_drawdown_pct', 0):.2f}%")
    print(f"Average win: {stats.get('avg_win', 0) * 100:.2f}%")
    print(f"Average loss: {stats.get('avg_loss', 0) * 100:.2f}%")
    
    return df

def create_feature_selection_history():
    """Create sample feature selection history for timeline testing"""
    print("\n\nCreating feature selection history...")
    
    # Create results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)
    
    # Define some feature names
    all_features = [
        'close_norm', 'volume_norm', 'high_norm', 'low_norm', 'open_norm',
        'returns_1', 'returns_2', 'returns_3', 'returns_5', 'returns_10',
        'volatility_5', 'volatility_10', 'volatility_20',
        'rsi_14', 'macd_signal', 'bb_position',
        'volume_ratio', 'price_position', 'trend_strength'
    ]
    
    # Create history of feature selection over walk-forward steps
    history = []
    for step in range(50):  # 50 walk-forward steps
        # Randomly select 5-10 features for each step
        n_features = np.random.randint(5, 11)
        # Bias towards selecting certain features more often
        weights = np.array([0.8 if 'returns' in f or 'volatility' in f else 0.3 for f in all_features])
        weights = weights / weights.sum()
        selected = np.random.choice(all_features, size=n_features, replace=False, p=weights)
        
        history.append({
            'step': step + 1,
            'selected_features': list(selected),
            'n_features': n_features
        })
    
    # Save to JSON
    history_data = {
        'all_features': all_features,
        'history': history,
        'config': {
            'model_type': 'longonly',
            'min_features': 5,
            'max_features': 10,
            'importance_threshold': 0.01
        }
    }
    
    with open('results/feature_selection_history.json', 'w') as f:
        json.dump(history_data, f, indent=2)
    
    print(f"Created feature_selection_history.json with {len(history)} steps")
    print(f"Features pool: {len(all_features)} features")
    
def main():
    """Run test data generation"""
    print("=" * 60)
    print("TESTING GUI FIXES - CREATING TEST DATA")
    print("=" * 60)
    
    # Create sample results
    df = create_sample_results()
    
    # Create feature selection history
    create_feature_selection_history()
    
    print("\n" + "=" * 60)
    print("TEST DATA CREATED SUCCESSFULLY")
    print("=" * 60)
    print("\nYou can now:")
    print("1. Open the OMtree GUI")
    print("2. Go to Model Tester tab")
    print("3. Click 'Load Results'")
    print("4. View Performance Statistics - should match the expected values above")
    print("5. View Equity Curve - should show compound returns matching stats")
    print("6. Try different chart types:")
    print("   - Feature Timeline: Should display the selection history")
    print("   - Comprehensive: Should show consistent portfolio values and drawdown")
    print("   - Progression: Should show walk-forward progression")
    print("\nAll statistics and charts should now be consistent!")

if __name__ == "__main__":
    main()