"""
Sample Data Generator for OMtree Trading System
Generates sample trading data for testing the system
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

def generate_sample_data(n_rows=10000, output_file='sample_trading_data.csv'):
    """
    Generate sample trading data with the required columns
    """
    np.random.seed(42)
    
    # Generate dates and times
    start_date = datetime(2020, 1, 1, 9, 0, 0)
    dates = []
    times = []
    
    current_date = start_date
    for i in range(n_rows):
        dates.append(current_date.strftime('%Y-%m-%d'))
        times.append(current_date.strftime('%H:%M:%S'))
        # Increment by 1 hour
        current_date += timedelta(hours=1)
        # Skip weekends
        if current_date.weekday() >= 5:
            current_date += timedelta(days=2)
    
    # Generate base price with trend and volatility
    base_price = 100
    prices = [base_price]
    for i in range(1, n_rows):
        # Random walk with slight upward trend
        change = np.random.normal(0.0001, 0.01)
        prices.append(prices[-1] * (1 + change))
    
    prices = np.array(prices)
    
    # Calculate returns at different horizons
    def calculate_forward_returns(prices, horizon):
        returns = []
        for i in range(len(prices)):
            if i + horizon < len(prices):
                ret = (prices[i + horizon] / prices[i] - 1) * 100
            else:
                ret = np.nan
            returns.append(ret)
        return returns
    
    # Create DataFrame
    df = pd.DataFrame({
        'Date': dates,
        'Time': times,
        'Ticker': 'NQ',  # Default ticker
        'Open': prices * np.random.uniform(0.995, 1.005, n_rows),
        'High': prices * np.random.uniform(1.005, 1.02, n_rows),
        'Low': prices * np.random.uniform(0.98, 0.995, n_rows),
        'Close': prices,
        'Volume': np.random.randint(100000, 1000000, n_rows)
    })
    
    # Add forward returns
    df['Ret_fwd1hr'] = calculate_forward_returns(prices, 1)
    df['Ret_fwd3hr'] = calculate_forward_returns(prices, 3)
    df['Ret_fwd6hr'] = calculate_forward_returns(prices, 6)
    df['Ret_fwd12hr'] = calculate_forward_returns(prices, 12)
    df['Ret_fwd1d'] = calculate_forward_returns(prices, 24)
    
    # Generate PIR (Price Interest Ratio) features
    # These are synthetic momentum/volume indicators
    for window in ['64-128hr', '32-64hr', '16-32hr', '8-16hr', '4-8hr', '2-4hr', '1-2hr', '0-1hr']:
        # Create synthetic PIR values with some correlation to returns
        base_pir = np.random.normal(0, 1, n_rows)
        if '0-1hr' in window:
            # Most recent PIR has higher correlation with forward returns
            df[f'PIR_{window}'] = base_pir * 0.3 + df['Ret_fwd1hr'].fillna(0) * 0.1 + np.random.normal(0, 0.5, n_rows)
        else:
            df[f'PIR_{window}'] = base_pir
    
    # Add some additional technical indicators
    df['RSI'] = 50 + np.random.normal(0, 15, n_rows)
    df['MACD'] = np.random.normal(0, 2, n_rows)
    df['ATR'] = abs(np.random.normal(1, 0.5, n_rows))
    
    # Save to CSV
    df.to_csv(output_file, index=False)
    print(f"Sample data generated: {output_file}")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
    
    return df

if __name__ == "__main__":
    # Generate sample data
    df = generate_sample_data(n_rows=10000)
    
    # Also create a smaller test file
    df_small = generate_sample_data(n_rows=1000, output_file='sample_trading_data_small.csv')
    
    print("\nSample data files created in the data/ directory")
    print("You can now update the config file to point to these files")