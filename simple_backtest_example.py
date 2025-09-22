"""
Simple Backtesting Example
This script demonstrates how to run a basic backtest using the OMtree ML system
"""

import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime

# Add src to path
sys.path.insert(0, 'src')

def run_simple_backtest():
    """Run a simple ML-based backtest"""

    print("="*60)
    print("Simple ML Backtest Example")
    print("="*60)
    print()

    # 1. Check data exists
    data_file = 'data/sample_trading_data.csv'
    if not os.path.exists(data_file):
        print("Error: Sample data not found. Run 'python test_system.py' first to create it.")
        return

    # 2. Load and preview data
    print("Loading data...")
    df = pd.read_csv(data_file)
    print(f"Data shape: {df.shape}")
    print(f"Date range: {df['Date'].iloc[0]} to {df['Date'].iloc[-1]}")
    print()

    # 3. Simple moving average crossover strategy (for comparison)
    print("Running simple MA crossover strategy...")
    df['MA_fast'] = df['Close'].rolling(window=10).mean()
    df['MA_slow'] = df['Close'].rolling(window=30).mean()
    df['Signal_MA'] = (df['MA_fast'] > df['MA_slow']).astype(int)
    df['Returns'] = df['Close'].pct_change()
    df['Strategy_Returns_MA'] = df['Signal_MA'].shift(1) * df['Returns']

    # Calculate performance
    total_return_ma = (1 + df['Strategy_Returns_MA'].dropna()).prod() - 1
    sharpe_ma = df['Strategy_Returns_MA'].mean() / df['Strategy_Returns_MA'].std() * np.sqrt(252 * 24)

    print(f"MA Crossover Performance:")
    print(f"  Total Return: {total_return_ma:.2%}")
    print(f"  Sharpe Ratio: {sharpe_ma:.2f}")
    print()

    # 4. ML-based strategy
    print("Running ML-based strategy...")
    try:
        from src.OMtree_model import DirectionalTreeEnsemble
        from src.OMtree_preprocessing import DataPreprocessor

        # Initialize model
        model = DirectionalTreeEnsemble(config_path='OMtree_config.ini', verbose=False)
        preprocessor = DataPreprocessor(config_path='OMtree_config.ini', verbose=False)

        # Prepare data
        features = ['ATR', 'MACD', 'RSI', 'RandomNoise_1', 'RandomNoise_2', 'RandomNoise_3']
        target = 'Ret_fwd6hr'

        # Simple train/test split
        train_size = int(len(df) * 0.7)
        train_data = df.iloc[:train_size]
        test_data = df.iloc[train_size:]

        # Preprocess
        X_train_processed, y_train_processed, _ = preprocessor.preprocess(
            train_data,
            features,
            target
        )

        X_test_processed, y_test_processed, _ = preprocessor.preprocess(
            test_data,
            features,
            target
        )

        # Train model
        print("Training ML model...")
        model.fit(X_train_processed, y_train_processed, features)

        # Make predictions
        predictions = model.predict(X_test_processed)

        # Generate signals (1 if positive prediction, 0 otherwise)
        test_data.loc[:, 'Signal_ML'] = (predictions > 0).astype(int)
        test_data.loc[:, 'Strategy_Returns_ML'] = test_data['Signal_ML'].shift(1) * test_data['Returns']

        # Calculate ML performance
        total_return_ml = (1 + test_data['Strategy_Returns_ML'].dropna()).prod() - 1
        sharpe_ml = test_data['Strategy_Returns_ML'].mean() / test_data['Strategy_Returns_ML'].std() * np.sqrt(252 * 24)

        print(f"\nML Strategy Performance (Out-of-Sample):")
        print(f"  Total Return: {total_return_ml:.2%}")
        print(f"  Sharpe Ratio: {sharpe_ml:.2f}")
        print(f"  Predictions > 0: {(predictions > 0).sum()} / {len(predictions)}")

        # Save results
        results = test_data[['Date', 'Time', 'Close', 'Returns', 'Signal_ML', 'Strategy_Returns_ML']].copy()
        results.to_csv('backtest_results.csv', index=False)
        print(f"\nResults saved to: backtest_results.csv")

    except Exception as e:
        print(f"Error running ML strategy: {e}")
        print("Make sure all dependencies are installed and modules are properly configured.")

    print("\n" + "="*60)
    print("Backtest Complete")
    print("="*60)

def visualize_results():
    """Simple visualization of backtest results"""
    print("\nTo visualize the results:")
    print("1. Run: python integrated_trading_launcher.py")
    print("2. Load the backtest_results.csv file")
    print("3. Or use: python launch_pyqtgraph_with_selector.py")

if __name__ == "__main__":
    run_simple_backtest()
    visualize_results()