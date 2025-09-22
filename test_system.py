"""
Test launcher for backtesting and visualization system
This script helps you test the integrated OMtree + ABtoPython system
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import sys

def create_sample_data():
    """
    Creates sample CSV data in the format expected by OMtree ML system

    Required columns:
    - Date: Date in YYYY-MM-DD format
    - Time: Time in HH:MM:SS format
    - Ticker: Symbol (e.g., 'NQ', 'ES')
    - Various return columns (Ret_fwd1hr, Ret_fwd3hr, etc.)
    - Feature columns (ATR, MACD, RSI, PIR_*, etc.)
    """

    # Generate dates and times
    dates = []
    times = []
    start_date = datetime(2020, 1, 1, 9, 0, 0)

    for i in range(5000):  # 5000 hourly bars
        current_dt = start_date + timedelta(hours=i)
        dates.append(current_dt.strftime('%Y-%m-%d'))
        times.append(current_dt.strftime('%H:%M:%S'))

    # Create price series with trend and noise
    np.random.seed(42)
    price = 100
    prices = []
    for i in range(5000):
        returns = np.random.normal(0.0001, 0.01)  # Mean return with volatility
        price = price * (1 + returns)
        prices.append(price)

    prices = np.array(prices)

    # Calculate returns at different horizons
    def calculate_forward_returns(prices, periods):
        returns = []
        for i in range(len(prices)):
            if i + periods < len(prices):
                ret = (prices[i + periods] - prices[i]) / prices[i]
                returns.append(ret)
            else:
                returns.append(0)
        return returns

    # Create DataFrame
    df = pd.DataFrame({
        'Date': dates,
        'Time': times,
        'Ticker': 'NQ',
        'Close': prices,

        # Forward returns (targets)
        'Ret_fwd1hr': calculate_forward_returns(prices, 1),
        'Ret_fwd3hr': calculate_forward_returns(prices, 3),
        'Ret_fwd6hr': calculate_forward_returns(prices, 6),
        'Ret_fwd12hr': calculate_forward_returns(prices, 12),
        'Ret_fwd1d': calculate_forward_returns(prices, 24),

        # Technical indicators (features)
        'ATR': np.abs(np.random.normal(0.01, 0.002, 5000)),
        'MACD': np.random.normal(0, 0.01, 5000),
        'RSI': np.random.uniform(30, 70, 5000),

        # PIR features (Price Impact Ratios at different horizons)
        'PIR_0-1hr': np.random.normal(0, 0.01, 5000),
        'PIR_1-2hr': np.random.normal(0, 0.01, 5000),
        'PIR_2-4hr': np.random.normal(0, 0.01, 5000),
        'PIR_4-8hr': np.random.normal(0, 0.01, 5000),
        'PIR_8-16hr': np.random.normal(0, 0.01, 5000),
        'PIR_16-32hr': np.random.normal(0, 0.01, 5000),
        'PIR_32-64hr': np.random.normal(0, 0.01, 5000),
        'PIR_64-128hr': np.random.normal(0, 0.01, 5000),

        # Random noise features for testing
        'RandomNoise_1': np.random.normal(0, 1, 5000),
        'RandomNoise_2': np.random.normal(0, 1, 5000),
        'RandomNoise_3': np.random.normal(0, 1, 5000),
    })

    # Save to data directory
    os.makedirs('data', exist_ok=True)
    csv_path = 'data/sample_trading_data.csv'
    df.to_csv(csv_path, index=False)
    print(f"Sample data created: {csv_path}")
    print(f"Shape: {df.shape}")
    print(f"Columns: {', '.join(df.columns)}")
    return csv_path

def test_ml_system():
    """Test the OMtree ML system"""
    print("\n" + "="*50)
    print("Testing OMtree ML System")
    print("="*50)

    # Check if main script exists
    if os.path.exists('OMtree_walkforward.py'):
        print("Found: OMtree_walkforward.py")
        print("\nTo run walk-forward validation:")
        print("  python OMtree_walkforward.py")
    else:
        print("Warning: OMtree_walkforward.py not found")

    if os.path.exists('OMtree_model.py'):
        print("\nFound: OMtree_model.py")
        print("\nTo train a single model:")
        print("  python OMtree_model.py")
    else:
        print("Warning: OMtree_model.py not found")

def test_visualization():
    """Test the visualization system"""
    print("\n" + "="*50)
    print("Testing Visualization System")
    print("="*50)

    # Check launcher scripts
    launchers = [
        'integrated_trading_launcher.py',
        'launch_integrated_system.bat',
        'launch_pyqtgraph_chart.py',
        'launch_pyqtgraph_with_selector.py'
    ]

    for launcher in launchers:
        if os.path.exists(launcher):
            print(f"Found: {launcher}")

    print("\nTo launch the integrated GUI:")
    print("  python integrated_trading_launcher.py")
    print("\nOr use the batch file:")
    print("  launch_integrated_system.bat")

def check_dependencies():
    """Check if required packages are installed"""
    print("\n" + "="*50)
    print("Checking Dependencies")
    print("="*50)

    required = {
        'pandas': 'Data manipulation',
        'numpy': 'Numerical computations',
        'sklearn': 'Machine learning models',
        'matplotlib': 'Basic plotting',
        'pyqtgraph': 'High-performance charting',
        'PyQt5': 'GUI framework',
        'vectorbt': 'Backtesting engine (optional)'
    }

    for package, description in required.items():
        try:
            if package == 'sklearn':
                from sklearn.tree import DecisionTreeClassifier
            else:
                __import__(package.replace('-', '_'))
            print(f"[OK] {package}: {description}")
        except ImportError:
            print(f"[MISSING] {package}: {description}")
            print(f"  Install with: pip install {package if package != 'sklearn' else 'scikit-learn'}")

def test_existing_system():
    """Test the existing OMtree system components"""
    print("\n" + "="*50)
    print("Testing Existing OMtree Components")
    print("="*50)

    # Add src to path
    sys.path.insert(0, 'src')

    # Test core modules
    print("\nCore modules:")
    try:
        from src.OMtree_model import DirectionalTreeEnsemble
        print("[OK] Model module (DirectionalTreeEnsemble)")
    except ImportError as e:
        print(f"[MISSING] Model module: {e}")

    try:
        from src.OMtree_preprocessing import DataPreprocessor
        print("[OK] Preprocessing module (DataPreprocessor)")
    except ImportError as e:
        print(f"[MISSING] Preprocessing module: {e}")

    try:
        from src.OMtree_validation import WalkForwardValidator
        print("[OK] Validation module (WalkForwardValidator)")
    except ImportError as e:
        print(f"[MISSING] Validation module: {e}")

    # Test configuration
    print("\nConfiguration:")
    config_file = "OMtree_config.ini"
    if os.path.exists(config_file):
        print(f"[OK] Configuration file: {config_file}")
        import configparser
        config = configparser.ConfigParser()
        config.read(config_file)
        print(f"  - Data file: {config['data']['csv_file']}")
        print(f"  - Target: {config['data']['target_column']}")
        print(f"  - Model type: {config['model']['model_type']}")
        print(f"  - Trees: {config['model']['n_trees']}")
    else:
        print(f"[MISSING] Configuration file: {config_file}")

def main():
    """Main test function"""
    print("="*50)
    print("Testing Backtesting and Visualization System")
    print("="*50)

    # Check dependencies first
    check_dependencies()

    # Test existing system
    test_existing_system()

    # Create sample data if it doesn't exist
    data_file = 'data/sample_trading_data.csv'
    if not os.path.exists(data_file):
        print(f"\nCreating sample data...")
        create_sample_data()
    else:
        print(f"\nSample data exists: {data_file}")
        df = pd.read_csv(data_file)
        print(f"Shape: {df.shape}")

    # Test ML system
    test_ml_system()

    # Test visualization
    test_visualization()

    print("\n" + "="*50)
    print("Quick Start Guide")
    print("="*50)
    print("\n1. For ML backtesting:")
    print("   python OMtree_walkforward.py")
    print("\n2. For visualization:")
    print("   python integrated_trading_launcher.py")
    print("\n3. For data selector + charts:")
    print("   python launch_pyqtgraph_with_selector.py")

    print("\n" + "="*50)
    print("Data Format Requirements")
    print("="*50)
    print("\nFor ML System (CSV):")
    print("- Date: YYYY-MM-DD format")
    print("- Time: HH:MM:SS format")
    print("- Ticker: Symbol (e.g., 'NQ', 'ES')")
    print("- Return columns: Ret_fwd1hr, Ret_fwd3hr, etc.")
    print("- Feature columns: ATR, MACD, RSI, PIR_*, etc.")

    print("\nFor Visualization (CSV or Parquet):")
    print("- DateTime or Date+Time columns")
    print("- OHLCV data (Open, High, Low, Close, Volume)")
    print("- Optional: Trade signals, indicators")

if __name__ == "__main__":
    main()