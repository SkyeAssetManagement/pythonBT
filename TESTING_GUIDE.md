# Testing and Visualization Guide

## Quick Start

### 1. System Check
First, verify your system is ready:
```bash
python test_system.py
```
This will:
- Check all dependencies
- Create sample data if needed
- Verify all components are installed

### 2. Data Requirements

#### For ML Backtesting (CSV format)
Your CSV files need these columns:
- **Date**: YYYY-MM-DD format (e.g., "2024-01-15")
- **Time**: HH:MM:SS format (e.g., "09:30:00")
- **Ticker**: Symbol (e.g., "NQ", "ES")
- **Close**: Price data
- **Forward Returns** (targets): Ret_fwd1hr, Ret_fwd3hr, Ret_fwd6hr, Ret_fwd12hr, Ret_fwd1d
- **Features**: ATR, MACD, RSI, PIR_0-1hr, PIR_1-2hr, etc.

#### For Visualization (CSV or Parquet)
- **DateTime** or separate Date+Time columns
- **OHLCV**: Open, High, Low, Close, Volume
- Optional: Trade signals, indicators

### 3. Running Backtests

#### Simple Backtest Example
```bash
python simple_backtest_example.py
```
This runs a basic ML backtest and saves results to `backtest_results.csv`

#### Full Walk-Forward Validation
```bash
python OMtree_walkforward.py
```
This runs the complete walk-forward ML validation system

### 4. Visualization

#### Integrated Trading GUI (Recommended)
```bash
python integrated_trading_launcher.py
```
Or use the batch file:
```bash
launch_integrated_system.bat
```

#### PyQtGraph with Data Selector
```bash
python launch_pyqtgraph_with_selector.py
```
This opens a file selector to choose your data files (CSV or Parquet)

#### Direct Chart Launcher
```bash
python launch_pyqtgraph_chart.py
```

## Step-by-Step Testing Process

### Step 1: Prepare Your Data

If you have parquet range bar data:
- Place files in `parquetData/` directory
- Files should have OHLCV columns

If you have CSV data:
- Ensure it matches the format requirements above
- Place in `data/` directory

### Step 2: Configure the System

Edit `OMtree_config.ini`:
```ini
[data]
csv_file = data/your_data.csv
target_column = Ret_fwd6hr  # Or your preferred target
selected_features = ATR,MACD,RSI  # Your features

[model]
n_trees = 100  # Number of decision trees
max_depth = 3  # Tree depth

[validation]
train_size = 2000  # Training window
test_size = 100   # Test window
step_size = 100   # Step size for walk-forward
```

### Step 3: Run ML Backtest

For a simple test:
```bash
python simple_backtest_example.py
```

For full walk-forward:
```bash
python OMtree_walkforward.py
```

### Step 4: Visualize Results

Launch the GUI:
```bash
python integrated_trading_launcher.py
```

In the GUI:
1. Click "Load Data" to select your data file
2. View charts with candlesticks and indicators
3. Check the "ML Analysis" tab for model results
4. Use "Performance" tab for metrics

## Common Data Formats

### Example CSV Structure
```csv
Date,Time,Ticker,Close,Ret_fwd1hr,Ret_fwd6hr,ATR,MACD,RSI
2024-01-01,09:00:00,NQ,16500,0.002,0.005,150,10,55
2024-01-01,10:00:00,NQ,16533,0.001,0.003,152,12,58
```

### Example Parquet Range Bars
Your parquet files should contain:
- DateTime or timestamp column
- Open, High, Low, Close, Volume
- Any additional indicators

## Troubleshooting

### Missing Dependencies
If you get import errors, install missing packages:
```bash
pip install pandas numpy scikit-learn matplotlib pyqtgraph PyQt5 vectorbt
```

### Data Not Loading
- Check file path is correct
- Verify CSV has required columns
- For parquet, ensure pyarrow is installed: `pip install pyarrow`

### ML Model Errors
- Ensure you have enough data (minimum 2000 rows recommended)
- Check that feature columns exist in your data
- Verify target column has valid values (not all NaN)

### Visualization Issues
- For PyQtGraph issues, try: `pip install --upgrade pyqtgraph`
- If GUI doesn't open, check PyQt5 installation
- On Windows, you might need: `pip install pyqt5-tools`

## Simple Strategy Testing

### Moving Average Crossover
The `simple_backtest_example.py` includes a basic MA crossover for comparison:
- Fast MA: 10 periods
- Slow MA: 30 periods
- Signal: Long when fast > slow

### ML-Based Strategy
The ML system uses:
- Decision tree ensemble
- Walk-forward validation
- Multiple timeframe predictions
- Feature importance analysis

## Performance Metrics

The system calculates:
- Total Return
- Sharpe Ratio
- Win Rate
- Maximum Drawdown
- Information Coefficient (IC)
- Feature Importance

## Next Steps

1. **Load Your Own Data**: Replace sample data with your trading data
2. **Tune Parameters**: Adjust model settings in config file
3. **Add Features**: Include your own technical indicators
4. **Optimize**: Use walk-forward results to improve strategy
5. **Deploy**: Integrate with your trading system

## Support Files Created

- `test_system.py` - System checker and sample data generator
- `simple_backtest_example.py` - Basic backtest demonstration
- `backtest_results.csv` - Output from backtests
- `data/sample_trading_data.csv` - Sample data for testing