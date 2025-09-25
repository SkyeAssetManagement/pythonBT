# Testing and Visualization Guide

## Headless Backtesting with P&L Calculations

### Overview
The system now supports headless backtesting that runs completely independent of the chart visualizer. Results are saved to organized folders with trade lists including P&L calculations and execution metrics.

### Architecture Components

#### 1. Headless Backtester
- **File**: `src/trading/backtesting/headless_backtester.py`
- **Purpose**: Runs backtests independently, saves to organized folders
- **Features**: Standard and TWAP execution modes, P&L calculations, chunked processing

#### 2. Integrated Chart Runner
- **File**: `src/trading/visualization/integrated_chart_runner.py`
- **Purpose**: Chart visualizer with dual mode - load existing results OR run new backtests
- **Features**: Mode selector dialog, parameter input, progress tracking

#### 3. Result Storage Structure
```
backtest_results/
├── sma_crossover_20250925_163706/
│   ├── parameters/run_params.json          # Strategy parameters and metadata
│   ├── code/strategy_snapshot.py           # Strategy code at time of run
│   ├── trades/trade_list.csv              # Complete trade list with P&L data
│   └── equity/equity_curve.csv            # Equity curve data
```

## Testing Protocol for P&L System

### Test 1: Headless Backtest Execution
```bash
# Test direct headless backtesting
python -c "
from src.trading.backtesting.headless_backtester import HeadlessBacktester
backtester = HeadlessBacktester()
params = {
    'fast_period': 10,
    'slow_period': 30,
    'long_only': False,
    'signal_lag': 2,
    'position_size': 1.0,
    'min_execution_time': 5.0
}
run_id = backtester.run_backtest('sma_crossover', params, 'data/ES_5min.csv', 'standard')
print(f'Backtest completed: {run_id}')
"
```

**Expected Results:**
- Creates timestamped folder in `backtest_results/`
- Generates `trade_list.csv` with P&L and cumulative profit columns
- No % signs in P&L values (raw decimal format)
- Parameters saved to `run_params.json`

### Test 2: Trade List P&L Verification
```bash
# Check P&L calculations in trade list
head -10 backtest_results/sma_crossover_*/trades/trade_list.csv
```

**Expected CSV Format:**
```csv
datetime,bar_index,trade_type,price,size,exec_bars,execution_time_minutes,num_phases,pnl,cumulative_profit
2023-01-04 10:59:00,31,BUY,4230.58,1.0,1,0.0,1,0.0,0.0
2023-01-04 11:55:00,43,SELL,4227.37,1.0,1,0.0,1,-3.21,-3.21
```

**Validation Checks:**
- [PASS] P&L values are raw decimals (no % sign)
- [PASS] Cumulative profit accumulates correctly
- [PASS] Entry trades have pnl=0.0
- [PASS] Exit trades show actual profit/loss
- [PASS] execBars column shows execution period length

### Test 3: Chart Visualizer Load Mode
```bash
# Test loading existing results
python src/trading/visualization/integrated_chart_runner.py
```

**Test Steps:**
1. Select "Load Existing Backtest Results" radio button
2. Choose a run from the available runs list
3. Verify run details display correctly
4. Click OK to load results

**Expected Results:**
- [PASS] Available runs list populated from backtest_results/
- [PASS] Run details show strategy, execution mode, parameters
- [PASS] Trades load successfully with P&L metadata
- [PASS] Chart displays loaded trade data

### Test 4: Chart Visualizer Run Mode
```bash
# Test running new backtest from chart visualizer
python src/trading/visualization/integrated_chart_runner.py
```

**Test Steps:**
1. Select "Run New Headless Backtest" radio button
2. Configure strategy parameters (fast_period=10, slow_period=30)
3. Select data file (ES range bars)
4. Choose execution mode (TWAP)
5. Click OK to run backtest

**Expected Results:**
- [PASS] Progress dialog shows "Running headless backtest..."
- [PASS] New timestamped folder created in backtest_results/
- [PASS] Trade list generated with P&L calculations
- [PASS] Results automatically loaded into chart visualizer

### Test 5: TWAP Error Handling (No Fallbacks)
```bash
# Test TWAP mode - should throw error when TWAP not available
python -c "
from src.trading.backtesting.headless_backtester import HeadlessBacktester
backtester = HeadlessBacktester()
params = {
    'fast_period': 10,
    'slow_period': 30,
    'signal_lag': 2,
    'min_execution_time': 5.0,
    'position_size': 1.0
}
try:
    run_id = backtester.run_backtest('sma_crossover', params, 'data/sample_trading_data_small.csv', 'twap')
    print('ERROR: Should have thrown ImportError')
except ImportError as e:
    print(f'CORRECT: Got expected error - {e}')
"
```

**Expected Results:**
- [PASS] Throws ImportError: "TWAP execution mode requested but TWAP system not available"
- [PASS] Clear error message with specific import path information
- [PASS] No silent fallback to standard execution
- [PASS] Detailed diagnostic information about missing dependencies

### Test 5b: Valid Execution Modes
```bash
# Test standard mode (should work)
python -c "
from src.trading.backtesting.headless_backtester import HeadlessBacktester
backtester = HeadlessBacktester()
params = {'fast_period': 10, 'slow_period': 30, 'signal_lag': 2}
run_id = backtester.run_backtest('sma_crossover', params, 'data/sample_trading_data_small.csv', 'standard')
print(f'Standard backtest completed: {run_id}')
"

# Test invalid mode (should throw error)
python -c "
from src.trading.backtesting.headless_backtester import HeadlessBacktester
backtester = HeadlessBacktester()
try:
    backtester.run_backtest('sma_crossover', {}, 'data/sample_trading_data_small.csv', 'invalid_mode')
except ValueError as e:
    print(f'CORRECT: Got expected error - {e}')
"
```

**Expected Results:**
- [PASS] Standard mode executes successfully
- [PASS] Invalid modes throw ValueError with supported modes listed
- [PASS] No silent mode switching or fallback behavior

### Test 6: Large Dataset Processing
```bash
# Test with production ES 0.05 range bars (192k+ signals)
python -c "
from src.trading.backtesting.headless_backtester import HeadlessBacktester
backtester = HeadlessBacktester()
params = {
    'fast_period': 10,
    'slow_period': 30,
    'signal_lag': 2,
    'min_execution_time': 5.0
}
run_id = backtester.run_backtest('sma_crossover', params, 'data/ES_0.05_RangeBars.csv', 'twap')
print(f'Large dataset test completed: {run_id}')
"
```

**Expected Results:**
- [PASS] No hanging during processing (chunked processing works)
- [PASS] All trades processed and saved to CSV
- [PASS] P&L calculations accurate for large dataset
- [PASS] Memory usage remains reasonable

### Test 7: Result Loader Integration
```bash
# Test CSV result loading system
python -c "
from src.trading.visualization.backtest_result_loader import BacktestResultLoader
loader = BacktestResultLoader()
runs = loader.list_available_runs()
print(f'Found {len(runs)} runs')
if runs:
    trades = loader.load_trade_list(runs[0]['run_id'])
    print(f'Loaded {len(trades)} trades from latest run')
    if hasattr(trades[0], 'metadata'):
        print(f'Trade metadata: {trades[0].metadata}')
"
```

**Expected Results:**
- [PASS] All available runs discovered and listed
- [PASS] Trades loaded successfully from CSV
- [PASS] TWAP metadata preserved (exec_bars, execution_time_minutes)
- [PASS] P&L data accessible through trade objects

## Error Scenarios and Validation

### Edge Cases to Test:
1. **Signal at End of Data**: Verify no hanging when signal occurs near data end
2. **Zero Volume Bars**: Handle bars with zero volume in TWAP calculations
3. **Missing Data Files**: Proper error handling when data files don't exist
4. **Invalid Parameters**: Validation of strategy parameters
5. **Existing Results Overwrite**: Same parameters should overwrite previous results

### Performance Benchmarks:
- **Small Dataset (1k bars)**: < 5 seconds processing time
- **Medium Dataset (50k bars)**: < 30 seconds processing time
- **Large Dataset (200k bars)**: < 2 minutes processing time
- **Memory Usage**: Should not exceed 1GB RAM for any dataset size

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