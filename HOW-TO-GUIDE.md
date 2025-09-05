# OMtree Trading System - How-To Guide

## Table of Contents
1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Using the GUI](#using-the-gui)
4. [Running Walk-Forward Validation](#running-walk-forward-validation)
5. [Configuration Guide](#configuration-guide)
6. [Data Format Requirements](#data-format-requirements)
7. [Advanced Usage](#advanced-usage)
8. [Troubleshooting](#troubleshooting)

---

## Installation

### Step 1: Install Python Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Prepare Your Data
- Use the provided sample data generator or prepare your own CSV file
- Generate sample data:
```bash
cd data
python sample_data_generator.py
```

### Step 3: Configure the System
- Edit `OMtree_config.ini` to point to your data file
- Adjust model parameters as needed

---

## Quick Start

### Option 1: Using the GUI (Recommended for Beginners)
```bash
python OMtree_gui.py
```

### Option 2: Command-Line Walk-Forward Validation
```bash
python OMtree_walkforward.py
```

---

## Using the GUI

### Main Features

#### 1. Configuration Tab
- **Data Settings**: Load and configure your data source
- **Model Parameters**: Adjust tree ensemble settings
- **Feature Selection**: Choose which features to use
- **Save/Load Configurations**: Store different configurations for reuse

#### 2. Walk-Forward Tab
- **Run Analysis**: Execute walk-forward validation
- **View Results**: See performance metrics and trade statistics
- **Export Results**: Save results to CSV

#### 3. Performance Tab
- **Equity Curves**: Visualize cumulative returns
- **Performance Metrics**: View Sharpe ratio, win rate, etc.
- **Trade Distribution**: Analyze trade patterns

#### 4. Tree Visualizer Tab
- **Model Structure**: View decision tree structures
- **Feature Importance**: See which features drive predictions
- **Split Points**: Understand decision boundaries

#### 5. Data View Tab
- **Raw Data**: Inspect your input data
- **Preprocessed Data**: View transformed features
- **Predictions**: See model predictions alongside actual values

#### 6. Regression Analysis Tab
- **Statistical Analysis**: Run regression diagnostics
- **Correlation Studies**: Analyze feature relationships
- **Residual Plots**: Check model assumptions

### GUI Workflow

1. **Load Configuration**
   - Click "Load Config" button
   - Select a `.ini` file or use default

2. **Select Data File**
   - Click "Browse" in Data Settings
   - Choose your CSV file
   - Verify columns are detected correctly

3. **Configure Model**
   - Set model type (longonly/shortonly)
   - Adjust tree parameters
   - Enable/disable feature selection

4. **Run Walk-Forward**
   - Go to Walk-Forward tab
   - Click "Run Walk-Forward"
   - Monitor progress in console

5. **Analyze Results**
   - Review performance metrics
   - Export results if needed
   - Adjust parameters and repeat

---

## Running Walk-Forward Validation

### What is Walk-Forward Validation?
Walk-forward validation is a backtesting method that:
1. Trains on a historical window
2. Tests on the next period
3. Slides the window forward
4. Repeats until end of data

### Command-Line Usage
```bash
python OMtree_walkforward.py
```

### Understanding the Output
- **predictions_*.csv**: Raw predictions for each period
- **performance_report.txt**: Summary statistics
- **equity_curve.png**: Visual performance over time

### Key Metrics Explained
- **Sharpe Ratio**: Risk-adjusted returns (higher is better)
- **Win Rate**: Percentage of profitable trades
- **Average Return**: Mean return per trade
- **Max Drawdown**: Largest peak-to-trough decline
- **Calmar Ratio**: Annual return / Max drawdown

---

## Configuration Guide

### Essential Parameters

#### Data Section
```ini
[data]
csv_file = path/to/your/data.csv
target_column = Ret_fwd6hr  # Column with forward returns
feature_columns = PIR_64-128hr,PIR_32-64hr,...  # Input features
date_column = Date
time_column = Time
ticker_filter = NQ  # Optional: filter specific ticker
hour_filter = 10  # Optional: filter specific hour
```

#### Model Section
```ini
[model]
model_type = longonly  # or shortonly
n_trees = 100  # Number of trees in ensemble
max_depth = 3  # Maximum tree depth
bootstrap_fraction = 0.6  # Sampling fraction
min_leaf_fraction = 0.1  # Minimum samples in leaf
target_threshold = 0.0  # Return threshold for signals
vote_threshold = 0.5  # Ensemble voting threshold
```

#### Validation Section
```ini
[validation]
train_size = 2000  # Training window size
test_size = 100  # Testing window size
step_size = 100  # Window step size
validation_start_date = 2020-01-01
validation_end_date = 2021-12-31
```

### Advanced Parameters

#### Feature Selection
```ini
[feature_selection]
enabled = true
min_features = 2
max_features = 6
importance_threshold = 0.125
```

#### Preprocessing
```ini
[preprocessing]
normalize_features = false
normalize_target = true
normalization_method = LOGIT_RANK
vol_window = 90
```

---

## Data Format Requirements

### Required Columns
1. **Date**: YYYY-MM-DD format
2. **Time**: HH:MM:SS format (optional)
3. **Target**: Forward returns (e.g., Ret_fwd6hr)
4. **Features**: Input variables for prediction

### Example CSV Structure
```csv
Date,Time,Ticker,Open,High,Low,Close,Volume,Ret_fwd1hr,PIR_0-1hr,...
2020-01-01,09:00:00,NQ,100.5,101.2,100.1,100.8,500000,0.5,-0.3,...
```

### Data Preparation Tips
- Ensure no missing values in feature columns
- Forward returns should be percentage returns
- Features can be technical indicators, price ratios, etc.
- Include sufficient history (minimum 3000 rows recommended)

---

## Advanced Usage

### 1. Feature Engineering
Create custom features in your data:
- Moving averages
- Volatility measures
- Price patterns
- Volume indicators

### 2. Model Optimization
Use the GUI to experiment with:
- Different tree depths (2-5 recommended)
- Bootstrap fractions (0.5-0.8)
- Vote thresholds (0.4-0.6)
- Feature combinations

### 3. Multi-Timeframe Analysis
- Test different forward return periods
- Combine signals from multiple models
- Use different features for different timeframes

### 4. Risk Management
- Set appropriate position sizing
- Use stop-losses based on model confidence
- Monitor drawdowns and adjust accordingly

### 5. Permutation Testing
Run statistical significance tests:
```python
python run_permutation_isolated.py
```

---

## Troubleshooting

### Common Issues and Solutions

#### 1. "File not found" Error
- Check file path in config
- Use absolute paths
- Ensure CSV file exists

#### 2. "Column not found" Error
- Verify column names match exactly
- Check for extra spaces
- Case-sensitive column names

#### 3. Poor Model Performance
- Increase training data size
- Try different feature combinations
- Adjust target threshold
- Check for data leakage

#### 4. GUI Not Responding
- Check console for errors
- Ensure all dependencies installed
- Try smaller data files first

#### 5. Memory Issues
- Reduce data size
- Lower n_trees parameter
- Use feature selection

### Debug Mode
Enable verbose logging:
```ini
[output]
verbose_logging = true
```

### Getting Help
1. Check console output for detailed error messages
2. Review log files in output directory
3. Verify data format matches requirements
4. Test with sample data first

---

## Best Practices

1. **Start Simple**: Begin with default parameters
2. **Iterate Gradually**: Change one parameter at a time
3. **Monitor Overfitting**: Watch out-of-sample performance
4. **Document Changes**: Keep notes on what works
5. **Validate Results**: Use multiple validation periods
6. **Risk First**: Focus on drawdown before returns

---

## Example Workflow

### Complete Example: Long-Only Strategy

1. **Prepare Data**
```bash
cd data
python sample_data_generator.py
```

2. **Configure Model**
Edit `OMtree_config.ini`:
```ini
[data]
csv_file = data/sample_trading_data.csv
target_column = Ret_fwd6hr

[model]
model_type = longonly
n_trees = 100
max_depth = 3
```

3. **Run GUI**
```bash
python OMtree_gui.py
```

4. **Execute Walk-Forward**
- Click "Run Walk-Forward" in GUI
- Wait for completion
- Review results

5. **Analyze Performance**
- Check Sharpe ratio > 1.0
- Verify win rate > 50%
- Review equity curve smoothness

6. **Optimize Parameters**
- Try max_depth = 4
- Test bootstrap_fraction = 0.7
- Compare results

7. **Export Results**
- Click "Export to CSV"
- Save performance metrics
- Document configuration

---

## Additional Resources

- Configuration templates in `/configs/`
- Example data in `/data/`
- Performance reports in `/output/`
- Archived experiments in `/archive/`

For questions or issues, refer to the code documentation or check the console output for detailed error messages.