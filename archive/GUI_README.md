# OMtree Trading Model GUI

A graphical user interface for configuring, running, and analyzing the OMtree walk-forward validation trading model.

## Features

### 1. Configuration Tab
- Edit all model parameters in an intuitive interface
- Organized sections for:
  - Data Settings (file, features, columns)
  - Preprocessing (normalization, volatility windows)
  - Model Parameters (trees, thresholds, voting)
  - Validation Settings (train/test windows, dates)
- Save/Load configuration to `OMtree_config.ini`
- Reset to default values

### 2. Run Validation Tab
- One-click walk-forward validation execution
- Real-time output display
- Progress bar showing validation progress
- Stop button to interrupt long-running validations
- Console output with detailed logging

### 3. Performance Stats Tab
- Comprehensive performance metrics:
  - Trading frequency and hit rates
  - Monthly P&L statistics
  - Annualized Sharpe ratio
  - Yearly performance breakdown
- Auto-refreshes after validation completes

### 4. Charts Tab
- View generated performance charts:
  - `OMtree_comprehensive_*.png` - 6-panel detailed analysis
  - `OMtree_progression_*.png` - 4-panel yearly progression
- Dropdown selection for multiple charts
- Auto-scaling to fit window

## Usage

### Quick Start
1. Run the GUI:
   ```bash
   python launch_gui.py
   ```
   Or directly:
   ```bash
   python OMtree_gui.py
   ```

2. Configure your model parameters in the Configuration tab

3. Go to Run Validation tab and click "Run Walk-Forward Validation"

4. View results in Performance Stats and Charts tabs

### Configuration Guide

#### Key Parameters to Adjust:

**Model Type**
- `longonly`: Profit from upward moves
- `shortonly`: Profit from downward moves

**Vote Threshold** (0.5-0.9)
- Higher values = more selective trading
- Lower values = more frequent trading

**Target Threshold** (0-0.3)
- Minimum return to consider a trade profitable
- Higher values = looking for bigger moves

**Min Leaf Fraction** (0.05-0.4)
- Controls tree complexity
- Higher values = simpler trees, less overfitting

**Volatility Window** (10-300)
- Period for volatility normalization
- Shorter = more responsive to recent volatility

### Tips

1. **Start with defaults**: The default configuration is well-tested
2. **Out-of-sample testing**: Set `validation_start_date` to ensure fair evaluation
3. **Monitor Sharpe ratio**: Target > 0.5 for decent performance, > 1.0 for good
4. **Check hit rate**: Should be consistently above base rate (42%)
5. **Review charts**: Look for stable cumulative P&L growth

## Requirements

- Python 3.7+
- tkinter (usually included with Python)
- pandas
- numpy
- matplotlib
- Pillow (PIL)

Install requirements:
```bash
pip install pandas numpy matplotlib pillow
```

## Files Generated

- `OMtree_results.csv` - Detailed validation results
- `OMtree_performance.csv` - Performance metrics log
- `OMtree_comprehensive_*.png` - Analysis charts
- `OMtree_progression_*.png` - Progression charts

## Troubleshooting

**GUI won't start**: 
- Check Python version (3.7+)
- Install missing packages: `pip install pillow`

**Validation fails**:
- Check data file exists (`DTSnnData.csv`)
- Verify date format matches data
- Reduce vote_threshold if no trades generated

**Charts not showing**:
- Run validation first to generate charts
- Check PNG files exist in directory
- Try Refresh button in Charts tab