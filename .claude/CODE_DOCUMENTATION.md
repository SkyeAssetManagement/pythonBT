# CODE DOCUMENTATION - PythonBT Trading System

## Project Overview
PythonBT is a comprehensive backtesting platform combining machine learning (decision trees) with high-performance PyQtGraph visualization. Features range bar generation, walk-forward optimization, and advanced trade analytics with properly compounded P&L calculations.

## System Architecture

### Primary Components
```
PythonBT/
├── Main GUI Application
│   └── OMtree_gui.py                      # Main tabbed interface for ML models
│
├── PyQtGraph Visualization System
│   ├── launch_unified_system.py           # Primary launcher for charts
│   └── src/trading/visualization/
│       ├── pyqtgraph_range_bars_final.py  # High-performance candlestick chart
│       ├── enhanced_trade_panel.py        # Trade list with compounded P&L %
│       ├── simple_white_x_trades.py       # Trade mark overlays (white X)
│       └── trade_data.py                  # Core trade data structures
│
├── Machine Learning System
│   └── src/
│       ├── OMtree_model.py               # Random forest decision trees
│       ├── OMtree_preprocessing.py       # Feature engineering pipeline
│       ├── OMtree_validation.py          # Backtesting & validation
│       └── OMtree_walkforward.py         # Walk-forward optimization
│
├── Data Processing
│   └── createRangeBars/                  # Range bar generation tools
│       ├── main.py                       # Entry point
│       └── parallel_range_bars.py        # Parallel processing
│
└── Configuration
    ├── OMtree_config.ini                 # ML model configuration
    └── tradingCode/config.yaml           # Trading system settings
```

## Critical Design Patterns

### 1. P&L Calculation System (FIXED 2025-09-24)
**Properly Compounded Returns**
- Individual trades: Store as decimal (0.0238 = 2.38% gain)
- Total P&L: `(1 + r1) * (1 + r2) * ... * (1 + rn) - 1`
- Display: Multiply by 100 for percentage
- All calculations based on $1 invested per trade

### 2. Trade List Sorting (ADDED 2025-09-24)
**Column Sorting Features**
- Click header to sort ascending
- Click again for descending
- Supports all columns: Trade #, DateTime, Type, Price, P&L %, etc.
- Maintains data integrity during sort

### 3. Trade Type Classification
- **Long Trades**: BUY (entry) → SELL (exit)
- **Short Trades**: SHORT (entry) → COVER (exit)
- **Counting**: Properly categorizes exits with their entries

### 4. Data Structure Standards
```python
# Trade data with kwargs support
TradeData(
    bar_index=100,
    price=4200.50,
    trade_type='BUY',
    timestamp='2021-01-04 10:30:00',
    pnl_percent=0.0238,    # 2.38% gain
    trade_id=1,
    size=1,
    fees=2.0
)

# Chart data dictionary (lowercase required)
{
    'timestamp': np.array,  # Required for hover
    'open': np.array,       # Must be lowercase
    'high': np.array,
    'low': np.array,
    'close': np.array,
    'volume': np.array,
    'aux1': np.array,       # ATR values
    'aux2': np.array        # Range multiplier
}
```

## Key Components Detail

### OMtree_gui.py - Main Application
**Tabs:**
- Data & Fields: Load CSV, select features
- Model Tester: Configure and run backtests
- Performance Stats: View results and metrics
- Tree Visualizer: Visualize decision trees
- PermuteAlpha: Feature importance analysis
- Regression Analysis: Statistical testing

### enhanced_trade_panel.py - Trade Analytics
**Features:**
- P&L as percentages (properly compounded)
- Sortable columns (all fields)
- Summary statistics:
  - Total trades & win rate
  - Total/Average P&L %
  - Long/Short counts
  - Commission totals
  - Execution lag tracking

### pyqtgraph_range_bars_final.py - Chart System
**Capabilities:**
- Auto Y-axis scaling on zoom
- Dynamic data loading on pan
- Multi-monitor DPI awareness
- Enhanced time axis (HH:MM:SS)
- Hover display with full data
- Trade marks as white X symbols

### OMtree Model System
**Workflow:**
1. **Preprocessing**: Feature engineering, technical indicators
2. **Model Training**: Random forest with configurable parameters
3. **Validation**: Out-of-sample testing with trade generation
4. **Walk-Forward**: Rolling window optimization

## Performance Specifications
- **Data Capacity**: 6M+ bars handled efficiently
- **Trade Capacity**: 100K+ trades without lag
- **Chart Performance**: 60 FPS with viewport optimization
- **Memory Usage**: ~68MB for 377K bars
- **Load Time**: 377K bars in ~0.6s

## Recent Fixes (2025-09-24)

### Issue 1: Total P&L Calculation
- **Problem**: Summing percentages instead of compounding
- **Fix**: Implemented `(1+r1)*(1+r2)*...-1` formula
- **Files**: enhanced_trade_panel.py (lines 52-63, 359-370)

### Issue 2: Column Sorting
- **Problem**: No sorting functionality in trade list
- **Fix**: Added click-to-sort on all columns
- **Files**: enhanced_trade_panel.py (added sort methods)

### Issue 3: Trade Data Attributes
- **Problem**: kwargs not accessible as attributes
- **Fix**: Added setattr loop in TradeData.__init__
- **Files**: trade_data.py (lines 23-25)

## Testing Infrastructure
```bash
# Test P&L and sorting fixes
python test_enhanced_trade_panel.py

# Launch visualization system
python launch_unified_system.py

# Run ML model GUI
python OMtree_gui.py

# Test components individually
python src/trading/visualization/pyqtgraph_range_bars_final.py
```

## Configuration Files

### OMtree_config.ini
- Model parameters (max_depth, min_samples)
- Feature selections
- Data paths
- Backtest settings

### tradingCode/config.yaml
- Execution settings (signal_lag, formulas)
- Commission rates
- Strategy parameters

## Usage Workflow

### Machine Learning Pipeline
1. Load data in OMtree_gui
2. Select features and configure model
3. Run validation or walk-forward
4. Export results to CSV

### Visualization Pipeline
1. Launch pyqtgraph system
2. Select data source (parquet/CSV)
3. Configure trade source (System/CSV)
4. Run strategy or load trades
5. Analyze with sortable trade list

## Dependencies
- **Core**: pandas, numpy, scikit-learn
- **GUI**: PyQt5, pyqtgraph
- **Visualization**: matplotlib, PIL
- **Data**: parquet, pickle

## Known Limitations
- Windows-specific paths in some scripts
- Maximum practical dataset: ~10M bars
- PyQt5 required (not PyQt6 compatible)

## Future Enhancements
- Real-time data feed integration
- Portfolio-level analytics
- Additional ML models (XGBoost, LSTM)
- Cloud storage for models/data