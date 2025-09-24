# CODEBASE DOCUMENTATION - PythonBT

## Project Overview
PythonBT is a comprehensive backtesting and trading visualization system that combines machine learning models (decision trees) with advanced charting capabilities using PyQtGraph.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                      PythonBT System                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────────┐        ┌─────────────────────┐      │
│  │   GUI Layer      │        │  Visualization Layer │      │
│  │                  │        │                      │      │
│  │ - OMtree_gui.py  │◄──────►│ - PyQtGraph Charts  │      │
│  │ - Data Tab       │        │ - Trade Panel       │      │
│  │ - Config Tab     │        │ - Range Bars        │      │
│  │ - Results Tab    │        │ - Trade Marks       │      │
│  └──────────────────┘        └─────────────────────┘      │
│           │                           │                     │
│           ▼                           ▼                     │
│  ┌──────────────────┐        ┌─────────────────────┐      │
│  │   Core Trading   │        │   Data Processing   │      │
│  │                  │        │                      │      │
│  │ - Model Training │◄──────►│ - Preprocessing     │      │
│  │ - Validation     │        │ - Feature Engineer  │      │
│  │ - Walk Forward   │        │ - Range Bar Create  │      │
│  └──────────────────┘        └─────────────────────┘      │
│           │                           │                     │
│           ▼                           ▼                     │
│  ┌──────────────────────────────────────────────────┐      │
│  │              Data Layer & Storage                 │      │
│  │                                                   │      │
│  │  - CSV Files    - Pickle Models    - Config      │      │
│  └──────────────────────────────────────────────────┘      │
└─────────────────────────────────────────────────────────────┘
```

## Directory Structure

```
C:\code\PythonBT\
├── src/                           # Core source code
│   ├── trading/                   # Trading system components
│   │   ├── visualization/         # Chart and UI components
│   │   │   ├── pyqtgraph_range_bars_final.py  # Main chart window
│   │   │   ├── enhanced_trade_panel.py        # Trade list with P&L
│   │   │   ├── trade_panel.py                 # Base trade panel
│   │   │   ├── simple_white_x_trades.py       # Trade mark overlays
│   │   │   ├── trade_data.py                  # Trade data structures
│   │   │   └── range_bar_chart.py             # Range bar rendering
│   │   ├── data/                  # Data handling
│   │   │   ├── loaders.py         # Data loading utilities
│   │   │   ├── trade_data.py      # Trade data models
│   │   │   └── unified_pipeline.py # Data pipeline
│   │   ├── strategies/            # Trading strategies
│   │   │   ├── base.py            # Base strategy class
│   │   │   ├── sma_crossover.py   # SMA strategy
│   │   │   └── rsi_momentum.py    # RSI strategy
│   │   └── core/                  # Core trading engine
│   │       ├── trade_types.py     # Trade type definitions
│   │       └── standalone_execution.py # Execution engine
│   │
│   ├── OMtree_model.py           # Decision tree model
│   ├── OMtree_preprocessing.py   # Data preprocessing
│   ├── OMtree_validation.py      # Model validation
│   ├── OMtree_walkforward.py     # Walk-forward analysis
│   ├── config_manager.py         # Configuration management
│   ├── performance_stats.py      # Performance metrics
│   ├── regression_gui_module_v3.py # Regression analysis
│   ├── tree_visualizer.py        # Decision tree visualization
│   └── timeline_component_v2.py  # Timeline visualization
│
├── OMtree_gui.py                 # Main GUI application
├── OMtree_config.ini             # Configuration file
├── launch_unified_system.py      # PyQtGraph launcher
├── createRangeBars/              # Range bar creation tools
└── archive/                      # Archived code versions
```

## Module Descriptions

### GUI Components

#### **OMtree_gui.py**
Main application window with tabbed interface:
- Data & Fields Tab: Load data, select features
- Model Tester Tab: Configure and run models
- Performance Stats Tab: View backtest results
- Tree Visualizer Tab: Visualize decision trees
- PermuteAlpha Tab: Feature importance analysis
- Regression Analysis Tab: Statistical analysis

#### **src/trading/visualization/**

##### **pyqtgraph_range_bars_final.py**
High-performance candlestick chart implementation:
- Auto Y-axis scaling on zoom
- Dynamic data loading
- Multi-monitor DPI awareness
- Enhanced time axis labeling
- Hover display with timestamps
- Integration with trade panel

##### **enhanced_trade_panel.py**
Advanced trade list panel with:
- **P&L displayed as percentages** (properly compounded)
- **Sortable columns** (click headers to sort)
- Trade summary statistics
- Win rate calculation
- Commission tracking
- Execution lag display

##### **trade_panel.py**
Base trade panel implementation:
- Trade list display
- Trade source selection
- Trade navigation
- Basic trade metrics

##### **simple_white_x_trades.py**
Trade visualization overlay:
- White X marks for trades
- Entry/exit indicators
- Trade highlighting

### Core Trading System

#### **src/OMtree_model.py**
Decision tree model implementation:
- Random forest classifier
- Feature importance calculation
- Model persistence
- Prediction generation

#### **src/OMtree_preprocessing.py**
Data preprocessing pipeline:
- Feature engineering
- Technical indicators
- Data normalization
- Missing value handling

#### **src/OMtree_validation.py**
Model validation and backtesting:
- Out-of-sample testing
- Performance metrics
- Trade signal generation
- P&L calculation

#### **src/OMtree_walkforward.py**
Walk-forward optimization:
- Rolling window analysis
- Parameter optimization
- Performance tracking
- Result aggregation

### Data Management

#### **src/config_manager.py**
Configuration file management:
- INI file parsing
- Parameter validation
- Default values
- Configuration persistence

#### **src/performance_stats.py**
Performance metrics calculation:
- Sharpe ratio
- Win rate
- Maximum drawdown
- Return statistics

### Recent Fixes (2025-09-24)

#### **Fixed: Total P&L Calculation**
- **Issue**: Total P&L was incorrectly summing individual percentages
- **Solution**: Implemented proper return compounding formula
- **Location**: `enhanced_trade_panel.py` lines 52-63, 359-370
- **Formula**: `(1 + r1) * (1 + r2) * ... * (1 + rn) - 1`

#### **Added: Column Sorting**
- **Feature**: Click column headers to sort trade list
- **Implementation**: Added sorting handlers to all columns
- **Location**: `enhanced_trade_panel.py` lines 220-490
- **Columns**: Trade #, DateTime, Type, Price, Size, P&L %, Cum P&L %, Bar #

## Key Features

### 1. Machine Learning Models
- Random Forest decision trees
- Walk-forward optimization
- Feature importance analysis
- Out-of-sample validation

### 2. Advanced Visualization
- PyQtGraph-based candlestick charts
- Real-time trade overlays
- Performance metrics display
- Multi-monitor support

### 3. Trade Analysis
- Percentage-based P&L tracking
- Properly compounded returns
- Commission and slippage modeling
- Execution lag tracking

### 4. Data Processing
- Range bar generation
- Technical indicator calculation
- Feature engineering
- Data normalization

## Configuration Files

### **OMtree_config.ini**
Main configuration file containing:
- Model parameters
- Data paths
- Feature selections
- Backtest settings

### **config.yaml** (if present in tradingCode/)
Trading system configuration:
- Execution parameters
- Signal lag settings
- Commission rates
- Strategy settings

## Usage Workflow

1. **Data Loading**
   - Load CSV data via Data & Fields tab
   - Select features for model training
   - Configure preprocessing options

2. **Model Training**
   - Set model parameters in Config tab
   - Run validation or walk-forward analysis
   - Review performance metrics

3. **Visualization**
   - Launch PyQtGraph charts
   - Load trade results
   - Analyze with sortable trade list
   - Review P&L statistics

4. **Analysis**
   - Use Tree Visualizer for model insights
   - Run regression analysis
   - Perform feature importance studies

## Testing

Run tests with:
```bash
# Test the enhanced trade panel
python test_profit_calculation.py

# Launch the unified system
python launch_unified_system.py

# Test specific components
python src/trading/visualization/pyqtgraph_range_bars_final.py
```

## Dependencies

- PyQt5: GUI framework
- pyqtgraph: High-performance charting
- pandas: Data manipulation
- numpy: Numerical computing
- scikit-learn: Machine learning
- matplotlib: Additional plotting
- PIL: Image processing

## Performance Considerations

- PyQtGraph used for real-time chart updates
- Vectorized operations for data processing
- Lazy loading for large datasets
- Efficient memory management
- Multi-threaded execution where applicable

## Known Issues & Limitations

- Large datasets (>100k bars) may impact chart performance
- Some legacy code in archive/ folder not maintained
- Windows-specific paths in some configurations

## Future Enhancements

- [ ] Add more trading strategies
- [ ] Implement real-time data feeds
- [ ] Add portfolio optimization
- [ ] Enhance risk management tools
- [ ] Add cloud storage integration