# CODE DOCUMENTATION - PythonBT Trading System

## Overview
Advanced trading system with two main components: PyQtGraph visualization for range bar charting and OMTree machine learning for trade signal generation.

## Architecture

```
PythonBT Trading System
├── PyQtGraph Visualization (377k+ bars support)
│   ├── Real-time charting with trade overlays
│   ├── Strategy execution & backtesting
│   └── Technical indicators (SMA, RSI)
└── OMTree ML Pipeline
    ├── Decision tree classification
    ├── Walk-forward validation
    └── Feature engineering
```

## Directory Structure

```
C:\code\PythonBT\
├── .claude\                              # Project documentation
│   ├── CLAUDE.md                        # Development standards
│   ├── CODE_DOCUMENTATION.md           # This file
│   └── projecttodos.md                 # Task tracking
│
├── src\trading\                         # PyQtGraph components
│   ├── visualization\
│   │   ├── pyqtgraph_range_bars_final.py   # Main chart (377k bars)
│   │   ├── trade_panel.py                  # Trade list display
│   │   ├── simple_white_x_trades.py        # Trade markers
│   │   ├── strategy_runner.py              # Strategy execution
│   │   └── csv_trade_loader.py            # Trade import
│   └── strategies\
│       ├── base.py                         # Base strategy (lag=0)
│       ├── sma_crossover.py               # SMA strategy
│       └── rsi_momentum.py                # RSI strategy
│
├── OMTree ML Components\
│   ├── OMtree_gui.py                      # ML model GUI
│   ├── OMtree_walkforward.py              # Walk-forward validation
│   ├── OMtree_model.py                    # Decision tree model
│   ├── OMtree_preprocessing.py            # Feature engineering
│   ├── OMtree_validation.py               # Model validation
│   └── OMtree_config.ini                  # Configuration
│
├── Launch Scripts\
│   ├── launch_pyqtgraph_with_selector.py  # Main PyQtGraph launcher
│   ├── launch_trading_dashboard.py        # Trading dashboard
│   ├── launch_pyqtgraph_chart.py         # Direct chart launch
│   └── pyqtgraph_data_selector.py        # Data selection dialog
│
└── dataRaw\                              # Market data
    └── range-ATR30x0.05\ES\               # 377,690 range bars
```

## Component Details

### PyQtGraph Visualization System

#### Core Features
- **Data Capacity**: Handles 377,690+ bars (2021-2025)
- **Dynamic Loading**: Efficient rendering with downsampling
- **Real-time Updates**: Live chart updates with trade execution
- **Multi-monitor Support**: DPI-aware rendering

#### Key Fixes (2025-09-22)
1. **Timestamp Display**: Fixed DateTime column mapping
2. **ViewBox Limits**: Removed 200k hardcoded limit
3. **Navigation**: Fixed jump_to_trade position retention
4. **Trade Markers**: Enhanced visibility (18px, bold white)
5. **Indicator Rendering**: Fixed rendering across all ranges
6. **Initial View**: Opens with last 500 bars

#### Main Classes
- `RangeBarChart`: Core chart widget with candle rendering
- `TradePanel`: Trade list with DateTime display
- `DataSelector`: File browser for data/trade selection
- `StrategyRunner`: Executes trading strategies

### OMTree Machine Learning Pipeline

#### Core Components
- **Decision Tree Model**: Custom implementation with bootstrapping
- **Walk-forward Validation**: Time-series aware validation
- **Feature Engineering**: 100+ technical indicators
- **GUI Interface**: Real-time model training/testing

#### Key Parameters
- Bootstrap fraction: 0.5-0.67
- Min leaf fraction: 0.5-1.0%
- Vote threshold: 70-80%
- Target threshold: 0.4-0.6%

#### Processing Pipeline
```
Raw Data → Feature Engineering →
Normalization → Tree Training →
Signal Generation → Validation
```

## Data Flow

### PyQtGraph Visualization
```
CSV/Parquet Data → DataFrame →
NumPy Arrays → Chart Rendering →
Trade Overlay → Interactive Display
```

### OMTree ML Pipeline
```
Market Data → Feature Extraction →
Model Training → Signal Generation →
Walk-forward Validation → Performance Metrics
```

## Configuration Files

### OMtree_config.ini
- Feature selection
- Model parameters
- Data paths
- Processing options

### Strategy Parameters
- Signal lag: 0 (execute on signal bar)
- SMA periods: Configurable
- RSI thresholds: 30/70 default

## Performance Optimizations

### Chart Rendering
- Downsampling for >2000 visible bars
- Float32 arrays for memory efficiency
- Incremental indicator updates
- Z-ordering for proper layering

### ML Processing
- Vectorized NumPy operations
- Parallel tree training
- Cached feature calculations
- Memory-mapped data access

## Testing & Validation

### Chart System Tests
- All 377,690 bars accessible
- Timestamp display accuracy
- Trade navigation reliability
- Indicator rendering consistency

### ML Model Tests
- Walk-forward validation results
- Feature importance analysis
- Performance metrics tracking
- Parameter sensitivity testing

## Usage Examples

### Launch PyQtGraph Chart
```bash
python launch_pyqtgraph_with_selector.py
```

### Run OMTree GUI
```bash
python OMtree_gui.py
```

### Walk-forward Validation
```bash
python OMtree_walkforward.py
```

## Key Achievements

### PyQtGraph System
- Full 377k+ bar support without limits
- Accurate timestamp display throughout
- Smooth navigation and trade jumping
- Clear indicator and trade visualization

### OMTree System
- Robust decision tree implementation
- Comprehensive feature engineering
- Time-series aware validation
- GUI for interactive model development

---

*Version: 3.0.0*
*Last Updated: 2025-09-23*
*Status: Production Ready*