# CODE DOCUMENTATION - Unified Trading System

## Overview
This document provides comprehensive documentation for the Unified Trading System, which merges:
- **OMtree**: Machine learning decision tree trading system with walk-forward validation
- **ABtoPython**: VectorBT integration with PyQtGraph visualization and trade analysis
- **PyQtGraph Range Bar Visualization**: High-performance charting with 122,609+ bar support

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    Unified Trading System GUI                    │
│                  (integrated_trading_launcher.py)                │
├─────────────────────────────────────────────────────────────────┤
│  Data Viz │ Walk-Forward │ Performance │ Trade List │ Strategy  │
└────────┬───────┴──────┬───────┴──────┬──────┴─────┬─────┴───┬───┘
         │              │              │             │         │
    ┌────▼────┐    ┌────▼────┐   ┌────▼────┐  ┌────▼────┐ ┌──▼──┐
    │PyQtGraph│    │OMtree   │   │Stats    │  │Trade    │ │Strat│
    │Range Bar│    │Model    │   │Module   │  │Panel    │ │Runner│
    └─────────┘    └─────────┘   └─────────┘  └─────────┘ └─────┘
                        │                            │
                   ┌────▼─────────────────────────┐  │
                   │  Unified Data Pipeline       │──┘
                   │  (ES-DIFF Range Bars)        │
                   └───────────────────────────────┘
                              │
                   ┌──────────▼──────────┐
                   │   Feature Flags     │
                   │  (feature_flags.py) │
                   └─────────────────────┘
```

## Directory Structure

```
C:\code\PythonBT\
├── .claude\                      # Project management & docs
│   ├── CLAUDE.md                 # Development standards
│   ├── CODE_DOCUMENTATION.md     # This file
│   └── projecttodos.md          # Cleared - ready for new tasks
├── src\                          # Source code
│   ├── trading\                  # Trading system components
│   │   ├── data\                 # Data structures & pipeline
│   │   │   ├── __init__.py
│   │   │   ├── trade_data.py     # Core trade data structures
│   │   │   └── unified_pipeline.py  # Data format conversion
│   │   ├── visualization\        # Visualization components
│   │   │   ├── __init__.py
│   │   │   ├── pyqtgraph_range_bars_final.py  # Main chart (122k+ bars)
│   │   │   ├── trade_panel.py    # Trade list with DateTime
│   │   │   ├── strategy_runner.py # Strategy execution
│   │   │   └── trade_visualization.py # Trade markers on chart
│   │   ├── strategies\           # Trading strategies
│   │   │   ├── __init__.py
│   │   │   ├── base.py           # Base strategy class
│   │   │   ├── sma_crossover.py  # SMA crossover strategy
│   │   │   └── rsi_momentum.py   # RSI momentum strategy
│   │   └── integration\          # External integrations
│   │       ├── __init__.py
│   │       └── vbt_loader.py     # VectorBT integration
├── parquetData\                  # Data files
│   └── range\                    # Range bar data
│       └── ATR30x0.1\
│           └── ES-DIFF-range-ATR30x0.1-amibroker.parquet  # 122,609 bars
├── integrated_trading_launcher.py  # Main application entry
├── launch_integrated_system.bat    # Windows launcher
├── requirements.txt              # Python dependencies
└── README.md                     # Project overview
```

## Recent Critical Fixes (2025-09-22)

### 1. X-Axis DateTime Labels - FIXED ✓
**Problem**: X-axis showed "0:00:00" for all timestamps
**Solution**:
- Fixed x-coordinate mapping in `format_time_axis()` (lines 305-389)
- Added debug logging to trace timestamp processing
- Properly maps bar indices to timestamp labels

**Files Modified**:
- `src/trading/visualization/pyqtgraph_range_bars_final.py`

### 2. Trade List DateTime Display - FIXED ✓
**Problem**: Trade list showed "-" for all DateTime values
**Solution**:
- Verified DataFrame includes DateTime column in StrategyRunner
- Strategy base class extracts DateTime from DataFrame
- TradeData objects receive timestamp parameter

**Files Modified**:
- `src/trading/strategies/base.py` (DateTime extraction)
- `src/trading/visualization/strategy_runner.py` (DataFrame creation)

### 3. Dynamic Data Loading (122,609 bars) - FIXED ✓
**Problem**: Chart appeared to stop loading data when panning
**Solution**:
- Increased ViewBox limits from 100,000 to 200,000 bars (lines 176-184)
- Added `on_x_range_changed()` handler for dynamic loading (lines 514-531)
- Fixed `is_rendering` flag management to prevent stuck state
- Fixed `.items()` method call syntax

**Files Modified**:
- `src/trading/visualization/pyqtgraph_range_bars_final.py`

## Module Documentation

### Core Visualization Components

#### 1. PyQtGraph Range Bar Chart (`pyqtgraph_range_bars_final.py`)
**Purpose**: High-performance charting for 122,609+ range bars

**Key Classes**:
- `RangeBarChartFinal`: Main chart window
  - `load_data()`: Load ES-DIFF range bar parquet data
  - `render_range(start, end)`: Render specific bar range
  - `format_time_axis()`: Format DateTime labels on x-axis
  - `on_x_range_changed()`: Handle dynamic data loading on pan
  - `jump_to_trade()`: Navigate to specific trade location

**Key Features**:
- Supports 122,609+ bars with dynamic loading
- Real-time DateTime labels on x-axis
- Trade overlay visualization
- Indicator overlays (SMA, RSI)
- Adaptive candle spacing based on zoom
- Multi-monitor DPI awareness

#### 2. Trade Panel (`trade_panel.py`)
**Purpose**: Display and manage trade list with filtering

**Key Classes**:
- `TradePanel`: Main trade panel widget
  - `load_trades()`: Load trades into table
  - `on_trade_double_clicked()`: Jump to trade on chart
  - Signal: `trade_selected` - Emitted for chart navigation

- `TradeTableModel`: QTableModel for trade display
  - Columns: ID, DateTime, Type, Price, Size, P&L, Strategy

#### 3. Strategy Runner (`strategy_runner.py`)
**Purpose**: Execute trading strategies on chart data

**Key Classes**:
- `StrategyRunner`: Strategy execution widget
  - `set_chart_data()`: Load OHLCV + DateTime data
  - `run_strategy()`: Execute selected strategy
  - Signals: `trades_generated`, `indicators_calculated`

**Supported Strategies**:
- SMA Crossover (configurable periods)
- RSI Momentum (configurable levels)

### Trading Strategies

#### Base Strategy (`strategies/base.py`)
**Purpose**: Abstract base class for all strategies

**Key Methods**:
- `generate_signals(df)`: Generate buy/sell signals
- `signals_to_trades(signals, df)`: Convert signals to trades
  - Extracts DateTime from DataFrame
  - Calculates P&L for exit trades
  - Creates TradeData objects with timestamps

### Data Pipeline

#### Trade Data (`data/trade_data.py`)
**Purpose**: Core trade representation

**Key Classes**:
- `TradeData`: Individual trade
  - Required: bar_index, trade_type, price
  - Optional: trade_id, timestamp, size, pnl, strategy, symbol

- `TradeCollection`: Collection of trades
  - `add_trade()`, `remove_trade()`
  - `get_by_date_range()`: Filter by date
  - `calculate_total_pnl()`: Total P&L
  - `to_dataframe()`: Export to pandas

## Data Flow

### 1. Chart Data Loading
```
ES-DIFF Parquet (122,609 bars) → load_data() →
full_data dict → render_range() → PyQtGraph Display
```

### 2. Strategy Execution
```
Chart Data → StrategyRunner → DataFrame with DateTime →
Strategy.generate_signals() → signals_to_trades() →
TradeCollection → Trade Panel Display
```

### 3. Dynamic Rendering Pipeline
```
User Pans → on_x_range_changed() → Calculate visible range →
render_range() → Slice data → Update candlesticks →
format_time_axis() → Update DateTime labels
```

## Configuration

### ViewBox Limits (Supporting 122,609+ bars)
```python
viewBox.setLimits(
    xMin=-100,
    xMax=200000,      # Supports up to 200k bars
    yMin=0,
    yMax=100000,
    minXRange=10,     # Min 10 bars visible
    maxXRange=150000, # Max 150k bars visible
    minYRange=1,
    maxYRange=50000
)
```

## Performance Optimizations

### 1. Rendering
- Downsampling when >2000 bars visible
- Incremental indicator rendering
- ScatterPlotItem for trade marks (fast for thousands of trades)
- OpenGL disabled for stability

### 2. Data Management
- Float32 arrays for memory efficiency
- Vectorized operations with NumPy
- Lazy loading of indicators

### 3. GUI Responsiveness
- `is_rendering` flag prevents overlapping renders
- Signal throttling with SignalProxy
- Separate threads for data loading

## Testing

### Test Scripts Created
- `test_large_dataset.py`: Generate 10k bar test data
- `test_debug_logging.py`: Verify debug infrastructure
- `test_range_bars_direct.py`: Test rendering at all positions
- `test_all_fixes.py`: Comprehensive fix verification
- `test_panning_edge.py`: Test edge cases

### Verified Working
- ✓ Renders all 122,609 bars
- ✓ DateTime labels show real timestamps
- ✓ Trades have DateTime values
- ✓ Can pan to any position (0-122,609)
- ✓ Jump-to-trade works at all positions

## Debug Infrastructure

### Debug Logging Added
```python
[RENDER_RANGE] Called with start=122109, end=122609
[FORMAT_TIME_AXIS] Timestamps type: <class 'pandas.core.series.Series'>
[STRATEGY] DataFrame columns: ['DateTime', 'Open', 'High', 'Low', 'Close']
[ON_X_RANGE_CHANGED] X range changed: (0.0, 500.0)
[JUMP_TO_TRADE] Called for trade: BUY, bar_index=5000
```

## Known Issues & Solutions

### Issue: "Skipping - already rendering"
**Cause**: Multiple pan events while rendering
**Solution**: This is normal behavior - prevents overlapping renders

### Issue: Trade DateTime shows "-"
**Cause**: Missing timestamp in TradeData
**Solution**: Fixed - DataFrame now includes DateTime column

### Issue: Can't pan past certain point
**Cause**: ViewBox xMax limit too low
**Solution**: Fixed - increased to 200,000 bars

## Future Enhancements

1. Real-time data feed integration
2. Additional ML models integration
3. Advanced risk management tools
4. Multi-timeframe support
5. Cloud deployment capabilities

## Development Workflow

### Adding New Features
1. Create feature flag if needed
2. Implement with proper error handling
3. Add debug logging
4. Write tests
5. Update documentation
6. Test with full 122k bar dataset

### Commit Guidelines
- Branch from `main` or `merge-with-ABToPython`
- Clear, descriptive commit messages
- Reference issue numbers
- Update CODE_DOCUMENTATION.md

---

Last Updated: 2025-09-22
Version: 2.0.0 (PyQtGraph Integration Complete)