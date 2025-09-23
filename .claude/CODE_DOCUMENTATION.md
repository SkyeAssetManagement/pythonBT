# CODE DOCUMENTATION - PyQtGraph Trading Visualization System

## Overview
Comprehensive trading visualization system with PyQtGraph for high-performance charting of range bar data, supporting 377,690+ bars with real-time updates and trade analysis.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                  PyQtGraph Trading Visualization                 │
│                  (launch_pyqtgraph_with_selector.py)            │
├─────────────────────────────────────────────────────────────────┤
│  Chart View │ Trade Panel │ Strategy Runner │ Data Selector     │
└────────┬───────┴──────┬───────┴──────┬──────────┴────┬─────────┘
         │              │              │               │
    ┌────▼────┐    ┌────▼────┐   ┌────▼────┐    ┌────▼────┐
    │Range Bar│    │Trade    │   │Strategy │    │Data     │
    │Chart    │    │List     │   │Execution│    │Loader   │
    └─────────┘    └─────────┘   └─────────┘    └─────────┘
         │              │              │               │
         └──────────────┴──────────────┴───────────────┘
                              │
                   ┌──────────▼──────────┐
                   │   Unified Data      │
                   │   Pipeline          │
                   │  (377,690 bars)     │
                   └─────────────────────┘
```

## Directory Structure

```
C:\code\PythonBT\
├── .claude\
│   ├── CLAUDE.md                    # Development standards
│   ├── CODE_DOCUMENTATION.md        # This file
│   └── projecttodos.md             # Task tracking
├── src\
│   └── trading\
│       ├── visualization\
│       │   ├── pyqtgraph_range_bars_final.py  # Main chart (377k+ bars)
│       │   ├── trade_panel.py                 # Trade list with DateTime
│       │   ├── simple_white_x_trades.py       # Trade markers (bold, 18px)
│       │   ├── strategy_runner.py             # Strategy execution
│       │   └── csv_trade_loader.py           # Trade import
│       └── strategies\
│           ├── base.py                       # Base strategy (lag=0)
│           ├── sma_crossover.py              # SMA strategy
│           └── rsi_momentum.py               # RSI strategy
├── dataRaw\
│   └── range-ATR30x0.05\
│       └── ES\
│           └── diffAdjusted\
│               └── ES-DIFF-range-ATR30x0.05-dailyATR.csv  # 377,690 bars
├── launch_pyqtgraph_with_selector.py    # Main entry point
├── pyqtgraph_data_selector.py           # Data selection dialog
└── integrated_trading_launcher.py       # Alternative launcher
```

## Critical Fixes Implemented (2025-09-22)

### 1. Timestamp Display Issue - FIXED ✅
**Problem**: All timestamps showed "00:00:00" on x-axis and trade list
**Root Cause**: Column mapping renamed 'Date' to 'DateTime' before combining with 'Time'
**Solution**:
- Removed 'Date' from column_mapping dictionary (line 167)
- Now properly combines Date+Time before any renaming
- Timestamps display correctly: "2021-01-03 17:01:00"

### 2. ViewBox Hardcoded Limit - FIXED ✅
**Problem**: Could not pan past June 15, 2023 (bar 199,628)
**Root Cause**: Hardcoded xMax=200000 in ViewBox limits
**Solution**:
- Removed hardcoded limits
- Set limits dynamically: xMax = total_bars + 100
- Now supports all 377,690 bars (2021-2025 data)

### 3. Chart Navigation - FIXED ✅
**Problem**: Chart jumped back to previous position after jump_to_trade
**Solution**:
- Temporarily disable range handlers during jump
- Use QTimer to re-enable after 100ms
- Prevents reversion to previous position

### 4. Trade Marker Visibility - ENHANCED ✅
**Changes**:
- Size increased 25% (14px → 18px)
- Bold white pen (width=3)
- Z-value set to 1000 (renders on top)
- Pure white color for maximum visibility

### 5. SMA/Indicator Rendering - FIXED ✅
**Problem**: Indicators didn't render when jumping ahead
**Solution**:
- Improved visible range calculation
- Proper z-ordering (indicators=500, trades=1000)
- Line width increased to 2 for visibility

### 6. Trade Data Display - ENHANCED ✅
**Added to hover info**:
- DateTime in format YYYY-MM-DD HH:MM:SS
- Bar number
- Complete trade details

### 7. Initial View - IMPROVED ✅
**Change**: Chart now opens showing last 500 bars instead of first 500
**Benefit**: Recent data immediately visible

## Key Components

### PyQtGraph Range Bar Chart (`pyqtgraph_range_bars_final.py`)
**Features**:
- Dynamic data loading for 377,690+ bars
- Real-time timestamp labels
- Trade overlay visualization
- Indicator overlays (SMA, RSI)
- Adaptive candle spacing
- Multi-monitor DPI awareness

**Key Methods**:
- `load_data()`: Load CSV/Parquet data
- `render_range(start, end)`: Render specific range
- `format_time_axis()`: Format DateTime labels
- `jump_to_trade(trade)`: Navigate to trade
- `on_x_range_changed()`: Handle panning

### Trade Panel (`trade_panel.py`)
**Features**:
- Sortable trade list
- DateTime display for all trades
- Color-coded trade types
- Double-click to jump to trade
- PyQt signal type fixed (object)

### Data Selector (`pyqtgraph_data_selector.py`)
**Features**:
- File browser for data selection
- Trade source options (None/CSV/System)
- Strategy selection
- Indicator checkboxes

## Data Pipeline

### CSV Loading
```
CSV File → pd.read_csv() →
Combine Date+Time → pd.to_datetime() →
Convert to numpy arrays → full_data dict →
Set dynamic ViewBox limits → Render
```

### Trade Generation
```
Chart Data → Strategy.generate_signals() →
signals_to_trades(lag=0) → TradeCollection →
Trade Panel Display with DateTime
```

## Performance Optimizations

### Rendering
- Downsampling when >2000 bars visible
- Incremental indicator rendering
- ScatterPlotItem for trades (fast)
- is_rendering flag prevents overlap

### Data Management
- Float32 arrays for memory efficiency
- Vectorized NumPy operations
- Dynamic ViewBox limits

## Configuration

### Dynamic ViewBox Limits
```python
viewBox.setLimits(
    xMin=-100,
    xMax=self.total_bars + 100,    # Dynamic
    yMin=0,
    yMax=100000,
    minXRange=10,
    maxXRange=self.total_bars + 1000,
    minYRange=1,
    maxYRange=50000
)
```

### Signal Bar Lag
- Currently set to: **0** (trades execute on signal bar)
- Location: `src/trading/strategies/base.py`

## Testing Verification

### Data Access Tests
- ✅ All 377,690 bars accessible
- ✅ June 15, 2023 at bar 199,628
- ✅ Can pan to any date (2021-2025)
- ✅ Jump to trades at any position

### Display Tests
- ✅ Timestamps show correct time
- ✅ Trade list shows DateTime
- ✅ Indicators render at all positions
- ✅ Trade markers visible on top

## Known Issues - ALL RESOLVED

Previously fixed issues:
- ~~Timestamps showing 00:00:00~~ → Fixed with column mapping
- ~~Cannot pan past June 2023~~ → Fixed with dynamic limits
- ~~Chart jumps back after navigation~~ → Fixed with handler management
- ~~Indicators not rendering~~ → Fixed with range calculation
- ~~Trade markers hard to see~~ → Fixed with size/color/z-order

## Usage

### Basic Launch
```bash
python launch_pyqtgraph_with_selector.py
```

### Workflow
1. Select data file (CSV/Parquet)
2. Choose trade source
3. Select indicators
4. Chart opens with last 500 bars
5. Pan/zoom with mouse
6. Double-click trades to jump

### Controls
- Mouse wheel: Zoom
- Click+drag: Pan
- Hover: See price/time/trade info
- Trade list: Double-click to jump

---

*Last Updated: 2025-09-22*
*Version: 2.1.0 - All Critical Issues Resolved*