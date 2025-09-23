# CODE DOCUMENTATION - PythonBT Trading System

## Project Overview
PythonBT is a comprehensive trading system visualization and backtesting platform with PyQtGraph-based charting for handling large datasets (377,690+ bars) efficiently.

## Current Architecture (2025-09-23)

### Directory Structure
```
C:\code\PythonBT\
├── .claude\
│   ├── CLAUDE.md                    # Development standards
│   └── projectToDos.md             # Project task tracking
├── src\trading\
│   ├── core\                       # NEW: Unified execution engine
│   │   ├── standalone_execution.py # Execution engine with lag/formulas
│   │   ├── trade_types.py         # Enhanced trade data structures
│   │   └── strategy_runner_adapter.py # Adapter for legacy/unified
│   ├── visualization\
│   │   ├── pyqtgraph_range_bars_final.py  # Main chart (DO NOT MODIFY)
│   │   ├── trade_panel.py                 # Trade list panel
│   │   ├── enhanced_trade_panel.py        # NEW: P&L % display
│   │   ├── simple_white_x_trades.py       # Trade markers
│   │   └── csv_trade_loader.py           # CSV import
│   └── strategies\
│       ├── base.py                        # Base strategy class
│       ├── enhanced_base.py              # Enhanced with lag support
│       ├── strategy_wrapper.py           # NEW: Metadata wrapper
│       ├── sma_crossover.py             # SMA strategy
│       └── rsi_momentum.py              # RSI strategy
├── tradingCode\
│   └── config.yaml                       # Execution configuration
├── dataRaw\                             # ES futures range bar data
├── launch_pyqtgraph_with_selector.py   # Original launcher (stable)
├── launch_unified_system.py            # NEW: Unified launcher
├── test_execution_engine.py            # Test unified engine
└── test_side_by_side_comparison.py     # Compare old vs new
```

## Key Components

### 1. Chart Rendering (PRESERVED - DO NOT MODIFY)
**File**: `src/trading/visualization/pyqtgraph_range_bars_final.py`
- `RangeBarChartFinal` class with sequential `render_range()` method
- `EnhancedCandlestickItem` with 5 args: x, opens, highs, lows, closes
- Handles 377,690 bars efficiently with viewport rendering
- Working hover data display with OHLC values
- Time axis formatting with proper timestamps

### 2. Unified Execution Engine (NEW)
**File**: `src/trading/core/standalone_execution.py`
```python
class StandaloneExecutionEngine:
    - Signal lag: N bars between signal and execution
    - Price formulas: "(H+L+C)/3", "(H+L)/2", etc.
    - Position sizing: value, amount, or percentage
    - Friction costs: fees, slippage
```

**Configuration** (`tradingCode/config.yaml`):
```yaml
use_unified_engine: true  # Enable new engine
backtest:
  signal_lag: 1           # Bars between signal and execution
  execution_price: "formula"
  buy_execution_formula: "(H + L + C) / 3"
  sell_execution_formula: "(H + L + C) / 3"
```

### 3. Trade Data Structures
**File**: `src/trading/core/trade_types.py`
- `TradeRecord`: Enhanced trade with signal_bar, execution_bar, lag, P&L %
- `TradeRecordCollection`: Collection with metrics calculation
- Compatible with existing UI while adding percentage P&L

### 4. Strategy System
**Base Strategies**:
- `SMACrossoverStrategy`: Simple moving average crossover
- `RSIMomentumStrategy`: RSI with oversold/overbought levels

**Enhanced Features**:
- `StrategyWrapper`: Adds metadata and indicator definitions
- `StrategyFactory`: Easy instantiation with config

### 5. Launchers

**Original** (`launch_pyqtgraph_with_selector.py`):
- Stable, working version
- Uses inline trade generation
- No lag implementation

**Unified** (`launch_unified_system.py`):
- Uses new execution engine
- Proper lag implementation (1 bar default)
- Formula-based execution prices
- P&L as percentages
- Issues: Hover data and trade loading need fixes

## Data Flow

### Trade Execution with Lag
1. **Signal Generation** at bar N
2. **Lag Application**: Wait signal_lag bars (default 1)
3. **Execution** at bar N+lag
4. **Price Calculation**: Use formula or fixed price type
5. **P&L Tracking**: Calculate as percentage

### Chart Data Structure
```python
full_data = {
    'timestamp': array,  # Required for chart
    'open': array,      # Lowercase for chart
    'high': array,
    'low': array,
    'close': array,
    'volume': array,
    'DateTime': array   # For compatibility
}
```

## Testing

### Verification Tests
- `test_execution_engine.py`: Tests lag and formulas
- `test_side_by_side_comparison.py`: Compares old vs new
- `test_unified_integration.py`: Tests with real ES data

### Results
- ✅ Lag calculations work (signal bar 5 → execution bar 6)
- ✅ Price formulas evaluate correctly
- ✅ P&L tracked as percentages
- ⚠️ Chart hover data needs fix in unified launcher
- ⚠️ Trade loading integration incomplete

## Known Issues

### Unified Launcher
1. Hover data not displaying (data structure mismatch)
2. Trades not loading when "System" selected
3. Need to ensure `self.current_x_range` initialized

### Solutions in Progress
- Ensure `timestamp` key in full_data
- Initialize chart attributes before parent init
- Convert data properly for strategy execution

## Usage

### Running the System
```bash
# Stable original version
python launch_pyqtgraph_with_selector.py

# New unified version (partial functionality)
python launch_unified_system.py
```

### Enabling Unified Engine
Edit `tradingCode/config.yaml`:
```yaml
use_unified_engine: true
execution_price: "formula"
signal_lag: 1
```

## Performance
- Handles 377,690 ES futures bars
- Smooth pan/zoom with viewport rendering
- Memory efficient numpy arrays
- Optimized trade markers with ScatterPlotItem

---
*Last Updated: 2025-09-23*
*Version: 2.1.0 - Unified Engine Partially Integrated*