# CODE DOCUMENTATION - PythonBT Trading System

## Project Overview
PythonBT is a comprehensive backtesting and trading visualization platform built with PyQt5 and PyQtGraph. Features real-time charting with range bars, unified execution engine for realistic trade simulation, and support for massive datasets (6M+ bars).

## System Architecture

### Core Directory Structure
```
PythonBT/
├── Launch Scripts
│   ├── launch_unified_system.py             # Primary launcher with unified engine
│   ├── launch_pyqtgraph_with_selector.py    # Legacy launcher with data selector
│   └── integrated_trading_launcher.py       # Alternative launcher
│
├── src/trading/
│   ├── visualization/                       # UI Components
│   │   ├── pyqtgraph_range_bars_final.py   # Main chart (RangeBarChartFinal)
│   │   ├── enhanced_trade_panel.py         # Trade panel with % P&L display
│   │   ├── strategy_runner.py              # Strategy execution UI
│   │   ├── pyqtgraph_data_selector.py      # Data source selection dialog
│   │   └── trade_panel.py                  # Legacy trade panel
│   │
│   ├── strategies/                         # Trading Strategies
│   │   ├── base.py                         # TradingStrategy abstract base
│   │   ├── sma_crossover.py               # SMA crossover implementation
│   │   ├── rsi_momentum.py                # RSI momentum strategy
│   │   ├── enhanced_sma_crossover.py      # SMA with signal lag support
│   │   └── strategy_wrapper.py            # StrategyFactory and wrappers
│   │
│   ├── core/                              # Execution Engine
│   │   ├── standalone_execution.py        # ExecutionEngine with lag/formulas
│   │   ├── strategy_runner_adapter.py     # Adapter for legacy/unified routing
│   │   └── trade_types.py                 # TradeRecord, TradeRecordCollection
│   │
│   └── data/                              # Data Management
│       ├── trade_data.py                  # TradeData, TradeCollection
│       └── csv_trade_loader.py            # CSV trade import
│
├── tradingCode/
│   └── config.yaml                        # System configuration
│
└── Data/                                  # Data files (parquet/csv)
```

## Critical Design Patterns

### 1. Unified Execution Engine
The system supports two execution modes controlled by `config.yaml`:

```yaml
use_unified_engine: true    # Enable realistic execution
backtest:
  signal_lag: 1            # Bars between signal and execution
  execution_price: "formula"
  buy_execution_formula: "(H + L + C) / 3"
  sell_execution_formula: "(H + L + C) / 3"
  fees: 2.0               # Commission per trade
  slippage: 0.25          # Slippage per trade
```

**Execution Flow:**
```
Signal Generation → Lag Application → Price Formula → Commission → P&L Calculation
       ↓                    ↓              ↓             ↓              ↓
   Bar N detected      Bar N+lag      Custom price    Deduct fees   % based on $1
```

### 2. P&L Calculation System
All P&L calculations are normalized to $1 invested for consistent percentage returns:

```python
# Long position P&L
pnl_percent = ((exit_price / entry_price) - 1) * 100

# Short position P&L
pnl_percent = (1 - (exit_price / entry_price)) * 100

# Commission deduction
pnl_percent -= (commission / entry_price) * 100
```

### 3. Data Structure Requirements
Chart components expect specific data formats:

```python
# Chart data dictionary (lowercase keys required)
full_data = {
    'timestamp': np.array,  # Required for hover
    'open': np.array,       # Must be lowercase
    'high': np.array,
    'low': np.array,
    'close': np.array,
    'volume': np.array,     # Optional
    'aux1': np.array,       # ATR values
    'aux2': np.array        # Range multiplier
}
```

### 4. Trade Type Conversion
System handles conversion between unified and legacy trade formats:

```python
TradeRecordCollection → to_legacy_collection() → TradeCollection
        ↓                                              ↓
  (Unified format)                              (Legacy format)
  Has lag tracking                              Compatible with UI
  % P&L built-in                                Requires conversion
```

## Key Components

### Chart System (pyqtgraph_range_bars_final.py)
- **RangeBarChartFinal**: Main chart widget with viewport optimization
- **Features**:
  - Range bar rendering with ATR-based sizing
  - Hover data display (OHLC, volume, ATR)
  - Trade markers as white X symbols
  - Crosshair tracking
  - Viewport-based rendering for performance

### Trade Panel (enhanced_trade_panel.py)
- **EnhancedTradeListPanel**: Displays trades with percentage P&L
- **Features**:
  - P&L shown as percentage to 2 decimal places
  - Cumulative P&L tracking
  - Backtest summary with:
    - Total trades count
    - Win rate percentage
    - Total/Average P&L percentages
    - Commission totals
    - Average execution lag

### Strategy Runner (strategy_runner.py)
- **StrategyRunner**: UI for executing trading strategies
- **Features**:
  - Dynamic parameter controls
  - Color-coded feedback (green/orange/red)
  - Excessive trade warnings (>1000)
  - Integration with unified execution engine
  - Proper TradeCollection type handling

### Execution Engine (standalone_execution.py)
- **StandaloneExecutionEngine**: Core execution logic
- **ExecutionConfig**: Configuration management
- **Features**:
  - Signal lag implementation (1-10 bars)
  - Custom price formula evaluation
  - Commission calculation (fees + slippage)
  - Position tracking
  - $1-based P&L calculation

## Data Flow Pipeline

```
1. Data Loading
   Parquet/CSV → DataFrame → Column mapping → Numpy arrays

2. Strategy Execution
   Data → Signal generation → Lag application → Trade creation

3. Trade Display
   TradeRecordCollection → Legacy conversion → UI emission → Panel display

4. P&L Tracking
   Entry price → Exit price → % calculation → Cumulative tracking
```

## ATR Data Handling
ATR (Average True Range) data is used for range bar sizing:

1. **Column Detection**: Checks for AUX1, ATR, or atr columns
2. **Calculation**: If missing, calculates 14-period ATR
3. **Display**: Shows in data window as "ATR" and "mult"
4. **Storage**: Saved as AUX1 (ATR) and AUX2 (multiplier)

## Performance Specifications
- **Data Loading**: 377K bars in ~0.6s, 6.6M bars in ~14s
- **Memory Usage**: ~68MB for 377K bars
- **Trade Capacity**: Handles 100K+ trades efficiently
- **Rendering**: 60 FPS with viewport optimization
- **Hover Response**: <16ms update time

## Recent Fixes (2025-09-24)

### Critical Issues Resolved
1. **Strategy Runner Type Error**: Fixed TradeCollection emission type mismatch
2. **ATR Display**: Fixed 0.00 display issue by adding column detection
3. **Signal Lag**: Properly implemented with bar delay tracking
4. **Execution Formulas**: Working with custom price calculations
5. **Commission Integration**: Fees and slippage properly deducted
6. **P&L Percentages**: All calculations based on $1 invested

## Testing Infrastructure
- `test_all_fixes_comprehensive.py`: Validates all recent fixes
- `test_strategy_runner_fix.py`: TradeCollection type conversion
- `test_atr_data.py`: ATR calculation and display
- `test_all_fixes.py`: Integration testing

## Usage Instructions

### Quick Start
```bash
# Launch with unified engine
python launch_unified_system.py

# Select data file with ATR
Choose: test_data_10000_bars_with_atr.parquet

# Configure trades
Trade Source: System
Strategy: Simple Moving Average
```

### Adding ATR to Data Files
```bash
# Generate ATR for existing data
python test_atr_data.py

# Creates: original_file_with_atr.parquet
```

## Configuration Best Practices
1. Set `use_unified_engine: true` for realistic backtesting
2. Use `signal_lag: 1` minimum for realistic execution
3. Configure commission (fees + slippage) for accurate P&L
4. Use formula-based execution prices for realism

## Known Limitations
- Maximum practical dataset: ~10M bars
- PyQt5 required (not compatible with PyQt6)
- Windows-specific file paths in some scripts