# PythonBT Trading System - Code Documentation

## Overview
High-performance backtesting platform combining VectorBT Pro for vectorized backtesting with PyQtGraph for real-time visualization. Features config.yaml-driven execution with bar lag, OHLC price formulas, and $1 position sizing for clean percentage-based profit calculations.

## System Architecture

```
PythonBT/
├── tradingCode/
│   ├── config.yaml                 # Central configuration file
│   ├── main_simple.py              # Main entry point for backtesting
│   ├── src/
│   │   ├── backtest/
│   │   │   ├── vbt_engine.py      # VectorBT backtesting engine with config integration
│   │   │   └── phased_trading_engine.py  # Gradual entry/exit logic
│   │   ├── data/
│   │   │   └── parquet_converter.py  # Data loading and conversion
│   │   └── utils/
│   │       └── price_formulas.py   # OHLC formula evaluation engine
│   └── strategies/
│       ├── base_strategy.py        # Abstract base strategy class
│       └── time_window_strategy_vectorized.py  # Time-based trading strategy
├── src/trading/
│   ├── pyqtgraph_main.py          # PyQtGraph visualization entry point
│   └── visualization/
│       └── trade_panel.py         # Trade list and analysis panel
└── OMtree/                        # Machine learning models
```

## Key Components

### 1. Configuration System (config.yaml)

The system is driven by a comprehensive YAML configuration file that controls:

#### Backtesting Parameters
- **Position Sizing**: $1 fixed value for clean percentage calculations
- **Signal Lag**: Configurable bar delay between signal and execution (default: 1)
- **Execution Formulas**: Mathematical expressions using OHLC data
  - Buy formula: `(H + L + C) / 3` (typical price)
  - Sell formula: `(H + L + C) / 3` (typical price)
- **Direction**: "both" for long and short trades

#### Key Innovation: $1 Position Sizing
With $1 position sizing, the PnL in dollars directly represents the percentage return:
- Long trade: $1 position, 10% price increase = $0.10 profit (10%)
- Short trade: $1 position, 10% price decrease = $0.10 profit (10%)

### 2. VectorBT Engine (src/backtest/vbt_engine.py)

The core backtesting engine with the following features:

#### Signal Lag Implementation
```python
# Apply signal lag if configured
signal_lag = self.config['backtest'].get('signal_lag', 0)
if signal_lag > 0:
    entries = np.roll(entries, signal_lag, axis=0)
    exits = np.roll(exits, signal_lag, axis=0)
    entries[:signal_lag] = False
    exits[:signal_lag] = False
```

#### Execution Price Calculation
- Supports formula-based pricing: `(H + L + C) / 3`
- Calculates different prices for buy vs sell signals
- Properly aligned with lagged signals

#### Performance Characteristics
- Sub-linear scaling: 0.2-0.5x efficiency
- Handles 50,000+ bars in under 0.02 seconds
- Vectorized operations throughout

### 3. Price Formula Evaluator (src/utils/price_formulas.py)

Safe mathematical expression evaluator for OHLC-based formulas:

#### Supported Variables
- O: Open price
- H: High price
- L: Low price
- C: Close price

#### Common Formulas
- Typical price: `(H + L + C) / 3`
- Median price: `(H + L) / 2`
- Weighted close: `(H + L + 2*C) / 4`
- OHLC4: `(O + H + L + C) / 4`

#### Security Features
- Validates formulas for safe characters only
- Restricted evaluation namespace
- Fallback to close prices on error

### 4. Strategy System

#### Base Strategy (strategies/base_strategy.py)
Abstract base class defining the strategy interface:
- `generate_signals()`: Create entry/exit signals
- `get_parameter_combinations()`: Parameter sweep support
- `run_vectorized_backtest()`: Execute backtest with VectorBT

#### Time Window Strategy (strategies/time_window_strategy_vectorized.py)
Vectorized implementation with:
- Time-based entry/exit windows
- Peak detection for stops
- Full parameter optimization support

### 5. PyQtGraph Visualization

#### Main Chart (src/trading/pyqtgraph_main.py)
- Handles 377,690+ bars without performance issues
- Real-time pan and zoom
- Trade markers with hover information
- SMA overlays

#### Trade Panel (src/trading/visualization/trade_panel.py)
- Trade list with sorting and filtering
- CSV and backtester data sources
- Click-to-navigate functionality
- Performance metrics display

## Testing Infrastructure

### Test Suite Components

1. **test_config_backtest.py**: Comprehensive config validation
   - Bar lag verification
   - Execution formula testing
   - $1 position sizing validation
   - Non-linear scaling analysis

2. **test_real_data_config.py**: Real market data testing
   - Integration testing with actual data
   - Trade-by-trade analysis
   - Performance metric verification

## Performance Metrics

### Scaling Analysis (50,000 bars test)
- 1,000 bars: 0.009 seconds
- 5,000 bars: 0.009 seconds (0.2x scaling)
- 10,000 bars: 0.009 seconds (0.5x scaling)
- 50,000 bars: 0.011 seconds (0.23x scaling)

### Memory Efficiency
- Vectorized numpy operations throughout
- Efficient array broadcasting
- Minimal data copying

## Configuration Guide

### Essential Settings

```yaml
backtest:
  position_size: 1                    # $1 for clean percentage calculations
  position_size_type: "value"         # Fixed dollar amount
  signal_lag: 1                       # Execute on next bar after signal
  buy_execution_formula: "(H + L + C) / 3"   # Typical price for entries
  sell_execution_formula: "(H + L + C) / 3"  # Typical price for exits
  direction: "both"                   # Allow both long and short trades
```

### Formula Examples

#### Conservative Execution
```yaml
buy_execution_formula: "H"          # Buy at high (worst price)
sell_execution_formula: "L"         # Sell at low (worst price)
```

#### Aggressive Execution
```yaml
buy_execution_formula: "L"          # Buy at low (best price)
sell_execution_formula: "H"         # Sell at high (best price)
```

## Usage Examples

### Basic Backtest
```python
from src.backtest.vbt_engine import VectorBTEngine

# Initialize engine with config
engine = VectorBTEngine("config.yaml")

# Run backtest
pf = engine.run_vectorized_backtest(data, entries, exits)

# Get results
total_return = pf.total_return
trades = pf.trades.records_readable
```

### Strategy Execution
```python
from strategies.time_window_strategy_vectorized import TimeWindowVectorizedStrategy

# Initialize strategy
strategy = TimeWindowVectorizedStrategy()

# Generate signals
entries, exits = strategy.generate_signals(data, **params)

# Run backtest with config
pf = strategy.run_vectorized_backtest(data, config['backtest'])
```

## Critical Design Principles

### Signal Lag Implementation
- Signals detected at bar N execute at bar N+lag (default lag=1)
- Original signals tracked separately from lagged execution signals
- Execution prices calculated using OHLC data from execution bar, not signal bar

### Position Sizing Strategy
- $1 fixed position size enables PnL in dollars to equal percentage returns
- Long trades: $1 position, 10% price increase = $0.10 profit (10%)
- Short trades: $1 position, 10% price decrease = $0.10 profit (10%)
- Works with direction="both" for alternating long/short trades

### Performance Optimizations
- Vectorized numpy operations throughout
- Sub-linear scaling: 5x data increase = 1.2x time increase
- Efficient array broadcasting without copying
- 50,000 bars process in ~0.011 seconds

## Testing Infrastructure

- **test_config_backtest.py**: Validates lag, formulas, sizing, scaling
- **test_sma_execution.py**: Tests real strategy execution with config
- **test_real_data_config.py**: Integration testing with market data

## Current Status

✅ **Fully Operational:**
- Bar lag from config.yaml (`signal_lag: 1`)
- Position sizing from config.yaml (`position_size: 1`)
- Execution formulas from config.yaml (`(H + L + C) / 3`)
- PyQtGraph visualization with 377,690+ bars
- Both long and short trade support

---

*Last Updated: 2025-09-23*
*Version: 2.1.0 - Full Config Integration*