# CODE DOCUMENTATION - PythonBT Trading System

## CRITICAL REVERSION NOTICE (2025-09-23)

### Why We Reverted to Commit d22acf6

We have reverted from commit `48830fb` back to commit `d22acf6` ("Implement execution price formulas and signal lag tracking") because the subsequent attempts to unify the trading system resulted in severe degradation of chart rendering quality.

**Working Version (d22acf6)**: As shown in Screenshot 2025-09-22 164318.png
- Clean, crisp candlestick rendering with proper green/red colors
- Clear time axis labels (19:17:30, 20:46:06, etc.)
- Proper data hover window showing Bar, DateTime, OHLC, ATR, Range
- White X trade markers clearly visible
- Trade list panel with DateTime, Type, Price, Size columns
- Auto Y-axis scaling working perfectly
- Smooth panning and zooming across 377,690 bars

**Failed Attempts (commits after d22acf6)**:
- `bbbf2e9`: Modular backtesting architecture - broke chart rendering
- `6a6d072`: PyQtGraph integration - data not displaying
- `d0d19cb`: PyQtGraph launcher - rendering issues
- `52a5561`: P&L calculation fix - still had rendering problems
- `5df8579`: Unified trading system - chart completely broken
- `8dbb5ce`: EnhancedCandlestickItem fix attempt - didn't resolve issues
- `48830fb`: Complete implementation - chart still not working properly

### What We Were Trying to Achieve

The goal of the failed commits was to create a unified system that would:

1. **Unified Configuration**: Single config.yaml controlling all components
   - Execution lag settings
   - Price formulas (OHLC expressions)
   - Position sizing
   - Strategy parameters

2. **Modular Architecture**: Self-contained strategies with metadata
   - Base strategy class with indicator definitions
   - Auto-plotting of indicators
   - Consistent execution across visualization and backtesting

3. **Unified Execution Engine**: Single engine for both visualization and backtesting
   - Consistent trade execution logic
   - Proper lag implementation (signal bar + N)
   - Price formula evaluation

4. **Clean P&L Display**: Show P&L as percentage not dollars
   - With $1 position size, P&L = percentage gain
   - Display as "1.25%" not "$0.0125"

### What Went Wrong

The modular refactoring introduced too many changes at once:
- Lost the sequential rendering approach (`render_range()` method)
- Broke the coordinate system between components
- Changed how candlesticks were instantiated
- Lost proper time axis formatting
- Data hover window stopped working
- Chart wouldn't display data at all in some versions

## Current Working Architecture (Commit d22acf6)

### Directory Structure
```
C:\code\PythonBT\
├── src\trading\
│   ├── visualization\
│   │   ├── pyqtgraph_range_bars_final.py   # WORKING chart with 377k bars
│   │   ├── trade_panel.py                  # Trade list with DateTime
│   │   ├── simple_white_x_trades.py        # White X markers
│   │   ├── strategy_runner.py              # Strategy execution
│   │   └── csv_trade_loader.py            # CSV import
│   └── strategies\
│       ├── base.py                         # Base strategy with lag
│       ├── sma_crossover.py               # SMA crossover
│       └── rsi_momentum.py                # RSI strategy
├── config.yaml                             # Execution configuration
├── launch_pyqtgraph_with_selector.py      # Main launcher
└── parquetData\                           # Range bar data
```

### Key Working Components

#### pyqtgraph_range_bars_final.py
- `RangeBarChartFinal` class with proper `render_range()` method
- `EnhancedCandlestickItem` with 5 arguments: x, opens, highs, lows, closes
- Sequential rendering that handles 377,690 bars efficiently
- Proper time axis formatting with `format_time_axis()`
- Working hover data display with crosshairs

#### Trade Visualization
- `SimpleWhiteXTrades` class creating size 18 white X markers
- Trade panel showing DateTime, Type, Price, Size
- Proper trade jumping and highlighting

#### Strategy Execution (config.yaml driven)
```yaml
execution:
  bar_lag: 1  # Signal at bar N, execute at bar N+1
  buy_execution_price: "(H + L + C) / 3"
  sell_execution_price: "(H + L + C) / 3"
  exit_execution_price: "C"
  short_execution_price: "O"
  cover_execution_price: "C"
```

## Path Forward

See projecttodos.md for detailed stepwise plan to achieve unification without breaking the working chart.

## Commit Reference Timeline

### Working Baseline
- `d22acf6` (2025-09-23): Implement execution price formulas and signal lag tracking - CURRENT
- `1bd562d`: Implement config.yaml-driven execution with bar lag and price formulas
- `c4bf923`: Update documentation with all critical fixes completed

### Failed Attempts (DO NOT MERGE)
- `bbbf2e9`: Implement comprehensive modular backtesting architecture
- `6a6d072`: Complete PyQtGraph integration with modular architecture
- `d0d19cb`: Add PyQtGraph launcher with correct P&L calculation
- `52a5561`: Fix P&L calculation for $1 position = percentage gains
- `5df8579`: Create unified trading system combining all functionality
- `8dbb5ce`: Fix EnhancedCandlestickItem initialization error
- `48830fb`: Complete implementation of execution lag and price formulas

### Branch Structure
- `feature/modular-backtesting-refactor`: Current branch at d22acf6
- `FAILED-attempt`: Contains failed unification attempts for reference

---
*Last Updated: 2025-09-23*
*Status: Reverted to stable baseline, planning careful incremental changes*