# CODE DOCUMENTATION - PythonBT Trading System

## Project Overview
PythonBT is a comprehensive backtesting and trading visualization platform built with PyQt5 and PyQtGraph. Features real-time charting, strategy execution, and unified execution engine for realistic trade simulation with support for massive datasets (6M+ bars).

## System Architecture

### Core Components
```
PythonBT/
├── Launch Scripts
│   ├── launch_pyqtgraph_with_selector.py    # Main entry with data selector dialog
│   ├── launch_unified_system.py             # Unified execution engine launcher
│   └── integrated_trading_launcher.py       # Alternative launcher
│
├── src/trading/
│   ├── visualization/
│   │   ├── pyqtgraph_range_bars_final.py   # Main chart widget (RangeBarChartFinal class)
│   │   ├── trade_panel.py                  # Trade list panel (TradeListPanel class)
│   │   ├── enhanced_trade_panel.py         # Enhanced panel with P&L percentages
│   │   ├── strategy_runner.py              # Strategy execution UI (StrategyRunner class)
│   │   └── chart_components.py             # Reusable chart UI components
│   │
│   ├── strategies/
│   │   ├── base.py                         # TradingStrategy base class
│   │   ├── sma_crossover.py               # SMACrossoverStrategy implementation
│   │   ├── rsi_momentum.py                # RSIMomentumStrategy implementation
│   │   └── strategy_wrapper.py            # StrategyFactory and metadata wrapper
│   │
│   ├── core/
│   │   ├── standalone_execution.py        # ExecutionEngine with lag/price formulas
│   │   ├── strategy_runner_adapter.py     # Routes between legacy/unified engines
│   │   └── trade_types.py                 # TradeRecord, TradeRecordCollection classes
│   │
│   └── data/
│       ├── trade_data.py                  # TradeData, TradeCollection classes
│       └── csv_trade_loader.py            # CSV trade import functionality
```

## Critical Design Patterns & Implementation

### 1. Data Structure Requirements
```python
# Chart expects lowercase keys - CRITICAL for hover functionality
full_data = {
    'timestamp': numpy_array,  # Required by parent RangeBarChartFinal
    'open': numpy_array,       # Must be lowercase
    'high': numpy_array,
    'low': numpy_array,
    'close': numpy_array,
    'volume': numpy_array,     # Optional - check existence before access
    'aux1': numpy_array,       # Optional - ATR values
    'aux2': numpy_array        # Optional - range multiplier
}
```

### 2. Data Flow Pipeline
```
CSV/Parquet → DataFrame → Column Standardization → Numpy Arrays → Chart Display
                              ↓
                    Strategy Runner → Signal Generation → Trade Creation
```

### 3. Critical Connections

#### Chart to Strategy Runner
```python
# MUST call this after loading data to enable strategy execution
if self.trade_panel:
    bar_data = {
        'timestamp': self.full_data['timestamp'],
        'open': self.full_data['open'],
        'high': self.full_data['high'],
        'low': self.full_data['low'],
        'close': self.full_data['close']
    }
    self.trade_panel.set_bar_data(bar_data)  # Connects to strategy runner
```

#### Error Prevention Pattern
```python
# Prevent KeyError for optional fields
value = data['aux1'][idx] if 'aux1' in data and data['aux1'] is not None else 0
```

### 4. Strategy Execution Pipeline
- **Signal Generation**: Returns Series with 1 (long), 0 (flat), -1 (short)
- **Trade Conversion**: Signal changes trigger trades (BUY/SELL/SHORT/COVER)
- **P&L Tracking**: Enhanced trades include entry/exit prices and P&L calculations

### 5. Unified Execution Engine
- **Config-driven**: YAML configuration controls execution behavior
- **Signal Lag**: Realistic delay between signal and execution (default 1 bar)
- **Price Formulas**: Custom execution prices like "(H+L+C)/3"
- **Dual Mode**: Switch between legacy (immediate) and unified (realistic) execution

## UI Components & Features

### Trade Panel (3 tabs)
1. **Trades Tab**: Sortable list with DateTime, Price, Type, P&L%
2. **Source Selector**: Choose None/CSV/System-generated trades
3. **Run Strategy**: Execute strategies with parameter controls

### Strategy Runner Features
- **Color-coded feedback**: Green (success), Orange (warning/excessive trades)
- **Trade limits**: Warns when >1000 trades generated
- **Dynamic parameters**: UI adapts to selected strategy

### Chart Features
- **Viewport rendering**: Only visible bars rendered for performance
- **Hover data**: Shows OHLC, volume, timestamp on mouse movement
- **Crosshair**: Tracks mouse with price/time display
- **Trade markers**: White X marks at trade positions

## Performance Specifications
- **Loading**: 377K bars in ~0.6s, 6.6M bars in ~14s
- **Rendering**: Smooth pan/zoom with viewport optimization
- **Memory**: ~68MB for 377K bar dataset
- **Trade capacity**: Handles 100K+ trades efficiently

## Configuration System

### config.yaml
```yaml
use_unified_engine: false    # Enable new execution engine
backtest:
  signal_lag: 1              # Bars between signal and execution
  execution_price: "close"   # or "formula"
  buy_execution_formula: "(H + L + C) / 3"
  sell_execution_formula: "(H + L + C) / 3"
```

## Recent Fixes & Improvements

### Version 2.3.0 (Current Session)
- **Fixed**: Strategy runner "no chart data" in unified system
- **Added**: `pass_data_to_trade_panel()` method in UnifiedConfiguredChart
- **Improved**: Strategy feedback with color coding and excessive trade warnings

### Version 2.2.0
- **Fixed**: Hover data KeyErrors with dictionary existence checks
- **Fixed**: Trade panel scrolling with `get_first_visible_trade()`
- **Verified**: Performance with 6.6M bar datasets

## Testing Infrastructure
- `test_hover_proof.py`: Validates hover functionality
- `test_strategy_execution.py`: Tests strategy pipeline
- `test_unified_strategy_runner.py`: Verifies unified system
- `test_full_dataset.py`: Performance validation

## Usage

### Launch Commands
```bash
python launch_pyqtgraph_with_selector.py    # Standard with dialog
python launch_unified_system.py              # Unified execution engine
```

### Typical Workflow
1. Select data file (CSV/Parquet) from dialog
2. Choose trade source (None/CSV/System)
3. Navigate to "Run Strategy" tab
4. Select strategy and parameters
5. Click "Run Strategy" button
6. View generated trades with P&L tracking

## Known Issues & Limitations
- SMA strategy can generate excessive trades with short periods
- Maximum practical dataset ~10M bars
- Unified engine in migration phase (use_unified_engine: false by default)

## Future Enhancements
- Real-time data feed integration
- Strategy optimization framework
- Multi-timeframe analysis
- Additional execution price formulas