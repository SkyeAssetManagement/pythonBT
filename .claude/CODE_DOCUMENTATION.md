# CODE DOCUMENTATION - PythonBT Trading System

## Project Overview
PythonBT is a high-performance backtesting platform combining machine learning models with PyQtGraph visualization. Features vectorized P&L calculations, range bar processing, and advanced trade analytics.

## System Architecture

```
PythonBT/
├── Visualization Layer
│   ├── launch_unified_system.py              # Primary launcher
│   └── src/trading/visualization/
│       ├── pyqtgraph_range_bars_final.py     # High-performance charts
│       ├── enhanced_trade_panel.py           # Trade list with vectorized P&L
│       ├── strategy_runner.py                # Real-time strategy execution
│       └── trade_data.py                     # Core data structures
│
├── Trading Engine
│   └── src/trading/
│       ├── core/
│       │   ├── strategy_runner_adapter.py    # VectorBT integration
│       │   └── trade_types.py                # Modern trade records
│       ├── strategies/
│       │   ├── base.py                       # Strategy base class
│       │   └── sma_crossover.py             # Signal generation only
│       └── integration/
│           └── vbt_integration.py            # VectorBT Portfolio adapter
│
├── Machine Learning
│   └── src/
│       ├── OMtree_model.py                   # Random forest models
│       ├── OMtree_preprocessing.py           # Feature engineering
│       ├── OMtree_validation.py              # Backtesting framework
│       └── OMtree_walkforward.py             # Walk-forward optimization
│
└── Configuration
    ├── OMtree_config.ini                     # ML model settings
    └── tradingCode/config.yaml               # Trading execution settings
```

## Critical Design Patterns

### 1. Vectorized P&L Calculation (VectorBT Integration)
```python
# Strategies generate signals only
signals = strategy.generate_signals(df)

# VectorBT handles all P&L calculations
portfolio = vbt.Portfolio.from_signals(
    close=close_prices,
    entries=entry_signals,
    exits=exit_signals,
    init_cash=1.0  # $1 investment = percentage returns
)

# Results: Proper vectorized P&L with consistent calculations
```

### 2. Trade Data Flow
```python
# Modern format (VectorBT → TradeRecord)
TradeRecord(
    bar_index=100,
    trade_type='BUY',
    price=4200.50,
    pnl_percent=0.0238,  # Decimal format (2.38%)
    timestamp=pd.Timestamp('2021-01-04 10:30:00')
)

# Legacy format (UI compatibility)
TradeData(
    bar_index=100,
    trade_type='BUY',
    price=4200.50,
    pnl_percent=0.0238   # Copied from TradeRecord
)
```

### 3. Chart Data Standards
```python
# Required format for PyQtGraph charts
chart_data = {
    'timestamp': pd.DatetimeIndex,  # Required for hover display
    'open': np.array,               # OHLCV data
    'high': np.array,
    'low': np.array,
    'close': np.array,
    'volume': np.array,
    'aux1': np.array               # Additional indicators (ATR, etc.)
}
```

## Key Components

### Enhanced Trade Panel
- **P&L Display**: Simple summation of percentage returns
- **Sortable Columns**: Click headers for ascending/descending sort
- **Summary Statistics**: Win rate, total P&L, trade counts
- **Color Coding**: Green/red for profits/losses

### Strategy Runner Adapter
- **Signal Generation**: Strategies produce position signals only
- **VectorBT Integration**: Portfolio.from_signals() for P&L calculation
- **Legacy Compatibility**: Converts to existing UI format
- **Fallback Support**: Manual calculation if VectorBT unavailable

### PyQtGraph Chart System
- **Performance**: 60+ FPS with viewport optimization
- **Scalability**: Handles 6M+ bars efficiently
- **Features**: Auto Y-scaling, hover data, trade overlays
- **Multi-monitor**: DPI-aware rendering

## Configuration System

### config.yaml Structure
```yaml
backtest:
  signal_lag: 2                    # Bars delay for execution
  commission: 0.01                 # Per trade commission
  slippage: 0.001                  # Price slippage percentage

execution:
  formulas:
    entry_price: "close[i+1]"      # Entry price calculation
    exit_price: "close[i+1]"       # Exit price calculation
```

### OMtree_config.ini
```ini
[model]
max_depth = 10
min_samples_split = 5
n_estimators = 100

[data]
features = price_change,volume,atr
target = future_return
```

## Performance Specifications
- **Chart Rendering**: 60 FPS with 377K bars
- **Memory Usage**: ~68MB for large datasets
- **Load Time**: 377K bars in ~0.6 seconds
- **Trade Processing**: 100K+ trades without lag

## Testing Framework
```bash
# Test vectorized P&L calculations
python test_vectorbt_integration.py

# Launch main visualization system
python launch_unified_system.py

# Test individual components
python src/trading/visualization/strategy_runner.py
python src/trading/core/strategy_runner_adapter.py
```

## Recent Architectural Changes (2025-09-24)

### VectorBT P&L Integration
- **Purpose**: Eliminate inconsistent P&L calculations across strategies
- **Implementation**: Centralized Portfolio.from_signals() approach
- **Benefits**: Vectorized calculations, consistent results, better performance

### Trade Display Fixes
- **Issue**: P&L values off by factor of 100
- **Fix**: Removed double multiplication in display code
- **Result**: Trade list and summary panel now show matching values

## Dependencies
- **Core**: pandas, numpy, scikit-learn
- **Visualization**: PyQt5, pyqtgraph
- **Trading**: vectorbtpro (optional, with fallback)
- **Data**: parquet, pickle, yaml

## Usage Patterns

### Strategy Development
```python
class CustomStrategy(TradingStrategy):
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        # Generate position signals only
        # Return: 1 (long), -1 (short), 0 (no position)
        return signals

    # No P&L calculation needed - handled by VectorBT
```

### Chart Integration
```python
# Load data and run strategy
runner.set_chart_data(data_dict)
runner.run_strategy()  # Generates trades with vectorized P&L

# Display results in enhanced trade panel
trade_panel.load_trades(trades)  # Shows sorted, colored results
```

## Known Limitations
- VectorBT Pro required for full functionality
- Windows-specific paths in some modules
- PyQt5 dependency (PyQt6 not supported)
- Maximum practical dataset: ~10M bars