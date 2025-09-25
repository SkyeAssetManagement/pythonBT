# CODE DOCUMENTATION - PythonBT Trading System

## Project Overview
PythonBT is a high-performance backtesting platform with VectorBT Pro integration, pure array processing execution engine, and PyQtGraph visualization system.

## System Architecture

```
PythonBT/
├── Execution Engine (O(1) Array Processing)
│   └── src/trading/core/
│       ├── pure_array_execution.py          # True O(1) scaling engine
│       ├── standalone_execution.py          # Legacy O(n) fallback
│       └── phased_entry.py                  # VectorBT Pro accumulate features
│
├── VectorBT Pro Integration
│   └── src/trading/
│       ├── core/strategy_runner_adapter.py  # Unified execution interface
│       └── integration/vbt_integration.py   # Portfolio.from_signals wrapper
│
├── Visualization Layer
│   └── src/trading/visualization/
│       ├── pyqtgraph_range_bars_final.py    # 60+ FPS charts (6M+ bars)
│       └── enhanced_trade_panel.py          # Vectorized P&L display
│
├── Strategy Framework
│   └── src/trading/strategies/
│       ├── base.py                          # Signal generation interface
│       └── sma_crossover.py                 # Reference implementation
│
└── Configuration
    ├── config.yaml                          # VectorBT Pro execution settings
    ├── .mcp.json                           # VectorBT Pro MCP server config
    └── .claude/projecttodos.md             # Phased entry implementation plan
```

## Key Components

### Pure Array Execution Engine
**Performance**: O(1) scaling - processes ALL trades simultaneously
```python
# NO loops over trades - pure vectorized operations
entry_bars, exit_bars = create_trade_pairs_numba(change_bars, change_values)
exec_entry_bars, exec_exit_bars, entry_prices, exit_prices = vectorized_execution_lag_and_prices(
    entry_bars, exit_bars, prices_array, signal_lag, max_bar
)
pnl_percentages = vectorized_pnl_calculation(entry_prices, exit_prices)
```

### VectorBT Pro Integration
**Features**: Native execution pricing, accumulate parameter, hlc3 property
```python
# Native hlc3 pricing (NOT custom formula)
execution_price = data.hlc3  # Built-in (H+L+C)/3 calculation

# Position scaling with accumulate parameter
portfolio = vbt.Portfolio.from_signals(
    close=close_prices,
    entries=entry_signals,
    exits=exit_signals,
    accumulate=True,  # Enable phased entries
    price=execution_price  # Custom price arrays supported
)
```

### Phased Entry System (Planned)
**Architecture**: Two-sweep O(1) array processing
1. **Signal Sweep**: Generate position signals using pure array operations
2. **Execution Sweep**: Calculate volume/time-based execution plans vectorized
3. **Result**: VWAP/TWAP execution with O(1) performance scaling

## Configuration System

### VectorBT Pro Settings (config.yaml)
```yaml
# Native VectorBT Pro execution pricing
execution_price: "close"              # Options: "close", "open", "hlc3", etc.
signal_lag: 2                         # VectorBT Pro native lag support

# Phased Entry Configuration
phased_entries:
  enabled: true
  use_vectorbt_accumulate: true       # Native accumulate=True feature
  execution_method: "volume"          # "volume", "time", "hybrid"

  # Volume-based VWAP execution
  volume_based:
    target_participation: 0.10        # 10% of average volume per phase
    use_vwap_pricing: true            # VWAP over execution period
```

### MCP Server Configuration (.mcp.json)
```json
{
  "mcpServers": {
    "VectorBT PRO": {
      "command": "C:\\Python313\\python.exe",
      "args": ["-m", "vectorbtpro.mcp_server"],
      "env": {
        "GITHUB_TOKEN": "${GITHUB_TOKEN}",
        "OPENAI_API_KEY": "${OPENAI_API_KEY}"
      }
    }
  }
}
```

## Performance Specifications
- **Execution Engine**: True O(1) scaling - constant time regardless of dataset size
- **Chart Rendering**: 60+ FPS with 6M+ bars using viewport optimization
- **Memory Usage**: ~68MB for 377K bars with full OHLCV data
- **Trade Processing**: 100K+ trades processed simultaneously without loops

## Development Workflow

### Strategy Development
```python
class CustomStrategy(TradingStrategy):
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        # Generate signals only - NO P&L calculation
        # VectorBT Pro handles all execution and P&L
        return signals  # 1 (long), -1 (short), 0 (no position)
```

### VectorBT Pro Integration Pattern
```python
# Use unified execution engine configuration
adapter = StrategyRunnerAdapter()
adapter.use_unified_engine = True
adapter.use_pure_array_processing = True

# Run strategy with VectorBT Pro backend
trades = adapter.run_strategy(strategy_name, parameters, df)
```

## Technical Architecture Notes

### Array Processing Design
- **Pure Vectorization**: All operations use NumPy broadcasting across entire datasets
- **No Trade Loops**: Individual trade processing replaced with simultaneous array operations
- **Numba Optimization**: Critical functions compiled with @jit(nopython=True, cache=True)
- **Memory Efficiency**: Pre-allocated arrays with known sizes

### VectorBT Pro Features
- **Native Properties**: Use `data.hlc3`, `data.ohlc4` instead of custom formulas
- **Accumulate Parameter**: Enables position scaling and phased entries
- **Custom Price Arrays**: Pass calculated execution prices to Portfolio.from_signals()
- **Signal Lag Support**: Native execution delay implementation

## Current Implementation Status
- ✅ Pure array execution engine (O(1) scaling verified)
- ✅ VectorBT Pro MCP server integration (workspace-specific)
- ✅ Native hlc3 pricing configuration
- ✅ PyQtGraph visualization with vectorized P&L display
- 🔄 Volume/time-based phased entry implementation (in planning)
- 🔄 VWAP/TWAP execution pricing (in planning)

## Testing Framework
```bash
# Test pure array engine performance
python test_integrated_pure_array.py

# Launch main visualization system
python launch_unified_system.py

# Test VectorBT Pro integration
python src/trading/core/strategy_runner_adapter.py
```

## Dependencies
- **Core**: pandas, numpy, numba
- **Trading**: vectorbtpro (with MCP server)
- **Visualization**: PyQt5, pyqtgraph
- **Configuration**: yaml, python-dotenv