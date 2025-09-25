# CODE DOCUMENTATION - PythonBT Trading System

## Project Overview
PythonBT is a modular headless backtesting platform with CSV-based result storage, PyQtGraph visualization, and separated chart/backtesting architecture designed to eliminate GUI hanging issues.

## Current Architecture (Post-Separation)

```
PythonBT/
├── Headless Backtesting Engine (No GUI)
│   └── src/trading/backtesting/
│       ├── headless_backtester.py          # Independent backtest execution
│       └── result_storage.py               # Organized CSV/JSON storage
│
├── CSV Result System
│   └── backtest_results/                   # Timestamped result folders
│       ├── [strategy]_[timestamp]/
│       │   ├── parameters/run_params.json  # Strategy parameters & metadata
│       │   ├── trades/trade_list.csv       # Trades with P&L calculations
│       │   ├── equity/equity_curve.csv     # Equity curve data
│       │   └── code/strategy_snapshot.py   # Strategy code at execution
│
├── Chart Visualization Layer
│   └── src/trading/visualization/
│       ├── pyqtgraph_range_bars_final.py   # 1000-bar rendering, instant scrolling
│       ├── backtest_result_loader.py       # CSV to chart data conversion
│       ├── integrated_chart_runner.py      # Load existing OR run new workflow
│       └── trade_data.py                   # Chart-compatible trade format
│
├── Integrated GUI System
│   ├── launch_unified_system.py            # Main GUI with button controls
│   ├── launch_chart_only.py               # Chart-only system (no backtesting)
│   └── run_headless_only.py               # Headless-only backtesting
│
├── Strategy Implementation
│   └── src/trading/backtesting/
│       ├── sma_crossover.py                # SMA crossover with P&L
│       └── rsi_momentum.py                 # RSI momentum (planned)
│
└── TWAP/Execution Systems (Planned)
    └── src/trading/core/
        ├── time_based_twap_execution.py    # Volume-weighted natural phases
        ├── vectorbt_twap_adapter.py        # VectorBT TWAP integration
        └── minimum_time_phased_entry.py    # Range bar time enforcement
```

## Key Design Decisions

### Separation of Concerns
**Problem Solved**: Original `launch_unified_system.py` hung on strategy execution with 184k+ signals
**Solution**: Complete separation of backtesting from chart visualization

### Headless Backtesting Architecture
**Core Principle**: Backtests run completely independent of GUI, save to organized CSV structure

```python
# Headless execution flow
backtester = HeadlessBacktester()
run_id = backtester.run_backtest(
    strategy_name='sma_crossover',
    parameters=params,
    data_file='data/ES_5min.csv',
    execution_mode='standard'  # or 'twap'
)
# Results automatically saved to backtest_results/[strategy]_[timestamp]/
```

### CSV-Based Result Storage
**Format**: Organized folder structure with timestamped results
```
backtest_results/sma_crossover_20250926_051405/
├── parameters/run_params.json          # Complete run metadata
├── trades/trade_list.csv              # Trades with P&L calculations
├── equity/equity_curve.csv            # Equity curve progression
└── code/strategy_snapshot.py          # Strategy code at execution time
```

### Trade List CSV Format
**Key Feature**: Raw P&L values without % formatting
```csv
datetime,bar_index,trade_type,price,size,exec_bars,pnl,cumulative_profit,is_entry,is_exit
2023-01-04 10:59:00,31,BUY,4230.58,1.0,1,0.0,0.0,True,False
2023-01-04 11:55:00,43,SELL,4227.37,1.0,1,-3.21,-3.21,False,True
```

## Component Architecture

### 1. Headless Backtesting Engine
**File**: `src/trading/backtesting/headless_backtester.py`
**Purpose**: Run backtests completely independent of GUI
**Key Features**:
- No GUI dependencies
- Chunked processing for large datasets (5k signals at a time)
- P&L calculation integration
- Automatic parameter comparison and file overwrites
- Error-first approach (no fallbacks)

```python
class HeadlessBacktester:
    def run_backtest(self, strategy_name, parameters, data_file, execution_mode):
        # Load data -> Generate signals -> Execute trades -> Calculate P&L -> Save CSV
        if execution_mode == "twap" and not self.twap_available:
            raise ImportError("TWAP requested but not available")  # No fallbacks
```

### 2. CSV Result Loader
**File**: `src/trading/visualization/backtest_result_loader.py`
**Purpose**: Load CSV results into chart-compatible format
**Key Features**:
- Automatic result discovery from folder structure
- Metadata preservation (execBars, execution_time_minutes)
- Chart data format conversion

```python
class BacktestResultLoader:
    def list_available_runs(self) -> List[Dict[str, Any]]:
        # Scan backtest_results/ for all available runs

    def load_trade_list(self, run_id: str) -> TradeCollection:
        # Load CSV trades and convert to chart format
```

### 3. Integrated Chart System
**File**: `launch_unified_system.py`
**Purpose**: Main GUI with button-based controls
**Architecture**: Same chart functionality with new button controls

#### Button Functionality:
1. **📁 Load Previous Backtests**: Browse folder structure, load CSV trades
2. **🔄 Run Strategy + Load**: Execute headless backtest + auto-load results
3. **🧹 Clear**: Remove trades from chart

```python
class UnifiedConfiguredChart(RangeBarChartFinal):
    def load_previous_backtests(self):
        # Button 1: Load from backtest_results/ folder structure

    def run_strategy_and_load(self):
        # Button 2: Run headless -> auto-load -> display results
```

### 4. TWAP System (Planned Implementation)
**Architecture**: Volume-weighted natural phases for range bars
**Key Concept**: Each bar = one phase, size proportional to bar volume

```python
# TWAP execution logic (planned)
def volume_weighted_execution(signal_bar, min_execution_time, df):
    # 1. Find bars covering minimum time from signal
    # 2. Calculate volume proportion per bar
    # 3. Allocate position size by volume weight
    # 4. Natural phases (no artificial splitting)
```

## Performance Characteristics

### Chart Rendering
- **1000 Bar Viewport**: Instant rendering with scrolling
- **Memory Efficient**: Only renders visible bars
- **No Hanging**: Chart display completely separated from backtesting

### Backtesting Performance
- **Small Datasets (1k bars)**: < 5 seconds
- **Medium Datasets (50k bars)**: < 30 seconds
- **Large Datasets (200k bars)**: < 2 minutes (chunked processing)
- **Memory Usage**: < 1GB RAM regardless of dataset size

### Error Handling Philosophy
**Error-First Approach**: No silent fallbacks or mode switching
```python
# Example: TWAP mode handling
if execution_mode == "twap":
    if not self.twap_available:
        raise ImportError("TWAP system not available")  # Clear error
    results = self._run_twap_backtest()
elif execution_mode == "standard":
    results = self._run_standard_backtest()
else:
    raise ValueError(f"Unknown execution mode: {execution_mode}")
```

## Configuration System

### MCP Server Integration
**File**: `.mcp.json`
```json
{
  "mcpServers": {
    "VectorBT PRO": {
      "command": "C:\\Python313\\python.exe",
      "args": ["-m", "vectorbtpro.mcp_server"]
    }
  }
}
```

### Strategy Parameters
**Format**: JSON metadata in each backtest run
```json
{
  "run_id": "sma_crossover_20250926_051405",
  "strategy_name": "sma_crossover",
  "execution_mode": "standard",
  "parameters": {
    "fast_period": 20,
    "slow_period": 50,
    "signal_lag": 2,
    "min_execution_time": 5.0
  }
}
```

## Development Workflow

### Running Backtests
```bash
# Option 1: Headless only (no GUI)
python run_headless_only.py

# Option 2: Chart only (no backtesting)
python launch_chart_only.py

# Option 3: Integrated system (recommended)
python launch_unified_system.py
```

### Integrated System Workflow
1. **Launch**: `python launch_unified_system.py`
2. **Select Data**: Choose range bars or sample data
3. **Chart Loads**: 1000 bars render instantly
4. **Button Options**:
   - Load previous backtests from CSV files
   - Run new strategy + auto-load results
   - Clear trades from display

### Adding New Strategies
1. Implement in `src/trading/backtesting/headless_backtester.py`
2. Add to `_generate_signals()` method
3. Configure parameters in button handlers
4. Test with headless execution first

## Current Implementation Status

### ✅ Completed Components
- Headless backtesting engine with P&L calculations
- CSV result storage with organized folder structure
- Chart system with 1000-bar rendering
- Integrated GUI with button controls
- Result loading from CSV files
- Error-first approach (no fallbacks)
- Trade list with raw P&L values (no % formatting)

### 🔄 Planned Implementation
- TWAP system with volume-weighted natural phases
- Time-based execution for range bars (minimum 5-minute enforcement)
- VectorBT TWAP adapter integration
- RSI momentum strategy implementation
- Additional execution modes and strategies

## Testing Framework

### Component Testing
```bash
# Test headless backtesting
python test_headless_system.py

# Test chart components
python test_chart_only.py

# Test updated integrated system
python test_updated_system.py
```

### Integration Testing
```bash
# Test complete workflow
python run_headless_only.py    # Generate CSV results
python launch_chart_only.py    # Load and display results
```

## Dependencies
- **Core**: pandas, numpy, PyQt5
- **Visualization**: pyqtgraph
- **Trading**: vectorbtpro (with MCP server)
- **Configuration**: yaml, json, pathlib

## Architecture Benefits

### 1. Elimination of Hanging Issues
- Complete separation prevents GUI freezing during backtesting
- Headless execution runs independently
- Chart system only handles display

### 2. Modular Design
- Each component has single responsibility
- Easy to test and debug individual parts
- Flexible deployment options

### 3. Result Persistence
- All backtest results saved to CSV
- Historical backtests can be reloaded
- Organized folder structure for easy browsing

### 4. User Experience
- Progress dialogs during backtesting
- Clear error messages instead of hanging
- Button-based controls for intuitive operation