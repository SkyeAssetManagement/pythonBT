# PythonBT - Code Documentation

## Project Overview
PythonBT is a comprehensive trading system visualization and backtesting platform built with Python. It features advanced charting capabilities using PyQtGraph, real-time trading visualization, and integration with multiple trading strategies.

## Architecture Diagram
```
PythonBT/
    |
    +-- Core Launcher Scripts
    |     |-- launch_pyqtgraph_with_selector.py (Main Entry Point)
    |     |-- integrated_trading_launcher.py
    |
    +-- UI Components
    |     |-- pyqtgraph_data_selector.py (Data Selection Dialog)
    |
    +-- src/trading/
          |
          +-- visualization/
          |     |-- pyqtgraph_range_bars_final.py (Main Chart Widget)
          |     |-- trade_panel.py (Trade List Display)
          |     |-- chart_components.py (Chart UI Elements)
          |
          +-- strategies/
          |     |-- base.py (Strategy Base Class)
          |
          +-- data/
                |-- trade_data.py (Trade Data Structures)
                |-- csv_trade_loader.py (CSV Trade Import)
```

## Directory Structure

```
C:\code\PythonBT\
├── .claude/
│   ├── CLAUDE.md                    # Development standards and instructions
│   └── projecttodos.md              # Project task tracking and issue log
├── src/
│   └── trading/
│       ├── visualization/
│       │   ├── pyqtgraph_range_bars_final.py  # Main chart implementation
│       │   ├── trade_panel.py                 # Trade list panel widget
│       │   ├── chart_components.py            # Chart UI components
│       │   └── trade_data.py                  # Trade data structures
│       └── strategies/
│           └── base.py                        # Trading strategy base class
├── dataRaw/                         # Market data directory
│   └── range-ATR30x0.05/
│       └── ES/
│           └── diffAdjusted/
│               └── *.csv            # ES futures range bar data
├── launch_pyqtgraph_with_selector.py  # Main application launcher
├── pyqtgraph_data_selector.py         # Data selection dialog
├── integrated_trading_launcher.py      # Alternative launcher
├── test_*.py                           # Various test scripts
└── verify_timestamp_fix.py            # Timestamp verification utility
```

## Module Functional Summary

### Core Launcher Scripts

#### launch_pyqtgraph_with_selector.py
- **Purpose**: Main entry point for the PyQtGraph trading visualization
- **Key Functions**:
  - `ConfiguredChart`: Custom chart class that loads configured data
  - `load_data()`: Loads CSV/Parquet data with proper timestamp handling
  - `load_configured_trades()`: Loads trades from CSV or generates from strategy
  - `generate_system_trades()`: Creates trades using SMA or RSI strategies
- **Critical Fix Applied**: Removed 'Date' from column_mapping to preserve Date+Time combination
- **Data Flow**:
  1. Shows data selector dialog
  2. Loads selected data file (CSV/Parquet)
  3. Combines Date and Time columns into DateTime
  4. Standardizes column names (Open, High, Low, Close, Volume)
  5. Loads or generates trades based on configuration
  6. Displays interactive chart with trades

#### pyqtgraph_data_selector.py
- **Purpose**: Dialog for selecting data files and trade sources
- **Features**:
  - File browser for data selection
  - Trade source options (None, CSV, System-generated, Sample)
  - Trading system selection (SMA, RSI Momentum)
  - Indicator checkboxes
- **Returns**: Configuration dictionary with selected options

### Visualization Components

#### src/trading/visualization/pyqtgraph_range_bars_final.py
- **Purpose**: Main charting widget using PyQtGraph
- **Key Features**:
  - Range bar candlestick display
  - Interactive zoom and pan
  - Crosshair with price/time display
  - Trade marker overlay
  - X-axis timestamp formatting
- **Key Methods**:
  - `render_range()`: Renders visible bars in viewport
  - `format_time_axis()`: Formats x-axis with proper timestamps
  - `update_crosshair()`: Updates crosshair position and info
  - `load_trades()`: Loads and displays trade markers
- **Performance**: Optimized for large datasets using viewport rendering

#### src/trading/visualization/trade_panel.py
- **Purpose**: Side panel showing trade list and details
- **Features**:
  - Sortable trade list with columns (Index, DateTime, Price, Type)
  - Color-coded trade types (Buy=Green, Sell=Light Red, Short=Light Red)
  - Trade selection with chart synchronization
  - Trade statistics summary
- **Signal Fix Applied**: Changed PyQt signal from `TradeData` to `object` type
- **Color Fix Applied**: Improved SELL/SHORT contrast (255, 120, 120)

#### src/trading/visualization/chart_components.py
- **Purpose**: Reusable chart UI components
- **Components**:
  - Crosshair overlay
  - Info display panels
  - Zoom controls
  - Time axis formatting utilities

### Data Management

#### trade_data.py
- **Classes**:
  - `TradeData`: Individual trade with bar_index, price, type, datetime
  - `TradeCollection`: Container for multiple trades with utility methods
- **Features**:
  - Trade filtering by date range
  - Trade statistics calculation
  - Trade type enumeration (Buy, Sell, Short, Cover)

#### csv_trade_loader.py
- **Purpose**: Loads trades from CSV files
- **Supported Formats**:
  - Standard format: DateTime, Price, Type columns
  - Alternative formats with various column names
- **Features**:
  - Automatic column detection
  - DateTime parsing with multiple formats
  - Trade validation

### Trading Strategies

#### src/trading/strategies/base.py
- **Purpose**: Base class for trading strategies
- **Interface**:
  - `execute()`: Main strategy execution method
  - `on_bar()`: Called for each new bar
  - `generate_signal()`: Signal generation logic
- **Features**:
  - Timestamp propagation to strategies
  - Trade signal generation
  - Performance tracking

## Recent Changes and Fixes

### Timestamp Display Fix (2025-09-22)
**Problem**: X-axis and trade list showed dates without times (00:00:00)

**Root Cause**: Column mapping was renaming 'Date' to 'DateTime' before combining with 'Time' column

**Solution**:
```python
# BEFORE (BUG):
column_mapping = {
    'Date': 'DateTime',  # This prevented Date+Time combination
    ...
}

# AFTER (FIXED):
column_mapping = {
    # 'Date': 'DateTime',  # Removed - allows proper combination
    ...
}
```

**Files Modified**:
- `launch_pyqtgraph_with_selector.py` (line 167) - Critical fix
- `src/trading/visualization/trade_panel.py` - Signal type and color fixes
- Debug print statements removed from all files

### PyQt Signal Fix
**Problem**: TypeError when clicking trades in trade list

**Solution**: Changed signal from `pyqtSignal(TradeData)` to `pyqtSignal(object)`

### SELL/SHORT Visibility Fix
**Problem**: Dark red text hard to read on dark background

**Solution**: Changed color from (150, 0, 0) to (255, 120, 120) for better contrast

## Key Data Flows

### CSV Data Loading Pipeline
1. **Input**: CSV with Date, Time, OHLC columns
2. **Column Mapping**: Standardize names (preserve Date/Time separate)
3. **DateTime Creation**: Combine Date + Time into DateTime
4. **Type Conversion**: Convert to numpy arrays with proper dtypes
5. **Storage**: Store in full_data dictionary
6. **Display**: Render with preserved timestamps

### Trade Display Pipeline
1. **Load/Generate**: Load from CSV or generate from strategy
2. **Map to Bars**: Match trades to bar indices
3. **Create Markers**: Generate X markers at trade positions
4. **Update Panel**: Populate trade list with details
5. **Enable Selection**: Allow click to highlight trade

## Testing and Verification

### Test Scripts
- `verify_timestamp_fix.py`: Verifies timestamp preservation
- `test_data_loading.py`: Tests data loading pipeline
- `test_chart_timestamps.py`: Tests chart timestamp display
- `test_datetime_flow.py`: Tests datetime through full pipeline
- `test_selector_workflow.py`: Tests complete user workflow

### Verification Results
- Timestamps correctly parsed: "2021-01-03 17:01:00"
- Time components preserved: hour=17, minute=1, second=0
- X-axis displays actual times from data
- Trade list shows complete DateTime

## Performance Considerations

### Optimizations
- Viewport rendering: Only render visible bars
- Numpy arrays: Use vectorized operations
- Lazy loading: Load data on demand
- Caching: Cache formatted timestamps

### Large Dataset Handling
- Tested with 100,000+ bars
- Smooth pan/zoom via viewport updates
- Memory efficient data structures

## Future Enhancements

### Planned Features
- Real-time data feed integration
- Timezone support
- Additional trading strategies
- Performance analytics dashboard
- Multi-timeframe analysis

### Technical Debt
- Add comprehensive test suite
- Implement proper logging framework
- Add configuration file support
- Create user documentation

## Dependencies

### Core Libraries
- **PyQt5**: GUI framework
- **PyQtGraph**: High-performance plotting
- **pandas**: Data manipulation
- **numpy**: Numerical operations

### Python Version
- Requires Python 3.7+
- Tested on Python 3.9 and 3.10

## Configuration

### Data Files
- Supports CSV and Parquet formats
- Requires Date, Time, OHLC columns
- Optional: Volume, AUX1, AUX2 columns

### Trade Files
- CSV format with DateTime, Price, Type columns
- Supports multiple trade type formats

## Usage

### Basic Launch
```bash
python launch_pyqtgraph_with_selector.py
```

### Workflow
1. Select data file from dialog
2. Choose trade source (None/CSV/System)
3. Select indicators to display
4. Chart opens with interactive controls
5. Use mouse wheel to zoom, drag to pan
6. Click trades in list to highlight on chart

## Troubleshooting

### Common Issues
1. **Timestamps show as 00:00:00**: Fixed - was column mapping issue
2. **Trade click error**: Fixed - signal type issue
3. **Dark text on dark background**: Fixed - color contrast issue

### Debug Mode
Set environment variable for verbose output:
```bash
set DEBUG=1
python launch_pyqtgraph_with_selector.py
```

---
*Last Updated: 2025-09-22*
*Version: 1.0.1 - Timestamp Fix Applied*