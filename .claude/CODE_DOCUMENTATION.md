# CODE DOCUMENTATION - Unified Trading System

## Overview
This document provides comprehensive documentation for the Unified Trading System, which merges:
- **OMtree**: Machine learning decision tree trading system with walk-forward validation
- **ABtoPython**: VectorBT integration with PyQtGraph visualization and trade analysis

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    Unified Trading System GUI                    │
│                        (unified_gui.py)                          │
├─────────────────────────────────────────────────────────────────┤
│  Configuration │ Walk-Forward │ Performance │ Trade Viz │ Charts │
└────────┬───────┴──────┬───────┴──────┬──────┴─────┬─────┴───┬───┘
         │              │              │             │         │
    ┌────▼────┐    ┌────▼────┐   ┌────▼────┐  ┌────▼────┐ ┌──▼──┐
    │Config   │    │OMtree   │   │Stats    │  │Trade    │ │PyQt │
    │Manager  │    │Model    │   │Module   │  │Data     │ │Graph│
    └─────────┘    └─────────┘   └─────────┘  └─────────┘ └─────┘
                        │                            │
                   ┌────▼─────────────────────────┐  │
                   │  Unified Data Pipeline       │──┘
                   │  (unified_pipeline.py)       │
                   └───────────────────────────────┘
                              │
                   ┌──────────▼──────────┐
                   │   Feature Flags     │
                   │  (feature_flags.py) │
                   └─────────────────────┘
```

## Directory Structure

```
C:\code\PythonBT\
├── .claude\                      # Project management & docs
│   ├── CLAUDE.md                 # Development standards
│   ├── projectStatus.md          # Integration progress tracker
│   ├── file-refactor.md          # Safety-first refactoring guide
│   └── CODE_DOCUMENTATION.md     # This file
├── src\                          # Source code
│   ├── trading\                  # Trading system components
│   │   ├── data\                 # Data structures & pipeline
│   │   │   ├── __init__.py
│   │   │   ├── trade_data.py     # Core trade data structures
│   │   │   ├── trade_data_extended.py  # Extended functionality
│   │   │   ├── loaders.py        # CSV/data loaders
│   │   │   └── unified_pipeline.py  # Data format conversion
│   │   ├── visualization\        # Visualization components
│   │   │   ├── __init__.py
│   │   │   ├── charts.py         # PyQtGraph chart components
│   │   │   └── trade_marks.py    # Trade overlay visualization
│   │   └── integration\          # External integrations
│   │       ├── __init__.py
│   │       └── vbt_loader.py     # VectorBT integration
│   ├── config_manager.py         # Configuration management
│   ├── performance_stats.py      # Performance calculation
│   ├── date_parser.py            # Flexible date parsing
│   └── feature_flags.py          # Feature flag system
├── tests\                        # Test suite
│   ├── test_trade_data.py        # Trade data tests
│   ├── test_unified_pipeline.py  # Pipeline tests
│   └── test_gui_integration.py   # GUI integration tests
├── OMtree_model.py               # ML decision tree model
├── OMtree_preprocessing.py       # Data preprocessing
├── OMtree_validation.py          # Walk-forward validation
├── OMtree_config.ini             # Model configuration
├── OMtree_gui.py                 # Original OMtree GUI
├── unified_gui.py                # Unified GUI (main entry)
├── feature_flags.json            # Feature flag configuration
├── requirements.txt              # Python dependencies
└── README.md                     # Project overview
```

## Module Documentation

### Core Components

#### 1. Feature Flag System (`src/feature_flags.py`)
**Purpose**: Safe, gradual feature rollout following deploy-dark pattern

**Key Classes**:
- `FeatureFlags`: Manages feature toggles
  - `enable(flag_name)`: Enable a feature
  - `disable(flag_name)`: Disable a feature
  - `is_enabled(flag_name)`: Check if feature is active
  - `get_all_flags()`: Get all flag states

**Key Functions**:
- `@feature_flag(flag_name)`: Decorator for feature-protected functions

**Usage Example**:
```python
from feature_flags import feature_flag, get_feature_flags

flags = get_feature_flags()
flags.enable('new_feature')

@feature_flag('new_feature')
def risky_function():
    # Only runs if flag enabled
    pass
```

#### 2. Unified Data Pipeline (`src/trading/data/unified_pipeline.py`)
**Purpose**: Bridge OMtree DataFrame format with ABtoPython TradeData format

**Key Classes**:
- `UnifiedDataFormat`: Common data structure for both systems
  - Fields: timestamp, symbol, price, features, target, prediction, trade_type, size, pnl
  
- `DataPipelineAdapter`: Converts between formats
  - `omtree_to_unified()`: Convert OMtree DataFrame to unified
  - `unified_to_omtree()`: Convert unified back to OMtree
  - `trades_to_unified()`: Convert trades to unified
  - `unified_to_trades()`: Convert unified to trades
  - `merge_data_sources()`: Merge data from both systems

- `OMtreeAdapter`: OMtree-specific operations
  - `prepare_data_for_model()`: Prepare data for ML model
  - `process_predictions()`: Process model predictions

- `ABtoPythonAdapter`: ABtoPython-specific operations
  - `import_from_vectorbt()`: Import VectorBT data
  - `export_to_visualization()`: Export for PyQtGraph

#### 3. Trade Data Structures (`src/trading/data/trade_data.py`)
**Purpose**: Core trade representation and collection management

**Key Classes**:
- `TradeData`: Individual trade representation
  - Fields: trade_id, timestamp, bar_index, trade_type, price, size, pnl, strategy, symbol
  - Validation: Ensures valid trade types and positive prices/sizes

- `TradeCollection`: Efficient trade collection management
  - `add_trade()`: Add a trade to collection
  - `remove_trade()`: Remove a trade
  - `get_by_id()`: Fast O(1) lookup by ID
  - `get_by_date_range()`: Filter trades by date
  - `get_by_type()`: Filter by trade type
  - `calculate_total_pnl()`: Calculate total P&L
  - `to_dataframe()`: Convert to pandas DataFrame

#### 4. Unified GUI (`unified_gui.py`)
**Purpose**: Main application interface combining all features

**Key Components**:
- **Configuration Tab**: Model parameters and data paths
- **Walk-Forward Tab**: Backtesting and validation
- **Performance Tab**: Metrics and equity curves
- **Trade Visualization Tab**: Trade list and details (feature-flagged)
- **Advanced Charts Tab**: PyQtGraph integration (feature-flagged)
- **VectorBT Import Tab**: Data import utilities (feature-flagged)

**Key Methods**:
- `create_*_tab()`: Tab creation methods
- `launch_pyqtgraph_window()`: Launch PyQtGraph in separate window
- `import_from_vectorbt()`: Import VectorBT data
- `update_trade_display()`: Refresh trade visualization

### OMtree Components

#### 1. OMtree Model (`OMtree_model.py`)
**Purpose**: Machine learning decision tree ensemble

**Key Functions**:
- `train_model()`: Train decision tree ensemble
- `predict()`: Generate predictions
- `calculate_feature_importance()`: Analyze feature importance

#### 2. OMtree Preprocessing (`OMtree_preprocessing.py`)
**Purpose**: Data preprocessing for ML model

**Key Functions**:
- `load_data()`: Load and parse CSV data
- `engineer_features()`: Create technical indicators
- `normalize_features()`: Normalize feature values
- `split_data()`: Train/validation/test split

#### 3. OMtree Validation (`OMtree_validation.py`)
**Purpose**: Walk-forward validation implementation

**Key Functions**:
- `walk_forward_validation()`: Main validation loop
- `evaluate_window()`: Evaluate single window
- `aggregate_results()`: Combine window results

### Integration Components

#### 1. VectorBT Loader (`src/trading/integration/vbt_loader.py`)
**Purpose**: Import data from VectorBT backtesting

**Key Methods**:
- `load_trades()`: Load trade records
- `load_portfolio()`: Load portfolio data
- `load_signals()`: Load signal data
- `load_metrics()`: Load performance metrics

#### 2. PyQtGraph Charts (`src/trading/visualization/charts.py`)
**Purpose**: High-performance real-time charting

**Key Classes**:
- `PyQtGraphWindow`: Main chart window
- `CandlestickItem`: Candlestick chart implementation
- `TradeMarker`: Trade visualization overlay

## Data Flow

### 1. OMtree to Trade Visualization
```
CSV Data → OMtree Preprocessing → Feature Engineering → 
ML Model → Predictions → Unified Pipeline → Trade Visualization
```

### 2. VectorBT to GUI
```
VectorBT Data → VBT Loader → Unified Pipeline → 
TradeCollection → GUI Display
```

### 3. Unified Pipeline Conversion
```
OMtree DataFrame ←→ UnifiedDataFormat ←→ TradeData/Collection
                            ↓
                    Common Interface for
                    Both Systems
```

## Configuration Files

### feature_flags.json
Controls feature availability:
```json
{
  "flags": {
    "use_unified_gui": true,
    "show_trade_visualization_tab": true,
    "show_pyqtgraph_charts": true,
    "show_vectorbt_import": true,
    "unified_data_pipeline": true
  }
}
```

### OMtree_config.ini
Model configuration:
```ini
[DEFAULT]
n_estimators = 100
max_depth = 5
min_samples_split = 20
validation_split = 0.2
walk_forward_windows = 10
```

## Testing Strategy

### Unit Tests
- `test_trade_data.py`: Trade data structure tests
- `test_unified_pipeline.py`: Pipeline conversion tests
- Individual module tests for each component

### Integration Tests
- `test_gui_integration.py`: GUI component integration
- End-to-end data flow tests
- Performance tests with large datasets

### Coverage Requirements
- Minimum 80% code coverage
- 100% coverage for critical paths (trade data, pipeline)

## Performance Considerations

### Optimizations
1. **TradeCollection**: Uses dict for O(1) ID lookups
2. **Pipeline**: Vectorized operations with pandas/numpy
3. **GUI**: Lazy loading of tabs, threaded operations
4. **Charts**: PyQtGraph for GPU-accelerated rendering

### Benchmarks
- Trade collection: 10,000 trades < 0.01s lookup
- Pipeline conversion: 10,000 rows < 5s
- GUI responsiveness: < 100ms for user actions

## Security Considerations

1. **Input Validation**: All trade data validated on creation
2. **Feature Flags**: Gradual rollout of new features
3. **Error Handling**: Comprehensive try/catch blocks
4. **File Access**: Restricted to project directories

## Future Enhancements

### Planned Features
1. Real-time data feed integration
2. Additional ML models (LSTM, Random Forest)
3. Advanced risk management tools
4. Multi-asset portfolio support
5. Cloud deployment capabilities

### Technical Debt
1. Some GUI tests need fixing for full coverage
2. Performance optimization for very large datasets (>100k trades)
3. Better error recovery mechanisms
4. More comprehensive logging

## Development Workflow

### Adding New Features
1. Create feature flag in `feature_flags.json`
2. Implement behind `@feature_flag` decorator
3. Write tests (minimum 80% coverage)
4. Deploy dark (flag disabled)
5. Monitor for 24 hours
6. Enable flag gradually
7. Remove flag after stable

### Refactoring Guidelines
- Follow `.claude/file-refactor.md` principles
- Maximum 150 lines per refactor
- Always maintain backward compatibility
- Test before and after changes

## API Reference

### Public APIs

#### FeatureFlags API
```python
flags = get_feature_flags()
flags.enable('feature_name')
flags.disable('feature_name')
is_enabled = flags.is_enabled('feature_name')
```

#### TradeCollection API
```python
collection = TradeCollection(trades)
collection.add_trade(trade)
trade = collection.get_by_id(trade_id)
df = collection.to_dataframe()
total_pnl = collection.calculate_total_pnl()
```

#### DataPipelineAdapter API
```python
adapter = get_data_adapter()
unified = adapter.omtree_to_unified(df, config)
df = adapter.unified_to_omtree(unified)
unified = adapter.trades_to_unified(trades)
trades = adapter.unified_to_trades(unified)
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies in requirements.txt are installed
2. **Feature Not Showing**: Check feature flags in feature_flags.json
3. **Data Conversion Errors**: Verify data format matches expected schema
4. **GUI Not Responding**: Check for blocking operations, use threading
5. **Test Failures**: Run with pytest -v for detailed output

### Debug Mode
Enable debug mode in feature_flags.json:
```json
{
  "flags": {
    "debug_mode": true,
    "verbose_logging": true
  }
}
```

## Support and Contact

For issues, questions, or contributions:
- GitHub Issues: https://github.com/skyeAssetManagement/pythonBT/issues
- Documentation: See HOW-TO-GUIDE.md for usage instructions

---

Last Updated: 2025-09-05
Version: 1.0.0