# Project TODOs - PythonBT Trading System

## Active Development

### PyQtGraph Trade Panel Enhancements
- [ ] Add export functionality for trade list (CSV/Excel)
- [ ] Implement trade filtering (by date range, P&L range, trade type)
- [ ] Add equity curve visualization below trade list
- [ ] Create performance metrics (Sharpe ratio, max drawdown)
- [ ] Add trade grouping by day/week/month

### Machine Learning Improvements
- [ ] Integrate ML predictions with PyQtGraph visualization
- [ ] Add feature importance overlay on chart
- [ ] Create prediction confidence bands
- [ ] Implement ensemble model support
- [ ] Add real-time model retraining capability

## Bug Fixes

### High Priority
- [ ] Fix memory leak in long-running chart sessions
- [ ] Resolve SMA strategy excessive trade generation (>1000 trades)
- [ ] Fix trade timestamp display showing None occasionally

### Medium Priority
- [ ] Windows-specific path issues in some scripts
- [ ] Config changes requiring restart
- [ ] PyQt6 compatibility (currently PyQt5 only)

## Performance Optimization

### Data Handling
- [ ] Optimize for 10M+ bar datasets
- [ ] Implement data chunking for memory efficiency
- [ ] Add progress indicators for long operations
- [ ] Profile and optimize viewport rendering

## Feature Roadmap

### Trading Strategies
- [ ] Add minimum bars between trades parameter
- [ ] Implement stop-loss and take-profit orders
- [ ] Create Bollinger Bands strategy
- [ ] Add MACD strategy
- [ ] Implement strategy combination framework

### User Interface
- [ ] Add configuration UI for config.yaml editing
- [ ] Create indicator overlay panel
- [ ] Add drawing tools (trend lines, channels)
- [ ] Implement dark/light theme toggle
- [ ] Add keyboard shortcuts

### Data Management
- [ ] Add real-time data feed integration
- [ ] Implement multi-symbol backtesting
- [ ] Support multiple timeframes
- [ ] Create data validation utilities
- [ ] Add automatic data updates

## Infrastructure

### Testing & Quality
- [ ] Increase test coverage to >80%
- [ ] Add performance benchmarks
- [ ] Create regression test suite
- [ ] Implement CI/CD pipeline

### Documentation
- [ ] Create comprehensive user guide
- [ ] Add video tutorials
- [ ] Write strategy development guide
- [ ] Create API documentation

### Platform Support
- [ ] Test Linux compatibility
- [ ] Ensure macOS compatibility
- [ ] Create standalone executables
- [ ] Add Docker support

## Completed (2025-09-24)

### Trade Panel Fixes ✅
- [x] **Fixed Total P&L Calculation**
  - Corrected from sum to proper compounding: (1+r1)*(1+r2)*...-1
  - Files: enhanced_trade_panel.py

- [x] **Added Column Sorting**
  - Click headers to sort ascending/descending
  - Works on all columns: Trade #, DateTime, P&L %, etc.
  - Files: enhanced_trade_panel.py

- [x] **Fixed Trade Data Attributes**
  - kwargs now accessible as attributes
  - Files: trade_data.py

### Documentation ✅
- [x] Created CODEBASE_DOCUMENTATION.md with complete architecture overview
- [x] Created projectToDos.md for task tracking
- [x] Updated .claude/CODE_DOCUMENTATION.md with current design patterns

## Known Issues

### High Impact
- Memory usage increases over time in long sessions
- SMA strategy can generate excessive trades with short periods
- Large datasets (>10M bars) may cause crashes

### Medium Impact
- PyQt6 not supported (PyQt5 required)
- Config changes require application restart
- Windows-specific file paths in some scripts

### Low Impact
- No keyboard shortcuts implemented
- Limited color theme options
- No undo/redo for operations