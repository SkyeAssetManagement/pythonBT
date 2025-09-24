# Project TODOs - PythonBT Trading System

## High Priority Development

### Commission & Slippage Testing Framework
- [ ] **Unit Testing for Config Flow**
  - Create test scripts to verify commission/slippage data flows from config.yaml → VectorBT → chart visualizer
  - Test edge cases: zero commission, high slippage, negative values
  - Validate proper P&L impact calculations with different commission structures

- [ ] **Integration Testing**
  - End-to-end tests from config.yaml settings to final trade P&L display
  - Verify commission deduction appears in trade list and summary panel
  - Test slippage impact on entry/exit prices in hover data pane

- [ ] **VectorBT Configuration Integration**
  - Ensure config.yaml commission/slippage values properly feed into Portfolio.from_signals()
  - Add validation for reasonable commission/slippage ranges
  - Create config validation utilities

### Phased Entry/Exit Implementation
- [ ] **Time-Based Entry Phasing**
  - Add config parameter for entry/exit phase duration (e.g., "30min", "2hours")
  - Implement bar count calculation based on data frequency
  - Support multiple entry/exit executions over phase period

- [ ] **Trade List Enhancements**
  - Add "Entry Bars" and "Exit Bars" columns showing phase duration
  - Display bar count in trade list sortable columns
  - Color code phased entries differently from instant entries

- [ ] **Hover Data Integration**
  - Show entry/exit bar counts in chart hover pane
  - Display phase progress information
  - Add visual indicators for phased vs instant trades

### Array Processing Preservation
- [ ] **Performance Validation**
  - Maintain current 60+ FPS chart rendering performance
  - Preserve vectorized P&L calculations with VectorBT
  - Ensure no regression in 6M+ bar data handling

- [ ] **Feature Compatibility**
  - Keep all existing trade syncing functionality
  - Maintain proper chart scaling with phased entries
  - Preserve sortable trade list performance

- [ ] **Memory Efficiency**
  - Ensure phased entry data doesn't increase memory usage significantly
  - Use array processing for phase calculations
  - Maintain efficient viewport rendering

## Medium Priority Features

### Enhanced Visualization
- [ ] Add equity curve below trade list
- [ ] Implement trade filtering by date/P&L range
- [ ] Create performance metrics dashboard (Sharpe, drawdown)
- [ ] Add export functionality for trade results

### Strategy Development
- [ ] Add minimum bars between trades parameter
- [ ] Implement stop-loss/take-profit orders
- [ ] Create additional strategy templates (Bollinger, MACD)
- [ ] Add strategy combination framework

### User Interface Improvements
- [ ] Add configuration UI for config.yaml editing
- [ ] Implement dark/light theme toggle
- [ ] Add keyboard shortcuts for common actions
- [ ] Create indicator overlay management panel

## Bug Fixes

### High Priority
- [ ] Fix memory leak in long-running chart sessions
- [ ] Resolve excessive trade generation in some strategies
- [ ] Fix occasional timestamp display showing None

### Medium Priority
- [ ] Windows-specific path issues in some scripts
- [ ] Config changes requiring application restart
- [ ] PyQt6 compatibility (currently PyQt5 only)

## Infrastructure

### Testing & Quality
- [ ] Increase test coverage to >80%
- [ ] Add performance benchmarks
- [ ] Create regression test suite
- [ ] Implement continuous integration

### Documentation
- [ ] Create comprehensive user guide
- [ ] Add video tutorials
- [ ] Write strategy development guide
- [ ] Generate API documentation

## Completed (2025-09-24)

### P&L System Overhaul ✅
- [x] **VectorBT Integration**
  - Implemented centralized P&L calculation using Portfolio.from_signals()
  - Eliminated inconsistent strategy-level P&L calculations
  - Added vectorized performance improvements

- [x] **Display Fixes**
  - Fixed trade list P&L values off by factor of 100
  - Corrected double multiplication in percentage display
  - Aligned trade list and summary panel values

- [x] **Trade Data Pipeline**
  - Added proper TradeRecord to TradeData conversion
  - Fixed missing pnl_percent attribute transfers
  - Implemented simple summation for cumulative P&L

### Documentation Updates ✅
- [x] Updated CODE_DOCUMENTATION.md with current architecture
- [x] Documented VectorBT integration patterns
- [x] Added performance specifications and usage examples
- [x] Created comprehensive component descriptions

## Known Issues
- VectorBT Pro dependency for full P&L functionality
- Some Windows-specific file paths in legacy modules
- PyQt5 requirement (PyQt6 not yet supported)
- Maximum practical dataset size ~10M bars