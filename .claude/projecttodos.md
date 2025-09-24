# Project TODOs - PythonBT Trading System

## Current Sprint Focus

### Critical Bug Fixes
- [ ] Fix memory leak in long-running chart sessions
- [ ] Resolve SMA strategy excessive trade generation
- [ ] Fix trade timestamp display showing None occasionally

### Performance Optimization
- [ ] Optimize for 10M+ bar datasets
- [ ] Implement data chunking for memory efficiency
- [ ] Add progress indicators for long operations
- [ ] Profile and optimize viewport rendering

## Next Release Features

### Strategy Enhancements
- [ ] Add minimum bars between trades parameter
- [ ] Implement stop-loss and take-profit orders
- [ ] Create Bollinger Bands strategy
- [ ] Add MACD strategy
- [ ] Implement strategy combination framework

### User Interface
- [ ] Add configuration UI for config.yaml editing
- [ ] Create indicator overlay panel
- [ ] Add trade statistics dashboard
- [ ] Implement equity curve visualization
- [ ] Add drawing tools (trend lines, channels)

### Data Management
- [ ] Add real-time data feed integration
- [ ] Implement multi-symbol backtesting
- [ ] Create data validation utilities
- [ ] Add automatic data updates
- [ ] Support multiple timeframes

## Infrastructure Improvements

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

### Data Accuracy Fixes ✅
- [x] **P&L Calculation Corrections**
  - Fixed 100x error by storing as decimal, displaying as percentage
  - Added legacy pnl field conversion (points/price)
  - Ensured $1 position size basis for all calculations

- [x] **Trade Classification Fixes**
  - Fixed shorts count: BUY/SELL = longs, SHORT/COVER = shorts
  - Corrected SELL being counted as short (now properly long exit)

- [x] **Dynamic Configuration**
  - Lag display now reads from config.yaml (signal_lag parameter)
  - Updates automatically when config changes
  - Shows actual configured lag (e.g., 2.0 bars)

- [x] **Enhanced Data Display**
  - Added indicator values to hover tooltip
  - Format: "SMA_10_30: 4275.50"
  - Shows all plotted indicators dynamically

### UI/UX Improvements ✅
- [x] **Simplified strategy runner interface**
  - Removed excessive warning messages about trade counts
  - Cleaned up status feedback to single line
  - Removed redundant trade type counts from status bar

- [x] **Enhanced trade summary panel**
  - Increased panel size by 50% (from 80-100px to 120-150px)
  - Added Longs/Shorts trade type counts
  - Fixed counts: BUY/SELL = longs, SHORT/COVER = shorts
  - Removed duplicate trade statistics display
  - Better organized 4-row layout for all metrics

- [x] **Fixed data window display issues**
  - ATR multiplier now shows 2 decimal places
  - Commission and slippage always display (show $0.00 if not available)
  - P&L shows as percentage matching trade list format
  - Fixed P&L calculation: now stored as decimal, displayed as percentage
  - Added proper lag calculation from actual trade data (reads from config.yaml)
  - Added indicator values to hover display (shows SMA_10_30, etc. when hovering)

### Major Fixes & Enhancements ✅
- [x] **Fixed strategy runner TradeCollection type error**
  - Corrected import paths in trade_types.py
  - Added proper type checking and conversion

- [x] **Fixed ATR data display (was showing 0.00)**
  - Added column detection for AUX1/ATR/atr
  - Created test_atr_data.py for ATR calculation
  - Fixed data loading in chart components

- [x] **Implemented signal lag system**
  - Configurable 1-10 bar delay between signal and execution
  - Proper tracking of signal_bar and execution_bar

- [x] **Added execution price formulas**
  - Support for custom formulas like "(H + L + C) / 3"
  - Formula evaluation at execution time

- [x] **Integrated commission calculations**
  - Fees and slippage properly deducted from P&L
  - Commission displayed in backtest summary

- [x] **Normalized P&L to $1 invested basis**
  - All profits calculated as percentage on $1
  - Consistent returns regardless of instrument price

- [x] **Enhanced trade panel display**
  - P&L shown as percentage to 2 decimal places
  - Cumulative P&L tracking
  - Backtest summary with win rate, total/avg P&L
  - Commission and execution lag statistics

## Completed (Previous Releases)

### Version 2.3.0
- Fixed strategy runner "no chart data" issue
- Added pass_data_to_trade_panel() method
- Improved strategy feedback with color coding
- Added excessive trade warnings

### Version 2.2.0
- Fixed hover data KeyErrors
- Fixed trade panel scrolling
- Verified 6.6M bar dataset performance

## Known Issues

### High Priority
- Memory usage increases over time in long sessions
- SMA strategy can generate 1000+ trades with short periods
- Windows-specific file paths in some scripts

### Medium Priority
- PyQt6 not supported (requires PyQt5)
- Large datasets (>10M bars) may cause crashes
- Config changes require restart

### Low Priority
- No keyboard shortcuts
- Limited color theme options
- No undo/redo for operations