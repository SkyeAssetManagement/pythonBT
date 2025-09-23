# Project TODOs - PythonBT Trading System

## Current Focus: System Integration & Enhancement

### Strategy System Improvements
- [ ] Add minimum bars between trades parameter to reduce noise
- [ ] Implement stop-loss and take-profit functionality
- [ ] Create strategy performance metrics dashboard
- [ ] Add strategy parameter optimization framework

### Unified Execution Engine Migration
- [ ] Complete integration of unified engine with all strategies
- [ ] Add UI toggle for switching between legacy/unified modes
- [ ] Create validation tests comparing legacy vs unified results
- [ ] Document migration guide for users

### User Interface Enhancements
- [ ] Add configuration UI for editing config.yaml settings
- [ ] Create indicator overlay panel for technical indicators
- [ ] Implement trade statistics summary panel
- [ ] Add export functionality for trades and performance metrics

### Data & Performance
- [ ] Profile and optimize for 10M+ bar datasets
- [ ] Implement data caching for faster repeated loads
- [ ] Add support for multiple timeframes
- [ ] Create data validation and cleaning utilities

### Documentation & Testing
- [ ] Create comprehensive user guide
- [ ] Add integration tests for full workflow
- [ ] Document all execution price formulas
- [ ] Create video tutorials for common workflows

## Completed (2025-09-23)

### Critical Fixes ✅
- Fixed hover data KeyErrors with proper dictionary checks
- Fixed strategy runner "no chart data" in unified system
- Added trade panel scrolling fix with get_first_visible_trade()
- Improved strategy execution feedback with color coding

### System Improvements ✅
- Verified performance with 6.6M bar datasets
- Added pass_data_to_trade_panel() for data connection
- Implemented excessive trade warnings (>1000 trades)
- Enhanced trade panel with P&L percentage display

## Known Issues to Address

### High Priority
- SMA strategy generates excessive trades with short periods
- Need better default parameters for strategies
- Unified engine not enabled by default

### Medium Priority
- Trade timestamp display sometimes shows None
- Large datasets (>10M bars) may cause memory issues
- Strategy runner UI could be more intuitive

### Low Priority
- Add more built-in strategies
- Improve error messages for data loading failures
- Add keyboard shortcuts for common operations

## Future Roadmap

### Phase 1: Stabilization (Current)
Focus on fixing critical bugs and improving user experience

### Phase 2: Enhancement
Add advanced features like optimization and real-time data

### Phase 3: Production
Polish for production use with comprehensive documentation