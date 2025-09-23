# Project TODOs - PythonBT Trading System

## CRITICAL - Fix Unified Launcher Issues
### Hover Data Not Working
- [ ] Fix data structure mismatch - hover expects specific keys
- [ ] Ensure proxy widget connects to chart properly
- [ ] Debug why "no chart data available" appears despite rendering

### Trade Loading Not Working
- [ ] Fix trade generation when "System" selected
- [ ] Ensure trades appear on chart with proper markers
- [ ] Verify trade panel displays trades with P&L %

### Chart Integration
- [ ] Ensure `self.current_x_range` properly initialized
- [ ] Fix timing of `load_configured_trades()` call
- [ ] Verify data conversion for strategy execution

## Next Phase - Complete Integration
### Trade Panel Enhancement
- [ ] Replace standard panel with enhanced_trade_panel.py
- [ ] Display P&L as percentages not dollars
- [ ] Add cumulative P&L summary row

### Strategy Runner Integration
- [ ] Hook up strategy_runner_adapter to UI
- [ ] Allow switching between legacy and unified engines
- [ ] Test all strategies with proper lag

### Configuration Management
- [ ] Create UI for editing config.yaml settings
- [ ] Add validation for execution formulas
- [ ] Allow per-strategy configuration overrides

## Testing & Validation
### Comprehensive Testing
- [ ] Test with full 377,690 bar dataset
- [ ] Verify performance with unified engine
- [ ] Compare trade results old vs new
- [ ] Ensure no regression in chart performance

### Bug Fixes
- [ ] Fix FutureWarning in rsi_momentum.py (fillna method)
- [ ] Clean up import warnings
- [ ] Remove debug print statements

## Documentation
### User Documentation
- [ ] Create user guide for unified system
- [ ] Document config.yaml options
- [ ] Add examples of execution formulas

### Developer Documentation
- [ ] Document API for execution engine
- [ ] Create integration guide
- [ ] Add architecture diagrams

## Future Enhancements
### Performance Optimization
- [ ] Profile unified engine performance
- [ ] Optimize trade generation for large datasets
- [ ] Implement caching for indicator calculations

### Feature Additions
- [ ] Add more execution price formulas
- [ ] Implement stop-loss and take-profit
- [ ] Add multi-timeframe support
- [ ] Create backtesting report generator

---
*Last Updated: 2025-09-23*