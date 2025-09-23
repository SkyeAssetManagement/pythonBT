# Project TODOs - PythonBT Trading System

## Immediate Priorities

### Data Integration
- [ ] **Live Data Feed** - Connect to real-time market data providers
- [ ] **Database Backend** - Migrate from CSV to TimescaleDB for time-series data
- [ ] **Data Validation** - Add checks for data quality and missing bars

### Trading Features
- [ ] **Stop Loss/Take Profit** - Implement automatic exit conditions
- [ ] **Portfolio Management** - Support multiple simultaneous strategies
- [ ] **Commission Models** - Add realistic broker fee structures

## PyQtGraph Enhancements

### Visualization
- [ ] **Multi-timeframe Display** - Show M5, H1, D1 on same chart
- [ ] **Custom Indicators** - GUI for adding technical indicators
- [ ] **Performance Dashboard** - Live P&L, drawdown, win rate display
- [ ] **Trade Annotations** - Show entry/exit reasons on chart

### User Interface
- [ ] **Keyboard Shortcuts** - Quick navigation and actions
- [ ] **Chart Templates** - Save/load custom layouts
- [ ] **Dark/Light Themes** - User preference support

## ML/AI Integration

### OMTree Improvements
- [ ] **GPU Acceleration** - CUDA support for faster training
- [ ] **Feature Engineering** - Automatic technical indicator generation
- [ ] **Walk-Forward Analysis** - Automated parameter optimization
- [ ] **Ensemble Models** - Combine multiple strategies

## Testing & Quality

### Coverage Goals
- [ ] **Unit Tests** - Reach 80% code coverage
- [ ] **Integration Tests** - Test strategy combinations
- [ ] **Performance Tests** - Benchmark with 1M+ bars
- [ ] **Regression Suite** - Prevent breaking changes

## Recently Completed (2025-09-23)
✅ Config.yaml integration with bar lag and execution formulas
✅ $1 position sizing for clean percentage calculations
✅ Sub-linear performance scaling (0.2-0.5x efficiency)
✅ PyQtGraph handling 377,690+ bars smoothly

## Notes
- PyQtGraph system fully functional with 377,690 bars
- All critical timestamp and navigation issues resolved
- OMTree ML pipeline operational with walk-forward validation
- Config.yaml fully integrated with proper bar lag and execution formulas
- $1 position sizing enables clean percentage-based profit calculations
- Both systems ready for production use

---

*Last Updated: 2025-09-23*
*Status: Active Development*