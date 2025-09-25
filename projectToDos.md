# Project TODOs - PythonBT

## Current Status
Last Updated: 2025-09-26 - Architectural Separation Complete

## COMPLETED - P&L Calculation Fixes (2025-09-24)

### ✅ All P&L Issues Resolved
User Request: Fix P&L calculation issues in PyQtGraph display
- ✅ Total P&L showing correct values in summary pane
- ✅ P&L % in trade list now matches data pane values
- ✅ Implemented VectorBT-based P&L calculations for consistency

### Final Fixes Applied:

1. **✅ VectorBT Integration for P&L** (COMPLETED 14:05)
   - **Issue**: Strategies were calculating P&L individually instead of using vectorized approach
   - **Fix Applied**: Integrated VectorBT Portfolio.from_signals() for proper P&L calculation
   - **File Modified**: `src/trading/core/strategy_runner_adapter.py` - Added _calculate_trades_with_vectorbt method

2. **✅ Trade List Display Fix** (COMPLETED 14:06)
   - **Issue**: Trade list P&L values were off by factor of 100 (double multiplication)
   - **Fix Applied**: Removed extra *100 multiplication in display code
   - **Files Modified**:
     - `src/trading/visualization/enhanced_trade_panel.py` - Lines 123-124, 134-135

## Previous Fixes

### 1. ✅ Total P&L Calculation Fix (COMPLETED)
- **Issue**: Total P&L was incorrectly summing individual trade percentages instead of properly compounding returns
- **Fix Applied**: Modified `enhanced_trade_panel.py` to use proper compounding formula: (1 + r1) * (1 + r2) * ... * (1 + rn) - 1
- **Files Modified**:
  - `src/trading/visualization/enhanced_trade_panel.py` - Lines 52-63 and 359-370

### 2. ✅ Trade List Column Sorting (COMPLETED)
- **Issue**: Trade list table did not support column sorting
- **Fix Applied**: Added sorting functionality to all columns in the trade list
- **Features Added**:
  - Click any column header to sort ascending
  - Click again to sort descending
  - Works for all columns: Trade #, DateTime, Type, Price, Size, P&L %, Cum P&L %, Bar #
- **Files Modified**:
  - `src/trading/visualization/enhanced_trade_panel.py` - Added sorting methods and handlers

## Testing Requirements

### Test the following scenarios:
1. ✅ Verify Total P&L calculation shows correct compounded return
2. ✅ Test sorting on each column (ascending and descending)
3. ✅ Verify P&L % column sorting works correctly (largest wins/losses)
4. ✅ Confirm DateTime sorting maintains chronological order
5. ✅ Test with trades that have both positive and negative returns

## System Architecture

### PyQtGraph Range Bars Components:
- **Main Chart**: `src/trading/visualization/pyqtgraph_range_bars_final.py`
- **Trade Panel**: `src/trading/visualization/enhanced_trade_panel.py`
- **Trade Data**: `src/trading/visualization/trade_data.py`
- **Trade Marks**: `src/trading/visualization/simple_white_x_trades.py`

## COMPLETED - Minor Fixes (2025-09-25)

### ✅ Requirements & Code Cleanup
- ✅ **CRITICAL**: Added missing PyQt5 and pyqtgraph dependencies to requirements.txt
- ✅ Cleaned up TODO comments in launch_unified_system.py and trade_panel.py
- ✅ Conducted comprehensive codebase analysis and architecture review

### ✅ Initial Time-Based TWAP Implementation (NEEDS REFINEMENT)
- ✅ Built 5-step vectorized TWAP process for range bars
- ✅ Implemented time calculation for variable bar durations
- ✅ Created VectorBT Pro integration adapter
- ✅ Comprehensive test suite (ALL TESTS PASSED)
- **ISSUE IDENTIFIED**: Artificial phase splitting and inefficient time calculations

## COMPLETED - Architectural Separation (2025-09-26)

### ✅ Complete System Overhaul - Headless Architecture
**Problem Solved**: Original unified system hanging with 192k+ signals on strategy execution
**Solution**: Complete separation of backtesting from chart visualization

#### **✅ Key Achievements:**
1. **✅ Headless Backtesting Engine**: Independent execution with organized CSV storage
2. **✅ Chart-Only Visualization**: Same 1000-bar rendering performance without hanging
3. **✅ Button-Based Integration**: Load previous backtests OR run new + auto-load
4. **✅ P&L Integration**: Raw P&L and cumulative profit columns in CSV output
5. **✅ Error-First Approach**: Clear errors instead of silent fallbacks
6. **✅ Folder Structure Organization**: Timestamped results with complete metadata

#### **✅ Files Created/Modified:**
- **✅ src/trading/backtesting/headless_backtester.py** - Core headless engine
- **✅ src/trading/visualization/backtest_result_loader.py** - CSV to chart converter
- **✅ launch_unified_system.py** - Updated with button controls (no separate files)
- **✅ src/trading/core/optimized_twap_adapter.py** - Chunked processing for large datasets

#### **✅ Architecture Benefits:**
- **No Hanging**: Backtesting runs independent of GUI
- **Same Chart Performance**: Exact 1000-bar rendering as before
- **Result Persistence**: All backtests saved to organized CSV structure
- **User Control**: Manual button triggers prevent unwanted executions

## ACTIVE DEVELOPMENT - TWAP System Implementation (2025-09-26)

### 🔄 NEXT: Volume-Weighted Natural Phases
**Current Focus**: Complete TWAP system with volume-proportional allocation

#### **Implementation Requirements:**
1. **🔄 Volume-Weighted Allocation**: Size per bar = (bar_volume/total_volume) × position_size
2. **🔄 Natural Phases**: Each bar = one phase (no artificial splitting)
3. **🔄 Minimum Time Enforcement**: 5-minute execution spread across available bars
4. **🔄 Import Fixes**: Resolve module import issues for TWAP adapter

#### **Files Needing Completion:**
- `src/trading/core/time_based_twap_execution.py` - Volume-weighted algorithm
- `src/trading/core/vectorbt_twap_adapter.py` - VectorBT integration
- `src/trading/core/minimum_time_phased_entry.py` - Range bar time enforcement

#### **Success Criteria:**
- [ ] TWAP imports resolve correctly
- [ ] Volume-proportional allocation working
- [ ] Minimum 5-minute execution time enforced
- [ ] Natural phases (7 bars = 7 phases) confirmed via CSV output

## Future Enhancements (Optional)
- [ ] Add export functionality for trade list
- [ ] Add filtering options (by date range, P&L range, trade type)
- [ ] Add performance metrics (Sharpe ratio, max drawdown)
- [ ] Add equity curve visualization

## Notes
- The system uses PyQt5 and pyqtgraph for visualization
- Trade P&L is calculated based on $1 invested per trade
- Cumulative P&L properly compounds returns
- Commission and execution lag are tracked and displayed in summary