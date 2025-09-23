# Project TODOs - Stepwise Plan for Unified Trading System

## CRITICAL REQUIREMENT
**The chart rendering MUST remain IDENTICAL to Screenshot 2025-09-22 164318.png**
- No changes to pyqtgraph_range_bars_final.py rendering logic
- Preserve render_range() method exactly as is
- Keep EnhancedCandlestickItem with 5 arguments
- Maintain time axis formatting
- Keep hover data window working

## Phase 1: Foundation (DO NOT MODIFY CHART)
### Step 1.1: Create Standalone Execution Engine
- [ ] Create `src/trading/core/standalone_execution.py`
- [ ] Copy execution logic from strategies/base.py
- [ ] Add config.yaml price formula evaluation
- [ ] Test independently without touching visualization
- [ ] Verify lag and formula calculations work

### Step 1.2: Create Trade Data Structure
- [ ] Create `src/trading/core/trade_types.py`
- [ ] Define TradeRecord class with all fields needed
- [ ] Include: signal_bar, execution_bar, lag, prices, P&L
- [ ] Make compatible with existing trade_panel.py

### Step 1.3: Test Execution Engine Separately
- [ ] Create test_execution_engine.py
- [ ] Verify lag calculations (signal bar + N)
- [ ] Verify price formulas work correctly
- [ ] Test with sample data, NOT integrated yet

## Phase 2: P&L Display Update (MINIMAL CHANGES)
### Step 2.1: Update Trade Panel Display Only
- [ ] Modify trade_panel.py P&L display format
- [ ] Change from "$0.0125" to "1.25%" display
- [ ] Keep internal values as decimals
- [ ] DO NOT change any calculation logic

### Step 2.2: Add P&L Summary Row
- [ ] Add cumulative P&L % to trade panel
- [ ] Add win rate percentage
- [ ] Keep existing columns intact

## Phase 3: Strategy Interface (NO CHART CHANGES)
### Step 3.1: Create Strategy Wrapper
- [ ] Create `src/trading/strategies/strategy_wrapper.py`
- [ ] Wrap existing strategies (SMA, RSI)
- [ ] Add metadata without changing strategy logic
- [ ] Keep backward compatibility

### Step 3.2: Add Indicator Metadata
- [ ] Add indicator definitions to wrapped strategies
- [ ] Define colors, line styles, plot types
- [ ] DO NOT auto-plot yet, just define

## Phase 4: Integration Layer (PRESERVE EXISTING)
### Step 4.1: Create Adapter for Strategy Runner
- [ ] Create `src/trading/strategy_runner_adapter.py`
- [ ] Adapter between old strategy_runner.py and new engine
- [ ] Keep existing strategy_runner.py untouched
- [ ] Route execution through new engine if available

### Step 4.2: Add Configuration Switch
- [ ] Add use_unified_engine flag to config.yaml
- [ ] Default to FALSE (use existing code)
- [ ] Allow testing new engine without breaking old

## Phase 5: Careful Integration
### Step 5.1: Test Side-by-Side
- [ ] Run old system and new system in parallel
- [ ] Compare trade results
- [ ] Verify identical P&L calculations
- [ ] Check execution prices match

### Step 5.2: Create New Launcher (Keep Old)
- [ ] Create launch_unified_system.py
- [ ] Keep launch_pyqtgraph_with_selector.py unchanged
- [ ] New launcher uses unified components
- [ ] Old launcher remains as fallback

## Phase 6: Validation
### Step 6.1: Comprehensive Testing
- [ ] Test with all 377,690 bars
- [ ] Verify chart rendering unchanged
- [ ] Check trade markers still work
- [ ] Confirm hover data working
- [ ] Test strategy execution with lag

### Step 6.2: Performance Verification
- [ ] Measure rendering speed
- [ ] Check memory usage
- [ ] Verify no regression in performance

## What NOT to Do
❌ DO NOT modify pyqtgraph_range_bars_final.py
❌ DO NOT change EnhancedCandlestickItem initialization
❌ DO NOT alter render_range() method
❌ DO NOT modify coordinate systems
❌ DO NOT change time axis formatting
❌ DO NOT break hover data window
❌ DO NOT merge failed commits

## Success Criteria
✓ Chart looks EXACTLY like Screenshot 2025-09-22 164318.png
✓ All 377,690 bars accessible
✓ Trade execution uses config.yaml settings
✓ P&L displays as percentage
✓ Unified engine for visualization and backtesting
✓ Old system still works as fallback

## Current Status
- **Baseline Commit**: `d22acf6` - Working chart with config.yaml execution
- **Branch**: feature/modular-backtesting-refactor
- **Next Step**: Start Phase 1.1 - Create standalone execution engine

---
*Last Updated: 2025-09-23*
*Approach: Incremental changes preserving working chart*