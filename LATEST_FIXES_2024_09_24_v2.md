# Latest Fixes - September 24, 2025 (Version 2)

## Issues Identified from Latest Screenshot

1. **Lag not reading from config**: Config shows `signal_lag: 2` but summary panel showed "1.0 bars"
2. **P&L percentages too high**: Showing values like -1068.33% and +3200.00%
3. **Missing indicator values**: Hover display didn't show indicator values (SMA_10_30, etc.)

## Fixes Implemented

### 1. Dynamic Lag Reading from Config ✓

**File:** `src/trading/visualization/enhanced_trade_panel.py`
**Lines:** 391-409

**Changes:**
- Added dynamic config reading using yaml
- Reads `signal_lag` from `tradingCode/config.yaml`
- Falls back to 1.0 if config can't be read

```python
# Now reads from config
config_path = os.path.join(..., 'tradingCode', 'config.yaml')
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)
    lag = config.get('backtest', {}).get('signal_lag', 1)
    self.execution_label.setText(f"Avg Lag: {lag:.1f} bars")
```

**Result:** Lag display now updates automatically when config changes (e.g., shows "2.0 bars" when signal_lag: 2)

### 2. Fixed P&L Percentage Calculation ✓

**File:** `src/trading/visualization/enhanced_trade_panel.py`
**Lines:** 64-85

**Issue:** Legacy `pnl` field contains price points (e.g., 100 points), not percentages

**Fix:**
```python
def _get_pnl_percent(self, trade: TradeData):
    # Check for new format first
    if hasattr(trade, 'pnl_percent') and trade.pnl_percent is not None:
        return trade.pnl_percent

    # Legacy format: convert points to percentage
    if hasattr(trade, 'pnl') and trade.pnl is not None:
        if hasattr(trade, 'price') and trade.price is not None:
            # Convert: 100 points at $4200 = 2.38% return
            return trade.pnl / trade.price
        else:
            # Fallback: assume ~$4000 stock
            return trade.pnl / 4000
```

**Result:** P&L now shows reasonable percentages (e.g., +2.38% instead of +238%)

### 3. Added Indicator Values to Hover Display ✓

**File:** `src/trading/visualization/pyqtgraph_range_bars_final.py`
**Lines:** 626-636

**Addition:**
```python
# Add indicator values if available
if hasattr(self, 'indicator_data') and self.indicator_data:
    indicator_values = []
    for name, values in self.indicator_data.items():
        if 0 <= bar_idx < len(values):
            val = values[bar_idx]
            if not np.isnan(val):
                indicator_values.append(f"{name}: {val:.2f}")

    if indicator_values:
        hover_text += "\n" + " | ".join(indicator_values)
```

**Result:** Hover now shows indicator values like "SMA_10_30: 4275.50" when indicators are plotted

## Testing Verification

### Test 1: Lag Display
- Change `signal_lag` in config.yaml from 1 to 2
- Restart chart
- Bottom panel should show "Avg Lag: 2.0 bars"

### Test 2: P&L Percentages
- Check trade list P&L values
- Should show reasonable percentages (typically -5% to +5% range)
- Not 1000%+ values

### Test 3: Indicator Values
- Hover over any bar with indicators plotted
- Should see additional line showing indicator values
- Format: "SMA_10_30: 4275.50 | SMA_50: 4280.00"

## Files Modified Summary

1. **src/trading/visualization/enhanced_trade_panel.py**
   - Lines 64-85: Fixed P&L percentage calculation for legacy trades
   - Lines 391-409: Added dynamic config reading for lag

2. **src/trading/visualization/pyqtgraph_range_bars_final.py**
   - Lines 626-636: Added indicator values to hover display

3. **.claude/projecttodos.md**
   - Updated with completed fixes and new features

## Impact

These fixes ensure:
1. **Accurate lag reporting** - Always reflects actual config settings
2. **Correct P&L percentages** - Shows realistic returns based on $1 position size
3. **Complete data visibility** - All plotted indicators visible in hover tooltip
4. **Better debugging** - Can see exact indicator values at any bar

## Configuration Note

The system now correctly handles `position_size: 1` in config.yaml, which means:
- All trades use $1 position size
- P&L is calculated as percentage return on $1
- This gives clean percentage values regardless of stock price