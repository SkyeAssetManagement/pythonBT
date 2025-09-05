# GitHub Issue: Add Trade Triangle Markers to Chart

## Issue Title
Feature Request: Draw trade entry/exit triangles on price chart

## Description
Currently, trade markers are displayed as small arrows that can be difficult to see. We need to implement more visible trade markers using triangles, similar to how professional trading platforms like AmiBroker display trades.

## Current Behavior
- Trade markers use PyQtGraph's 't' (down arrow) and 't1' (up arrow) symbols
- Markers are small (size=25) and can be hard to spot on busy charts
- No clear visual distinction between entry and exit points

## Desired Behavior
- **Entry trades (Buy)**: Green upward-pointing triangles below the candle low
- **Exit trades (Sell)**: Red downward-pointing triangles above the candle high
- **Short entries**: Red downward-pointing triangles above the candle high
- **Short covers**: Green upward-pointing triangles below the candle low
- Triangles should be larger and more visible than current arrows
- Should scale appropriately with zoom level

## Implementation Details

### Proposed Changes

1. **Replace arrow symbols with triangle markers**
   - Use PyQtGraph's triangle symbols or custom path drawing
   - Increase marker size for better visibility

2. **Color coding**
   - Long entries: Green triangles pointing up (▲)
   - Long exits: Red triangles pointing down (▼)
   - Short entries: Red triangles pointing down (▼)
   - Short covers: Green triangles pointing up (▲)

3. **Positioning**
   - Entry triangles: Below candle low with 2-3% price offset
   - Exit triangles: Above candle high with 2-3% price offset
   - Ensure triangles don't overlap with price candles

4. **Files to modify**
   ```
   src/dashboard/chart_widget.py - Main chart rendering
   step6_complete_final.py - VisPy chart implementation
   ```

## Code Example
```python
# Instead of current implementation:
buy_scatter = pg.ScatterPlotItem(
    x=buy_x, y=buy_y,
    symbol='t1', size=25,  # Small arrow
    brush='#00CC00',
    pen=pg.mkPen('#000000', width=2)
)

# Proposed implementation:
buy_scatter = pg.ScatterPlotItem(
    x=buy_x, y=buy_y,
    symbol='t',  # Triangle
    size=40,     # Larger size
    brush=pg.mkBrush(0, 255, 0, 200),  # Semi-transparent green
    pen=pg.mkPen('darkgreen', width=2)  # Dark green outline
)
```

## Benefits
- Improved trade visibility on charts
- Better visual feedback for trade analysis
- Consistent with professional trading platforms
- Easier to spot entry/exit points during backtesting review

## Acceptance Criteria
- [ ] Trade triangles are clearly visible at all zoom levels
- [ ] Colors correctly represent trade direction (green=buy, red=sell)
- [ ] Triangles positioned appropriately relative to candles
- [ ] No performance degradation with many trades
- [ ] Works with both PyQtGraph and VisPy implementations

## Additional Context
This enhancement will make it easier for traders to visually analyze their backtesting results and understand trade placement relative to price action.

## Labels
- enhancement
- visualization
- chart
- trades

## Priority
Medium - This is a quality of life improvement that enhances the user experience but doesn't block core functionality.