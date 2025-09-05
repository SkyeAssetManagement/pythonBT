# VectorBT Pro: Uncompounded Cumulative Returns Guide

## Objective
Create an equity curve representing **cumulative uncompounded percentage returns** of a trading system using VectorBT Pro.

## Key Concept: Compounded vs Uncompounded Returns

| Return Type | Calculation | VectorBT Pro Command |
|------------|-------------|---------------------|
| **Uncompounded** | Sum of period returns | `pf.returns.cumsum()` |
| **Compounded** | Multiply (1+return) factors | `pf.get_cumulative_returns()` |

**Uncompounded returns** show the simple arithmetic sum of all period returns, providing a linear view of performance without reinvestment effects.

## Essential VectorBT Pro Commands

### Core Portfolio Properties
```python
pf.value                    # Portfolio equity curve (dollar values)
pf.returns                  # Period returns (daily/hourly returns)
pf.init_cash               # Initial cash amount
pf.total_return()          # Total return (compounded)
pf.drawdown()              # Drawdown series
pf.stats()                 # Complete performance statistics
```

### Uncompounded Returns Extraction

#### Method 1: From Period Returns (Recommended)
```python
# Get uncompounded cumulative returns as percentages
uncompounded_returns_pct = pf.returns.cumsum() * 100
```

#### Method 2: From Equity Values
```python
equity_curve = pf.value
initial_cash = pf.init_cash
uncompounded_pct = ((equity_curve - initial_cash) / initial_cash) * 100
```

#### Method 3: Complete Data Extraction
```python
def get_uncompounded_equity_data(pf):
    return {
        'equity_values': pf.value,
        'period_returns': pf.returns,
        'uncompounded_cumulative_pct': pf.returns.cumsum() * 100,
        'initial_cash': pf.init_cash
    }
```

## Position Sizing: $1 Per Trade Setup

To make **dollar returns equal percentage returns**, set each position size to exactly **$1**:

### Key Parameters
```python
pf = vbt.Portfolio.from_signals(
    data.close, 
    entries, 
    exits,
    size=1,                    # $1 per position
    size_type='value',         # Dollar value (not shares or %)
    init_cash=10000,          # Sufficient cash for multiple $1 trades
    fees=0.001
)
```

### Size Type Options
| `size_type` | Description | Example |
|-------------|-------------|---------|
| `'value'` | **Dollar amount** | `size=1` = $1 position |
| `'percent'` | **Percentage of equity** | `size=0.1` = 10% of portfolio |
| `'shares'` | **Number of shares** | `size=10` = 10 shares |
| `'valuepercent'` | **Percentage of current value** | `size=0.05` = 5% of current equity |

### Why $1 Positions Work
- **10% gain on $1** = $0.10 return = 10% in decimal form
- **Dollar return equals percentage return** (in decimal)
- **Simplified performance analysis** - easy to interpret results

## Complete Working Examples

### Example 1: Standard Portfolio
```python
import vectorbtpro as vbt

# 1. Create Portfolio (Standard)
data = vbt.YFData.pull("BTC-USD", start="2020", end="2023")
fast_ma = data.run("talib:sma", timeperiod=10)
slow_ma = data.run("talib:sma", timeperiod=20)
entries = fast_ma.real.vbt.crossed_above(slow_ma.real)
exits = fast_ma.real.vbt.crossed_below(slow_ma.real)

pf_standard = vbt.Portfolio.from_signals(
    data.close, entries, exits, 
    init_cash=10000, fees=0.001
)

# 2. Extract Uncompounded Returns
uncompounded_equity_curve = pf_standard.returns.cumsum() * 100

# 3. Results
print(f"Standard Portfolio Final Return: {uncompounded_equity_curve.iloc[-1]:.2f}%")
```

### Example 2: $1 Position Portfolio
```python
# 1. Create $1 Position Portfolio
pf_dollar = vbt.Portfolio.from_signals(
    data.close, 
    entries, 
    exits,
    size=1,                   # $1 per position
    size_type='value',        # Dollar value
    init_cash=10000,         # Ample cash for trades
    fees=0.001
)

# 2. Extract Returns
dollar_returns = pf_dollar.returns.cumsum() * 100
trade_pnl = pf_dollar.trades.pnl  # Dollar P&L per trade

# 3. Verify: Dollar returns ~= Percentage returns
print(f"$1 Position Portfolio Final Return: {dollar_returns.iloc[-1]:.2f}%")
print(f"Trade P&L Summary:")
print(pf_dollar.trades.pnl.describe())
```

## Verification Methods

```python
# Compare different calculation methods
method1 = (pf.returns.cumsum() * 100).iloc[-1]
method2 = ((pf.value.iloc[-1] - pf.init_cash) / pf.init_cash) * 100
vbt_builtin = pf.total_return() * 100

print(f"Method 1 (uncompounded): {method1:.2f}%")
print(f"Method 2 (equity-based): {method2:.2f}%")
print(f"VBT built-in (compounded): {vbt_builtin:.2f}%")
```

## Plotting the Equity Curve

```python
import plotly.graph_objects as go

# Get uncompounded returns
uncompounded_pct = pf.returns.cumsum() * 100

# Create plot
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=uncompounded_pct.index,
    y=uncompounded_pct.values,
    mode='lines',
    name='Uncompounded Cumulative Returns %'
))

fig.update_layout(
    title='Trading System Equity Curve - Uncompounded Returns',
    xaxis_title='Date',
    yaxis_title='Cumulative Returns (%)'
)

fig.show()
```

## Key VectorBT Pro Advantages

1. **Performance**: Numba-compiled functions for extremely fast computation
2. **Vectorization**: Process multiple strategies simultaneously 
3. **Flexibility**: Supports complex scenarios (leverage, stops, multi-asset)
4. **Built-in Analytics**: Comprehensive performance metrics included

## One-Liner Solutions

| Task | Command |
|------|---------|
| **Uncompounded equity curve (%)** | `pf.returns.cumsum() * 100` |
| **Portfolio value over time** | `pf.value` |
| **Final uncompounded return** | `(pf.returns.cumsum() * 100).iloc[-1]` |
| **Period returns** | `pf.returns` |
| **$1 position setup** | `size=1, size_type='value'` |
| **Trade P&L dollars** | `pf.trades.pnl` |
| **Trade returns percentage** | `pf.trades.returns * 100` |
| **Individual trade data** | `pf.trades.records_readable` |

## Trade-Level Analysis for $1 Positions

```python
# Access individual trade data
trades = pf_dollar.trades

# Key trade metrics
trade_returns_pct = trades.returns * 100        # Trade returns as %
trade_pnl_dollars = trades.pnl                  # Trade P&L in $
trade_duration = trades.duration                # Trade duration
trade_entry_price = trades.entry_price          # Entry prices
trade_exit_price = trades.exit_price            # Exit prices

# Verify dollar/percentage relationship for each trade
verification_df = pd.DataFrame({
    'Return_%': trade_returns_pct,
    'PnL_

## Summary

### Uncompounded Returns
The key to extracting **uncompounded cumulative returns** from VectorBT Pro is using:

```python
uncompounded_returns_pct = pf.returns.cumsum() * 100
```

### $1 Position Strategy
To make **dollar returns equal percentage returns**, use:

```python
pf = vbt.Portfolio.from_signals(
    data, entries, exits,
    size=1,                  # $1 per position
    size_type='value',       # Dollar value
    init_cash=10000         # Sufficient cash
)
```

This provides:
- **Clean equity curves** showing arithmetic accumulation of returns
- **Direct relationship** between dollar P&L and percentage returns
- **Simplified analysis** where $0.10 profit = 10% return

## Best Practices

### For Uncompounded Returns:
1. Always verify your calculations using multiple methods
2. Use `pf.returns.cumsum()` for true uncompounded returns
3. Multiply by 100 to convert to percentage terms
4. Consider using `pf.value` for dollar-based equity curves
5. Leverage VectorBT Pro's built-in plotting methods for quick visualization

### For $1 Position Strategy:
1. **Set sufficient initial cash** (e.g., `init_cash=100000`) to handle all signals
2. **Consider zero fees** (`fees=0`) for clean analysis of strategy performance
3. **Verify the relationship** between trade P&L and returns percentage
4. **Use for strategy comparison** where position sizing isn't the focus
5. **Be aware of limitations** - may not reflect real-world trading constraints

### Verification Checklist:
- [ ] Trade P&L ~= Trade return % (for $1 positions)
- [ ] Uncompounded returns calculated correctly
- [ ] Sufficient cash for all signals
- [ ] Expected number of trades executed
- [ ] Returns match across different calculation methods: trade_pnl_dollars,
    'Difference': abs(trade_pnl_dollars - (trade_returns_pct/100))
})

print("Trade-by-trade verification:")
print(verification_df.head())
print(f"Max difference: {verification_df['Difference'].max():.6f}")
```

## Summary

The key to extracting **uncompounded cumulative returns** from VectorBT Pro is using:

```python
uncompounded_returns_pct = pf.returns.cumsum() * 100
```

This provides a clean equity curve showing the simple arithmetic accumulation of returns over time, which is ideal for analyzing trading system performance without compounding effects.

## Best Practices

1. Always verify your calculations using multiple methods
2. Use `pf.returns.cumsum()` for true uncompounded returns
3. Multiply by 100 to convert to percentage terms
4. Consider using `pf.value` for dollar-based equity curves
5. Leverage VectorBT Pro's built-in plotting methods for quick visualization