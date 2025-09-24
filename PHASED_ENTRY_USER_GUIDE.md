# Phased Entry System - User Guide

## Overview

The Phased Entry System allows you to implement pyramid/scaling trading strategies where positions are built gradually over multiple entries rather than all at once. This approach can reduce initial risk while maximizing trend-following potential.

## Key Benefits

- **Risk Reduction**: Start with smaller initial positions
- **Trend Following**: Add to winning positions as trends develop
- **Better Average Prices**: Scale in at favorable levels
- **Flexible Configuration**: Customize phase triggers and sizing
- **Advanced Visualization**: See individual phases and connections on charts

## How It Works

### Traditional vs Phased Entries

**Traditional Entry:**
- Signal detected → Enter full position immediately
- All-or-nothing approach
- Higher initial risk

**Phased Entry:**
- Signal detected → Enter partial position (e.g., 40% of target)
- Price moves favorably → Add second phase (e.g., 30% more)
- Continued favorable movement → Add final phase (e.g., 30% more)
- Gradual position building with reduced risk

### Example Scenario

Let's say you want to invest $1000 in a stock at $100:

**Traditional:** Buy 10 shares at $100 = $1000 invested immediately

**Phased Entry (3 phases):**
1. **Phase 1**: Buy 4 shares at $100 (40% = $400)
2. **Phase 2**: Price hits $102 (+2%) → Buy 3 shares at $102 (30% = $306)
3. **Phase 3**: Price hits $104 (+4%) → Buy 3 shares at $104 (30% = $312)

**Result:** Total investment $1018, but with reduced initial risk and confirmation of trend strength.

## Configuration

### Basic Setup

Enable phased entries in your `config.yaml`:

```yaml
phased_entries:
  enabled: true                    # Enable phased entries
  max_phases: 3                   # Maximum phases per position
  initial_size_percent: 40.0      # First phase as % of total position
```

### Phase Triggers

Control when additional phases are added:

```yaml
phase_trigger:
  type: "percent"                 # Options: "percent", "points", "atr_multiple"
  value: 1.5                      # 1.5% favorable move triggers next phase
```

**Trigger Types:**
- **percent**: Percentage move from initial entry (e.g., 1.5% = 1.5% price increase for long positions)
- **points**: Absolute point move from initial entry (e.g., 2.0 = $2.00 price increase)
- **atr_multiple**: Multiple of Average True Range (e.g., 0.5 = half ATR move)

### Phase Sizing

Configure how position size is distributed across phases:

```yaml
phase_sizing:
  type: "equal"                   # Options: "equal", "decreasing", "increasing", "custom"
  multiplier: 1.0                 # For decreasing/increasing sizing
  custom_ratios: [0.4, 0.35, 0.25]  # Custom ratios (must sum to 1.0)
```

**Sizing Types:**
- **equal**: Each phase gets equal share of remaining size
- **decreasing**: Later phases get progressively smaller (safer)
- **increasing**: Later phases get progressively larger (more aggressive)
- **custom**: Specify exact ratios for each phase

### Risk Management

Built-in risk controls protect against adverse scenarios:

```yaml
risk_management:
  max_adverse_move: 3.0           # Stop scaling if 3% adverse move
  require_profit: true            # Only add phases when profitable
  time_limit_bars: 50             # Max time to complete all phases
```

**Risk Controls:**
- **max_adverse_move**: Stop adding phases if price moves against you by this percentage
- **require_profit**: Only trigger new phases when current position is profitable
- **time_limit_bars**: Maximum number of bars to complete all phases (prevents stale scaling)

### Stop Loss Adaptation

Adjust stop losses as phases are added:

```yaml
stop_loss:
  adapt_to_phases: true           # Adjust SL as phases added
  use_average_price: true         # Base SL on average entry vs first entry
```

## Using Phased Entries

### 1. Strategy Implementation

Extend your strategy from `PhasedTradingStrategy`:

```python
from src.trading.strategies.phased_strategy_base import PhasedTradingStrategy

class MyPhasedStrategy(PhasedTradingStrategy):
    def __init__(self):
        super().__init__("MyPhasedStrategy")

    def generate_signals(self, df):
        # Your signal generation logic
        return signals
```

### 2. Running Backtests

```python
strategy = MyPhasedStrategy()
trades, performance = strategy.run_backtest_with_phases(df, "SYMBOL")

# Get phased-specific metrics
phased_stats = strategy.get_phased_performance_metrics()
print(f"Average phases per position: {phased_stats['avg_phases_per_position']}")
```

### 3. Visualization

The system provides enhanced visualization for phased entries:

#### Trade Panel Features
- **Phase Column**: Shows which phase each trade represents
- **Average Entry Price**: Displays weighted average entry price
- **Total Position Size**: Shows cumulative position size
- **Phase Statistics**: Summary of phased entry performance

#### Chart Overlays
- **Different Markers**: Each phase has a unique marker style
  - Phase 1: Triangle (▲) in green
  - Phase 2: Diamond (♦) in orange
  - Phase 3: Square (■) in red
  - Phase 4+: Circle (●) in purple
- **Connection Lines**: Dotted lines connect phases of the same position
- **Size-Weighted Markers**: Larger markers for bigger position sizes

## Performance Analysis

### Key Metrics

The system tracks additional metrics for phased entries:

- **Average Phases per Position**: How many phases typically get executed
- **Completion Rate**: Percentage of positions that reach maximum phases
- **Phase P&L Breakdown**: Performance by individual phase
- **Scaling Effectiveness**: How well the phasing improves returns

### Interpretation

**Good Phased Entry Performance:**
- Higher completion rates (positions reach later phases)
- Later phases show positive P&L (confirms trend-following)
- Average phases > 1.5 (actual scaling is occurring)

**Poor Phased Entry Performance:**
- Low completion rates (positions stopped early)
- Later phases show negative P&L (poor timing)
- Average phases ≈ 1 (essentially single entries)

## Common Strategies

### Conservative Scaling
```yaml
phased_entries:
  max_phases: 3
  initial_size_percent: 50.0      # Larger initial position
  phase_trigger:
    value: 2.0                    # Wait for 2% confirmation
  phase_sizing:
    type: "decreasing"            # Smaller later phases
  risk_management:
    require_profit: true          # Only scale when winning
```

### Aggressive Scaling
```yaml
phased_entries:
  max_phases: 4
  initial_size_percent: 25.0      # Smaller initial position
  phase_trigger:
    value: 1.0                    # Quick 1% triggers
  phase_sizing:
    type: "equal"                 # Equal phase sizes
  risk_management:
    require_profit: false         # Scale even when not profitable
```

### ATR-Based Scaling
```yaml
phased_entries:
  max_phases: 3
  initial_size_percent: 40.0
  phase_trigger:
    type: "atr_multiple"
    value: 0.5                    # Half ATR move triggers phase
  risk_management:
    max_adverse_move: 1.0         # Tight adverse move limit
```

## Troubleshooting

### Common Issues

**Problem**: Phases never trigger
- **Cause**: Trigger value too high or adverse move limit too tight
- **Solution**: Lower trigger value or increase adverse move limit

**Problem**: All positions only have 1 phase
- **Cause**: Time limit too short or market not trending
- **Solution**: Increase time_limit_bars or adjust trigger sensitivity

**Problem**: Poor performance with phased entries
- **Cause**: Market not suitable for trend-following or wrong configuration
- **Solution**: Test different trigger values and sizing approaches

### Debugging

Enable debug logging to see phased entry decisions:

```python
import logging
logging.getLogger('trading.core.phased_entry').setLevel(logging.DEBUG)
```

Debug output shows:
- Phase trigger evaluations
- Risk management decisions
- Execution details for each phase

### Performance Tips

1. **Backtest First**: Always backtest phased settings before live trading
2. **Market Conditions**: Phased entries work best in trending markets
3. **Position Sizing**: Start conservative with initial position sizes
4. **Time Limits**: Set reasonable time limits to prevent stale scaling
5. **Monitor Completion Rates**: Aim for 60%+ completion rates

## Advanced Features

### Custom Phase Logic

Override methods in `PhasedTradingStrategy` for custom behavior:

```python
def calculate_dynamic_phase_size(self, phase_number, current_price, entry_price, df, bar_index):
    # Custom size calculation based on volatility
    atr = df['ATR'].iloc[bar_index]
    volatility_factor = min(2.0, atr / df['ATR'].mean())
    return 1.0 / volatility_factor  # Smaller phases in high volatility

def calculate_adaptive_triggers(self, df, entry_bar, is_long):
    # Custom trigger levels based on market conditions
    # Implementation depends on your strategy
    return custom_triggers
```

### Parameter Optimization

Optimize phased entry parameters:

```python
param_ranges = {
    'phase_trigger_value': [1.0, 1.5, 2.0, 2.5],
    'max_phases': [2, 3, 4],
    'initial_size_percent': [25, 33, 40, 50]
}

results = strategy.optimize_phase_parameters(df, param_ranges)
print(f"Best parameters: {results['best_params']}")
```

## Integration with Existing System

The phased entry system is designed for backward compatibility:

- **Existing strategies continue to work** when phased entries are disabled
- **Gradual adoption** - enable phased entries per strategy
- **Fallback behavior** - automatically falls back to single entries if configuration issues occur
- **Existing visualizations** still work alongside phased entry features

## Best Practices

1. **Start Simple**: Begin with 2-3 phases and equal sizing
2. **Test Thoroughly**: Backtest extensively before live trading
3. **Monitor Performance**: Track completion rates and phase P&L
4. **Adjust Gradually**: Make small configuration changes and observe results
5. **Consider Market Type**: Trending markets work best for phased entries
6. **Risk Management**: Always use adverse move limits and profit requirements
7. **Documentation**: Keep notes on configuration changes and their effects

This completes the comprehensive user guide for the Phased Entry System. The system provides powerful tools for implementing pyramid trading strategies while maintaining the safety and reliability of the existing trading framework.