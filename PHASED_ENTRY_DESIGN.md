# Phased Entry System Design

## Overview
This document outlines the design for implementing phased/pyramid entry strategies in the PythonBT trading system. Phased entries allow traders to scale into positions gradually, reducing initial risk while maximizing trend-following potential.

## Architecture Components

### 1. Core Classes

#### PhasedEntryConfig
```python
@dataclass
class PhasedEntryConfig:
    enabled: bool = False
    max_phases: int = 3
    initial_size_percent: float = 33.33  # % of total position for first entry
    phase_trigger_type: str = "percent"  # "percent", "points", "atr"
    phase_trigger_value: float = 2.0    # 2% move in favor for next phase
    phase_size_type: str = "equal"      # "equal", "decreasing", "increasing"
    phase_size_multiplier: float = 1.0  # Multiplier for subsequent phases
    max_adverse_move: float = 5.0       # Max % move against before stopping scaling
    require_profit: bool = True         # Only add phases when in profit
```

#### PhasedEntry
```python
class PhasedEntry:
    def __init__(self, config: PhasedEntryConfig):
        self.config = config
        self.phases: List[TradePhase] = []
        self.total_size = 0.0
        self.entry_signal_bar = None
        self.last_phase_bar = None
        self.is_complete = False
```

#### TradePhase
```python
@dataclass
class TradePhase:
    phase_number: int
    trigger_price: float
    execution_price: float
    size: float
    bar_index: int
    timestamp: Optional[datetime]
    trigger_met: bool = False
    executed: bool = False
```

### 2. Enhanced Signal Processing

#### Current System
- Simple signals: 1 (long), -1 (short), 0 (flat)
- Single entry/exit per signal change

#### Phased System
- Extended signals: Support for partial position building
- Signal strength: 0.33, 0.67, 1.0 for different phase triggers
- Position state tracking: Active phases, remaining capacity

### 3. Position Management

#### PhasedPositionTracker
```python
class PhasedPositionTracker:
    def __init__(self):
        self.active_positions: Dict[str, PhasedEntry] = {}
        self.completed_positions: List[PhasedEntry] = []
        self.position_history: List[Dict] = []

    def should_add_phase(self, symbol: str, current_price: float, bar_index: int) -> bool
    def calculate_next_phase_size(self, entry: PhasedEntry) -> float
    def check_phase_triggers(self, entry: PhasedEntry, current_price: float) -> bool
    def get_total_position_size(self, symbol: str) -> float
```

### 4. Integration Points

#### A. Enhanced Execution Engine
- Extend `StandaloneExecutionEngine` with phased entry support
- Add phase trigger evaluation logic
- Track multiple entries per position

#### B. Strategy Integration
- Extend `EnhancedTradingStrategy` with phased entry methods
- Add phase configuration loading from YAML
- Signal generation for phase triggers

#### C. Visualization Updates
- Enhanced trade panel showing phase breakdown
- Chart overlays for phase entry points
- P&L calculation across all phases

## Implementation Strategy

### Phase 1: Core Infrastructure
1. Create phased entry configuration classes
2. Implement basic phase tracking logic
3. Extend execution engine with phase support

### Phase 2: Strategy Integration
1. Update strategy base classes
2. Add phase trigger logic
3. Implement phase size calculations

### Phase 3: Visualization
1. Update trade panel for phase display
2. Add chart markers for phase entries
3. Enhanced P&L calculations

### Phase 4: Configuration & Testing
1. Add YAML configuration support
2. Create comprehensive test cases
3. Performance validation

## Configuration Example

```yaml
# Phased Entry Configuration
phased_entries:
  enabled: true
  max_phases: 3
  initial_size_percent: 40.0

  # Trigger Configuration
  phase_trigger:
    type: "percent"     # percent, points, atr_multiple
    value: 1.5         # 1.5% favorable move to trigger next phase

  # Size Configuration
  phase_sizing:
    type: "equal"       # equal, decreasing, increasing, custom
    multiplier: 1.0     # For non-equal sizing
    custom_ratios: [0.4, 0.35, 0.25]  # Custom phase ratios

  # Risk Management
  risk_management:
    max_adverse_move: 3.0      # Stop scaling if 3% adverse move
    require_profit: true       # Only scale when profitable
    time_limit_bars: 50        # Max bars to complete all phases

  # Stop Loss Adaptation
  stop_loss:
    adapt_to_phases: true      # Adjust SL as phases are added
    use_average_price: true    # Base SL on average entry price
```

## Risk Management Features

### 1. Phase Limiting
- Maximum number of phases per position
- Time-based limits for completing phases
- Adverse move limits to stop scaling

### 2. Size Management
- Configurable phase size ratios
- Maximum total position size limits
- Dynamic sizing based on market conditions

### 3. Stop Loss Adaptation
- Adjust stop losses as phases are added
- Use average entry price for SL calculation
- Individual phase exit capabilities

## P&L Calculation

### Traditional vs Phased
- **Traditional**: Single entry/exit price
- **Phased**: Weighted average entry price across phases
- **Per-Phase P&L**: Individual phase performance tracking
- **Cumulative P&L**: Total position performance

### Formula for Weighted Average Entry
```
Average Entry Price = Σ(Phase Size × Phase Price) / Total Size
```

## Visualization Enhancements

### Trade Panel
- Phase breakdown table showing each entry
- Cumulative size and average price display
- Individual phase P&L tracking

### Chart Overlays
- Different markers for each phase (▲, ♦, ●)
- Size-weighted markers (larger = bigger phase)
- Connection lines between phases

### Performance Metrics
- Average phases per trade
- Phase completion rate
- Scaling effectiveness metrics

## Testing Strategy

### Unit Tests
- Phase trigger logic
- Size calculations
- Risk management rules

### Integration Tests
- End-to-end phased entry execution
- Configuration loading
- Visualization updates

### Performance Tests
- Large dataset backtests
- Memory usage with many phases
- Execution speed comparison

## Migration Path

### Backward Compatibility
- All existing strategies continue to work
- Phased entries opt-in via configuration
- Fallback to single-phase behavior

### Gradual Rollout
1. Core infrastructure (no UI changes)
2. Basic phased entry support
3. Advanced features and visualization
4. Full GUI integration

## Success Metrics

### Functionality
- Accurate phase execution
- Proper P&L calculation
- Risk management effectiveness

### Performance
- Minimal impact on single-phase trades
- Efficient memory usage
- Fast phase trigger evaluation

### Usability
- Clear visualization of phases
- Intuitive configuration
- Comprehensive documentation

This design provides a robust foundation for implementing phased entries while maintaining system performance and backward compatibility.