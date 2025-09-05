# Dashboard Rebuild Plan - Complete Architecture Redesign

## Executive Summary

The current dashboard architecture using manual OpenGL shader programming for candlestick rendering has proven fundamentally flawed. We need a complete rebuild using a professional financial charting library that handles candlestick visualization natively.

## Current Architecture Problems

### 1. Manual Candlestick Drawing Issues
- **Problem**: Manually calculating vertex positions for each candlestick using OpenGL shaders
- **Impact**: Complex geometry calculations, rendering artifacts when zoomed, poor performance
- **Evidence**: Strange candle rendering at different zoom levels, overlapping candles

### 2. Non-Modular Design
- **Problem**: Tight coupling between chart engine, trade rendering, and UI components
- **Impact**: Changes in one component break others, difficult to debug
- **Evidence**: Trade navigation affects Y-axis scaling, data windows overlap

### 3. Y-Axis Autoscaling Failures
- **Problem**: Manual calculation of viewport bounds and projection matrices
- **Impact**: Y-axis doesn't properly scale to visible data
- **Evidence**: Charts showing incorrect price ranges, data cut off

### 4. Trade Navigation Issues
- **Problem**: Complex timestamp-to-index mapping, viewport synchronization problems
- **Impact**: Click-to-trade doesn't work reliably
- **Evidence**: Multiple duplicate methods, navigation callbacks not triggering

## Proposed New Architecture

### Technology Stack

#### Option 1: Plotly + Dash (Recommended)
**Pros:**
- Native candlestick chart support with `plotly.graph_objects.Candlestick`
- Built-in zoom, pan, hover tooltips
- Handles millions of points with datashader integration
- Trade markers as native scatter plots
- Web-based for better performance

**Cons:**
- Requires browser component
- Different from current PyQt5 approach

**Implementation:**
```python
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output

# Native candlestick - no manual drawing!
fig = go.Figure(data=[go.Candlestick(
    x=df['datetime'],
    open=df['open'],
    high=df['high'],
    low=df['low'],
    close=df['close']
)])

# Trade markers as native scatter
fig.add_trace(go.Scatter(
    x=trades['entry_time'],
    y=trades['entry_price'],
    mode='markers',
    marker=dict(symbol='triangle-up', size=12, color='green')
))
```

#### Option 2: Bokeh
**Pros:**
- Server-side rendering for large datasets
- Native candlestick support
- Good interactivity
- Python-native

**Cons:**
- Learning curve
- Less financial-specific than Plotly

#### Option 3: LightweightCharts (TradingView)
**Pros:**
- Professional trading interface
- Extremely fast with millions of points
- Native candlestick and indicators
- Used by professional trading platforms

**Cons:**
- JavaScript library (needs Python wrapper)
- Licensing considerations

#### Option 4: Matplotlib + mplfinance (Fallback)
**Pros:**
- Pure Python, works with PyQt5
- Native candlestick support
- Simple integration

**Cons:**
- Slower with large datasets
- Less interactive

### Architecture Design

```
┌─────────────────────────────────────────────┐
│            Main Application                 │
│                                             │
│  ┌────────────────────────────────────┐    │
│  │      Data Management Layer         │    │
│  │  - Efficient data storage          │    │
│  │  - Decimation for large datasets   │    │
│  │  - Caching layer                   │    │
│  └────────────────────────────────────┘    │
│                    │                        │
│  ┌─────────────────┴──────────────────┐    │
│  │                                     │    │
│  ▼                                     ▼    │
│ ┌──────────────┐  ┌──────────────────┐     │
│ │ Chart Widget │  │ Trade List Widget │     │
│ │              │  │                   │     │
│ │ - Plotly     │  │ - PyQt5 Table     │     │
│ │   Candlestick│  │ - Click handler   │     │
│ │ - Native     │◄─┤ - Signals/Slots   │     │
│ │   zoom/pan   │  │                   │     │
│ │ - Trade      │  └──────────────────┘     │
│ │   markers    │                            │
│ └──────────────┘                            │
│                                             │
│  ┌────────────────────────────────────┐    │
│  │         Event Bus                  │    │
│  │  - Trade selection events          │    │
│  │  - Viewport sync events            │    │
│  │  - Data update events              │    │
│  └────────────────────────────────────┘    │
└─────────────────────────────────────────────┘
```

### Key Design Principles

1. **Use Native Components**
   - NO manual drawing of candlesticks
   - Use library's built-in candlestick chart type
   - Native zoom/pan controls

2. **Modular Architecture**
   - Clear separation of concerns
   - Event-driven communication
   - Loose coupling between components

3. **Performance Optimization**
   - Data decimation for initial load
   - Progressive loading for zoom
   - Efficient memory management
   - Web workers for data processing

4. **Professional UX**
   - Smooth 60fps interactions
   - Instant trade navigation
   - Clear visual feedback
   - Responsive to all actions

## Implementation Plan

### Phase 1: Proof of Concept (2 days)
1. Create simple Plotly candlestick chart
2. Load 1 million data points
3. Add trade markers
4. Test performance

### Phase 2: Core Dashboard (3 days)
1. Implement data management layer
2. Create main chart widget
3. Add trade list with click handling
4. Implement viewport synchronization

### Phase 3: Features (2 days)
1. Add technical indicators
2. Implement equity curve
3. Add hover tooltips
4. Create crosshair

### Phase 4: Optimization (1 day)
1. Performance tuning
2. Memory optimization
3. Progressive loading
4. Caching strategy

### Phase 5: Testing (1 day)
1. Load testing with 7M+ points
2. User interaction testing
3. Trade navigation verification
4. Performance benchmarking

## Success Criteria

### Performance Requirements
- [ ] Load 7 million data points in under 2 minutes
- [ ] Render 1000 candlesticks at 60fps
- [ ] Instant trade navigation (<100ms)
- [ ] Smooth zoom/pan without lag

### Functional Requirements
- [ ] Native candlestick rendering (no manual drawing)
- [ ] Click-to-trade navigation working 100%
- [ ] Trade arrows visible and correct
- [ ] Y-axis autoscaling working
- [ ] No overlapping UI elements

### Quality Requirements
- [ ] Clean, modular codebase
- [ ] Comprehensive error handling
- [ ] Unit tests for critical paths
- [ ] Documentation for all components

## Risk Mitigation

### Risk 1: Library Performance
**Mitigation**: Prototype with multiple libraries early, benchmark performance

### Risk 2: PyQt5 Integration
**Mitigation**: Use QWebEngineView for web-based charts if needed

### Risk 3: Data Volume
**Mitigation**: Implement intelligent decimation and progressive loading

## Recommended Approach

Given the requirements and current issues, I strongly recommend:

**Plotly + Dash** for the main implementation because:
1. Native financial chart support
2. Proven ability to handle millions of points
3. Rich ecosystem of financial indicators
4. Active development and community
5. Professional appearance

With fallback to **mplfinance** if web-based approach is not acceptable.

## Next Steps

1. **Immediate**: Create new branch `dashboard-rebuild-plotly`
2. **Day 1**: Implement Plotly proof of concept with 1M points
3. **Day 2**: Add trade list integration
4. **Day 3-5**: Build complete dashboard
5. **Day 6-7**: Testing and optimization
6. **Day 8**: Documentation and deployment

## Conclusion

The current architecture's fundamental flaws require a complete rebuild. By using a professional charting library with native candlestick support, we can eliminate the complex manual rendering code and achieve better performance, reliability, and maintainability.

The investment in rebuilding will pay off immediately in:
- Reduced debugging time
- Better user experience
- Easier feature additions
- Professional appearance
- Maintainable codebase

This is not just a fix - it's an upgrade to a professional-grade trading dashboard.