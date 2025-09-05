# GitHub Issue: Rebuild Dashboard with Plotly

## Issue Title
Rebuild Dashboard with Native Plotly Candlestick Charts

## Labels
- enhancement
- architecture
- high-priority

## Description

### Problem
The current VisPy/OpenGL-based dashboard has fundamental architectural issues:
- Manual candlestick drawing with complex vertex calculations
- Y-axis autoscaling not working
- Trade navigation unreliable
- Poor zoom performance
- Non-modular architecture

### Solution
Complete rebuild using Plotly's native candlestick charts with Dash for interactivity.

### Requirements
1. Native candlestick rendering (no manual drawing)
2. Handle 7M+ data points efficiently
3. Fast rendering of 1000 bars from memory
4. Working trade list with jump-to-trade
5. Trade markers on chart

### Implementation Plan
- [x] Create rebuild plan document
- [ ] Create Plotly MVP branch
- [ ] Implement basic candlestick chart
- [ ] Add SMA indicator
- [ ] Implement trade navigation
- [ ] Performance testing with large datasets
- [ ] Full dashboard integration

### Acceptance Criteria
- Chart loads 1M+ points in <2 seconds
- Smooth 60fps zoom/pan
- Trade navigation works 100% reliably
- Clean, maintainable code

### Branch
`plotly-dashboard-mvp`

### Related
- See REBUILD_DASHBOARD_PLAN.md for detailed architecture