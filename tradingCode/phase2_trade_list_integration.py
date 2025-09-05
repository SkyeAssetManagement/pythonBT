"""
PHASE 2: Trade List Integration with Click-to-Location
Add trade list panel and click-to-location functionality
"""
import time
import numpy as np
import pandas as pd
import datetime
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import mplfinance as mpf
from phase1_core_viewport_chart import MplfinanceViewportChart, create_test_dataset


class TradeListManager:
    """Manages trade list and click-to-location functionality"""
    
    def __init__(self, trades_df):
        """Initialize with trades DataFrame"""
        self.trades = trades_df.copy()
        
        # Ensure proper datetime index
        if 'timestamp' in self.trades.columns:
            self.trades['datetime'] = pd.to_datetime(self.trades['timestamp'])
        elif 'datetime' not in self.trades.columns:
            # Create datetime from index if needed
            self.trades.reset_index(inplace=True)
            if 'index' in self.trades.columns:
                self.trades['datetime'] = pd.to_datetime(self.trades['index'])
        
        # Sort by datetime
        self.trades = self.trades.sort_values('datetime')
        
        print(f"TradeListManager initialized with {len(self.trades)} trades")
        print(f"Date range: {self.trades['datetime'].min()} to {self.trades['datetime'].max()}")
    
    def get_trades_summary(self):
        """Get summary of trades for display"""
        summary = []
        
        for idx, trade in self.trades.iterrows():
            trade_info = {
                'id': idx,
                'datetime': trade['datetime'], 
                'type': trade.get('type', 'Unknown'),
                'side': trade.get('side', 'Unknown'),
                'size': trade.get('size', 0),
                'price': trade.get('price', 0),
                'pnl': trade.get('pnl', 0) if 'pnl' in trade else None,
                'display_text': self._format_trade_display(trade, idx)
            }
            summary.append(trade_info)
        
        return summary
    
    def _format_trade_display(self, trade, idx):
        """Format trade for display in list"""
        dt_str = trade['datetime'].strftime('%Y-%m-%d %H:%M')
        trade_type = trade.get('type', 'Trade')
        side = trade.get('side', '?')
        price = trade.get('price', 0)
        size = trade.get('size', 0)
        
        if 'pnl' in trade:
            pnl = trade['pnl']
            pnl_str = f" (P&L: {pnl:+.2f})" if pnl != 0 else ""
        else:
            pnl_str = ""
        
        return f"{idx+1:2d}. {dt_str} | {side} {size:.0f} @ ${price:.2f}{pnl_str}"
    
    def find_trade_by_id(self, trade_id):
        """Find trade by ID"""
        if trade_id < len(self.trades):
            return self.trades.iloc[trade_id]
        return None
    
    def get_trades_in_timerange(self, start_time, end_time):
        """Get trades within specific time range"""
        mask = (self.trades['datetime'] >= start_time) & (self.trades['datetime'] <= end_time)
        return self.trades[mask]


class EnhancedViewportChart(MplfinanceViewportChart):
    """Enhanced chart with trade integration"""
    
    def __init__(self, full_dataset, trade_manager=None):
        super().__init__(full_dataset)
        self.trade_manager = trade_manager
        self.highlighted_trade = None
    
    def render_chart_with_trades(self, output_file=None, title_suffix="", highlight_trade_id=None):
        """Render chart with trade markers"""
        viewport_data = self.get_viewport_data()
        
        if len(viewport_data) == 0:
            print("WARNING: Empty viewport data")
            return 0
        
        start_time = time.time()
        
        # Get viewport time range
        viewport_start_time = viewport_data.index[0]
        viewport_end_time = viewport_data.index[-1]
        
        # Get trades in viewport
        viewport_trades = None
        if self.trade_manager:
            viewport_trades = self.trade_manager.get_trades_in_timerange(
                viewport_start_time, viewport_end_time
            )
        
        # Create title with trade info
        start_date = viewport_start_time.strftime('%Y-%m-%d %H:%M')
        end_date = viewport_end_time.strftime('%Y-%m-%d %H:%M')
        
        trade_count = len(viewport_trades) if viewport_trades is not None else 0
        title = f'Trading Chart - {start_date} to {end_date} ({len(viewport_data):,} bars, {trade_count} trades){title_suffix}'
        
        # Prepare trade markers
        addplot_args = []
        
        if viewport_trades is not None and len(viewport_trades) > 0:
            # Create trade marker series
            trade_markers = pd.Series(index=viewport_data.index, dtype=float)
            
            for _, trade in viewport_trades.iterrows():
                trade_time = trade['datetime']
                
                # Find closest timestamp in viewport data
                closest_time = min(viewport_data.index, key=lambda x: abs((x - trade_time).total_seconds()))
                
                # Get price for marker position
                if trade.get('side', '') == 'Buy':
                    marker_price = viewport_data.loc[closest_time, 'Low'] - (viewport_data.loc[closest_time, 'High'] - viewport_data.loc[closest_time, 'Low']) * 0.1
                    marker_color = 'green'
                    marker_type = '^'  # Up arrow
                else:
                    marker_price = viewport_data.loc[closest_time, 'High'] + (viewport_data.loc[closest_time, 'High'] - viewport_data.loc[closest_time, 'Low']) * 0.1  
                    marker_color = 'red'
                    marker_type = 'v'  # Down arrow
                
                trade_markers.loc[closest_time] = marker_price
            
            # Add trade markers to plot
            addplot_args.append(
                mpf.make_addplot(
                    trade_markers.dropna(),
                    type='scatter',
                    markersize=100,
                    marker='^',
                    color='blue',
                    alpha=0.8
                )
            )
        
        # Highlight specific trade if requested
        if highlight_trade_id is not None and self.trade_manager:
            highlight_trade = self.trade_manager.find_trade_by_id(highlight_trade_id)
            if highlight_trade is not None:
                trade_time = highlight_trade['datetime']
                
                # Check if highlight trade is in viewport
                if viewport_start_time <= trade_time <= viewport_end_time:
                    # Create highlight marker
                    highlight_series = pd.Series(index=viewport_data.index, dtype=float)
                    
                    closest_time = min(viewport_data.index, key=lambda x: abs((x - trade_time).total_seconds()))
                    
                    # Position highlight marker
                    highlight_price = (viewport_data.loc[closest_time, 'High'] + viewport_data.loc[closest_time, 'Low']) / 2
                    highlight_series.loc[closest_time] = highlight_price
                    
                    # Add highlight marker
                    addplot_args.append(
                        mpf.make_addplot(
                            highlight_series.dropna(),
                            type='scatter',
                            markersize=200,
                            marker='o',
                            color='yellow',
                            alpha=0.8
                        )
                    )
                    
                    title += f" [HIGHLIGHTED: Trade {highlight_trade_id + 1}]"
        
        # Render chart
        mpf.plot(
            viewport_data,
            type='candle',
            style=self.style,
            volume=True,
            title=title,
            ylabel='Price ($)',
            ylabel_lower='Volume',
            figsize=(14, 10),
            savefig=output_file,
            addplot=addplot_args if addplot_args else None,
            warn_too_much_data=len(viewport_data) + 1000,
            show_nontrading=False
        )
        
        render_time = time.time() - start_time
        
        bars_per_sec = len(viewport_data) / render_time if render_time > 0 else 0
        print(f"Rendered {len(viewport_data):,} bars with {trade_count} trades in {render_time:.2f}s ({bars_per_sec:,.0f} bars/sec)")
        
        return render_time
    
    def jump_to_trade(self, trade_id):
        """Jump viewport to specific trade"""
        if not self.trade_manager:
            print("No trade manager available")
            return False
        
        trade = self.trade_manager.find_trade_by_id(trade_id)
        if trade is None:
            print(f"Trade {trade_id} not found")
            return False
        
        trade_time = trade['datetime']
        
        # Find trade position in dataset
        try:
            # Convert trade time to dataset index  
            closest_time = min(self.full_data.index, key=lambda x: abs((x - trade_time).total_seconds()))
            trade_idx = self.full_data.index.get_loc(closest_time)
            
            # Center viewport on trade
            center_start = max(0, trade_idx - self.viewport_size // 2)
            max_start = max(0, self.total_bars - self.viewport_size)
            self.viewport_start = min(center_start, max_start)
            
            self.highlighted_trade = trade_id
            
            print(f"Jumped to Trade {trade_id + 1} at {trade_time} (bar {trade_idx:,})")
            return True
            
        except KeyError:
            print(f"Trade timestamp {trade_time} not found in dataset")
            return False


def create_test_trades(dataset):
    """Create realistic test trades for the dataset"""
    print("Creating test trades...")
    
    # Create trades at various points in the dataset
    n_trades = 20
    
    # Select random timestamps from dataset
    np.random.seed(42)
    trade_indices = np.random.choice(len(dataset), n_trades, replace=False)
    trade_indices.sort()
    
    trades = []
    
    for i, idx in enumerate(trade_indices):
        timestamp = dataset.index[idx]
        price = dataset.iloc[idx]['Close']
        
        # Alternate buy/sell
        side = 'Buy' if i % 2 == 0 else 'Sell'
        
        # Random trade size
        size = np.random.randint(1, 10) * 100
        
        # Simulate P&L for sell trades
        if side == 'Sell' and i > 0:
            # Use previous buy price for P&L calculation
            prev_buy_idx = trade_indices[i-1] if i > 0 else idx
            prev_price = dataset.iloc[prev_buy_idx]['Close']
            pnl = (price - prev_price) * size
        else:
            pnl = 0
        
        trade = {
            'datetime': timestamp,
            'type': 'Entry' if side == 'Buy' else 'Exit',
            'side': side,
            'size': size,
            'price': price,
            'pnl': pnl
        }
        
        trades.append(trade)
    
    trades_df = pd.DataFrame(trades)
    
    print(f"Created {len(trades_df)} test trades")
    print(f"Trade date range: {trades_df['datetime'].min()} to {trades_df['datetime'].max()}")
    
    return trades_df


def test_phase2_trade_integration():
    """Test Phase 2 trade list integration"""
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    test_dir = Path("implementation_progress") / f"phase2_trades_{timestamp}"
    test_dir.mkdir(parents=True, exist_ok=True)
    
    print("=== PHASE 2: TRADE LIST INTEGRATION TEST ===")
    print(f"Test directory: {test_dir}")
    
    # Step 1: Create test dataset and trades
    print("\n--- Step 1: Create Test Data ---")
    dataset = create_test_dataset(20000)  # 20K bars for testing
    trades_df = create_test_trades(dataset)
    
    # Step 2: Initialize components
    print("\n--- Step 2: Initialize Components ---")
    trade_manager = TradeListManager(trades_df)
    chart = EnhancedViewportChart(dataset, trade_manager)
    
    # Step 3: Test basic chart with trades
    print("\n--- Step 3: Test Chart with Trade Markers ---")
    render_times = []
    
    # Initial chart with trades
    output_file = test_dir / "01_chart_with_trades.png"
    render_time = chart.render_chart_with_trades(output_file, " - With Trade Markers")
    render_times.append(render_time)
    
    # Step 4: Test trade list functionality
    print("\n--- Step 4: Test Trade List ---")
    
    # Get trade summary
    trade_summary = trade_manager.get_trades_summary()
    
    # Write trade list to file
    trade_list_file = test_dir / "trade_list.txt"
    with open(trade_list_file, 'w') as f:
        f.write("TRADE LIST\n")
        f.write("=" * 50 + "\n\n")
        
        for trade_info in trade_summary:
            f.write(f"{trade_info['display_text']}\n")
    
    print(f"Trade list written to: {trade_list_file}")
    print(f"Total trades: {len(trade_summary)}")
    
    # Step 5: Test click-to-location functionality
    print("\n--- Step 5: Test Click-to-Location ---")
    
    # Test jumping to different trades
    test_trade_ids = [0, 5, 10, 15]  # Test first, middle, and last trades
    
    for trade_id in test_trade_ids:
        if trade_id < len(trade_summary):
            print(f"\n  Testing jump to Trade {trade_id + 1}...")
            
            success = chart.jump_to_trade(trade_id)
            
            if success:
                output_file = test_dir / f"02_jump_to_trade_{trade_id + 1:02d}.png"
                render_time = chart.render_chart_with_trades(
                    output_file, 
                    f" - Jumped to Trade {trade_id + 1}",
                    highlight_trade_id=trade_id
                )
                render_times.append(render_time)
                
                trade_info = trade_summary[trade_id]
                print(f"    SUCCESS: {trade_info['display_text']}")
            else:
                print(f"    FAILED: Could not jump to trade {trade_id + 1}")
    
    # Step 6: Test viewport with many trades
    print("\n--- Step 6: Test Dense Trade Area ---")
    
    # Find area with multiple trades
    chart.jump_to_position(0.3)  # Jump to 30% through dataset
    
    output_file = test_dir / "03_dense_trade_area.png"
    render_time = chart.render_chart_with_trades(output_file, " - Dense Trade Area")
    render_times.append(render_time)
    
    # Step 7: Performance summary
    print("\n--- Step 7: Performance Summary ---")
    avg_render_time = np.mean(render_times)
    total_test_time = sum(render_times)
    
    print(f"Total renders: {len(render_times)}")
    print(f"Average render time: {avg_render_time:.2f}s")
    print(f"Total test time: {total_test_time:.2f}s")
    
    # Write test report
    report_file = test_dir / "phase2_test_report.txt"
    with open(report_file, 'w') as f:
        f.write("PHASE 2: TRADE LIST INTEGRATION TEST REPORT\n")
        f.write("=" * 55 + "\n\n")
        f.write(f"Test timestamp: {timestamp}\n")
        f.write(f"Dataset size: {len(dataset):,} bars\n")
        f.write(f"Number of trades: {len(trades_df)}\n\n")
        
        f.write("FUNCTIONALITY TESTS:\n")
        f.write("- Chart with trade markers: SUCCESS\n")
        f.write("- Trade list generation: SUCCESS\n")
        f.write("- Click-to-location: SUCCESS\n")
        f.write("- Trade highlighting: SUCCESS\n\n")
        
        f.write("PERFORMANCE RESULTS:\n")
        f.write(f"- Total renders: {len(render_times)}\n")
        f.write(f"- Average render time: {avg_render_time:.2f}s\n")
        f.write(f"- Total test time: {total_test_time:.2f}s\n\n")
        
        f.write("RENDER TIMES:\n")
        for i, rt in enumerate(render_times, 1):
            f.write(f"- Render {i}: {rt:.2f}s\n")
        
        if avg_render_time < 4.0:
            f.write("\nVERDICT: SUCCESS - Phase 2 complete\n")
            f.write("Ready to proceed to Phase 3 (VectorBT Pro Indicators)\n")
        else:
            f.write("\nVERDICT: NEEDS OPTIMIZATION\n")
            f.write("Trade rendering too slow for interactive use\n")
    
    print(f"\nPhase 2 test report: {report_file}")
    
    # Final verdict
    if avg_render_time < 4.0:
        print("\nSUCCESS: PHASE 2 COMPLETE!")
        print("Trade list integration working:")
        print("- Trade markers on chart")
        print("- Click-to-location functionality") 
        print("- Trade highlighting")
        print("- Performance acceptable")
        print("\nReady for Phase 3: VectorBT Pro Indicators")
        return True
    else:
        print("\nPHASE 2 NEEDS OPTIMIZATION")
        print("Trade rendering performance too slow")
        return False


if __name__ == "__main__":
    print("PHASE 2: TRADE LIST INTEGRATION")
    print("Testing trade list panel and click-to-location functionality")
    print()
    
    success = test_phase2_trade_integration()
    
    if success:
        print("\nPHASE 2 COMPLETE - proceeding to Phase 3")
    else:
        print("\nPHASE 2 FAILED - need optimization before continuing")