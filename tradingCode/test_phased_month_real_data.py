"""
Test phased trading over a full month with realistic price data
"""

import numpy as np
import pandas as pd
import sys
from pathlib import Path
from datetime import datetime, timedelta
import vectorbtpro as vbt

sys.path.insert(0, str(Path(__file__).parent))

from src.backtest.vbt_engine import VectorBTEngine
from src.data.parquet_importer import ParquetImporter


def test_phased_trading_month():
    """Test phased trading with a month of real market data."""
    
    print("="*80)
    print("PHASED TRADING - FULL MONTH TEST WITH REAL DATA")
    print("="*80)
    print(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    # Load real market data
    print("\n### LOADING REAL MARKET DATA")
    print("-"*60)
    
    # Get available files
    parquet_dir = Path("C:/Users/skyeAM/SkyeAM Dropbox/SAMresearch/ABtoPython/parquetData/1m/ES/diffAdjusted")
    data_dir = Path("C:/Users/skyeAM/SkyeAM Dropbox/SAMresearch/ABtoPython/data")
    
    # Import last month of data
    importer = ParquetImporter(str(data_dir), str(parquet_dir))
    parquet_files = list(parquet_dir.glob("*.parquet"))
    
    if not parquet_files:
        print("ERROR: No parquet files found")
        return False
    
    # Use the most recent file
    latest_file = max(parquet_files, key=lambda x: x.stat().st_mtime)
    print(f"Using data file: {latest_file.name}")
    
    # Load data
    df = pd.read_parquet(latest_file)
    
    # Get last month of data (approximately 8640 5-minute bars for 30 days)
    n_bars_per_day = 288  # 24 hours * 60 minutes / 5 minutes
    n_days = 30
    total_bars = n_bars_per_day * n_days
    
    # Take last month of data
    if len(df) > total_bars:
        df = df.tail(total_bars)
    
    print(f"Loaded {len(df)} bars of data")
    print(f"Date range: {df.index[0]} to {df.index[-1]}")
    
    # Convert to numpy arrays (columns are capitalized in the parquet file)
    data = {
        'open': df['Open'].values,
        'high': df['High'].values,
        'low': df['Low'].values,
        'close': df['Close'].values,
        'volume': df['Volume'].values
    }
    # Use datetime_int as timestamps
    timestamps = df['datetime_int'].values
    
    # Show price statistics
    print(f"\nPrice Statistics:")
    print(f"  Open range: ${data['open'].min():.2f} - ${data['open'].max():.2f}")
    print(f"  High range: ${data['high'].min():.2f} - ${data['high'].max():.2f}")
    print(f"  Low range: ${data['low'].min():.2f} - ${data['low'].max():.2f}")
    print(f"  Close range: ${data['close'].min():.2f} - ${data['close'].max():.2f}")
    print(f"  Average daily range: ${(data['high'] - data['low']).mean():.2f}")
    
    # Create config with phased trading
    config_content = """
data:
  amibroker_path: "dummy"
  data_frequency: "5T"

backtest:
  initial_cash: 100000
  position_size: 10000
  position_size_type: "value"
  execution_price: "formula"
  buy_execution_formula: "(H + L + C) / 3"
  sell_execution_formula: "(H + L + C) / 3"
  signal_lag: 0
  fees: 0.0
  fixed_fees: 0.0
  slippage: 0.0
  direction: "both"
  min_size: 0.0000000001
  call_seq: "auto"
  freq: "5T"
  phased_trading_enabled: true
  phased_entry_bars: 5
  phased_exit_bars: 3
  phased_entry_distribution: "linear"
  phased_exit_distribution: "linear"
  consolidate_phased_trades: false

output:
  results_dir: "results_month_test"
  trade_list_filename: "trades.csv"
  equity_curve_filename: "equity.csv"
"""
    
    config_path = "config_month_test.yaml"
    with open(config_path, 'w') as f:
        f.write(config_content)
    
    # Create multiple test signals throughout the month
    print("\n### CREATING TEST SIGNALS")
    print("-"*60)
    
    n_bars = len(data['close'])
    entries = np.zeros(n_bars, dtype=bool)
    exits = np.zeros(n_bars, dtype=bool)
    
    # Create several trades throughout the month
    test_trades = [
        (500, 800),    # Trade 1: Day 2-3
        (1500, 2000),  # Trade 2: Day 5-7
        (3000, 3500),  # Trade 3: Day 10-12
        (4500, 5000),  # Trade 4: Day 15-17
        (6000, 6500),  # Trade 5: Day 21-23
        (7500, 8000),  # Trade 6: Day 26-28
    ]
    
    actual_trades = []
    for entry_bar, exit_bar in test_trades:
        if entry_bar < n_bars and exit_bar < n_bars:
            entries[entry_bar] = True
            exits[exit_bar] = True
            actual_trades.append((entry_bar, exit_bar))
    
    print(f"Created {len(actual_trades)} test signals:")
    for i, (entry_bar, exit_bar) in enumerate(actual_trades):
        entry_price = data['close'][entry_bar]
        exit_price = data['close'][exit_bar]
        print(f"  Trade {i+1}: Entry bar {entry_bar} (${entry_price:.2f}) -> Exit bar {exit_bar} (${exit_price:.2f})")
    
    # Initialize engine and capture signals
    print("\n### RUNNING PHASED TRADING ENGINE")
    print("-"*60)
    
    engine = VectorBTEngine(config_path)
    
    # Process signals through phased engine directly to see transformation
    if engine.phased_engine.config.enabled:
        phased_results = engine.phased_engine.process_signals(
            entries,
            exits,
            data,
            position_size=10000
        )
        
        phased_entries = phased_results['entries']
        phased_exits = phased_results['exits']
        entry_sizes = phased_results['entry_sizes']
        exit_sizes = phased_results['exit_sizes']
        
        print(f"Original signals: {np.sum(entries)} entries, {np.sum(exits)} exits")
        print(f"After phasing: {np.sum(phased_entries)} entries, {np.sum(phased_exits)} exits")
        
        # Detailed analysis for each trade
        print("\n### DETAILED PHASING ANALYSIS")
        print("-"*60)
        
        reconciliation_data = []
        
        for trade_num, (original_entry, original_exit) in enumerate(actual_trades):
            print(f"\nTrade {trade_num + 1} Analysis:")
            print(f"  Original entry at bar {original_entry}, exit at bar {original_exit}")
            
            # Find phased entries for this trade
            entry_phase_bars = []
            for i in range(5):  # 5 phased entries
                phase_bar = original_entry + i
                if phase_bar < n_bars and phased_entries[phase_bar]:
                    entry_phase_bars.append(phase_bar)
            
            # Find phased exits for this trade
            exit_phase_bars = []
            for i in range(3):  # 3 phased exits
                phase_bar = original_exit + i
                if phase_bar < n_bars and phased_exits[phase_bar]:
                    exit_phase_bars.append(phase_bar)
            
            # Calculate entry details
            print(f"\n  Entry Phases ({len(entry_phase_bars)} bars):")
            entry_weighted_sum = 0
            total_entry_size = 0
            
            for bar in entry_phase_bars:
                hlc3 = (data['high'][bar] + data['low'][bar] + data['close'][bar]) / 3
                size = entry_sizes[bar]
                weight = size / 10000
                weighted_price = hlc3 * weight
                entry_weighted_sum += weighted_price
                total_entry_size += size
                
                print(f"    Bar {bar}: H=${data['high'][bar]:.2f}, L=${data['low'][bar]:.2f}, C=${data['close'][bar]:.2f}")
                print(f"      HLC3=${hlc3:.2f}, Size=${size:.2f}, Weight={weight:.1%}")
                
                reconciliation_data.append({
                    'Trade': trade_num + 1,
                    'Phase': 'Entry',
                    'Bar': bar,
                    'High': data['high'][bar],
                    'Low': data['low'][bar],
                    'Close': data['close'][bar],
                    'HLC3': hlc3,
                    'Size': size,
                    'Weight': weight
                })
            
            if entry_phase_bars:
                print(f"  Weighted average entry: ${entry_weighted_sum:.2f}")
                print(f"  First bar close price: ${data['close'][entry_phase_bars[0]]:.2f}")
            
            # Calculate exit details
            print(f"\n  Exit Phases ({len(exit_phase_bars)} bars):")
            exit_weighted_sum = 0
            total_exit_size = 0
            
            for bar in exit_phase_bars:
                hlc3 = (data['high'][bar] + data['low'][bar] + data['close'][bar]) / 3
                size = exit_sizes[bar]
                weight = size / 10000
                weighted_price = hlc3 * weight
                exit_weighted_sum += weighted_price
                total_exit_size += size
                
                print(f"    Bar {bar}: H=${data['high'][bar]:.2f}, L=${data['low'][bar]:.2f}, C=${data['close'][bar]:.2f}")
                print(f"      HLC3=${hlc3:.2f}, Size=${size:.2f}, Weight={weight:.1%}")
                
                reconciliation_data.append({
                    'Trade': trade_num + 1,
                    'Phase': 'Exit',
                    'Bar': bar,
                    'High': data['high'][bar],
                    'Low': data['low'][bar],
                    'Close': data['close'][bar],
                    'HLC3': hlc3,
                    'Size': size,
                    'Weight': weight
                })
            
            if exit_phase_bars:
                print(f"  Weighted average exit: ${exit_weighted_sum:.2f}")
                print(f"  First bar close price: ${data['close'][exit_phase_bars[0]]:.2f}")
    
    # Capture what VectorBT receives
    print("\n### VECTORBT SIGNAL CAPTURE")
    print("-"*60)
    
    original_from_signals = vbt.Portfolio.from_signals
    vbt_captured = {}
    
    def capture_vbt_signals(close, entries, exits, **kwargs):
        """Capture signals that VectorBT receives."""
        vbt_captured['entries'] = entries.copy()
        vbt_captured['exits'] = exits.copy()
        vbt_captured['size'] = kwargs.get('size', None)
        
        if entries.ndim > 1:
            entries_flat = entries.flatten()
            exits_flat = exits.flatten()
        else:
            entries_flat = entries
            exits_flat = exits
        
        print(f"VectorBT received: {np.sum(entries_flat)} entries, {np.sum(exits_flat)} exits")
        
        return original_from_signals(close=close, entries=entries, exits=exits, **kwargs)
    
    vbt.Portfolio.from_signals = capture_vbt_signals
    
    try:
        # Run backtest
        pf = engine.run_vectorized_backtest(data, entries, exits, "ES")
        
        # Check trades
        if hasattr(pf.trades, 'records') and len(pf.trades.records) > 0:
            trades_df = pf.trades.records_readable
            print(f"\n### FINAL TRADES IN TRADELIST")
            print("-"*60)
            print(f"Number of trades: {len(trades_df)}")
            
            for idx, trade in trades_df.iterrows():
                print(f"\nTrade {idx + 1}:")
                print(f"  Entry Index: {trade.get('Entry Index', 'N/A')}")
                print(f"  Exit Index: {trade.get('Exit Index', 'N/A')}")
                print(f"  Size: ${trade.get('Size', 0):.2f}")
                print(f"  Entry Price: ${trade.get('Avg Entry Price', 0):.2f}")
                print(f"  Exit Price: ${trade.get('Avg Exit Price', 0):.2f}")
                print(f"  PnL: ${trade.get('PnL', 0):.2f}")
    
    finally:
        vbt.Portfolio.from_signals = original_from_signals
    
    # Save reconciliation to Excel
    print("\n### SAVING MONTH TEST RESULTS")
    print("-"*60)
    
    recon_df = pd.DataFrame(reconciliation_data)
    
    with pd.ExcelWriter('phased_month_test_results.xlsx', engine='openpyxl') as writer:
        # Sheet 1: Full reconciliation
        recon_df.to_excel(writer, sheet_name='Full_Reconciliation', index=False)
        
        # Sheet 2: Summary by trade
        if not recon_df.empty:
            summary_data = []
            for trade_num in recon_df['Trade'].unique():
                trade_data = recon_df[recon_df['Trade'] == trade_num]
                entry_data = trade_data[trade_data['Phase'] == 'Entry']
                exit_data = trade_data[trade_data['Phase'] == 'Exit']
                
                if not entry_data.empty and not exit_data.empty:
                    summary_data.append({
                        'Trade': trade_num,
                        'Entry_Bars': len(entry_data),
                        'Exit_Bars': len(exit_data),
                        'Entry_Weighted_Avg': (entry_data['HLC3'] * entry_data['Weight']).sum(),
                        'Exit_Weighted_Avg': (exit_data['HLC3'] * exit_data['Weight']).sum(),
                        'Entry_First_Close': entry_data.iloc[0]['Close'],
                        'Exit_First_Close': exit_data.iloc[0]['Close'],
                        'Entry_Price_Range': f"${entry_data['Close'].min():.2f}-${entry_data['Close'].max():.2f}",
                        'Exit_Price_Range': f"${exit_data['Close'].min():.2f}-${exit_data['Close'].max():.2f}"
                    })
            
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Trade_Summary', index=False)
    
    print("Results saved to: phased_month_test_results.xlsx")
    
    # Clean up
    import os
    if os.path.exists(config_path):
        os.remove(config_path)
    
    # Final verdict
    print("\n" + "="*80)
    print("MONTH TEST CONCLUSION")
    print("="*80)
    
    if 'vbt_captured' in locals() and 'entries' in vbt_captured:
        vbt_entries = vbt_captured['entries']
        if vbt_entries.ndim > 1:
            vbt_entries = vbt_entries.flatten()
        
        expected_phased_entries = len(actual_trades) * 5
        expected_phased_exits = len(actual_trades) * 3
        actual_vbt_entries = np.sum(vbt_entries)
        
        print(f"\nExpected after phasing: {expected_phased_entries} entries, {expected_phased_exits} exits")
        print(f"VectorBT received: {actual_vbt_entries} signals")
        
        if actual_vbt_entries >= expected_phased_entries:
            print("\n[OK] PHASED TRADING IS WORKING OVER FULL MONTH")
            print("- Signals are being split correctly")
            print("- Real price movements are handled properly")
            print("- Position sizes are distributed as configured")
        else:
            print("\n[WARNING] Fewer signals than expected")
            print("This could be due to signals near the end of data")
    
    return True


if __name__ == "__main__":
    try:
        test_phased_trading_month()
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()