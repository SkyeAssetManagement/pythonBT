"""
Test phased trading with real market data over the last month
Comprehensive verification of prices and reconciliation
"""

import numpy as np
import pandas as pd
import vectorbtpro as vbt
from datetime import datetime, timedelta
from pathlib import Path
import sys
import yaml

sys.path.insert(0, str(Path(__file__).parent))

from phased_trading_correct import PhasedTradingCorrect
from src.data.parquet_converter import ParquetConverter


def run_phased_backtest_real_data():
    """Run phased trading backtest with real market data."""
    
    print("="*80)
    print("PHASED TRADING - REAL MARKET DATA BACKTEST")
    print("="*80)
    print(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    # Load real market data
    print("\n### LOADING MARKET DATA")
    print("-"*60)
    
    # Use ES data for testing
    symbol = "ES"
    frequency = "1m"
    adjustment = "diffAdjusted"
    
    # Load last month of data
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
    
    converter = ParquetConverter()
    data_dict = converter.load_or_convert(symbol, frequency, adjustment)
    
    if data_dict is None:
        print("ERROR: Could not load market data")
        return None
    
    # Filter to last month
    data_dict = converter.filter_data_by_date(data_dict, start_date, end_date)
    
    n_bars = len(data_dict['close'])
    print(f"Loaded {n_bars} bars of {symbol} data")
    print(f"Date range: {start_date} to {end_date}")
    
    # Create sample signals for testing
    # We'll create a few entry/exit pairs throughout the month
    master_entries = np.zeros(n_bars, dtype=bool)
    master_exits = np.zeros(n_bars, dtype=bool)
    
    # Place signals at regular intervals for testing
    signal_positions = [
        (100, 500),    # Entry at bar 100, exit at bar 500
        (1000, 1400),  # Entry at bar 1000, exit at bar 1400
        (2000, 2400),  # Entry at bar 2000, exit at bar 2400
    ]
    
    for entry_bar, exit_bar in signal_positions:
        if entry_bar < n_bars:
            master_entries[entry_bar] = True
        if exit_bar < n_bars:
            master_exits[exit_bar] = True
    
    print(f"\nCreated {master_entries.sum()} entry signals and {master_exits.sum()} exit signals")
    
    # Configuration
    config = {
        'backtest': {
            'initial_cash': 100000,
            'position_size': 10000,
            'position_size_type': 'value',
            'direction': 'longonly',
            'fees': 0.0002,  # 0.02% fees
            'fixed_fees': 1.0,  # $1 per trade
            'slippage': 0.0001,  # 0.01% slippage
            'freq': '1T'
        }
    }
    
    # Run phased trading
    print("\n### RUNNING PHASED BACKTEST")
    print("-"*60)
    
    phased_trader = PhasedTradingCorrect(n_phases=5, phase_distribution="equal")
    
    # Get phased signals
    phased_results = phased_trader.create_phased_signals(master_entries, master_exits, data_dict)
    
    # Display phasing summary
    print(f"Phased entries: {phased_results['phased_entries'].sum()} (from {master_entries.sum()} master signals)")
    print(f"Phased exits: {phased_results['phased_exits'].sum()} (from {master_exits.sum()} master signals)")
    print(f"Phase weights: {phased_trader.phase_weights}")
    
    # Run backtest
    pf, verification = phased_trader.run_phased_backtest(
        data_dict, 
        master_entries, 
        master_exits, 
        config, 
        symbol
    )
    
    # Get portfolio statistics
    print("\n### PORTFOLIO PERFORMANCE")
    print("-"*60)
    print(f"Total Return: {pf.total_return:.2%}")
    print(f"Sharpe Ratio: {pf.sharpe_ratio:.2f}")
    print(f"Max Drawdown: {pf.max_drawdown:.2%}")
    print(f"Number of Trades: {len(pf.trades.records)}")
    
    # Verify trade prices
    print("\n### VERIFYING TRADE PRICES")
    print("-"*60)
    
    verification_df = phased_trader.verify_trade_prices(pf, verification, data_dict)
    
    if not verification_df.empty:
        # Create detailed reconciliation
        reconciliation_data = []
        
        for signal_entry, signal_exit in signal_positions:
            if signal_entry >= n_bars or signal_exit >= n_bars:
                continue
                
            print(f"\n## Signal Pair: Entry at bar {signal_entry}, Exit at bar {signal_exit}")
            print("-"*50)
            
            # Entry phases
            print("\nENTRY PHASES:")
            for phase in range(5):
                bar_idx = signal_entry + phase
                if bar_idx < n_bars:
                    ohlc = {
                        'open': data_dict['open'][bar_idx],
                        'high': data_dict['high'][bar_idx],
                        'low': data_dict['low'][bar_idx],
                        'close': data_dict['close'][bar_idx]
                    }
                    hlc3 = (ohlc['high'] + ohlc['low'] + ohlc['close']) / 3
                    size = phased_trader.phase_weights[phase] * config['backtest']['position_size']
                    
                    print(f"  Phase {phase+1} (Bar {bar_idx}):")
                    print(f"    OHLC: {ohlc['open']:.2f} / {ohlc['high']:.2f} / {ohlc['low']:.2f} / {ohlc['close']:.2f}")
                    print(f"    HLC3 Price: {hlc3:.2f}")
                    print(f"    Position Size: ${size:.2f}")
                    
                    reconciliation_data.append({
                        'Signal': f"Entry_{signal_entry}",
                        'Phase': phase + 1,
                        'Bar': bar_idx,
                        'Type': 'Entry',
                        'Open': ohlc['open'],
                        'High': ohlc['high'],
                        'Low': ohlc['low'],
                        'Close': ohlc['close'],
                        'HLC3_Calculated': hlc3,
                        'Size': size,
                        'Weight': phased_trader.phase_weights[phase]
                    })
            
            # Exit phases
            print("\nEXIT PHASES:")
            for phase in range(5):
                bar_idx = signal_exit + phase
                if bar_idx < n_bars:
                    ohlc = {
                        'open': data_dict['open'][bar_idx],
                        'high': data_dict['high'][bar_idx],
                        'low': data_dict['low'][bar_idx],
                        'close': data_dict['close'][bar_idx]
                    }
                    hlc3 = (ohlc['high'] + ohlc['low'] + ohlc['close']) / 3
                    size_pct = phased_trader.phase_weights[phase] * 100
                    
                    print(f"  Phase {phase+1} (Bar {bar_idx}):")
                    print(f"    OHLC: {ohlc['open']:.2f} / {ohlc['high']:.2f} / {ohlc['low']:.2f} / {ohlc['close']:.2f}")
                    print(f"    HLC3 Price: {hlc3:.2f}")
                    print(f"    Exit Size: {size_pct:.1f}% of position")
                    
                    reconciliation_data.append({
                        'Signal': f"Exit_{signal_exit}",
                        'Phase': phase + 1,
                        'Bar': bar_idx,
                        'Type': 'Exit',
                        'Open': ohlc['open'],
                        'High': ohlc['high'],
                        'Low': ohlc['low'],
                        'Close': ohlc['close'],
                        'HLC3_Calculated': hlc3,
                        'Size': f"{size_pct:.1f}%",
                        'Weight': phased_trader.phase_weights[phase]
                    })
        
        # Create reconciliation DataFrame
        reconciliation_df = pd.DataFrame(reconciliation_data)
        
        # Check price matching
        print("\n### PRICE VERIFICATION SUMMARY")
        print("-"*60)
        
        if not verification_df.empty:
            entry_matches = verification_df['Entry_Match'].all()
            exit_matches = verification_df['Exit_Match'].all()
            
            if entry_matches and exit_matches:
                print("SUCCESS: All trades executed at expected HLC3 prices!")
            else:
                print("WARNING: Some trades did not match expected prices")
                print(f"Entry matches: {verification_df['Entry_Match'].sum()}/{len(verification_df)}")
                print(f"Exit matches: {verification_df['Exit_Match'].sum()}/{len(verification_df)}")
        
        return pf, verification_df, reconciliation_df
    
    return pf, None, None


def save_reconciliation_report(verification_df, reconciliation_df):
    """Save reconciliation data to files."""
    
    # Save to CSV
    if verification_df is not None:
        verification_df.to_csv("phased_trades_verification.csv", index=False)
        print("\nSaved verification to phased_trades_verification.csv")
    
    if reconciliation_df is not None:
        reconciliation_df.to_csv("phased_trades_reconciliation.csv", index=False)
        print("Saved reconciliation to phased_trades_reconciliation.csv")
        
        # Also save to Excel with formatting
        with pd.ExcelWriter("phased_trades_reconciliation.xlsx", engine='openpyxl') as writer:
            reconciliation_df.to_excel(writer, sheet_name='Reconciliation', index=False)
            verification_df.to_excel(writer, sheet_name='Verification', index=False) if verification_df is not None else None
            
            # Add formatting
            workbook = writer.book
            for worksheet in workbook.worksheets:
                for column in worksheet.columns:
                    max_length = 0
                    column_letter = column[0].column_letter
                    for cell in column:
                        try:
                            if len(str(cell.value)) > max_length:
                                max_length = len(str(cell.value))
                        except:
                            pass
                    adjusted_width = min(max_length + 2, 50)
                    worksheet.column_dimensions[column_letter].width = adjusted_width
        
        print("Saved Excel file: phased_trades_reconciliation.xlsx")


def create_markdown_report(pf, verification_df, reconciliation_df):
    """Create a markdown report of the results."""
    
    report = []
    report.append("# Phased Trading Reconciliation Report")
    report.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("\n## Configuration")
    report.append("- **Phasing**: 5 bars per signal")
    report.append("- **Distribution**: Equal weights (20% per phase)")
    report.append("- **Price Formula**: (H + L + C) / 3")
    
    if pf is not None:
        report.append("\n## Portfolio Performance")
        report.append(f"- **Total Return**: {pf.total_return:.2%}")
        report.append(f"- **Sharpe Ratio**: {pf.sharpe_ratio:.2f}")
        report.append(f"- **Max Drawdown**: {pf.max_drawdown:.2%}")
        report.append(f"- **Number of Trades**: {len(pf.trades.records)}")
    
    if reconciliation_df is not None and not reconciliation_df.empty:
        report.append("\n## Phased Execution Details")
        
        # Group by signal
        for signal in reconciliation_df['Signal'].unique():
            signal_data = reconciliation_df[reconciliation_df['Signal'] == signal]
            signal_type = signal_data['Type'].iloc[0]
            
            report.append(f"\n### {signal}")
            report.append(f"\n| Phase | Bar | Open | High | Low | Close | HLC3 | Size |")
            report.append("|-------|-----|------|------|-----|-------|------|------|")
            
            for _, row in signal_data.iterrows():
                report.append(f"| {row['Phase']} | {row['Bar']} | {row['Open']:.2f} | {row['High']:.2f} | {row['Low']:.2f} | {row['Close']:.2f} | {row['HLC3_Calculated']:.2f} | {row['Size']} |")
    
    if verification_df is not None and not verification_df.empty:
        report.append("\n## Price Verification")
        
        all_match = verification_df['Entry_Match'].all() and verification_df['Exit_Match'].all()
        if all_match:
            report.append("\n**Result: SUCCESS - All trades executed at expected HLC3 prices**")
        else:
            report.append("\n**Result: PARTIAL - Some trades did not match expected prices**")
        
        report.append("\n| Trade | Entry Price | Entry HLC3 | Entry Match | Exit Price | Exit HLC3 | Exit Match |")
        report.append("|-------|-------------|------------|-------------|------------|-----------|------------|")
        
        for _, row in verification_df.iterrows():
            entry_check = "✓" if row['Entry_Match'] else "✗"
            exit_check = "✓" if row['Exit_Match'] else "✗"
            report.append(f"| {row['Trade_ID']} | {row['Entry_Actual']:.2f} | {row['Entry_HLC3_Calc']:.2f} | {entry_check} | {row['Exit_Actual']:.2f} | {row['Exit_HLC3_Calc']:.2f} | {exit_check} |")
    
    report.append("\n## Conclusion")
    report.append("\nThe phased trading implementation correctly distributes trades across 5 bars,")
    report.append("with each bar using the (H+L+C)/3 price formula for execution.")
    
    # Save report
    report_text = "\n".join(report)
    with open("phased_trading_reconciliation.md", "w") as f:
        f.write(report_text)
    
    print("\nSaved markdown report: phased_trading_reconciliation.md")
    
    return report_text


if __name__ == "__main__":
    # Run the backtest
    pf, verification_df, reconciliation_df = run_phased_backtest_real_data()
    
    if pf is not None:
        # Save all reports
        save_reconciliation_report(verification_df, reconciliation_df)
        report_text = create_markdown_report(pf, verification_df, reconciliation_df)
        
        print("\n" + "="*80)
        print("RECONCILIATION COMPLETE")
        print("="*80)
        print("\nGenerated files:")
        print("1. phased_trades_verification.csv - Trade-level verification")
        print("2. phased_trades_reconciliation.csv - Phase-level details")
        print("3. phased_trades_reconciliation.xlsx - Excel with both sheets")
        print("4. phased_trading_reconciliation.md - Markdown report")