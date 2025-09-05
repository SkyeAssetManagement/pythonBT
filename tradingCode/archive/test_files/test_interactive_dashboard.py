# test_interactive_dashboard.py  
# Quick test to launch the interactive dashboard
#
# This demonstrates the full interactive experience with chart and trade list

import sys
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def launch_test_dashboard():
    """Launch the interactive dashboard with test data"""
    
    print(f"[LAUNCH] LAUNCHING INTERACTIVE TRADING DASHBOARD")
    print(f"=" * 60)
    
    try:
        from src.dashboard.dashboard_launcher import launch_interactive_dashboard
        from src.dashboard.vispy_candlestick_renderer import create_test_data
        import numpy as np
        import pandas as pd
        
        print(f"[CHART] Creating realistic test data...")
        
        # Create 100K candlesticks for realistic performance test
        price_data = create_test_data(100000)
        print(f"   [OK] Created {len(price_data['close']):,} candlesticks")
        
        # Create 500 realistic trades
        print(f"📋 Creating test trades...")
        trade_data = []
        
        for i in range(500):
            # Random entry within first 90% of data
            entry_time = np.random.randint(0, 90000)  
            # Random duration 1-200 bars
            exit_time = entry_time + np.random.randint(1, 200)
            
            # Get actual prices from our data
            entry_price = price_data['close'][entry_time]
            exit_price = price_data['close'][min(exit_time, 99999)]
            
            # Realistic forex-style trade
            direction = 'Long' if np.random.random() > 0.3 else 'Short'  # 70% longs
            size = np.random.uniform(0.1, 2.0)
            
            # Calculate PnL based on direction
            if direction == 'Long':
                pnl = (exit_price - entry_price) * size
            else:
                pnl = (entry_price - exit_price) * size
            
            trade_data.append({
                'EntryTime': entry_time,
                'ExitTime': exit_time, 
                'Direction': direction,
                'Avg Entry Price': entry_price,
                'Avg Exit Price': exit_price,
                'Size': size,
                'PnL': pnl
            })
        
        trades_df = pd.DataFrame(trade_data)
        print(f"   [OK] Created {len(trades_df)} test trades")
        
        # Calculate some statistics
        total_pnl = trades_df['PnL'].sum()
        win_rate = (trades_df['PnL'] > 0).mean() * 100
        print(f"   [GROWTH] Total PnL: {total_pnl:+.2f}")
        print(f"   [TARGET] Win Rate: {win_rate:.1f}%")
        
        print(f"\n🎮 LAUNCHING INTERACTIVE DASHBOARD...")
        print(f"   This will open TWO windows:")
        print(f"   1. 📋 Trade List Panel (Qt window)")
        print(f"   2. [CHART] VisPy Chart (separate window)")
        print(f"")
        print(f"[TARGET] FEATURES TO TEST:")
        print(f"   [OK] Click on trades in the list -> chart jumps to that trade")
        print(f"   [OK] Mouse wheel on chart -> zoom in/out")
        print(f"   [OK] Drag on chart -> pan around")
        print(f"   [OK] Press 'S' on chart -> take screenshot")
        print(f"   [OK] Press 'R' on chart -> reset to recent view")
        print(f"   [OK] Press 'Q' on chart -> quit chart")
        print(f"")
        
        # Launch the dashboard
        success = launch_interactive_dashboard(
            price_data=price_data,
            trade_data=trades_df,
            portfolio_data=None,
            show_chart=True,
            show_trade_list=True
        )
        
        if success:
            print(f"[OK] DASHBOARD TEST COMPLETED SUCCESSFULLY!")
        else:
            print(f"[X] DASHBOARD TEST FAILED")
            
        return success
        
    except ImportError as e:
        print(f"[X] MISSING DEPENDENCIES: {e}")
        print(f"")
        print(f"[TOOLS] TO FIX, RUN:")
        print(f"   pip install vispy imageio PyQt5")
        print(f"")
        return False
        
    except Exception as e:
        print(f"[X] ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

def launch_from_real_backtest():
    """Launch dashboard using real backtest results if available"""
    
    print(f"[SEARCH] LOOKING FOR REAL BACKTEST RESULTS...")
    
    import glob
    
    # Look for recent backtest results
    result_files = glob.glob("results/tradelist.csv") + glob.glob("tradingCode/results/tradelist.csv")
    
    if result_files:
        print(f"   [OK] Found real results: {result_files[0]}")
        
        try:
            from src.dashboard.dashboard_launcher import launch_dashboard_from_main_results
            import os
            
            results_dir = os.path.dirname(result_files[0])
            print(f"   📂 Results directory: {results_dir}")
            
            success = launch_dashboard_from_main_results(results_dir)
            
            if success:
                print(f"[OK] REAL DATA DASHBOARD COMPLETED!")
            else:
                print(f"[X] REAL DATA DASHBOARD FAILED")
                
            return success
            
        except Exception as e:
            print(f"[X] ERROR with real data: {e}")
            return False
    else:
        print(f"   ℹ️  No real backtest results found")
        print(f"   [IDEA] Run a backtest first:")
        print(f"      python main.py ES simpleSMA --useDefaults")
        print(f"")
        return False

if __name__ == "__main__":
    print(f"INTERACTIVE DASHBOARD TEST OPTIONS:")
    print(f"1. Test with synthetic data (always works)")
    print(f"2. Test with real backtest results (if available)")
    print(f"")
    
    # Try real data first, fallback to synthetic
    print(f"🔄 Trying real backtest results first...")
    real_success = launch_from_real_backtest()
    
    if not real_success:
        print(f"🔄 Falling back to synthetic test data...")
        test_success = launch_test_dashboard()
        
        if test_success:
            print(f"\n[IDEA] TO USE WITH REAL DATA:")
            print(f"   1. Run: python main.py ES simpleSMA --useDefaults")
            print(f"   2. Dashboard will auto-launch with real results!")
    
    print(f"\n[TARGET] DASHBOARD TESTING COMPLETE")