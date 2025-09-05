"""
Correct implementation of phased trading with VectorBT Pro
Implements proper price calculation for phased entries and exits
"""

import numpy as np
import pandas as pd
import vectorbtpro as vbt
from typing import Dict, Tuple, Optional
from datetime import datetime
import yaml


class PhasedTradingCorrect:
    """
    Implements correct phased trading with proper price tracking.
    
    When a master signal occurs, the position is phased over the next 5 bars,
    with each bar using the (H+L+C)/3 price formula for that specific bar.
    """
    
    def __init__(self, n_phases: int = 5, phase_distribution: str = "equal"):
        """
        Initialize phased trading.
        
        Args:
            n_phases: Number of bars to phase the entry/exit over
            phase_distribution: How to distribute the position ("equal", "front_loaded", "back_loaded")
        """
        self.n_phases = n_phases
        self.phase_distribution = phase_distribution
        self.phase_weights = self._calculate_phase_weights()
        
    def _calculate_phase_weights(self) -> np.ndarray:
        """Calculate the weights for each phase."""
        if self.phase_distribution == "equal":
            weights = np.ones(self.n_phases) / self.n_phases
        elif self.phase_distribution == "front_loaded":
            # More weight at the beginning
            weights = np.linspace(2, 1, self.n_phases)
            weights = weights / weights.sum()
        elif self.phase_distribution == "back_loaded":
            # More weight at the end
            weights = np.linspace(1, 2, self.n_phases)
            weights = weights / weights.sum()
        else:
            raise ValueError(f"Unknown distribution: {self.phase_distribution}")
        return weights
    
    def create_phased_signals(self, 
                            master_entries: np.ndarray,
                            master_exits: np.ndarray,
                            data: Dict[str, np.ndarray]) -> Dict:
        """
        Convert master signals into phased signals over multiple bars.
        
        Args:
            master_entries: Boolean array of master entry signals
            master_exits: Boolean array of master exit signals
            data: OHLCV data dictionary
            
        Returns:
            Dictionary with phased signals, sizes, and expected prices
        """
        n_bars = len(master_entries)
        
        # Initialize output arrays
        phased_entries = np.zeros(n_bars, dtype=bool)
        phased_exits = np.zeros(n_bars, dtype=bool)
        entry_sizes = np.zeros(n_bars, dtype=np.float64)
        exit_sizes = np.zeros(n_bars, dtype=np.float64)
        
        # Track expected execution prices for verification
        expected_entry_prices = np.full(n_bars, np.nan)
        expected_exit_prices = np.full(n_bars, np.nan)
        
        # Calculate HLC3 prices for each bar
        hlc3_prices = (data['high'] + data['low'] + data['close']) / 3.0
        
        # Process entry signals
        entry_indices = np.where(master_entries)[0]
        for master_idx in entry_indices:
            # Phase the entry over the next n_phases bars
            for phase_num in range(self.n_phases):
                phase_idx = master_idx + phase_num
                if phase_idx < n_bars:
                    phased_entries[phase_idx] = True
                    entry_sizes[phase_idx] = self.phase_weights[phase_num]
                    # Record the expected price for this phased entry
                    expected_entry_prices[phase_idx] = hlc3_prices[phase_idx]
        
        # Process exit signals
        exit_indices = np.where(master_exits)[0]
        for master_idx in exit_indices:
            # Phase the exit over the next n_phases bars
            for phase_num in range(self.n_phases):
                phase_idx = master_idx + phase_num
                if phase_idx < n_bars:
                    phased_exits[phase_idx] = True
                    exit_sizes[phase_idx] = self.phase_weights[phase_num]
                    # Record the expected price for this phased exit
                    expected_exit_prices[phase_idx] = hlc3_prices[phase_idx]
        
        return {
            'phased_entries': phased_entries,
            'phased_exits': phased_exits,
            'entry_sizes': entry_sizes,
            'exit_sizes': exit_sizes,
            'expected_entry_prices': expected_entry_prices,
            'expected_exit_prices': expected_exit_prices,
            'hlc3_prices': hlc3_prices
        }
    
    def run_phased_backtest(self,
                          data: Dict[str, np.ndarray],
                          master_entries: np.ndarray,
                          master_exits: np.ndarray,
                          config: Dict,
                          symbol: str = "TEST") -> Tuple[vbt.Portfolio, Dict]:
        """
        Run a backtest with phased entries and exits.
        
        Args:
            data: OHLCV data dictionary
            master_entries: Master entry signals
            master_exits: Master exit signals
            config: Backtest configuration
            symbol: Symbol name
            
        Returns:
            Tuple of (Portfolio, verification_results)
        """
        # Create phased signals
        phased_results = self.create_phased_signals(master_entries, master_exits, data)
        
        # Calculate execution prices using HLC3 formula
        execution_prices = phased_results['hlc3_prices']
        
        # Create size array for VectorBT (in dollars based on config)
        position_size = config['backtest']['position_size']
        
        # Create custom size array that varies per bar based on phase weights
        custom_sizes = np.zeros(len(data['close']))
        
        # Apply entry sizes
        entry_mask = phased_results['phased_entries']
        custom_sizes[entry_mask] = phased_results['entry_sizes'][entry_mask] * position_size
        
        # Apply exit sizes (for exits, we need to size based on current position)
        # VectorBT will handle this automatically with percent sizing
        exit_mask = phased_results['phased_exits']
        
        # For exits, we use percent sizing to close the appropriate fraction
        exit_size_type = np.full(len(data['close']), vbt.pf_enums.SizeType.Value)
        exit_size_type[exit_mask] = vbt.pf_enums.SizeType.Percent
        
        # Exit sizes as percentages (each phase exits its weight fraction)
        exit_sizes_pct = np.zeros(len(data['close']))
        exit_sizes_pct[exit_mask] = phased_results['exit_sizes'][exit_mask] * 100  # Convert to percentage
        
        # Combine size arrays
        final_sizes = custom_sizes.copy()
        final_sizes[exit_mask] = exit_sizes_pct[exit_mask]
        
        # Run the backtest with VectorBT
        pf = vbt.Portfolio.from_signals(
            close=execution_prices,
            entries=phased_results['phased_entries'],
            exits=phased_results['phased_exits'],
            size=final_sizes,
            size_type=exit_size_type,
            init_cash=config['backtest']['initial_cash'],
            direction=config['backtest']['direction'],
            fees=config['backtest'].get('fees', 0),
            fixed_fees=config['backtest'].get('fixed_fees', 0),
            slippage=config['backtest'].get('slippage', 0),
            freq=config['backtest'].get('freq', '1T')
        )
        
        # Prepare verification results
        verification = {
            'phased_results': phased_results,
            'execution_prices': execution_prices,
            'position_size': position_size,
            'phase_weights': self.phase_weights,
            'n_phases': self.n_phases
        }
        
        return pf, verification
    
    def verify_trade_prices(self,
                          pf: vbt.Portfolio,
                          verification: Dict,
                          data: Dict[str, np.ndarray]) -> pd.DataFrame:
        """
        Verify that trades executed at the expected prices.
        
        Args:
            pf: Portfolio from backtest
            verification: Verification data from run_phased_backtest
            data: Original OHLCV data
            
        Returns:
            DataFrame with verification results
        """
        # Get trades from portfolio
        trades = pf.trades.records_readable
        
        if len(trades) == 0:
            return pd.DataFrame()
        
        # Extract phased results
        phased_results = verification['phased_results']
        
        # Create verification dataframe
        verification_rows = []
        
        for idx, trade in trades.iterrows():
            entry_idx = int(trade['Entry Index'])
            exit_idx = int(trade['Exit Index'])
            
            # Get expected prices
            expected_entry = phased_results['expected_entry_prices'][entry_idx]
            expected_exit = phased_results['expected_exit_prices'][exit_idx]
            
            # Get actual prices from trade
            actual_entry = trade['Avg Entry Price']
            actual_exit = trade['Avg Exit Price']
            
            # Calculate HLC3 for those bars
            hlc3_entry = phased_results['hlc3_prices'][entry_idx]
            hlc3_exit = phased_results['hlc3_prices'][exit_idx]
            
            # Get OHLC values for reference
            entry_ohlc = {
                'open': data['open'][entry_idx],
                'high': data['high'][entry_idx],
                'low': data['low'][entry_idx],
                'close': data['close'][entry_idx]
            }
            exit_ohlc = {
                'open': data['open'][exit_idx],
                'high': data['high'][exit_idx],
                'low': data['low'][exit_idx],
                'close': data['close'][exit_idx]
            }
            
            verification_rows.append({
                'Trade_ID': idx,
                'Entry_Index': entry_idx,
                'Exit_Index': exit_idx,
                'Entry_Open': entry_ohlc['open'],
                'Entry_High': entry_ohlc['high'],
                'Entry_Low': entry_ohlc['low'],
                'Entry_Close': entry_ohlc['close'],
                'Entry_HLC3_Calc': hlc3_entry,
                'Entry_Expected': expected_entry,
                'Entry_Actual': actual_entry,
                'Entry_Match': np.abs(actual_entry - hlc3_entry) < 0.01,
                'Exit_Open': exit_ohlc['open'],
                'Exit_High': exit_ohlc['high'],
                'Exit_Low': exit_ohlc['low'],
                'Exit_Close': exit_ohlc['close'],
                'Exit_HLC3_Calc': hlc3_exit,
                'Exit_Expected': expected_exit,
                'Exit_Actual': actual_exit,
                'Exit_Match': np.abs(actual_exit - hlc3_exit) < 0.01,
                'Size': trade['Size'],
                'PnL': trade['PnL']
            })
        
        return pd.DataFrame(verification_rows)


def test_phased_trading_with_sample_data():
    """Test phased trading with sample data to verify pricing."""
    
    print("="*80)
    print("PHASED TRADING VERIFICATION TEST")
    print("="*80)
    
    # Create sample data with known OHLC values
    n_bars = 20
    np.random.seed(42)
    
    # Create realistic OHLC data
    base_price = 100
    data = {
        'open': np.array([base_price + i * 0.5 + np.random.uniform(-1, 1) for i in range(n_bars)]),
        'high': np.array([base_price + i * 0.5 + np.random.uniform(1, 3) for i in range(n_bars)]),
        'low': np.array([base_price + i * 0.5 + np.random.uniform(-3, -1) for i in range(n_bars)]),
        'close': np.array([base_price + i * 0.5 + np.random.uniform(-0.5, 0.5) for i in range(n_bars)]),
        'volume': np.ones(n_bars) * 1000
    }
    
    # Create master signals
    master_entries = np.zeros(n_bars, dtype=bool)
    master_exits = np.zeros(n_bars, dtype=bool)
    
    # Place signals at specific positions
    master_entries[2] = True   # Entry signal at bar 2
    master_exits[10] = True     # Exit signal at bar 10
    
    # Configuration
    config = {
        'backtest': {
            'initial_cash': 100000,
            'position_size': 10000,
            'position_size_type': 'value',
            'direction': 'longonly',
            'fees': 0,
            'fixed_fees': 0,
            'slippage': 0,
            'freq': '1T'
        }
    }
    
    # Run phased trading
    phased_trader = PhasedTradingCorrect(n_phases=5, phase_distribution="equal")
    
    # Get phased signals
    phased_results = phased_trader.create_phased_signals(master_entries, master_exits, data)
    
    # Display phasing details
    print("\n### PHASING DETAILS")
    print("-"*60)
    print(f"Master entry at bar 2 will phase over bars 2-6")
    print(f"Master exit at bar 10 will phase over bars 10-14")
    print(f"Phase weights: {phased_trader.phase_weights}")
    
    # Show expected prices for phased entries
    print("\n### EXPECTED ENTRY PRICES (Bars 2-6)")
    print("-"*60)
    for i in range(2, 7):
        if i < n_bars:
            hlc3 = (data['high'][i] + data['low'][i] + data['close'][i]) / 3
            print(f"Bar {i}: H={data['high'][i]:.2f}, L={data['low'][i]:.2f}, C={data['close'][i]:.2f}")
            print(f"        HLC3 = ({data['high'][i]:.2f} + {data['low'][i]:.2f} + {data['close'][i]:.2f}) / 3 = {hlc3:.2f}")
            print(f"        Size = {phased_trader.phase_weights[i-2] * config['backtest']['position_size']:.2f}")
    
    # Show expected prices for phased exits
    print("\n### EXPECTED EXIT PRICES (Bars 10-14)")
    print("-"*60)
    for i in range(10, 15):
        if i < n_bars:
            hlc3 = (data['high'][i] + data['low'][i] + data['close'][i]) / 3
            print(f"Bar {i}: H={data['high'][i]:.2f}, L={data['low'][i]:.2f}, C={data['close'][i]:.2f}")
            print(f"        HLC3 = ({data['high'][i]:.2f} + {data['low'][i]:.2f} + {data['close'][i]:.2f}) / 3 = {hlc3:.2f}")
            print(f"        Size = {phased_trader.phase_weights[i-10] * 100:.1f}% of position")
    
    # Run backtest
    pf, verification = phased_trader.run_phased_backtest(data, master_entries, master_exits, config)
    
    # Verify trade prices
    verification_df = phased_trader.verify_trade_prices(pf, verification, data)
    
    if not verification_df.empty:
        print("\n### TRADE VERIFICATION")
        print("-"*60)
        print(verification_df.to_string())
        
        # Check if all prices match
        all_match = verification_df['Entry_Match'].all() and verification_df['Exit_Match'].all()
        if all_match:
            print("\n✓ SUCCESS: All trades executed at expected HLC3 prices!")
        else:
            print("\n✗ FAILURE: Some trades did not execute at expected prices")
    
    return pf, verification_df


if __name__ == "__main__":
    # Run the test
    pf, verification_df = test_phased_trading_with_sample_data()
    
    # Save verification results
    if not verification_df.empty:
        verification_df.to_csv("phased_trading_verification.csv", index=False)
        print("\nVerification results saved to phased_trading_verification.csv")