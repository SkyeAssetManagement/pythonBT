"""
Basic test for phased entry system components
"""

import sys
import os
import pandas as pd
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_basic_imports():
    """Test that all phased entry modules can be imported"""
    try:
        from trading.core.phased_entry import PhasedEntryConfig, PhasedEntry, TradePhase, PhasedPositionTracker
        print("+ Phased entry core imports successful")

        from trading.core.phased_execution_engine import PhasedExecutionEngine, PhasedExecutionConfig
        print("+ Phased execution engine imports successful")

        from trading.strategies.phased_strategy_base import PhasedTradingStrategy
        print("+ Phased strategy base imports successful")

        return True
    except Exception as e:
        print(f"- Import failed: {e}")
        return False

def test_config_creation():
    """Test basic configuration creation"""
    try:
        from trading.core.phased_entry import PhasedEntryConfig

        config = PhasedEntryConfig()
        print(f"+ Default config created - enabled: {config.enabled}, max_phases: {config.max_phases}")

        config_enabled = PhasedEntryConfig(enabled=True, max_phases=4)
        print(f"+ Custom config created - enabled: {config_enabled.enabled}, max_phases: {config_enabled.max_phases}")

        return True
    except Exception as e:
        print(f"- Config creation failed: {e}")
        return False

def test_phased_entry_basic():
    """Test basic phased entry functionality"""
    try:
        from trading.core.phased_entry import PhasedEntryConfig, PhasedEntry

        config = PhasedEntryConfig(enabled=True, max_phases=3, initial_size_percent=40.0, phase_trigger_value=2.0)

        entry = PhasedEntry(
            config=config,
            initial_signal_bar=10,
            initial_price=100.0,
            is_long=True,
            total_target_size=1000.0
        )

        print(f"+ Phased entry created - phases: {len(entry.phases)}, first phase size: {entry.phases[0].size}")

        # Test phase execution
        entry.execute_phase(entry.phases[0], 100.5, 11)
        executed = entry.get_executed_phases()
        print(f"+ Phase executed - executed phases: {len(executed)}")

        # Test average price calculation
        avg_price = entry.get_average_entry_price()
        print(f"+ Average entry price: ${avg_price:.2f}")

        return True
    except Exception as e:
        print(f"- Phased entry test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run basic validation tests"""
    print("=== Basic Phased Entry System Validation ===\n")

    success = True

    success &= test_basic_imports()
    success &= test_config_creation()
    success &= test_phased_entry_basic()

    print(f"\n{'='*50}")
    if success:
        print("SUCCESS: BASIC VALIDATION PASSED!")
        print("Core phased entry components are working.")
    else:
        print("FAILED: VALIDATION FAILED!")
        print("There are issues with the phased entry implementation.")
    print(f"{'='*50}")

    return success

if __name__ == "__main__":
    main()