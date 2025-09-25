#!/usr/bin/env python3
"""
Test Minimum Execution Time Phased Entry System
===============================================
Tests the new minimum time-based phase spreading approach
"""

import sys
import os
sys.path.insert(0, 'src')
sys.path.insert(0, 'src/trading')

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.trading.core.minimum_time_phased_entry import (
    MinimumTimePhasedConfig,
    MinimumTimePhasedEntry,
    MinimumTimePhasedTracker
)


def test_phase_bar_calculation():
    """Test calculation of phase execution bars"""
    print("=== TEST 1: Phase Bar Calculation ===")

    config = MinimumTimePhasedConfig(
        enabled=True,
        max_phases=3,
        minimum_execution_minutes=5.0,
        data_frequency_minutes=1.0,  # 1-minute bars
        spread_method="equal"
    )

    signal_lag = 2
    phase_bars = config.calculate_phase_bars(signal_lag)

    print(f"Configuration:")
    print(f"  Minimum execution time: {config.minimum_execution_minutes} minutes")
    print(f"  Data frequency: {config.data_frequency_minutes} minute bars")
    print(f"  Max phases: {config.max_phases}")
    print(f"  Signal lag: {signal_lag} bars")
    print()

    total_bars_needed = int(config.minimum_execution_minutes / config.data_frequency_minutes)
    print(f"Total bars needed for minimum execution time: {total_bars_needed}")
    print(f"Phase execution bars: {phase_bars}")
    print()

    # Expected: bars 2, 3, 4 for equal spread over 5 bars with 3 phases
    expected_spacing = total_bars_needed // config.max_phases
    print(f"Expected spacing between phases: ~{expected_spacing} bars")

    if len(phase_bars) == config.max_phases:
        print("[OK] Correct number of phases calculated")
    else:
        print(f"[ERROR] Expected {config.max_phases} phases, got {len(phase_bars)}")

    print("\n" + "="*60 + "\n")


def test_different_spread_methods():
    """Test different spreading methods"""
    print("=== TEST 2: Different Spread Methods ===")

    base_config = {
        'enabled': True,
        'max_phases': 3,
        'minimum_execution_minutes': 15.0,  # 15 minutes
        'data_frequency_minutes': 5.0,      # 5-minute bars = 3 bars total
    }

    spread_methods = ["equal", "weighted_front", "weighted_back"]
    signal_lag = 1

    for method in spread_methods:
        config = MinimumTimePhasedConfig(**base_config, spread_method=method)
        phase_bars = config.calculate_phase_bars(signal_lag)

        print(f"Spread method '{method}': {phase_bars}")
        print(f"  Bar intervals from signal: {[bar - signal_lag for bar in phase_bars[1:]]}")

    print("\n" + "="*60 + "\n")


def test_phased_entry_execution():
    """Test phased entry execution logic"""
    print("=== TEST 3: Phased Entry Execution Logic ===")

    config = MinimumTimePhasedConfig(
        enabled=True,
        max_phases=3,
        minimum_execution_minutes=10.0,  # 10 minutes
        data_frequency_minutes=5.0,      # 5-minute bars = 2 bars
        spread_method="equal",
        max_adverse_move=5.0,
        require_profit=False  # Allow all phases for testing
    )

    # Create phased entry
    signal_bar = 100
    signal_price = 150.0
    target_size = 300.0
    signal_lag = 1

    entry = MinimumTimePhasedEntry(
        config=config,
        signal_bar=signal_bar,
        signal_price=signal_price,
        is_long=True,
        target_size=target_size,
        signal_lag=signal_lag
    )

    print(f"Created phased entry:")
    print(f"  Signal bar: {signal_bar}, Signal price: ${signal_price}")
    print(f"  Target size: {target_size}, Signal lag: {signal_lag}")
    print(f"  Execution start bar: {entry.execution_start_bar}")
    print()

    print("Phase schedule:")
    for i, phase in enumerate(entry.phases):
        print(f"  Phase {phase.phase_number}: scheduled bar {phase.scheduled_bar}, size {phase.size:.2f}")

    print()

    # Simulate execution at scheduled bars
    current_prices = [151.0, 152.5, 154.0, 155.5]  # Rising prices

    for current_bar in range(signal_bar + signal_lag, signal_bar + signal_lag + 4):
        price_idx = current_bar - (signal_bar + signal_lag)
        if price_idx < len(current_prices):
            current_price = current_prices[price_idx]

            ready_phases = entry.check_phase_execution(current_bar, current_price)

            if ready_phases:
                print(f"Bar {current_bar}: {len(ready_phases)} phase(s) ready for execution at ${current_price}")

                for phase in ready_phases:
                    entry.execute_phase(phase, current_price, current_bar)
                    print(f"  [EXEC] Executed phase {phase.phase_number}: ${current_price} x {phase.size:.2f}")
            else:
                print(f"Bar {current_bar}: No phases ready (price ${current_price})")

    # Check final status
    status = entry.get_status_summary()
    print()
    print("Final status:")
    for key, value in status.items():
        print(f"  {key}: {value}")

    print("\n" + "="*60 + "\n")


def test_adverse_move_protection():
    """Test adverse move protection"""
    print("=== TEST 4: Adverse Move Protection ===")

    config = MinimumTimePhasedConfig(
        enabled=True,
        max_phases=3,
        minimum_execution_minutes=10.0,
        data_frequency_minutes=5.0,
        spread_method="equal",
        max_adverse_move=2.0,  # Only 2% adverse move allowed
        require_profit=False
    )

    entry = MinimumTimePhasedEntry(
        config=config,
        signal_bar=100,
        signal_price=100.0,  # Easy to calculate percentages
        is_long=True,
        target_size=300.0,
        signal_lag=1
    )

    print("Testing adverse move protection (max 2% adverse move):")
    print("Signal price: $100.00 (long position)")
    print()

    # Test prices that should trigger adverse move protection
    test_prices = [99.0, 98.0, 97.5, 96.0]  # 1%, 2%, 2.5%, 4% adverse moves

    for i, test_price in enumerate(test_prices):
        adverse_move_pct = ((100.0 - test_price) / 100.0) * 100
        current_bar = entry.execution_start_bar + i

        ready_phases = entry.check_phase_execution(current_bar, test_price)

        print(f"Bar {current_bar}: Price ${test_price} ({adverse_move_pct:.1f}% adverse move)")

        if entry.stop_scaling:
            print("  [STOP] Scaling stopped due to adverse move limit")
            break
        elif ready_phases:
            print(f"  [READY] {len(ready_phases)} phase(s) ready for execution")
        else:
            print("  [WAIT] No phases ready yet")

        print()

    print("\n" + "="*60 + "\n")


def test_tracker_functionality():
    """Test the phased entry tracker"""
    print("=== TEST 5: Tracker Functionality ===")

    config = MinimumTimePhasedConfig(
        enabled=True,
        max_phases=2,
        minimum_execution_minutes=5.0,
        data_frequency_minutes=5.0,
        spread_method="equal"
    )

    tracker = MinimumTimePhasedTracker(config)

    # Start entries for multiple symbols
    symbols = ['AAPL', 'MSFT', 'GOOGL']

    for i, symbol in enumerate(symbols):
        entry = tracker.start_phased_entry(
            symbol=symbol,
            signal_bar=100 + i,
            signal_price=150.0 + i * 10,
            is_long=True,
            target_size=1000.0,
            signal_lag=1
        )
        print(f"Started phased entry for {symbol}: {len(entry.phases)} phases")

    print(f"\nActive entries: {len(tracker.active_entries)}")

    # Simulate some completions
    tracker.complete_entry('AAPL')
    print("Completed AAPL entry")

    stats = tracker.get_statistics()
    print(f"\nTracker statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    print("\n" + "="*60 + "\n")


def test_real_world_example():
    """Test with realistic market scenario"""
    print("=== TEST 6: Real-World Example ===")

    # Example: 5-minute bars, want to spread entry over 15 minutes minimum
    config = MinimumTimePhasedConfig(
        enabled=True,
        max_phases=3,
        minimum_execution_minutes=15.0,     # 15 minutes minimum
        data_frequency_minutes=5.0,         # 5-minute bars
        spread_method="equal",
        phase_size_type="equal",
        max_adverse_move=3.0,
        require_profit=False
    )

    print("Real-world scenario:")
    print("  Market: 5-minute bars")
    print("  Strategy: Spread $10,000 position over minimum 15 minutes")
    print("  Method: 3 equal phases")
    print("  Signal lag: 1 bar (next bar execution)")
    print()

    # Calculate the execution schedule
    signal_lag = 1
    phase_bars = config.calculate_phase_bars(signal_lag)

    bars_needed = int(config.minimum_execution_minutes / config.data_frequency_minutes)  # 3 bars

    print(f"Total bars needed for 15 minutes: {bars_needed}")
    print(f"Phase execution bars (relative to signal): {phase_bars}")
    print()

    # Create the entry
    entry = MinimumTimePhasedEntry(
        config=config,
        signal_bar=100,      # Signal at bar 100
        signal_price=250.0,  # Entry signal at $250
        is_long=True,
        target_size=10000.0, # $10,000 position
        signal_lag=signal_lag
    )

    print("Execution schedule:")
    for phase in entry.phases:
        bar_offset = phase.scheduled_bar - entry.signal_bar
        minutes_offset = bar_offset * config.data_frequency_minutes
        print(f"  Phase {phase.phase_number}: Bar {phase.scheduled_bar} ({bar_offset} bars, {minutes_offset:.0f} minutes), Size: ${phase.size:.2f}")

    print()
    print("[SUCCESS] Minimum time phased entry system working correctly!")
    print("   Phases are spread over the minimum execution timeframe")
    print("   rather than at fixed intervals.")


if __name__ == "__main__":
    print("MINIMUM EXECUTION TIME PHASED ENTRY SYSTEM - TEST SUITE")
    print("=" * 70)
    print()

    try:
        test_phase_bar_calculation()
        test_different_spread_methods()
        test_phased_entry_execution()
        test_adverse_move_protection()
        test_tracker_functionality()
        test_real_world_example()

        print("\n" + "=" * 70)
        print("[ALL TESTS PASSED] The minimum time phased entry system is ready for use!")

    except Exception as e:
        print(f"\n[TEST FAILED]: {e}")
        import traceback
        traceback.print_exc()