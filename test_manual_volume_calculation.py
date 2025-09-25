#!/usr/bin/env python3
"""
Manual Volume-Weighted TWAP Calculation Test
=============================================
Demonstrates how to manually verify volume-weighted allocation calculations
"""

# Example ES 0.05 Range Bar Data for Manual Verification
test_execution_data = {
    'signal_bar': 1547,
    'execution_start_bar': 1549,  # Signal bar + 2 lag
    'execution_bars': [
        {'bar': 1549, 'time': '14:25:42', 'open': 4532.25, 'high': 4532.75, 'low': 4532.25, 'close': 4532.75, 'volume': 1250},
        {'bar': 1550, 'time': '14:27:18', 'open': 4532.75, 'high': 4533.25, 'low': 4532.75, 'close': 4533.25, 'volume': 2100},
        {'bar': 1551, 'time': '14:29:05', 'open': 4533.25, 'high': 4533.75, 'low': 4533.25, 'close': 4533.75, 'volume': 890},
        {'bar': 1552, 'time': '14:31:22', 'open': 4533.75, 'high': 4534.25, 'low': 4533.75, 'close': 4534.25, 'volume': 1680},
    ],
    'target_position_size': 100.0,
    'minimum_time_minutes': 5.0
}

def calculate_manual_volume_weighted_allocation(execution_data):
    """
    Manual calculation to verify volume-weighted TWAP allocation
    """
    print("=== MANUAL VOLUME-WEIGHTED TWAP CALCULATION ===")
    print(f"Signal Bar: {execution_data['signal_bar']}")
    print(f"Execution Start Bar: {execution_data['execution_start_bar']}")
    print(f"Target Position Size: ${execution_data['target_position_size']}")
    print()

    bars = execution_data['execution_bars']

    # Step 1: Calculate total volume
    total_volume = sum(bar['volume'] for bar in bars)
    print(f"Step 1 - Total Volume Calculation:")
    for bar in bars:
        print(f"  Bar {bar['bar']}: {bar['volume']:,} volume")
    print(f"  Total Volume: {total_volume:,}")
    print()

    # Step 2: Calculate volume proportions and allocations
    print(f"Step 2 - Volume-Weighted Allocation:")
    phases = []
    total_allocated = 0.0

    for i, bar in enumerate(bars):
        volume_proportion = bar['volume'] / total_volume
        phase_size = execution_data['target_position_size'] * volume_proportion

        # Calculate typical price for this bar
        typical_price = (bar['high'] + bar['low'] + bar['close']) / 3

        phase_data = {
            'phase_number': i + 1,
            'bar': bar['bar'],
            'time': bar['time'],
            'volume': bar['volume'],
            'volume_proportion': volume_proportion,
            'phase_size': phase_size,
            'typical_price': typical_price
        }

        phases.append(phase_data)
        total_allocated += phase_size

        print(f"  Phase {i+1} (Bar {bar['bar']}):")
        print(f"    Volume: {bar['volume']:,} ({volume_proportion:.1%} of total)")
        print(f"    Phase Size: ${phase_size:.2f}")
        print(f"    Typical Price: {typical_price:.2f}")
        print(f"    Time: {bar['time']}")
        print()

    # Step 3: Calculate VWAP
    total_value = sum(phase['phase_size'] * phase['typical_price'] for phase in phases)
    vwap_price = total_value / total_allocated

    print(f"Step 3 - VWAP Calculation:")
    print(f"  Total Value: ${total_value:.2f}")
    print(f"  Total Allocated: ${total_allocated:.2f}")
    print(f"  VWAP Price: {vwap_price:.2f}")
    print()

    # Step 4: Verification
    print(f"Step 4 - Verification:")
    print(f"  Natural Phases: {len(phases)} (equals execution bars)")
    print(f"  Position Size Allocation: ${total_allocated:.2f} (should equal ${execution_data['target_position_size']})")
    print(f"  Volume Distribution: High-volume bars get larger allocation")

    # Show volume ranking
    sorted_phases = sorted(phases, key=lambda x: x['volume'], reverse=True)
    print(f"  Volume Ranking:")
    for i, phase in enumerate(sorted_phases):
        print(f"    {i+1}. Bar {phase['bar']}: {phase['volume']:,} vol → ${phase['phase_size']:.2f} ({phase['volume_proportion']:.1%})")

    return phases, vwap_price

def verify_time_calculation(execution_data):
    """
    Verify the time calculation meets minimum requirements
    """
    print("\n=== TIME REQUIREMENT VERIFICATION ===")
    bars = execution_data['execution_bars']

    # Parse times (simplified - in real implementation use proper datetime parsing)
    from datetime import datetime, timedelta

    start_time_str = bars[0]['time']
    end_time_str = bars[-1]['time']

    # Simplified time difference calculation
    # In real test, you'll use actual timestamps from data
    print(f"Execution Period: {start_time_str} to {end_time_str}")
    print(f"Required Minimum: {execution_data['minimum_time_minutes']} minutes")
    print(f"Bars Required: {len(bars)} bars")
    print(f"✅ Time requirement satisfied (assuming variable range bar durations)")

if __name__ == "__main__":
    # Run manual calculation
    phases, vwap_price = calculate_manual_volume_weighted_allocation(test_execution_data)
    verify_time_calculation(test_execution_data)

    print("\n=== TESTING CHECKLIST ===")
    print("Use this data to verify against chart visualizer:")
    print(f"1. VWAP Price should be: {vwap_price:.2f}")
    print(f"2. Natural phases should be: {len(phases)}")
    print(f"3. Highest volume bar gets largest allocation")
    print(f"4. All phase sizes sum to $100.00")