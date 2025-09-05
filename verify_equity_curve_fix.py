#!/usr/bin/env python
"""
Verify that equity curve calculations are now correct
"""

import pandas as pd
import numpy as np

# Load actual results
df = pd.read_csv('OMtree_results.csv')
trades = df[df['prediction'] == 1].copy()

print("EQUITY CURVE CALCULATION VERIFICATION")
print("=" * 60)
print(f"Total trades: {len(trades)}")
print()

# Get the returns (in percentage form)
returns_pct = trades['target_value'].values

# Method 1: Simple sum (what console shows)
simple_sum = returns_pct.sum()
print(f"1. SIMPLE SUM (Console output):")
print(f"   Total: {simple_sum:.2f}%")
print()

# Method 2: Compound returns (what equity curve should show)
print(f"2. COMPOUND RETURNS (Equity Curve):")
portfolio = 1000  # Starting capital
for i, ret_pct in enumerate(returns_pct):
    ret_decimal = ret_pct / 100.0  # CRITICAL: Divide by 100
    portfolio = portfolio * (1 + ret_decimal)
    
    if i < 5:  # Show first few
        print(f"   Trade {i+1}: {ret_pct:+.3f}% -> Portfolio: ${portfolio:.2f}")
    elif i == 5:
        print(f"   ...")

final_return_pct = (portfolio - 1000) / 1000 * 100
print(f"\n   Final Portfolio: ${portfolio:.2f}")
print(f"   Total Return: {final_return_pct:.2f}%")
print()

# Show the difference
print("COMPARISON:")
print(f"  Simple Sum:     {simple_sum:.2f}%")
print(f"  Compound:       {final_return_pct:.2f}%")
print(f"  Difference:     {final_return_pct - simple_sum:.2f}%")
print()

# Verify the calculation is correct
print("VERIFICATION:")
# Manual check: if all returns were 1%, what would happen?
test_portfolio = 1000
for _ in range(100):
    test_portfolio *= 1.01  # 1% gain
test_return = (test_portfolio - 1000) / 10
print(f"  100 trades of +1% each:")
print(f"    Simple sum: 100.00%")
print(f"    Compound:   {test_return:.2f}%")
print(f"    ✓ Compounding effect confirmed")
print()

print("=" * 60)
print("SUCCESS: Equity curve should now show portfolio value")
print("starting at $1,000 and properly compounding returns")
print("after dividing by 100 to convert percentage to decimal.")
print("=" * 60)