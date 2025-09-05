"""
Test script to verify regression filtering calculations
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from scipy import stats

# Load test data
df = pd.read_csv('DTSmlDATA7x7.csv')
print("="*60)
print("REGRESSION FILTERING TEST")
print("="*60)

# Select test variables
x_var = 'Ret_3-6hr'
y_var = 'Ret_fwd6hr'
threshold = 0.1

# Clean data
x_data = pd.to_numeric(df[x_var], errors='coerce')
y_data = pd.to_numeric(df[y_var], errors='coerce')
valid_mask = ~(x_data.isna() | y_data.isna())

# Test 1: ALL data
x_all = x_data[valid_mask].values.reshape(-1, 1)
y_all = y_data[valid_mask].values
print(f"\n1. ALL DATA:")
print(f"   Observations: {len(x_all)}")

if len(x_all) > 1:
    model = LinearRegression()
    model.fit(x_all, y_all)
    r2_all = r2_score(y_all, model.predict(x_all))
    coef_all = model.coef_[0]
    corr_all = np.corrcoef(x_all.flatten(), y_all)[0, 1]
    _, _, _, p_all, _ = stats.linregress(x_all.flatten(), y_all)
    
    print(f"   R²: {r2_all:.4f}")
    print(f"   Coefficient: {coef_all:.4f}")
    print(f"   Correlation: {corr_all:.4f}")
    print(f"   P-value: {p_all:.6f}")

# Test 2: UP filter (y > threshold)
up_mask = valid_mask & (y_data > threshold)
x_up = x_data[up_mask].values.reshape(-1, 1)
y_up = y_data[up_mask].values
print(f"\n2. UP FILTER (y > {threshold}):")
print(f"   Observations: {len(x_up)} ({len(x_up)/len(x_all)*100:.1f}% of ALL)")

if len(x_up) > 1:
    model = LinearRegression()
    model.fit(x_up, y_up)
    r2_up = r2_score(y_up, model.predict(x_up))
    coef_up = model.coef_[0]
    corr_up = np.corrcoef(x_up.flatten(), y_up)[0, 1]
    _, _, _, p_up, _ = stats.linregress(x_up.flatten(), y_up)
    
    print(f"   R²: {r2_up:.4f}")
    print(f"   Coefficient: {coef_up:.4f}")
    print(f"   Correlation: {corr_up:.4f}")
    print(f"   P-value: {p_up:.6f}")
    
    # Check target values
    print(f"   Target mean: {y_up.mean():.4f}")
    print(f"   Target min: {y_up.min():.4f}")
    print(f"   Target max: {y_up.max():.4f}")

# Test 3: DOWN filter (y < -threshold)
down_mask = valid_mask & (y_data < -threshold)
x_down = x_data[down_mask].values.reshape(-1, 1)
y_down = y_data[down_mask].values
print(f"\n3. DOWN FILTER (y < -{threshold}):")
print(f"   Observations: {len(x_down)} ({len(x_down)/len(x_all)*100:.1f}% of ALL)")

if len(x_down) > 1:
    model = LinearRegression()
    model.fit(x_down, y_down)
    r2_down = r2_score(y_down, model.predict(x_down))
    coef_down = model.coef_[0]
    corr_down = np.corrcoef(x_down.flatten(), y_down)[0, 1]
    _, _, _, p_down, _ = stats.linregress(x_down.flatten(), y_down)
    
    print(f"   R²: {r2_down:.4f}")
    print(f"   Coefficient: {coef_down:.4f}")
    print(f"   Correlation: {corr_down:.4f}")
    print(f"   P-value: {p_down:.6f}")
    
    # Check target values
    print(f"   Target mean: {y_down.mean():.4f}")
    print(f"   Target min: {y_down.min():.4f}")
    print(f"   Target max: {y_down.max():.4f}")

# Verify filtering is working correctly
print(f"\n4. FILTER VERIFICATION:")
print(f"   Total valid observations: {len(x_all)}")
print(f"   UP filtered: {len(x_up)}")
print(f"   DOWN filtered: {len(x_down)}")
print(f"   UP + DOWN: {len(x_up) + len(x_down)}")
print(f"   Middle (excluded): {len(x_all) - len(x_up) - len(x_down)}")

# Test P-value significance levels
print(f"\n5. P-VALUE SIGNIFICANCE CLASSIFICATION:")
test_p_values = [0.0001, 0.005, 0.02, 0.08, 0.5]
for p in test_p_values:
    if p < 0.001:
        sig = "*** (p < 0.001)"
    elif p < 0.01:
        sig = "** (p < 0.01)"
    elif p < 0.05:
        sig = "* (p < 0.05)"
    else:
        sig = "(not significant)"
    print(f"   p = {p:.4f} -> {sig}")

print("\n" + "="*60)