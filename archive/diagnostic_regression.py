import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

print("="*80)
print("DIAGNOSTIC ANALYSIS - CHECKING FOR DATA/REGRESSION ISSUES")
print("="*80)

# Load data
df = pd.read_csv('DTSmlDATA7x7.csv')
print(f"\nData shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")

# Check first few rows
print("\nFirst 5 rows of data:")
print(df.head())

# Select feature and target columns
feature_cols = ['Ret1', 'Ret2', 'Ret3', 'Ret4', 'Ret5', 'Ret6', 'Ret7', 'Ret8', 'Ret9']
target_cols = ['FRet1', 'FRet2', 'FRet3', 'FRet4', 'FRet5', 'FRet6', 'FRet7']

# Check for the suspicious relationship: Ret1 -> FRet6
print("\n" + "="*60)
print("INVESTIGATING SUSPICIOUS RELATIONSHIP: Ret1 -> FRet6")
print("="*60)

# Get clean data
df_clean = df[['Ret1', 'FRet6']].replace(0.0, np.nan).dropna()
print(f"\nClean data points: {len(df_clean)}")

# Check basic statistics
print("\nBasic Statistics:")
print(f"Ret1  - Mean: {df_clean['Ret1'].mean():.4f}, Std: {df_clean['Ret1'].std():.4f}, Min: {df_clean['Ret1'].min():.4f}, Max: {df_clean['Ret1'].max():.4f}")
print(f"FRet6 - Mean: {df_clean['FRet6'].mean():.4f}, Std: {df_clean['FRet6'].std():.4f}, Min: {df_clean['FRet6'].min():.4f}, Max: {df_clean['FRet6'].max():.4f}")

# Check correlation
correlation = df_clean['Ret1'].corr(df_clean['FRet6'])
print(f"\nCorrelation: {correlation:.4f}")

# Check if they're identical or nearly identical
print("\nChecking if columns are identical or nearly identical:")
are_identical = (df_clean['Ret1'] == df_clean['FRet6']).all()
print(f"Are columns exactly identical? {are_identical}")

# Check if one is a linear transformation of the other
diff = df_clean['Ret1'] + df_clean['FRet6']  # Since correlation is negative
print(f"Sum of Ret1 + FRet6 (should be ~0 if FRet6 = -Ret1):")
print(f"  Mean: {diff.mean():.6f}, Std: {diff.std():.6f}")

# Check if there's a constant offset
ratio = df_clean['FRet6'] / df_clean['Ret1']
ratio_clean = ratio[np.isfinite(ratio)]
print(f"\nRatio FRet6/Ret1:")
print(f"  Mean: {ratio_clean.mean():.4f}, Std: {ratio_clean.std():.4f}")

# Sample some actual values
print("\nSample of actual values (first 20 non-zero pairs):")
sample = df_clean[df_clean['Ret1'] != 0].head(20)
for idx, row in sample.iterrows():
    print(f"  Ret1: {row['Ret1']:8.4f}, FRet6: {row['FRet6']:8.4f}, Ratio: {row['FRet6']/row['Ret1']:8.4f}")

# Check uniqueness
print(f"\nNumber of unique values:")
print(f"  Ret1: {df_clean['Ret1'].nunique()} unique values out of {len(df_clean)}")
print(f"  FRet6: {df_clean['FRet6'].nunique()} unique values out of {len(df_clean)}")

# Perform regression
X = df_clean[['Ret1']].values
y = df_clean['FRet6'].values

model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)

r2 = r2_score(y, y_pred)
print(f"\nRegression Results:")
print(f"  R² Score: {r2:.6f}")
print(f"  Coefficient: {model.coef_[0]:.6f}")
print(f"  Intercept: {model.intercept_:.6f}")

# Check residuals
residuals = y - y_pred
print(f"\nResiduals:")
print(f"  Mean: {residuals.mean():.6f}")
print(f"  Std: {residuals.std():.6f}")
print(f"  Min: {residuals.min():.6f}")
print(f"  Max: {residuals.max():.6f}")

# Create scatter plot
plt.figure(figsize=(12, 5))

plt.subplot(1, 3, 1)
plt.scatter(df_clean['Ret1'], df_clean['FRet6'], alpha=0.5, s=1)
plt.plot(X, y_pred, 'r-', linewidth=2)
plt.xlabel('Ret1')
plt.ylabel('FRet6')
plt.title(f'Ret1 vs FRet6\nR-squared = {r2:.4f}')
plt.grid(True, alpha=0.3)

plt.subplot(1, 3, 2)
plt.hist(residuals, bins=50, edgecolor='black')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.title('Distribution of Residuals')
plt.grid(True, alpha=0.3)

plt.subplot(1, 3, 3)
plt.scatter(y_pred, residuals, alpha=0.5, s=1)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residuals vs Predicted')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('diagnostic_plot.png', dpi=150)
print("\nDiagnostic plot saved to 'diagnostic_plot.png'")

# Check other suspicious relationships
print("\n" + "="*60)
print("CHECKING OTHER RELATIONSHIPS")
print("="*60)

# Check all relationships with R² > 0.5
high_r2_pairs = []
for feat in feature_cols[:3]:  # Check first 3 features for speed
    for targ in target_cols:
        df_pair = df[[feat, targ]].replace(0.0, np.nan).dropna()
        if len(df_pair) > 10:
            X_temp = df_pair[[feat]].values
            y_temp = df_pair[targ].values
            model_temp = LinearRegression()
            model_temp.fit(X_temp, y_temp)
            y_pred_temp = model_temp.predict(X_temp)
            r2_temp = r2_score(y_temp, y_pred_temp)
            
            if r2_temp > 0.5:
                high_r2_pairs.append((feat, targ, r2_temp, model_temp.coef_[0]))

print("\nRelationships with R-squared > 0.5:")
for feat, targ, r2, coef in sorted(high_r2_pairs, key=lambda x: x[2], reverse=True):
    print(f"  {feat} -> {targ}: R-squared = {r2:.4f}, Coef = {coef:.4f}")

# Check for data leakage - are forward returns actually backward looking?
print("\n" + "="*60)
print("DATA LEAKAGE CHECK")
print("="*60)

# Check if FRet columns might be mislabeled or contain past data
print("\nChecking temporal alignment...")

# Get a sample where we have consecutive rows
sample_idx = 100
if len(df) > sample_idx + 10:
    print(f"\nSample starting at row {sample_idx}:")
    for i in range(5):
        idx = sample_idx + i
        print(f"\nRow {idx} (Date: {df.iloc[idx]['Date']}):")
        print(f"  Ret1: {df.iloc[idx]['Ret1']:.4f}")
        print(f"  FRet1: {df.iloc[idx]['FRet1']:.4f}")
        print(f"  FRet6: {df.iloc[idx]['FRet6']:.4f}")
        if idx > 0:
            print(f"  Previous row's Ret1: {df.iloc[idx-1]['Ret1']:.4f}")
            print(f"  Similarity to current FRet6: {abs(df.iloc[idx]['FRet6'] + df.iloc[idx-1]['Ret1']):.6f}")

print("\n" + "="*60)
print("CONCLUSION")
print("="*60)
print("""
The extremely high R-squared values (>0.95) between past and forward returns are highly suspicious.
Possible issues to investigate:
1. Data labeling error - forward returns might actually be past returns
2. Look-ahead bias - forward returns calculated using current period data
3. Sign reversal - one column might be the negative of another
4. Data duplication - columns might be copies with minor transformations

The analysis above should help identify which of these issues is present.
""")