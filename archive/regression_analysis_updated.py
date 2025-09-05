import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')

print("Loading data with new format...")
df = pd.read_csv('DTSmlDATA7x7.csv')

# Check if Date and Time columns exist
if 'Date' in df.columns and 'Time' in df.columns:
    # Combine Date and Time into a single datetime column
    df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
    print(f"Date range: {df['DateTime'].min()} to {df['DateTime'].max()}")

# New column definitions based on updated format
feature_cols = ['Ret_0-1hr', 'Ret_1-3hr', 'Ret_3-6hr', 'Ret_6-12hr', 
                'Ret_12-24hr', 'Ret_1-2d', 'Ret_2-4d', 'Ret_4-8d']
target_cols = ['Ret_fwd1hr', 'Ret_fwd3hr', 'Ret_fwd6hr', 'Ret_fwd12hr', 'Ret_fwd1d']

print(f"\nFeature columns (past returns): {feature_cols}")
print(f"Target columns (forward returns): {target_cols}")

# Select numeric columns - zeros may be valid returns now
df_numeric = df[feature_cols + target_cols]

# Remove rows with all NaN
df_clean = df_numeric.dropna()

print(f"Data shape after removing NaN values: {df_clean.shape}")
print(f"Number of samples: {len(df_clean)}")

print("\nPerforming regression analysis...")
r2_matrix = np.zeros((len(feature_cols), len(target_cols)))
coef_matrix = np.zeros((len(feature_cols), len(target_cols)))
pvalue_matrix = np.zeros((len(feature_cols), len(target_cols)))

for i, feature in enumerate(feature_cols):
    for j, target in enumerate(target_cols):
        X = df_clean[[feature]].values
        y = df_clean[target].values
        
        # Check for constant values
        if np.std(X) == 0 or np.std(y) == 0:
            r2_matrix[i, j] = np.nan
            coef_matrix[i, j] = np.nan
            pvalue_matrix[i, j] = np.nan
            continue
        
        model = LinearRegression()
        model.fit(X, y)
        y_pred = model.predict(X)
        
        r2_matrix[i, j] = r2_score(y, y_pred)
        coef_matrix[i, j] = model.coef_[0]
        
        from scipy import stats
        _, _, _, p_value, _ = stats.linregress(X.flatten(), y)
        pvalue_matrix[i, j] = p_value

fig, axes = plt.subplots(2, 2, figsize=(16, 14))

# R² heatmap
ax = axes[0, 0]
sns.heatmap(r2_matrix, 
            annot=True, 
            fmt='.3f',
            xticklabels=[col.replace('Ret_fwd', '') for col in target_cols],
            yticklabels=[col.replace('Ret_', '') for col in feature_cols],
            cmap='RdYlGn',
            center=0,
            cbar_kws={'label': 'R² Score'},
            ax=ax,
            vmin=0,
            vmax=0.1)
ax.set_title('R² Values: Past Returns vs Forward Returns', fontsize=14, fontweight='bold')
ax.set_xlabel('Forward Return Period', fontsize=12)
ax.set_ylabel('Past Return Period', fontsize=12)

# Coefficient heatmap
ax = axes[0, 1]
sns.heatmap(coef_matrix,
            annot=True,
            fmt='.3f',
            xticklabels=[col.replace('Ret_fwd', '') for col in target_cols],
            yticklabels=[col.replace('Ret_', '') for col in feature_cols],
            cmap='coolwarm',
            center=0,
            cbar_kws={'label': 'Regression Coefficient'},
            ax=ax)
ax.set_title('Regression Coefficients', fontsize=14, fontweight='bold')
ax.set_xlabel('Forward Return Period', fontsize=12)
ax.set_ylabel('Past Return Period', fontsize=12)

# Correlation heatmap
ax = axes[1, 0]
correlation_matrix = df_clean[feature_cols + target_cols].corr().loc[feature_cols, target_cols]
sns.heatmap(correlation_matrix,
            annot=True,
            fmt='.3f',
            cmap='coolwarm',
            center=0,
            cbar_kws={'label': 'Correlation'},
            ax=ax,
            vmin=-0.3,
            vmax=0.3)
ax.set_title('Correlation Matrix', fontsize=14, fontweight='bold')
ax.set_xlabel('Forward Return Period', fontsize=12)
ax.set_ylabel('Past Return Period', fontsize=12)

# Significance heatmap
ax = axes[1, 1]
significance_matrix = np.where(pvalue_matrix < 0.001, 3,
                      np.where(pvalue_matrix < 0.01, 2,
                      np.where(pvalue_matrix < 0.05, 1, 0)))
sns.heatmap(significance_matrix,
            annot=True,
            fmt='d',
            xticklabels=[col.replace('Ret_fwd', '') for col in target_cols],
            yticklabels=[col.replace('Ret_', '') for col in feature_cols],
            cmap='YlOrRd',
            cbar_kws={'label': 'Significance Level'},
            ax=ax,
            vmin=0,
            vmax=3)
ax.set_title('Statistical Significance (0=NS, 1=p<0.05, 2=p<0.01, 3=p<0.001)', fontsize=14, fontweight='bold')
ax.set_xlabel('Forward Return Period', fontsize=12)
ax.set_ylabel('Past Return Period', fontsize=12)

plt.suptitle('Regression Analysis: Past Returns to Forward Returns', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('regression_analysis_matrix_updated.png', dpi=300, bbox_inches='tight')
print("\nMatrix chart saved to 'regression_analysis_matrix_updated.png'")

print("\n" + "="*80)
print("REGRESSION ANALYSIS SUMMARY - NEW FORMAT")
print("="*80)

print("\nTop 10 Feature-Target Pairs by R² Score:")
print("-"*50)
r2_flat = []
for i, feature in enumerate(feature_cols):
    for j, target in enumerate(target_cols):
        if not np.isnan(r2_matrix[i, j]):
            r2_flat.append((feature, target, r2_matrix[i, j], coef_matrix[i, j], correlation_matrix.loc[feature, target]))

r2_flat.sort(key=lambda x: x[2], reverse=True)
print(f"{'Feature':<15} {'Target':<15} {'R²':<8} {'Coef':<10} {'Corr':<8}")
print("-"*50)
for feature, target, r2, coef, corr in r2_flat[:10]:
    print(f"{feature:<15} {target:<15} {r2:<8.4f} {coef:<10.4f} {corr:<8.4f}")

print("\n\nAverage R² by Feature (Past Returns):")
print("-"*40)
for i, feature in enumerate(feature_cols):
    avg_r2 = np.nanmean(r2_matrix[i, :])
    print(f"{feature}: {avg_r2:.4f}")

print("\n\nAverage R² by Target (Forward Returns):")
print("-"*40)
for j, target in enumerate(target_cols):
    avg_r2 = np.nanmean(r2_matrix[:, j])
    print(f"{target}: {avg_r2:.4f}")

overall_avg_r2 = np.nanmean(r2_matrix)
print(f"\n\nOverall Average R²: {overall_avg_r2:.4f}")

max_r2 = np.nanmax(r2_matrix)
max_idx = np.unravel_index(np.nanargmax(r2_matrix), r2_matrix.shape)
print(f"Maximum R²: {max_r2:.4f} ({feature_cols[max_idx[0]]} -> {target_cols[max_idx[1]]})")

print("\n\nStatistical Significance Summary:")
print("-"*30)
total_tests = np.sum(~np.isnan(r2_matrix))
sig_001 = np.sum(pvalue_matrix < 0.001)
sig_01 = np.sum(pvalue_matrix < 0.01)
sig_05 = np.sum(pvalue_matrix < 0.05)
print(f"p < 0.001: {sig_001}/{total_tests} ({100*sig_001/total_tests:.1f}%)")
print(f"p < 0.01:  {sig_01}/{total_tests} ({100*sig_01/total_tests:.1f}%)")
print(f"p < 0.05:  {sig_05}/{total_tests} ({100*sig_05/total_tests:.1f}%)")

# Check for suspiciously high R² values
high_r2_threshold = 0.3  # In financial markets, R² > 0.3 for return prediction is very rare
suspicious_pairs = [(feature_cols[i], target_cols[j], r2_matrix[i, j]) 
                   for i in range(len(feature_cols)) 
                   for j in range(len(target_cols)) 
                   if r2_matrix[i, j] > high_r2_threshold]

if suspicious_pairs:
    print("\n\n" + "!"*60)
    print("WARNING: Suspiciously high R² values detected!")
    print("!"*60)
    print("The following relationships have R² > 0.3, which is unusual for financial return prediction:")
    for feat, targ, r2 in suspicious_pairs:
        print(f"  {feat} -> {targ}: R² = {r2:.4f}")
    print("\nPlease verify the data integrity and check for:")
    print("- Look-ahead bias")
    print("- Data leakage")
    print("- Calculation errors in forward returns")
else:
    print("\n\nAll R² values are within expected range for financial data (< 0.3)")

results_df = pd.DataFrame(r2_flat, columns=['Feature', 'Target', 'R2_Score', 'Coefficient', 'Correlation'])
results_df.to_csv('regression_analysis_results_updated.csv', index=False)
print("\n\nDetailed results saved to 'regression_analysis_results_updated.csv'")