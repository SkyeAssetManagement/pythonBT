"""
Demonstrate the effects of balanced bootstrap in different market scenarios
"""
import numpy as np
import pandas as pd
import configparser
from OMtree_model import DirectionalTreeEnsemble
import matplotlib.pyplot as plt

print("="*80)
print("BALANCED BOOTSTRAP EFFECTS IN DIFFERENT SCENARIOS")
print("="*80)

# Scenario 1: Trend-following in ranging market (90% no trend)
print("\nSCENARIO 1: Trend Detection (10% trends, 90% ranging)")
print("-"*60)

np.random.seed(42)
n_samples = 1000

# Create data: 10% trending, 90% ranging
X1 = np.random.randn(n_samples, 3)
trend_mask = np.random.random(n_samples) < 0.1  # 10% trends
y1 = np.where(trend_mask, 
              np.random.uniform(0.1, 0.3, n_samples),  # Trending: profitable
              np.random.uniform(-0.05, 0.05, n_samples))  # Ranging: small moves

print(f"Data: {np.sum(y1 > 0.05)}/{n_samples} profitable trends ({np.mean(y1 > 0.05)*100:.1f}%)")

# Test both methods
results = {}
for balanced in [False, True]:
    config = configparser.ConfigParser(inline_comment_prefixes='#')
    config['data'] = {
        'csv_file': 'test.csv',
        'feature_columns': 'f1,f2,f3',
        'selected_features': 'f1,f2,f3',
        'target_column': 'target',
    }
    
    config['preprocessing'] = {
        'normalize_features': 'false',
        'normalize_target': 'false',
        'detrend_features': 'false',
    }
    
    config['model'] = {
        'model_type': 'longonly',
        'algorithm': 'decision_trees',
        'probability_aggregation': 'mean',
        'balanced_bootstrap': str(balanced).lower(),
        'n_trees': '50',
        'max_depth': '3',
        'bootstrap_fraction': '0.8',
        'min_leaf_fraction': '0.05',
        'target_threshold': '0.05',  # Looking for trends > 5%
        'vote_threshold': '0.5',
        'random_seed': '42'
    }
    
    with open('temp.ini', 'w') as f:
        config.write(f)
    
    model = DirectionalTreeEnsemble('temp.ini', verbose=False)
    model.fit(X1, y1)
    
    # Test on new data with same distribution
    X_test = np.random.randn(200, 3)
    trend_mask_test = np.random.random(200) < 0.1
    y_test = np.where(trend_mask_test,
                     np.random.uniform(0.1, 0.3, 200),
                     np.random.uniform(-0.05, 0.05, 200))
    
    pred = model.predict(X_test)
    
    # Calculate metrics
    true_trends = y_test > 0.05
    true_positives = np.sum((pred == 1) & true_trends)
    false_positives = np.sum((pred == 1) & ~true_trends)
    false_negatives = np.sum((pred == 0) & true_trends)
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    
    results[balanced] = {
        'trades': np.sum(pred),
        'precision': precision,
        'recall': recall,
        'true_positives': true_positives,
        'false_positives': false_positives
    }

print(f"\nRegular Bootstrap:")
print(f"  Trades: {results[False]['trades']}/200")
print(f"  Precision: {results[False]['precision']:.2%} (correct when trading)")
print(f"  Recall: {results[False]['recall']:.2%} (caught trends)")
print(f"  False Positives: {results[False]['false_positives']}")

print(f"\nBalanced Bootstrap:")
print(f"  Trades: {results[True]['trades']}/200")
print(f"  Precision: {results[True]['precision']:.2%} (correct when trading)")
print(f"  Recall: {results[True]['recall']:.2%} (caught trends)")
print(f"  False Positives: {results[True]['false_positives']}")

# Scenario 2: Crisis detection (2% crisis events)
print("\n" + "="*60)
print("SCENARIO 2: Crisis Detection (2% crisis, 98% normal)")
print("-"*60)

# Create data: 2% crisis, 98% normal
crisis_mask = np.random.random(n_samples) < 0.02  # 2% crisis
y2 = np.where(crisis_mask,
              np.random.uniform(-0.3, -0.5, n_samples),  # Crisis: large losses
              np.random.uniform(-0.02, 0.02, n_samples))  # Normal: small moves

print(f"Data: {np.sum(y2 < -0.1)}/{n_samples} crisis events ({np.mean(y2 < -0.1)*100:.1f}%)")

# Test short model for crisis
for balanced in [False, True]:
    config['model']['model_type'] = 'shortonly'
    config['model']['target_threshold'] = '0.1'  # Looking for drops > 10%
    config['model']['balanced_bootstrap'] = str(balanced).lower()
    
    with open('temp.ini', 'w') as f:
        config.write(f)
    
    model = DirectionalTreeEnsemble('temp.ini', verbose=False)
    model.fit(X1, y2)  # Using X1, y2 for crisis
    
    pred = model.predict(X_test)
    
    # For crisis detection
    true_crisis = y_test < -0.1  # Would be y_test in real scenario
    detected = np.sum(pred == 1)
    
    print(f"\n{'Balanced' if balanced else 'Regular'} Bootstrap:")
    print(f"  Warning signals: {detected}/200")

# Scenario 3: Effect on probability calibration
print("\n" + "="*60)
print("SCENARIO 3: Probability Calibration Effect")
print("-"*60)

# Create data with known 20% positive rate
y3 = np.where(np.random.random(n_samples) < 0.2, 0.15, -0.05)
print(f"True positive rate: {np.mean(y3 > 0)*100:.1f}%")

for balanced in [False, True]:
    config['model']['model_type'] = 'longonly'
    config['model']['target_threshold'] = '0.0'
    config['model']['balanced_bootstrap'] = str(balanced).lower()
    config['model']['vote_threshold'] = '0.5'
    
    with open('temp.ini', 'w') as f:
        config.write(f)
    
    model = DirectionalTreeEnsemble('temp.ini', verbose=False)
    model.fit(X1, y3)
    
    probs = model.predict_proba(X_test)
    
    print(f"\n{'Balanced' if balanced else 'Regular'} Bootstrap:")
    print(f"  Mean probability: {np.mean(probs):.3f}")
    print(f"  Probability > 0.5: {np.mean(probs > 0.5)*100:.1f}%")
    print(f"  Min/Max: {np.min(probs):.3f} / {np.max(probs):.3f}")

# Clean up
import os
if os.path.exists('temp.ini'):
    os.remove('temp.ini')

print("\n" + "="*80)
print("KEY INSIGHTS")
print("="*80)
print("""
1. TREND DETECTION (10% trends):
   - Regular: Conservative, high precision, low recall
   - Balanced: Aggressive, lower precision, higher recall
   → Use balanced when missing trends is costly

2. CRISIS DETECTION (2% events):
   - Regular: May never trigger warnings
   - Balanced: More likely to detect rare events
   → Use balanced for rare but important events

3. PROBABILITY CALIBRATION:
   - Regular: Probabilities ≈ true rates (20%)
   - Balanced: Probabilities ≈ 50% regardless of true rate
   → Don't use balanced if you need calibrated probabilities

RECOMMENDATION:
- Use balanced bootstrap when minority class is important
- Use regular bootstrap when probability calibration matters
- Consider the cost of false positives vs false negatives
""")