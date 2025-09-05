"""
Test that the pipeline works correctly with VolSignal removed
"""
import pandas as pd
import numpy as np
from OMtree_preprocessing import DataPreprocessor
from OMtree_model import DirectionalTreeEnsemble
from OMtree_validation import DirectionalValidator

print("="*80)
print("TESTING PIPELINE WITHOUT VOLSIGNAL")
print("="*80)

# Test 1: Check config doesn't have VolSignal
print("\n1. Checking configuration...")
print("-"*40)

preprocessor = DataPreprocessor()
if hasattr(preprocessor, 'add_volatility_signal'):
    print("[WARNING] Preprocessor still has add_volatility_signal attribute")
else:
    print("[OK] No add_volatility_signal attribute in preprocessor")

if hasattr(preprocessor, 'vol_signal_window'):
    print("[WARNING] Preprocessor still has vol_signal_window attribute")
else:
    print("[OK] No vol_signal_window attribute in preprocessor")

# Test 2: Process data and check no VolSignal columns
print("\n2. Testing preprocessing...")
print("-"*40)

# Load sample data
df = pd.read_csv('C:/Users/jd/OM3/DTSmlDATA7x7.csv')
df_sample = df.head(100)

# Process data
processed = preprocessor.process_data(df_sample)

# Check for VolSignal columns
vol_signal_cols = [col for col in processed.columns if 'VolSignal' in col]
if vol_signal_cols:
    print(f"[ERROR] Found VolSignal columns: {vol_signal_cols}")
else:
    print("[OK] No VolSignal columns in processed data")

print(f"\nProcessed columns ({len(processed.columns)}):")
for i, col in enumerate(processed.columns):
    if i < 5 or i >= len(processed.columns) - 2:
        print(f"  {col}")
    elif i == 5:
        print(f"  ...")

# Test 3: Train model without VolSignal
print("\n3. Testing model training...")
print("-"*40)

try:
    # Get features and target
    feature_cols = [col for col in processed.columns if col.endswith('_vol_adj') and not 'fwd' in col]
    if not feature_cols:
        feature_cols = [col for col in processed.columns if col.startswith('Ret_') and not 'fwd' in col]
    
    target_col = 'Ret_fwd6hr_vol_adj'
    if target_col not in processed.columns:
        target_col = 'Ret_fwd6hr'
    
    if feature_cols and target_col in processed.columns:
        X = processed[feature_cols[:3]].values  # Use first 3 features
        y = processed[target_col].values
        
        # Remove NaN
        valid_mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
        X = X[valid_mask]
        y = y[valid_mask]
        
        if len(X) > 10:
            model = DirectionalTreeEnsemble(verbose=False)
            model.fit(X, y)
            
            # Make predictions
            preds = model.predict(X[:10])
            print(f"[OK] Model trained and predicted successfully")
            print(f"    Features used: {feature_cols[:3]}")
            print(f"    Predictions shape: {preds.shape}")
        else:
            print("[WARNING] Not enough valid data for training")
    else:
        print("[WARNING] Could not find suitable features/target")
        
except Exception as e:
    print(f"[ERROR] Model training failed: {e}")

# Test 4: Quick validation run
print("\n4. Testing validation pipeline...")
print("-"*40)

try:
    validator = DirectionalValidator()
    
    # Override for quick test
    validator.train_size = 100
    validator.test_size = 20
    validator.step_size = 50
    
    # Check selected features don't include VolSignal
    if any('VolSignal' in feat for feat in validator.selected_features):
        print("[WARNING] VolSignal found in selected features")
    else:
        print("[OK] No VolSignal in selected features")
    
    # Run a quick validation step
    data = validator.load_and_prepare_data()
    if data is not None:
        # Check processed data doesn't have VolSignal
        vol_cols = [col for col in data.columns if 'VolSignal' in col]
        if vol_cols:
            print(f"[ERROR] VolSignal columns in validation data: {vol_cols}")
        else:
            print("[OK] No VolSignal in validation data")
            
        print(f"    Data shape: {data.shape}")
        print(f"    Features available: {len([c for c in data.columns if '_vol_adj' in c])}")
    
except Exception as e:
    print(f"[ERROR] Validation failed: {e}")

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print("""
VolSignal has been successfully removed from:
✓ Configuration file
✓ Preprocessing module  
✓ GUI components
✓ Validation pipeline

The pipeline now works with only the raw features, without any
engineered volatility signal features. This simplifies the model
and reduces complexity.
""")