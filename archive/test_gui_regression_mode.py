"""Quick test to verify regression_mode is in the GUI"""
import configparser

# Check if the config has regression_mode
config = configparser.ConfigParser(inline_comment_prefixes='#')
config.read('OMtree_config.ini')

if 'model' in config and 'regression_mode' in config['model']:
    print(f"[OK] regression_mode found in config: {config['model']['regression_mode']}")
else:
    print("[ERROR] regression_mode NOT found in config")

# Check if GUI code has the parameter
with open('OMtree_gui_v3.py', 'r', encoding='utf-8') as f:
    gui_code = f.read()
    
if "'model.regression_mode'" in gui_code:
    print("[OK] regression_mode found in GUI code")
    
    # Count how many times it appears
    count = gui_code.count("'model.regression_mode'")
    print(f"  Appears {count} time(s) in GUI code")
    
    # Check for the description
    if "'Use regressors (continuous targets)'" in gui_code:
        print("[OK] Description for regression_mode found")
else:
    print("[ERROR] regression_mode NOT found in GUI code")

print("\nGUI should now have:")
print("1. Regression Mode dropdown in Model Parameters section")
print("2. Options: 'false' (classifiers) or 'true' (regressors)")
print("3. Description that updates based on selection")
print("\nTo test: Run 'python OMtree_gui_v3.py' and check Model Parameters section")