#!/usr/bin/env python
"""
Fix GUI duplication by removing the duplicate feature selection from left column
"""

# Read the GUI file with UTF-8 encoding
with open('OMtree_gui.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

print(f"Total lines: {len(lines)}")

# Find lines containing the duplicate feature selection
found_start = -1
found_end = -1

for i in range(len(lines)):
    # Find the start of the duplicate block
    if i >= 797 and i <= 800 and 'Random Forest MDI Feature Selection Settings' in lines[i]:
        found_start = i
        print(f"Found duplicate start at line {i+1}: {lines[i].strip()}")
    
    # Find where the duplicate ends (before "Validation Parameters")
    if found_start > 0 and i >= 910 and i <= 915 and 'Validation Parameters' in lines[i]:
        found_end = i
        print(f"Found duplicate end before line {i+1}: {lines[i].strip()}")
        break

if found_start > 0 and found_end > 0:
    # Create new lines array
    new_lines = []
    
    # Copy everything before the duplicate
    new_lines.extend(lines[:found_start])
    
    # Add a comment about the move
    new_lines.append('        # Feature Selection has been moved to the middle column\n')
    new_lines.append('        \n')
    
    # Copy everything after the duplicate
    new_lines.extend(lines[found_end:])
    
    # Write back the fixed file
    with open('OMtree_gui.py', 'w', encoding='utf-8') as f:
        f.writelines(new_lines)
    
    print(f"Fixed GUI duplication - removed lines {found_start+1} to {found_end}")
    print(f"New file has {len(new_lines)} lines (removed {len(lines) - len(new_lines)} lines)")
else:
    print("Could not find the duplicate feature selection block")
    print("Searching for alternative markers...")
    
    # Search more broadly
    for i in range(790, min(920, len(lines))):
        if 'feature_selection_frame = ttk.LabelFrame(config_scrollable' in lines[i]:
            print(f"Found feature selection frame at line {i+1}")
        if 'toggle_feature_selection()' in lines[i] and 'Initialize state' in lines[i]:
            print(f"Found end of feature selection at line {i+1}")