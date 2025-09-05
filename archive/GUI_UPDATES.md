# GUI Updates - Constrained Value Selection

## What's New

The OMtree GUI has been updated with improved input controls for better user experience and data validation:

### 1. Dropdown Menus (Comboboxes)
Fields with limited valid options now use dropdown menus:

- **Model Type**: Select between `longonly` or `shortonly`
- **Normalize Features/Target**: Choose `true` or `false`  
- **Smoothing Type**: Pick from `exponential`, `linear`, or `none`

### 2. Numeric Spinboxes
Numeric parameters now use spinbox controls with:
- **Minimum and maximum values** to prevent invalid entries
- **Increment buttons** for easy adjustment
- **Decimal precision** where appropriate

#### Examples:
- **Vote Threshold**: Range 0.5-1.0, increments of 0.05
- **Number of Trees**: Range 10-1000, increments of 10
- **Volatility Window**: Range 10-500, increments of 10
- **Min Leaf Fraction**: Range 0.01-0.5, increments of 0.01

### 3. Text Entry Fields
Fields that accept custom values remain as text entries:
- CSV file names
- Feature lists (comma-separated)
- Date fields (YYYY-MM-DD format)

## Benefits

1. **Error Prevention**: Invalid values cannot be entered for constrained fields
2. **Faster Configuration**: Click to select instead of typing
3. **Clear Boundaries**: Users can see valid ranges for numeric parameters
4. **Better UX**: Increment/decrement buttons for fine-tuning

## Parameter Ranges

### Preprocessing
- **Volatility Window**: 10-500 days (step: 10)
- **Smoothing Alpha**: 0.01-1.0 (step: 0.01)

### Model Parameters  
- **Number of Trees**: 10-1000 (step: 10)
- **Max Depth**: 1-10 (step: 1)
- **Bootstrap Fraction**: 0.1-1.0 (step: 0.05)
- **Min Leaf Fraction**: 0.01-0.5 (step: 0.01)
- **Target Threshold**: 0.0-0.5 (step: 0.01)
- **Vote Threshold**: 0.5-1.0 (step: 0.05)

### Validation
- **Training Size**: 100-5000 observations (step: 100)
- **Test Size**: 10-500 observations (step: 10)
- **Step Size**: 1-200 days (step: 10)
- **Base Rate**: 0.3-0.7 (step: 0.01)

## Usage Tips

1. **Spinbox Controls**:
   - Click the up/down arrows to increment/decrement
   - Or type a value directly (will be validated against range)
   - Values outside the range will be automatically corrected

2. **Dropdown Menus**:
   - Click to see all available options
   - Cannot type custom values - must select from list

3. **Keyboard Navigation**:
   - Tab between fields
   - Use arrow keys in dropdowns and spinboxes
   - Enter to confirm selection

## Screenshots Example

Configuration tab now shows:
```
Model Type:        [longonly ▼]      (dropdown)
Number of Trees:   [200 ↑↓]          (spinbox) 
Vote Threshold:    [0.60 ↑↓]         (spinbox with decimals)
CSV File:          [DTSnnData.csv]   (text entry)
```

This makes the interface more intuitive and reduces configuration errors!