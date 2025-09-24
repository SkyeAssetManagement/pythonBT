# Debug Mode Usage for PyQtGraph Range Bars

## Overview
The PyQtGraph Range Bars system now includes a debug verbosity control to reduce terminal output in production mode.

## Features Implemented

### 1. ATR Multiplier Display
- The ATR multiplier in the data window hover display now shows 2 decimal places (e.g., 1.93 instead of 1.9)
- Format: `Range: {atr} x {multiplier:.2f} = {range_value}`

### 2. Trade P&L Display
- Trade profit/loss is now displayed as percentage to match the trade list panel
- Shows P&L % when available, falls back to dollar amount if percentage not provided
- Format: `P&L: +1.25%` or `P&L: -0.50%`

### 3. Commission and Slippage Display
- Commission and slippage are now displayed in the hover window when available
- Shows on a new line under the trade information
- Format: `Commission: $0.50 | Slippage: $0.25`

### 4. Debug Verbosity Control

#### Production Mode (Default)
By default, the system runs in production mode with minimal console output.

#### Debug/Verbose Mode
To enable verbose debug output, set the environment variable before running:

**Windows (Command Prompt):**
```cmd
set DEBUG_VERBOSE=TRUE
python your_script.py
```

**Windows (PowerShell):**
```powershell
$env:DEBUG_VERBOSE="TRUE"
python your_script.py
```

**Linux/Mac:**
```bash
export DEBUG_VERBOSE=TRUE
python your_script.py
```

#### What Debug Mode Shows
When `DEBUG_VERBOSE=TRUE`, you'll see:
- Data loading progress and statistics
- Time axis formatting details
- Trade loading information
- Indicator rendering details
- Warning messages
- Feature list on startup

#### Disable Debug Mode
To return to production mode:

**Windows (Command Prompt):**
```cmd
set DEBUG_VERBOSE=FALSE
```

**Windows (PowerShell):**
```powershell
$env:DEBUG_VERBOSE="FALSE"
```

**Linux/Mac:**
```bash
export DEBUG_VERBOSE=FALSE
```

Or simply unset the variable:
```bash
unset DEBUG_VERBOSE
```

## Summary of Changes

1. **pyqtgraph_range_bars_final.py**:
   - Added debug_verbose flag controlled by environment variable
   - Wrapped all print statements with debug verbosity check
   - Updated hover display to show ATR multiplier with 2 decimal places
   - Added P&L percentage display (matching trade list)
   - Added commission and slippage display when available

These changes ensure:
- Clean terminal output in production
- Professional data display matching trade list format
- Complete trade information including costs
- Easy toggle between debug and production modes