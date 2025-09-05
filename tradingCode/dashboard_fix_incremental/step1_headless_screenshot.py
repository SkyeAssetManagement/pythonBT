"""
Step 1: Headless screenshot generation using matplotlib
This creates candlestick charts and saves them as PNG files without GUI
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
from pathlib import Path
import time

def load_data():
    """Load ES data or create synthetic data"""
    print("Loading data for screenshot test...")
    
    # Try to load real ES data first
    es_file = Path(__file__).parent.parent.parent / "dataRaw" / "1m" / "ES" / "Current" / "ES-NONE-1m-EST-NoPad.csv"
    
    if es_file.exists():
        print(f"Loading real ES data: {es_file}")
        df = pd.read_csv(es_file)
        
        # Sample first 1000 for testing
        sample_size = min(1000, len(df))
        df_sample = df.head(sample_size)
        
        ohlc_data = []
        for i, row in df_sample.iterrows():
            ohlc_data.append([
                i,
                float(row['Open']),
                float(row['High']),
                float(row['Low']),
                float(row['Close'])
            ])
        
        print(f"Loaded {len(ohlc_data)} real ES bars")
        return ohlc_data, f"ES Real Data ({len(ohlc_data)} bars)"
        
    else:
        print("ES data not found, creating synthetic data...")
        # Create synthetic data
        n_bars = 1000
        np.random.seed(42)
        price = 4000.0
        ohlc_data = []
        
        for i in range(n_bars):
            open_price = price
            price_change = np.random.normal(0, 2.0)
            close_price = price + price_change
            
            high = max(open_price, close_price) + abs(np.random.normal(0, 1.0))
            low = min(open_price, close_price) - abs(np.random.normal(0, 1.0))
            
            ohlc_data.append([i, open_price, high, low, close_price])
            price = close_price
        
        print(f"Created {len(ohlc_data)} synthetic bars")
        return ohlc_data, f"Synthetic Data ({len(ohlc_data)} bars)"

def create_candlestick_chart(ohlc_data, num_bars, title_suffix="", width_factor=1.0):
    """Create candlestick chart with specific parameters"""
    
    # Take last num_bars
    if len(ohlc_data) > num_bars:
        start_idx = len(ohlc_data) - num_bars
        display_data = ohlc_data[start_idx:]
    else:
        display_data = ohlc_data
        start_idx = 0
    
    print(f"Creating chart with {len(display_data)} bars, width_factor={width_factor}")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    
    # Calculate candle width based on number of visible bars (THIN candlesticks)
    visible_bars = len(display_data)
    if visible_bars <= 50:
        base_width = 0.6
    elif visible_bars <= 200:
        base_width = 0.4
    elif visible_bars <= 1000:
        base_width = 0.3
    else:
        base_width = 0.2
    
    # Apply thinning factor to fix blob issue
    candle_width = base_width * width_factor * 0.7  # 30% thinner
    
    print(f"Using candle width: {candle_width:.3f} (base: {base_width}, factor: {width_factor})")
    
    # Draw candlesticks
    up_count = 0
    down_count = 0
    
    for i, (x, open_price, high, low, close) in enumerate(display_data):
        screen_x = start_idx + i
        
        # Skip invalid data
        if not all(np.isfinite([open_price, high, low, close])):
            continue
        if high <= 0 or low <= 0 or high < low:
            continue
        
        # Draw wick (thin vertical line)
        ax.plot([screen_x, screen_x], [low, high], color='black', linewidth=1)
        
        # Draw body (THIN rectangle)
        body_bottom = min(open_price, close)
        body_height = abs(close - open_price)
        
        # Handle doji candles
        min_height = (high - low) * 0.005
        if body_height < min_height:
            body_height = min_height
        
        # Color and draw rectangle
        if close >= open_price:  # Up candle
            color = 'white'
            edge_color = 'black'
            up_count += 1
        else:  # Down candle
            color = 'red'
            edge_color = 'black'
            down_count += 1
        
        rect = Rectangle(
            (screen_x - candle_width/2, body_bottom),
            candle_width,
            body_height,
            facecolor=color,
            edgecolor=edge_color,
            linewidth=1
        )
        ax.add_patch(rect)
    
    # Set up the chart
    min_price = min(row[3] for row in display_data)  # lows
    max_price = max(row[2] for row in display_data)  # highs
    price_padding = (max_price - min_price) * 0.05
    
    ax.set_xlim(start_idx - 1, start_idx + len(display_data))
    ax.set_ylim(min_price - price_padding, max_price + price_padding)
    
    ax.set_xlabel('Time (Bar Index)', fontsize=12)
    ax.set_ylabel('Price', fontsize=12)
    ax.set_title(f'Candlestick Chart - {len(display_data)} bars{title_suffix}\\n'
                f'Width: {candle_width:.3f} | Up: {up_count} | Down: {down_count}', 
                fontsize=14, fontweight='bold')
    
    ax.grid(True, alpha=0.3)
    
    # Add text box with details
    info_text = f'Candle Width: {candle_width:.3f}\\nVisible Bars: {visible_bars}\\nUp Candles: {up_count}\\nDown Candles: {down_count}'
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes, 
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    return fig

def test_different_configurations():
    """Test different candlestick configurations and save screenshots"""
    print("="*70)
    print("STEP 1: HEADLESS CANDLESTICK SCREENSHOT TEST")
    print("="*70)
    
    # Load data
    ohlc_data, data_description = load_data()
    
    # Test configurations
    configs = [
        (100, "100 bars", 1.0),
        (100, "100 bars THIN", 0.7),   # Thinner version
        (100, "100 bars VERY THIN", 0.5),  # Very thin
        (1000, "1000 bars", 1.0),
        (1000, "1000 bars THIN", 0.7),
    ]
    
    screenshots = []
    
    for i, (num_bars, title, width_factor) in enumerate(configs, 1):
        print(f"\\nCreating screenshot {i}: {title}")
        start_time = time.time()
        
        fig = create_candlestick_chart(ohlc_data, num_bars, f" - {title}", width_factor)
        
        # Save screenshot
        screenshot_file = Path(__file__).parent / f"step1_headless_{i:02d}_{title.replace(' ', '_').lower()}.png"
        fig.savefig(screenshot_file, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        creation_time = time.time() - start_time
        print(f"[OK] Saved: {screenshot_file}")
        print(f"  Creation time: {creation_time:.3f}s")
        
        screenshots.append(screenshot_file)
    
    print(f"\\n" + "="*70)
    print("SCREENSHOTS CREATED SUCCESSFULLY!")
    print("="*70)
    print(f"Data: {data_description}")
    print("Screenshots saved:")
    for i, screenshot in enumerate(screenshots, 1):
        print(f"  {i}. {screenshot.name}")
    
    print("\\nNow you can visually inspect these screenshots to verify:")
    print("  - Are candlesticks THIN (not fat blobs)?")
    print("  - Do they look professional like AmiBroker?")
    print("  - How does width change with different bar counts?")
    print("="*70)
    
    return screenshots

def compare_with_reference():
    """Instructions for comparing with reference screenshots"""
    print("\\nCOMPARISON INSTRUCTIONS:")
    print("1. Open the BAD screenshot: Screenshot 2025-08-05 060337.png")
    print("2. Open the GOOD reference: Screenshot 2025-08-04 105740.png") 
    print("3. Compare with our new screenshots:")
    print("   - step1_headless_01_100_bars.png")
    print("   - step1_headless_02_100_bars_thin.png")
    print("   - step1_headless_03_100_bars_very_thin.png")
    print("4. The new screenshots should look like the GOOD reference (thin)")
    print("5. NOT like the BAD screenshot (fat blobs)")

def main():
    """Main function"""
    screenshots = test_different_configurations()
    compare_with_reference()
    return screenshots

if __name__ == "__main__":
    main()