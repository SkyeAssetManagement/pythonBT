#!/usr/bin/env python3
"""
CSV Export Module for Range Bar Data
===================================
Converts parquet range bar data back to CSV format in standardized structure:
Symbol, DateTime(text), O, H, L, C, V

Output Structure: dataRaw/[same folder structure as parquetData]/
"""

import pandas as pd
import pyarrow.parquet as pq
from pathlib import Path
import logging
import time
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# ============================================================================
# CORE CSV EXPORT FUNCTIONS
# ============================================================================

def export_parquet_to_csv(parquet_file_path, csv_output_path, dataset_type="DIFF"):
    """
    Convert a single parquet range bar file to CSV format.
    
    Args:
        parquet_file_path: Path to input parquet file
        csv_output_path: Path for output CSV file
        dataset_type: Dataset type (DIFF or CURR) for symbol formatting
        
    Returns:
        dict: Export statistics and results
    """
    try:
        start_time = time.time()
        
        # Load parquet data
        logger.info(f"Loading parquet: {parquet_file_path}")
        df = pd.read_parquet(parquet_file_path)
        
        if df.empty:
            logger.warning(f"Empty parquet file: {parquet_file_path}")
            return {'success': False, 'reason': 'Empty file'}
        
        logger.info(f"Loaded {len(df):,} range bars")
        
        # Create correct symbol format: ES-DIFF or ES-NONE
        if dataset_type == "DIFF":
            symbol = "ES-DIFF"
        elif dataset_type == "CURR":
            symbol = "ES-NONE"  # Current data uses ES-NONE as requested
        else:
            symbol = "ES"
        
        # Create CSV format with required columns
        csv_df = pd.DataFrame({
            'Symbol': symbol,
            'DateTime': df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S.%f'),  # Text format
            'O': df['open'].round(4),    # Open
            'H': df['high'].round(4),    # High  
            'L': df['low'].round(4),     # Low
            'C': df['close'].round(4),   # Close
            'V': df['volume']            # Volume
        })
        
        # Ensure output directory exists
        csv_output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Export to CSV
        csv_df.to_csv(csv_output_path, index=False)
        
        processing_time = time.time() - start_time
        file_size_mb = csv_output_path.stat().st_size / (1024 * 1024)
        
        logger.info(f"Exported to CSV: {csv_output_path}")
        logger.info(f"  Bars: {len(csv_df):,}")
        logger.info(f"  Size: {file_size_mb:.2f} MB")
        logger.info(f"  Time: {processing_time:.2f}s")
        
        return {
            'success': True,
            'bars_exported': len(csv_df),
            'file_size_mb': file_size_mb,
            'processing_time_sec': processing_time
        }
        
    except Exception as e:
        logger.error(f"Failed to export {parquet_file_path}: {e}")
        return {'success': False, 'reason': str(e)}

def discover_parquet_files(parquet_base_dir="parquetData"):
    """
    Discover all range bar parquet files in the parquet data structure.
    
    Args:
        parquet_base_dir: Base directory containing parquet files
        
    Returns:
        list: List of discovered parquet files with metadata
    """
    parquet_files = []
    base_path = Path(parquet_base_dir)
    
    if not base_path.exists():
        logger.error(f"Parquet base directory not found: {parquet_base_dir}")
        return []
    
    # Search for range bar parquet files
    for parquet_file in base_path.rglob("*.parquet"):
        # Look for range-ATR files
        if "range-ATR" in str(parquet_file) and "EST.parquet" in str(parquet_file):
            # Skip lightning, elegant, and other test versions
            file_str = str(parquet_file)
            
            # Skip unwanted versions
            skip_versions = ["-lightning", "-elegant", "-final-", "-corrected", "-full-dataset", "-optimized"]
            if any(skip in file_str for skip in skip_versions):
                continue
            
            # Extract metadata from path
            path_parts = parquet_file.parts
            
            # Find ATR period and dataset type
            atr_period = None
            dataset_type = None
            instrument = None
            
            for part in path_parts:
                if "range-ATR" in part:
                    if "ATR14" in part:
                        atr_period = 14
                    elif "ATR30" in part:
                        atr_period = 30
                    elif "ATR90" in part:
                        atr_period = 90
                
                if part in ["Current"]:
                    dataset_type = "CURR"
                elif part in ["diffAdjusted"]:
                    dataset_type = "DIFF"
                
                if part == "ES":
                    instrument = "ES"
            
            parquet_files.append({
                'file_path': parquet_file,
                'atr_period': atr_period,
                'dataset_type': dataset_type,
                'instrument': instrument,
                'relative_path': parquet_file.relative_to(base_path)
            })
    
    logger.info(f"Discovered {len(parquet_files)} range bar parquet files")
    for pf in parquet_files:
        logger.info(f"  ATR-{pf['atr_period']} {pf['dataset_type']}: {pf['file_path']}")
    
    return parquet_files

def create_csv_output_path(parquet_file_info, csv_base_dir="dataRaw"):
    """
    Create corresponding CSV output path maintaining directory structure.
    
    Args:
        parquet_file_info: Dictionary with parquet file metadata
        csv_base_dir: Base directory for CSV output
        
    Returns:
        Path: Output CSV file path
    """
    # Create dataRaw equivalent of parquetData structure
    relative_path = parquet_file_info['relative_path']
    csv_path = Path(csv_base_dir) / relative_path
    
    # Change extension to .csv
    csv_path = csv_path.with_suffix('.csv')
    
    return csv_path

# ============================================================================
# BATCH EXPORT OPERATIONS
# ============================================================================

def export_all_range_bars_to_csv(parquet_base_dir="parquetData", csv_base_dir="dataRaw", symbol="ES"):
    """
    Export all discovered range bar parquet files to CSV format.
    
    Args:
        parquet_base_dir: Directory containing parquet range bars
        csv_base_dir: Output directory for CSV files
        symbol: Trading symbol for CSV export
        
    Returns:
        dict: Complete export results and statistics
    """
    logger.info("🚀 CSV EXPORT MODULE - RANGE BAR PARQUET TO CSV")
    logger.info("="*80)
    logger.info(f"Exporting from: {parquet_base_dir}")
    logger.info(f"Exporting to: {csv_base_dir}")
    logger.info(f"Symbol: {symbol}")
    logger.info("="*80)
    
    total_start_time = time.time()
    
    # Discover parquet files
    parquet_files = discover_parquet_files(parquet_base_dir)
    
    if not parquet_files:
        logger.error("No range bar parquet files found!")
        return {'success': False, 'reason': 'No files found'}
    
    # Export each file
    export_results = []
    successful_exports = 0
    failed_exports = 0
    total_bars = 0
    total_size_mb = 0
    
    for file_info in parquet_files:
        logger.info(f"\n📊 Exporting ATR-{file_info['atr_period']} {file_info['dataset_type']}")
        logger.info("-" * 60)
        
        # Create output path
        csv_output_path = create_csv_output_path(file_info, csv_base_dir)
        
        # Export to CSV
        result = export_parquet_to_csv(
            file_info['file_path'], 
            csv_output_path, 
            file_info['dataset_type']
        )
        
        result['file_info'] = file_info
        result['csv_path'] = csv_output_path
        export_results.append(result)
        
        if result['success']:
            successful_exports += 1
            total_bars += result['bars_exported']
            total_size_mb += result['file_size_mb']
        else:
            failed_exports += 1
            logger.error(f"❌ Export failed: {result['reason']}")
    
    # Final summary
    total_processing_time = time.time() - total_start_time
    
    logger.info(f"\n{'='*80}")
    logger.info("CSV EXPORT RESULTS SUMMARY")
    logger.info("="*80)
    logger.info(f"{'File':<25} {'ATR':<6} {'Dataset':<8} {'Bars':<10} {'Size (MB)':<10} {'Status':<8}")
    logger.info("-" * 80)
    
    for result in export_results:
        if result['success']:
            file_name = result['csv_path'].name[:24]
            atr_period = result['file_info']['atr_period'] or "N/A"
            dataset_type = result['file_info']['dataset_type'] or "N/A"
            bars = result['bars_exported']
            size_mb = result['file_size_mb']
            status = "✅ OK"
            
            logger.info(f"{file_name:<25} {atr_period:<6} {dataset_type:<8} {bars:<10,} {size_mb:<10.2f} {status}")
        else:
            file_name = result['file_info']['file_path'].name[:24]
            logger.info(f"{file_name:<25} {'N/A':<6} {'N/A':<8} {'N/A':<10} {'N/A':<10} ❌ FAIL")
    
    logger.info("-" * 80)
    logger.info(f"Total files processed: {len(export_results)}")
    logger.info(f"Successful exports: {successful_exports}")
    logger.info(f"Failed exports: {failed_exports}")
    logger.info(f"Total bars exported: {total_bars:,}")
    logger.info(f"Total CSV size: {total_size_mb:.2f} MB")
    logger.info(f"Total processing time: {total_processing_time:.1f} seconds")
    
    # Directory structure summary
    logger.info(f"\n📁 CSV OUTPUT DIRECTORY STRUCTURE:")
    logger.info(f"   {csv_base_dir}/")
    
    if successful_exports > 0:
        csv_base_path = Path(csv_base_dir)
        for result in export_results:
            if result['success']:
                rel_path = result['csv_path'].relative_to(csv_base_path)
                logger.info(f"   ├── {rel_path}")
    
    logger.info("="*80)
    if failed_exports == 0:
        logger.info("🎉 ALL RANGE BAR EXPORTS COMPLETED SUCCESSFULLY")
    else:
        logger.info(f"⚠️  {failed_exports} EXPORTS FAILED - CHECK LOGS ABOVE")
    
    return {
        'success': failed_exports == 0,
        'total_files': len(export_results),
        'successful_exports': successful_exports,
        'failed_exports': failed_exports,
        'total_bars_exported': total_bars,
        'total_size_mb': total_size_mb,
        'processing_time_sec': total_processing_time,
        'export_results': export_results
    }

def export_specific_atr_period(atr_period, parquet_base_dir="parquetData", csv_base_dir="dataRaw", symbol="ES"):
    """
    Export only a specific ATR period (e.g., just ATR-14 DIFF and CURR).
    
    Args:
        atr_period: ATR period to export (14, 30, or 90)
        parquet_base_dir: Directory containing parquet range bars
        csv_base_dir: Output directory for CSV files
        symbol: Trading symbol for CSV export
        
    Returns:
        dict: Export results for specified ATR period
    """
    logger.info(f"🎯 EXPORTING ATR-{atr_period} RANGE BARS TO CSV")
    logger.info("="*60)
    
    # Discover all parquet files
    all_files = discover_parquet_files(parquet_base_dir)
    
    # Filter for specific ATR period
    atr_files = [f for f in all_files if f['atr_period'] == atr_period]
    
    if not atr_files:
        logger.error(f"No ATR-{atr_period} range bar files found!")
        return {'success': False, 'reason': f'No ATR-{atr_period} files found'}
    
    logger.info(f"Found {len(atr_files)} ATR-{atr_period} files to export")
    
    export_results = []
    for file_info in atr_files:
        csv_output_path = create_csv_output_path(file_info, csv_base_dir)
        
        result = export_parquet_to_csv(
            file_info['file_path'],
            csv_output_path,
            file_info['dataset_type']
        )
        
        result['file_info'] = file_info
        export_results.append(result)
    
    successful = sum(1 for r in export_results if r['success'])
    total_bars = sum(r.get('bars_exported', 0) for r in export_results if r['success'])
    
    logger.info(f"\nATR-{atr_period} EXPORT COMPLETE:")
    logger.info(f"  Files exported: {successful}/{len(export_results)}")
    logger.info(f"  Total bars: {total_bars:,}")
    
    return {
        'success': successful == len(export_results),
        'atr_period': atr_period,
        'files_exported': successful,
        'total_bars': total_bars,
        'export_results': export_results
    }

# ============================================================================
# ENTRY POINT AND CLI INTERFACE
# ============================================================================

def main():
    """
    Main entry point for CSV export functionality.
    
    Usage examples:
    - Export all range bars: python csv_exporter.py
    - Export specific ATR: python csv_exporter.py --atr 14
    """
    import sys
    
    # Simple command line argument parsing
    if len(sys.argv) > 1 and sys.argv[1] == "--atr" and len(sys.argv) > 2:
        atr_period = int(sys.argv[2])
        result = export_specific_atr_period(atr_period)
    else:
        result = export_all_range_bars_to_csv()
    
    if result['success']:
        logger.info("\n✅ CSV export completed successfully!")
        exit(0)
    else:
        logger.error("\n❌ CSV export failed!")
        exit(1)

if __name__ == "__main__":
    main()