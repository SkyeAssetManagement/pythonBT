"""
Flexible column detection utilities for handling different data formats
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Optional


class ColumnDetector:
    """Intelligently detect feature and target columns from various data formats"""
    
    @staticmethod
    def detect_columns(df: pd.DataFrame, 
                      config_features: Optional[List[str]] = None,
                      config_targets: Optional[List[str]] = None) -> Dict[str, List[str]]:
        """
        Detect feature and target columns, using config if valid, otherwise auto-detect
        
        Returns:
            Dict with 'features' and 'targets' lists
        """
        result = {'features': [], 'targets': []}
        
        # First try to use config columns if they exist in the dataframe
        if config_features:
            valid_features = [col for col in config_features if col in df.columns]
            if valid_features:
                result['features'] = valid_features
        
        if config_targets:
            valid_targets = [col for col in config_targets if col in df.columns]
            if valid_targets:
                result['targets'] = valid_targets
        
        # If we don't have features or targets from config, auto-detect
        if not result['features'] or not result['targets']:
            auto_detected = ColumnDetector.auto_detect_columns(df)
            
            if not result['features']:
                result['features'] = auto_detected['features']
            if not result['targets']:
                result['targets'] = auto_detected['targets']
        
        return result
    
    @staticmethod
    def auto_detect_columns(df: pd.DataFrame) -> Dict[str, List[str]]:
        """
        Auto-detect feature and target columns based on naming patterns
        """
        features = []
        targets = []
        
        # Get numeric columns only
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        for col in numeric_cols:
            col_lower = col.lower()
            
            # Skip date/time/index columns
            if any(skip in col_lower for skip in ['date', 'time', 'index', 'id', 'timestamp']):
                continue
            
            # Detect targets (forward-looking returns)
            if any(pattern in col_lower for pattern in ['forward', 'fwd', 'future', 'target', 'label']):
                targets.append(col)
            # Detect features (historical returns or other patterns)
            elif any(pattern in col_lower for pattern in ['ret', 'return', 'price', 'volume', 
                                                          'close', 'open', 'high', 'low', 
                                                          'diff', 'none', 'overnight', 'day',
                                                          'pir', 'ratio', 'impact']):
                # Exclude if it looks like a forward return
                if not any(fwd in col_lower for fwd in ['forward', 'fwd', 'future']):
                    features.append(col)
        
        # If no features/targets found with patterns, use heuristics
        if not features and not targets:
            # Assume last few columns might be targets
            if len(numeric_cols) > 2:
                # Take last 20% as potential targets
                n_targets = max(1, len(numeric_cols) // 5)
                targets = numeric_cols[-n_targets:]
                features = numeric_cols[:-n_targets]
            elif len(numeric_cols) == 2:
                features = [numeric_cols[0]]
                targets = [numeric_cols[1]]
            else:
                # Single column - assume it's a feature
                features = numeric_cols
        
        return {'features': features, 'targets': targets}
    
    @staticmethod
    def validate_columns(df: pd.DataFrame, required_cols: List[str], 
                        col_type: str = "required") -> Tuple[List[str], List[str]]:
        """
        Validate which columns exist and which are missing
        
        Returns:
            Tuple of (existing_columns, missing_columns)
        """
        existing = [col for col in required_cols if col in df.columns]
        missing = [col for col in required_cols if col not in df.columns]
        
        if missing and col_type:
            print(f"Warning: Missing {col_type} columns: {missing}")
        
        return existing, missing
    
    @staticmethod
    def find_similar_columns(df: pd.DataFrame, target_col: str, threshold: float = 0.6) -> List[str]:
        """
        Find columns with similar names to target_col using string similarity
        """
        from difflib import SequenceMatcher
        
        similar = []
        target_lower = target_col.lower()
        
        for col in df.columns:
            col_lower = col.lower()
            similarity = SequenceMatcher(None, target_lower, col_lower).ratio()
            if similarity >= threshold and col != target_col:
                similar.append((col, similarity))
        
        # Sort by similarity
        similar.sort(key=lambda x: x[1], reverse=True)
        return [col for col, _ in similar]
    
    @staticmethod
    def get_safe_columns(df: pd.DataFrame, requested_cols: List[str], 
                        auto_detect: bool = True) -> List[str]:
        """
        Get columns that exist, with fallback to auto-detection
        """
        # First, get columns that actually exist
        existing = [col for col in requested_cols if col in df.columns]
        
        if existing:
            return existing
        
        if auto_detect:
            print(f"Requested columns not found: {requested_cols}")
            print("Auto-detecting columns...")
            
            # Try to find similar columns
            all_similar = []
            for req_col in requested_cols:
                similar = ColumnDetector.find_similar_columns(df, req_col)
                if similar:
                    print(f"  Found similar to '{req_col}': {similar[0]}")
                    all_similar.append(similar[0])
            
            if all_similar:
                return all_similar
            
            # Fall back to auto-detection
            detected = ColumnDetector.auto_detect_columns(df)
            if 'fwd' in str(requested_cols).lower() or 'forward' in str(requested_cols).lower():
                return detected['targets']
            else:
                return detected['features']
        
        return []