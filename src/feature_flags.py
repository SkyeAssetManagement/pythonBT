"""
Feature Flag System - Control new functionality deployment
==========================================================
Following safety-first principles: Deploy dark, validate, then activate
"""

import json
import os
from typing import Dict, Any, Optional
from datetime import datetime


class FeatureFlags:
    """
    Feature flag management for safe incremental deployment
    
    Principles:
    - ALL new code behind flags
    - Monitor for 24 hours before enabling
    - Easy rollback if issues arise
    """
    
    def __init__(self, config_file: str = 'feature_flags.json'):
        """Initialize feature flags from config file"""
        self.config_file = config_file
        self.flags: Dict[str, bool] = {}
        self.metadata: Dict[str, Dict[str, Any]] = {}
        self.load_flags()
        
    def load_flags(self):
        """Load flags from configuration file"""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    data = json.load(f)
                    self.flags = data.get('flags', {})
                    self.metadata = data.get('metadata', {})
            except Exception as e:
                print(f"Warning: Could not load feature flags: {e}")
                self._set_defaults()
        else:
            self._set_defaults()
            self.save_flags()
    
    def _set_defaults(self):
        """Set default feature flags (all disabled for safety)"""
        self.flags = {
            # Trading data integration
            'use_new_trade_data': False,
            'enable_vbt_integration': False,
            'use_pyqtgraph_charts': False,
            
            # Data pipeline
            'unified_data_pipeline': False,
            'new_csv_loader': False,
            
            # GUI features
            'show_trade_visualization_tab': False,
            'enable_range_bar_charts': False,
            'show_trade_panel': False,
            
            # Performance optimizations
            'use_parallel_processing': False,
            'enable_caching': False,
            
            # Testing features
            'verbose_logging': True,
            'debug_mode': False,
        }
        
        # Metadata for tracking
        for flag in self.flags:
            self.metadata[flag] = {
                'created': datetime.now().isoformat(),
                'enabled_at': None,
                'description': '',
                'risk_level': 'medium'
            }
    
    def save_flags(self):
        """Save current flags to configuration file"""
        try:
            data = {
                'flags': self.flags,
                'metadata': self.metadata
            }
            with open(self.config_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save feature flags: {e}")
    
    def is_enabled(self, flag_name: str) -> bool:
        """
        Check if a feature flag is enabled
        
        Args:
            flag_name: Name of the feature flag
            
        Returns:
            True if enabled, False otherwise (default safe)
        """
        return self.flags.get(flag_name, False)
    
    def enable(self, flag_name: str):
        """
        Enable a feature flag (with metadata tracking)
        
        Args:
            flag_name: Name of the feature flag to enable
        """
        if flag_name in self.flags:
            self.flags[flag_name] = True
            if flag_name in self.metadata:
                self.metadata[flag_name]['enabled_at'] = datetime.now().isoformat()
            self.save_flags()
            print(f"Feature flag '{flag_name}' enabled")
        else:
            print(f"Warning: Unknown feature flag '{flag_name}'")
    
    def disable(self, flag_name: str):
        """
        Disable a feature flag (for rollback)
        
        Args:
            flag_name: Name of the feature flag to disable
        """
        if flag_name in self.flags:
            self.flags[flag_name] = False
            self.save_flags()
            print(f"Feature flag '{flag_name}' disabled")
        else:
            print(f"Warning: Unknown feature flag '{flag_name}'")
    
    def get_all_flags(self) -> Dict[str, bool]:
        """Get all current feature flags"""
        return self.flags.copy()
    
    def get_flag_info(self, flag_name: str) -> Optional[Dict[str, Any]]:
        """Get metadata about a specific flag"""
        return self.metadata.get(flag_name)
    
    def set_risk_level(self, flag_name: str, risk_level: str):
        """
        Set risk level for a feature flag
        
        Args:
            flag_name: Name of the feature flag
            risk_level: 'low', 'medium', or 'high'
        """
        if flag_name in self.metadata:
            if risk_level in ['low', 'medium', 'high']:
                self.metadata[flag_name]['risk_level'] = risk_level
                self.save_flags()
            else:
                print(f"Invalid risk level: {risk_level}")
    
    def get_flags_by_risk(self, risk_level: str) -> Dict[str, bool]:
        """Get all flags of a specific risk level"""
        result = {}
        for flag_name, enabled in self.flags.items():
            if self.metadata.get(flag_name, {}).get('risk_level') == risk_level:
                result[flag_name] = enabled
        return result
    
    def enable_low_risk_features(self):
        """Enable all low-risk features (safe for early deployment)"""
        low_risk = self.get_flags_by_risk('low')
        for flag_name in low_risk:
            self.enable(flag_name)
    
    def disable_all_features(self):
        """Emergency rollback - disable all feature flags"""
        for flag_name in self.flags:
            self.flags[flag_name] = False
        self.save_flags()
        print("All feature flags disabled (emergency rollback)")


# Global feature flags instance
_feature_flags = None

def get_feature_flags() -> FeatureFlags:
    """Get global feature flags instance"""
    global _feature_flags
    if _feature_flags is None:
        _feature_flags = FeatureFlags()
    return _feature_flags


# Decorator for feature flag protection
def feature_flag(flag_name: str, default_return=None):
    """
    Decorator to protect functions with feature flags
    
    Usage:
        @feature_flag('new_feature')
        def my_new_function():
            # This only runs if 'new_feature' flag is enabled
            pass
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            flags = get_feature_flags()
            if flags.is_enabled(flag_name):
                return func(*args, **kwargs)
            else:
                if default_return is not None:
                    return default_return
                # For GUI functions, often return silently
                return None
        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        return wrapper
    return decorator