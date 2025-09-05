"""
Configuration management utilities for OMtree
"""

import configparser
import json
import os
from typing import Dict, Any, Optional, List
from constants import *

class ConfigManager:
    """Centralized configuration management"""
    
    def __init__(self, config_file: str = 'OMtree_config.ini'):
        self.config_file = config_file
        self.config = configparser.ConfigParser(inline_comment_prefixes='#')
        self.load()
    
    def load(self) -> None:
        """Load configuration from file"""
        if os.path.exists(self.config_file):
            self.config.read(self.config_file)
        else:
            self.create_default()
    
    def save(self) -> None:
        """Save configuration to file"""
        with open(self.config_file, 'w') as f:
            self.config.write(f)
    
    def create_default(self) -> None:
        """Create default configuration"""
        # Data section
        self.config['data'] = {
            'data_path': '',
            'target_column': '',
            'selected_features': ''
        }
        
        # Model section
        self.config['model'] = {
            'model_type': MODEL_LONGONLY,
            'n_trees': str(DEFAULT_N_TREES),
            'max_depth': str(DEFAULT_MAX_DEPTH),
            'min_samples_leaf': str(DEFAULT_MIN_SAMPLES_LEAF),
            'bootstrap_fraction': str(DEFAULT_BOOTSTRAP_FRACTION),
            'auto_calibrate_threshold': 'true',
            'target_prediction_rate': str(DEFAULT_TARGET_RATE),
            'probability_mode': PROB_RAW_AGGREGATION,
            'probability_aggregation': AGG_MEDIAN
        }
        
        # Validation section
        self.config['validation'] = {
            'train_size': str(DEFAULT_TRAIN_SIZE),
            'test_size': str(DEFAULT_TEST_SIZE),
            'step_size': str(DEFAULT_STEP_SIZE),
            'min_training_samples': str(MIN_TRAINING_SAMPLES),
            'calibration_lookback': str(DEFAULT_CALIBRATION_LOOKBACK)
        }
        
        # Feature selection section
        self.config['feature_selection'] = {
            'enabled': 'true',
            'min_features': str(DEFAULT_MIN_FEATURES),
            'max_features': str(DEFAULT_MAX_FEATURES),
            'importance_threshold': str(DEFAULT_IMPORTANCE_THRESHOLD),
            'selection_lookback': str(DEFAULT_SELECTION_LOOKBACK),
            'n_trees_method': 'per_feature',
            'n_trees_base': str(DEFAULT_N_TREES_BASE)
        }
        
        self.save()
    
    def get_data_config(self) -> Dict[str, Any]:
        """Get data configuration"""
        return dict(self.config['data']) if 'data' in self.config else {}
    
    def get_model_config(self) -> Dict[str, Any]:
        """Get model configuration"""
        if 'model' not in self.config:
            return {}
        
        config = dict(self.config['model'])
        
        # Convert types
        config['n_trees'] = int(config.get('n_trees', DEFAULT_N_TREES))
        config['max_depth'] = int(config.get('max_depth', DEFAULT_MAX_DEPTH))
        config['min_samples_leaf'] = int(config.get('min_samples_leaf', DEFAULT_MIN_SAMPLES_LEAF))
        config['bootstrap_fraction'] = float(config.get('bootstrap_fraction', DEFAULT_BOOTSTRAP_FRACTION))
        config['auto_calibrate_threshold'] = config.get('auto_calibrate_threshold', 'true').lower() == 'true'
        config['target_prediction_rate'] = float(config.get('target_prediction_rate', DEFAULT_TARGET_RATE))
        
        return config
    
    def get_validation_config(self) -> Dict[str, Any]:
        """Get validation configuration"""
        if 'validation' not in self.config:
            return {}
        
        config = dict(self.config['validation'])
        
        # Convert types
        config['train_size'] = int(config.get('train_size', DEFAULT_TRAIN_SIZE))
        config['test_size'] = int(config.get('test_size', DEFAULT_TEST_SIZE))
        config['step_size'] = int(config.get('step_size', DEFAULT_STEP_SIZE))
        config['min_training_samples'] = int(config.get('min_training_samples', MIN_TRAINING_SAMPLES))
        config['calibration_lookback'] = int(config.get('calibration_lookback', DEFAULT_CALIBRATION_LOOKBACK))
        
        return config
    
    def get_feature_selection_config(self) -> Dict[str, Any]:
        """Get feature selection configuration"""
        if 'feature_selection' not in self.config:
            return {}
        
        config = dict(self.config['feature_selection'])
        
        # Convert types
        config['enabled'] = config.get('enabled', 'true').lower() == 'true'
        config['min_features'] = int(config.get('min_features', DEFAULT_MIN_FEATURES))
        config['max_features'] = int(config.get('max_features', DEFAULT_MAX_FEATURES))
        config['importance_threshold'] = float(config.get('importance_threshold', DEFAULT_IMPORTANCE_THRESHOLD))
        config['selection_lookback'] = int(config.get('selection_lookback', DEFAULT_SELECTION_LOOKBACK))
        config['n_trees_base'] = int(config.get('n_trees_base', DEFAULT_N_TREES_BASE))
        
        return config
    
    def get_selected_features(self) -> List[str]:
        """Get list of selected features"""
        if 'data' not in self.config:
            return []
        
        features_str = self.config['data'].get('selected_features', '')
        if not features_str:
            return []
        
        return [f.strip() for f in features_str.split(',')]
    
    def set_selected_features(self, features: List[str]) -> None:
        """Set selected features"""
        if 'data' not in self.config:
            self.config['data'] = {}
        
        self.config['data']['selected_features'] = ','.join(features)
    
    def update_section(self, section: str, values: Dict[str, Any]) -> None:
        """Update a configuration section"""
        if section not in self.config:
            self.config[section] = {}
        
        for key, value in values.items():
            self.config[section][key] = str(value)
    
    def get_results_file_path(self, model_type: str) -> str:
        """Get results file path for given model type"""
        return os.path.join(RESULTS_DIR, RESULTS_FILE_TEMPLATE.format(model_type=model_type))
    
    def get_calibration_history_path(self, model_type: Optional[str] = None) -> str:
        """Get calibration history file path"""
        if model_type:
            filename = CALIBRATION_HISTORY_TEMPLATE.format(model_type=model_type)
        else:
            filename = CALIBRATION_HISTORY_FILE
        return os.path.join(RESULTS_DIR, filename)
    
    def get_feature_history_path(self, model_type: Optional[str] = None) -> str:
        """Get feature selection history file path"""
        if model_type:
            filename = FEATURE_HISTORY_TEMPLATE.format(model_type=model_type)
        else:
            filename = FEATURE_SELECTION_HISTORY_FILE
        return os.path.join(RESULTS_DIR, filename)

def load_json_config(filepath: str) -> Dict[str, Any]:
    """Load JSON configuration file"""
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            return json.load(f)
    return {}

def save_json_config(filepath: str, data: Dict[str, Any]) -> None:
    """Save JSON configuration file"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)

def merge_configs(*configs: Dict[str, Any]) -> Dict[str, Any]:
    """Merge multiple configuration dictionaries"""
    result = {}
    for config in configs:
        result.update(config)
    return result