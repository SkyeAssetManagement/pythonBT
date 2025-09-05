"""
Configuration Management System for OMtree
Handles data configs, model configs, and projects
"""

import json
import os
from datetime import datetime
import uuid
import pandas as pd

class ConfigurationManager:
    def __init__(self):
        self.data_configs_file = 'data_configs.json'
        self.model_configs_file = 'model_configs.json'
        self.projects_dir = 'projects'
        
        # Create projects directory if it doesn't exist
        if not os.path.exists(self.projects_dir):
            os.makedirs(self.projects_dir)
        
        # Load existing configurations
        self.data_configs = self.load_json(self.data_configs_file, default=[])
        self.model_configs = self.load_json(self.model_configs_file, default=[])
    
    def load_json(self, filename, default=None):
        """Load JSON file or return default"""
        if os.path.exists(filename):
            try:
                with open(filename, 'r') as f:
                    return json.load(f)
            except:
                return default if default is not None else {}
        return default if default is not None else {}
    
    def save_json(self, data, filename):
        """Save data to JSON file"""
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    
    def save_data_config(self, csv_file, validation_start, validation_end, 
                        date_column, time_column, features, targets):
        """Save a data configuration to history"""
        config_id = str(uuid.uuid4())
        config = {
            'id': config_id,
            'csv_file': csv_file,
            'validation_start': validation_start,
            'validation_end': validation_end,
            'date_column': date_column,
            'time_column': time_column,
            'features': features,
            'targets': targets,
            'saved_at': datetime.now().isoformat()
        }
        
        # Add to beginning of list (most recent first)
        self.data_configs.insert(0, config)
        
        # Keep only last 50 configs
        self.data_configs = self.data_configs[:50]
        
        # Save to file
        self.save_json(self.data_configs, self.data_configs_file)
        
        return config_id
    
    def save_model_config(self, selected_features, selected_target, model_params):
        """Save a model configuration to history"""
        config_id = str(uuid.uuid4())
        config = {
            'id': config_id,
            'selected_features': selected_features,
            'selected_target': selected_target,
            'model_type': model_params.get('model_type', 'longonly'),
            'n_trees': model_params.get('n_trees', 200),
            'max_depth': model_params.get('max_depth', 1),
            'min_leaf_fraction': model_params.get('min_leaf_fraction', 0.2),
            'target_threshold': model_params.get('target_threshold', 0.1),
            'vote_threshold': model_params.get('vote_threshold', 0.7),
            'normalize_features': model_params.get('normalize_features', True),
            'normalize_target': model_params.get('normalize_target', True),
            'vol_window': model_params.get('vol_window', 60),
            'saved_at': datetime.now().isoformat()
        }
        
        # Add to beginning of list
        self.model_configs.insert(0, config)
        
        # Keep only last 50 configs
        self.model_configs = self.model_configs[:50]
        
        # Save to file
        self.save_json(self.model_configs, self.model_configs_file)
        
        return config_id
    
    def get_data_configs_df(self):
        """Get data configurations as DataFrame for display"""
        if not self.data_configs:
            return pd.DataFrame()
        
        # Create simplified view for display
        display_configs = []
        for config in self.data_configs:
            display_configs.append({
                'CSV File': os.path.basename(config['csv_file']),
                'Val Start': config['validation_start'],
                'Val End': config['validation_end'],
                'Features': len(config.get('features', [])),
                'Targets': len(config.get('targets', [])),
                'Saved': config['saved_at'][:19].replace('T', ' ')
            })
        
        return pd.DataFrame(display_configs)
    
    def get_model_configs_df(self):
        """Get model configurations as DataFrame for display"""
        if not self.model_configs:
            return pd.DataFrame()
        
        # Create simplified view for display
        display_configs = []
        for config in self.model_configs:
            features_str = ','.join(config['selected_features'][:3])
            if len(config['selected_features']) > 3:
                features_str += f'... ({len(config["selected_features"])} total)'
            
            display_configs.append({
                'Features': features_str,
                'Target': config['selected_target'],
                'Model': config['model_type'],
                'Trees': config['n_trees'],
                'Threshold': config['target_threshold'],
                'Saved': config['saved_at'][:19].replace('T', ' ')
            })
        
        return pd.DataFrame(display_configs)
    
    def get_data_config(self, index):
        """Get a specific data configuration by index"""
        if 0 <= index < len(self.data_configs):
            return self.data_configs[index]
        return None
    
    def get_model_config(self, index):
        """Get a specific model configuration by index"""
        if 0 <= index < len(self.model_configs):
            return self.model_configs[index]
        return None
    
    def save_project(self, project_name, data_config_id, model_config_id, additional_settings=None):
        """Save a complete project"""
        project = {
            'name': project_name,
            'data_config_id': data_config_id,
            'model_config_id': model_config_id,
            'additional_settings': additional_settings or {},
            'saved_at': datetime.now().isoformat()
        }
        
        project_file = os.path.join(self.projects_dir, f"{project_name}.json")
        self.save_json(project, project_file)
        
        return project_file
    
    def load_project(self, project_file):
        """Load a project and return its configurations"""
        project = self.load_json(project_file)
        
        # Find the referenced configurations
        data_config = None
        model_config = None
        
        for dc in self.data_configs:
            if dc['id'] == project.get('data_config_id'):
                data_config = dc
                break
        
        for mc in self.model_configs:
            if mc['id'] == project.get('model_config_id'):
                model_config = mc
                break
        
        return {
            'project': project,
            'data_config': data_config,
            'model_config': model_config
        }
    
    def get_latest_configs(self):
        """Get the most recent data and model configs"""
        data_config = self.data_configs[0] if self.data_configs else None
        model_config = self.model_configs[0] if self.model_configs else None
        return data_config, model_config