# OMtree Trading System - Code Documentation

## System Architecture Overview

The OMtree Trading System is a machine learning-based trading strategy implementation that uses ensemble decision trees for directional market prediction. The system is designed for walk-forward validation and includes a comprehensive GUI for analysis.

## Project Structure

```
PythonBT/
├── src/                          # Core modules
│   ├── OMtree_model.py          # Main model implementation
│   ├── OMtree_validation.py    # Validation logic
│   ├── OMtree_preprocessing.py # Data preprocessing
│   ├── OMtree_walkforward.py   # Walk-forward engine
│   └── [supporting modules]
├── data/                        # Data directory
├── archive/                     # Historical experiments
├── OMtree_gui.py               # Main GUI application
├── OMtree_config.ini           # Configuration file
└── requirements.txt            # Dependencies
```

## Core Components

### 1. Model Module (`src/OMtree_model.py`)

#### Class: `DirectionalTreeEnsemble`
The main model class implementing the tree ensemble strategy.

**Key Methods:**
- `__init__(config_path)`: Initialize model with configuration
- `fit(X, y)`: Train the ensemble on features X and targets y
- `predict(X)`: Generate predictions for new data
- `predict_proba(X)`: Get probability estimates
- `create_directional_labels(y)`: Convert returns to binary labels

**Configuration Parameters:**
- `n_trees`: Number of trees in ensemble (default: 100)
- `max_depth`: Maximum tree depth (default: 3)
- `bootstrap_fraction`: Sampling fraction (default: 0.6)
- `min_leaf_fraction`: Minimum samples in leaf nodes (default: 0.1)
- `target_threshold`: Return threshold for signal generation
- `vote_threshold`: Ensemble agreement threshold
- `model_type`: 'longonly' or 'shortonly'

**Algorithm Options:**
- Decision Trees (standard)
- Extra Trees (randomized)
- Random Forest (with bagging)
- Regression mode (continuous targets)

### 2. Preprocessing Module (`src/OMtree_preprocessing.py`)

#### Class: `DataPreprocessor`
Handles all data transformation and feature engineering.

**Key Methods:**
- `preprocess(df, config)`: Main preprocessing pipeline
- `normalize_features()`: Apply normalization methods
- `calculate_volatility()`: Compute volatility measures
- `create_technical_indicators()`: Generate additional features

**Normalization Methods:**
- `NONE`: No normalization
- `STANDARD`: Z-score normalization
- `MINMAX`: Min-max scaling
- `ROBUST`: Robust scaling
- `LOGIT_RANK`: Logit transformation of ranks
- `PERCENTILE`: Percentile transformation

**Feature Engineering:**
- Rolling statistics
- Volatility adjustments
- Technical indicators
- Lag features

### 3. Validation Module (`src/OMtree_validation.py`)

#### Class: `WalkForwardValidator`
Implements walk-forward validation logic.

**Key Methods:**
- `run_validation()`: Execute full walk-forward analysis
- `train_test_split()`: Create time-based splits
- `calculate_performance()`: Compute performance metrics
- `generate_reports()`: Create output reports

**Validation Parameters:**
- `train_size`: Training window size
- `test_size`: Testing window size  
- `step_size`: Window advancement step
- `min_training_samples`: Minimum required training data

### 4. GUI Module (`OMtree_gui.py`)

#### Class: `OMtreeGUI`
Main GUI application using tkinter.

**Key Components:**
- **Configuration Tab**: Model and data settings
- **Walk-Forward Tab**: Run and monitor validation
- **Performance Tab**: View metrics and charts
- **Tree Visualizer Tab**: Inspect model structure
- **Data View Tab**: Explore raw and processed data
- **Regression Analysis Tab**: Statistical diagnostics

**Key Methods:**
- `setup_ui()`: Initialize GUI components
- `load_configuration()`: Load settings from file
- `run_walkforward()`: Execute walk-forward in thread
- `update_results()`: Display results in GUI
- `export_results()`: Save results to files

### 5. Performance Statistics (`src/performance_stats.py`)

#### Functions:
- `calculate_sharpe_ratio()`: Risk-adjusted returns
- `calculate_max_drawdown()`: Maximum peak-to-trough decline
- `calculate_win_rate()`: Percentage of profitable trades
- `calculate_calmar_ratio()`: Return to drawdown ratio
- `calculate_information_ratio()`: Active return vs tracking error

### 6. Feature Selection (`src/feature_selector.py`)

#### Class: `FeatureSelector`
Dynamic feature selection based on importance.

**Methods:**
- `select_features()`: Choose best features
- `calculate_importance()`: Compute feature importance
- `apply_threshold()`: Filter by importance threshold

### 7. Configuration Manager (`src/config_manager.py`)

#### Class: `ConfigurationManager`
Handles configuration file operations.

**Methods:**
- `load_config()`: Read configuration from INI file
- `save_config()`: Write configuration to file
- `validate_config()`: Check configuration validity
- `merge_configs()`: Combine multiple configurations

## Data Flow

```
1. Raw Data (CSV)
   ↓
2. Preprocessing
   - Date/time parsing
   - Feature engineering
   - Normalization
   ↓
3. Walk-Forward Splits
   - Training windows
   - Testing windows
   ↓
4. Model Training
   - Tree ensemble fitting
   - Feature selection
   ↓
5. Prediction Generation
   - Probability aggregation
   - Signal generation
   ↓
6. Performance Calculation
   - Trade statistics
   - Risk metrics
   ↓
7. Results Output
   - CSV files
   - Charts
   - Reports
```

## Key Algorithms

### 1. Ensemble Voting
```python
# Simplified voting logic
predictions = []
for tree in self.trees:
    pred = tree.predict(X)
    predictions.append(pred)

# Aggregate predictions
if self.probability_aggregation == 'mean':
    final_pred = np.mean(predictions, axis=0)
elif self.probability_aggregation == 'median':
    final_pred = np.median(predictions, axis=0)

# Apply threshold
signals = final_pred > self.vote_threshold
```

### 2. Bootstrap Sampling
```python
# Bootstrap with replacement
n_samples = int(len(X) * self.bootstrap_fraction)
indices = np.random.choice(len(X), n_samples, replace=True)
X_bootstrap = X[indices]
y_bootstrap = y[indices]
```

### 3. Walk-Forward Logic
```python
# Sliding window validation
for i in range(start, end, step_size):
    train_start = i
    train_end = i + train_size
    test_start = train_end
    test_end = test_start + test_size
    
    # Train on historical window
    model.fit(X[train_start:train_end], 
              y[train_start:train_end])
    
    # Test on forward period
    predictions = model.predict(X[test_start:test_end])
    
    # Store results
    results.append(predictions)
```

## Configuration File Structure

### Sections and Parameters

#### [data]
- `csv_file`: Path to input data
- `target_column`: Column with forward returns
- `feature_columns`: Comma-separated feature list
- `date_column`: Date column name
- `time_column`: Time column name
- `ticker_filter`: Filter specific ticker
- `hour_filter`: Filter specific hour

#### [model]
- `model_type`: longonly/shortonly
- `algorithm`: decision_trees/extra_trees
- `n_trees`: Number of trees
- `max_depth`: Tree depth limit
- `bootstrap_fraction`: Sampling ratio
- `min_leaf_fraction`: Minimum leaf size
- `target_threshold`: Signal threshold
- `vote_threshold`: Voting threshold

#### [preprocessing]
- `normalize_features`: Enable normalization
- `normalization_method`: Method to use
- `vol_window`: Volatility calculation window
- `winsorize_enabled`: Outlier capping
- `winsorize_percentile`: Capping threshold

#### [validation]
- `train_size`: Training window
- `test_size`: Testing window
- `step_size`: Window step
- `validation_start_date`: Start date
- `validation_end_date`: End date

#### [output]
- `results_file`: Output CSV path
- `verbose_logging`: Enable detailed logs
- `save_predictions`: Store all predictions

## Performance Metrics

### Core Metrics
1. **Sharpe Ratio**: `(mean_return - risk_free) / std_return`
2. **Win Rate**: `profitable_trades / total_trades`
3. **Profit Factor**: `gross_profits / gross_losses`
4. **Max Drawdown**: `max(peak - trough) / peak`
5. **Calmar Ratio**: `annual_return / max_drawdown`

### Trade Statistics
- Number of trades
- Average trade duration
- Average winning trade
- Average losing trade
- Win/loss ratio
- Consecutive wins/losses

## Error Handling

### Common Error Patterns
```python
try:
    # Data loading
    df = pd.read_csv(file_path)
except FileNotFoundError:
    print(f"Data file not found: {file_path}")
except pd.errors.ParserError:
    print(f"Error parsing CSV file: {file_path}")

try:
    # Model training
    model.fit(X, y)
except ValueError as e:
    print(f"Model training failed: {e}")
```

### Validation Checks
- Minimum data requirements
- Column existence validation
- Date format verification
- Feature availability checks

## GUI Event Handling

### Threading Model
```python
# Run long operations in separate thread
def run_walkforward(self):
    thread = threading.Thread(target=self._execute_walkforward)
    thread.daemon = True
    thread.start()

def _execute_walkforward(self):
    # Long-running operation
    results = walkforward.run()
    # Update GUI in main thread
    self.root.after(0, self.update_results, results)
```

### Tab Management
Each tab is a separate frame with its own:
- Event handlers
- Data management
- Visualization components
- Export functionality

## Extensibility

### Adding New Features
1. Update preprocessing pipeline
2. Add to feature_columns in config
3. Implement calculation in preprocessing

### Adding New Models
1. Inherit from base model class
2. Implement fit() and predict()
3. Register in model factory

### Adding New Metrics
1. Add calculation to performance_stats.py
2. Update reporting functions
3. Add to GUI display

## Testing Components

### Unit Testing Structure
```python
# Test model predictions
def test_model_prediction():
    model = DirectionalTreeEnsemble()
    X_test = generate_test_data()
    predictions = model.predict(X_test)
    assert len(predictions) == len(X_test)
```

### Integration Testing
- End-to-end walk-forward runs
- GUI interaction tests
- Configuration validation

## Memory Management

### Large Dataset Handling
- Chunked data processing
- Rolling window calculations
- Garbage collection hints
- Feature selection to reduce dimensionality

## Logging

### Log Levels
- `DEBUG`: Detailed diagnostic info
- `INFO`: General information
- `WARNING`: Warning messages
- `ERROR`: Error conditions
- `CRITICAL`: Critical failures

### Log Output
```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('omtree.log'),
        logging.StreamHandler()
    ]
)
```

## Security Considerations

### Data Validation
- Input sanitization
- Path traversal prevention
- SQL injection prevention (if using databases)

### Configuration Security
- No hardcoded credentials
- Secure file permissions
- Environment variable usage

## Performance Optimization

### Computational Efficiency
- Vectorized operations with NumPy
- Parallel tree training
- Caching repeated calculations
- Efficient data structures

### Memory Efficiency
- Generator expressions for large datasets
- In-place operations where possible
- Proper cleanup of large objects

## Deployment Considerations

### Environment Setup
```bash
# Virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

### Production Checklist
- [ ] Data pipeline validation
- [ ] Model versioning
- [ ] Performance monitoring
- [ ] Error alerting
- [ ] Backup strategies
- [ ] Resource limits

## Version Control

### Git Workflow
```bash
# Feature branch
git checkout -b feature/new-indicator

# Commit changes
git add .
git commit -m "Add new technical indicator"

# Push to remote
git push origin feature/new-indicator
```

### Versioning Strategy
- Major: Breaking changes
- Minor: New features
- Patch: Bug fixes

## Future Enhancements

### Planned Features
1. Real-time data integration
2. Multi-asset portfolio support
3. Advanced risk management
4. Cloud deployment support
5. API endpoints for external access

### Performance Improvements
1. GPU acceleration for tree training
2. Distributed computing support
3. Advanced caching mechanisms
4. Database integration for large datasets

## Support and Maintenance

### Common Maintenance Tasks
1. Update dependencies regularly
2. Monitor log files for errors
3. Validate data quality
4. Review model performance
5. Update documentation

### Troubleshooting Workflow
1. Check log files
2. Verify data format
3. Validate configuration
4. Test with sample data
5. Debug with verbose mode

---

*This documentation provides a comprehensive overview of the OMtree Trading System codebase. For specific implementation details, refer to the individual source files and inline documentation.*