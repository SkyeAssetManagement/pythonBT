"""
Constants and configuration values for OMtree
"""

# File paths and templates
RESULTS_FILE_TEMPLATE = "walkforward_results_{model_type}.csv"
CALIBRATION_HISTORY_FILE = "calibration_history.json"
FEATURE_SELECTION_HISTORY_FILE = "feature_selection_history.json"
CALIBRATION_HISTORY_TEMPLATE = "calibration_history_{model_type}.json"
FEATURE_HISTORY_TEMPLATE = "feature_selection_history_{model_type}.json"

# Default directories
DATA_DIR = "data"
CONFIG_DIR = "config"
RESULTS_DIR = "results"
PROJECTS_DIR = "projects"
PERMUTE_RESULTS_DIR = "PermuteAlpha_Results"

# Model defaults
DEFAULT_N_TREES = 100
DEFAULT_MAX_DEPTH = 3
DEFAULT_MIN_SAMPLES_LEAF = 100
DEFAULT_BOOTSTRAP_FRACTION = 0.8
DEFAULT_TARGET_RATE = 0.2
MAX_ITERATIONS = 1000
EPSILON = 1e-10

# Validation defaults
DEFAULT_TRAIN_SIZE = 2000
DEFAULT_TEST_SIZE = 100
DEFAULT_STEP_SIZE = 100
DEFAULT_CALIBRATION_LOOKBACK = 90
MIN_TRAINING_SAMPLES = 100

# Feature selection defaults
DEFAULT_MIN_FEATURES = 1
DEFAULT_MAX_FEATURES = 4
DEFAULT_IMPORTANCE_THRESHOLD = 0.15
DEFAULT_SELECTION_LOOKBACK = 2000
DEFAULT_N_TREES_BASE = 50

# Column names
DATE_COLUMN = 'date'
TIME_COLUMN = 'time'
PREDICTION_COLUMN = 'prediction'
ACTUAL_COLUMN = 'actual'
SIGNAL_COLUMN = 'signal'
PNL_COLUMN = 'pnl'

# Output columns for PermuteAlpha
PERMUTE_DATE_COL = 'Date'
PERMUTE_RETURN_COL = 'Return'
PERMUTE_TRADE_COL = 'TradeFlag'
PERMUTE_FEATURE_COL = 'FeatureReturn'

# Feature patterns
RETURN_FEATURE_PREFIX = 'Ret_'
FORWARD_RETURN_KEYWORD = 'fwd'
VOLATILITY_SIGNAL_COL = 'VolSignal'

# Model types
MODEL_LONGONLY = 'longonly'
MODEL_SHORTONLY = 'shortonly'

# Aggregation methods
AGG_MEAN = 'mean'
AGG_MEDIAN = 'median'

# Probability modes
PROB_RAW_AGGREGATION = 'raw_aggregation'

# GUI defaults
DEFAULT_WINDOW_WIDTH = 1920
DEFAULT_WINDOW_HEIGHT = 1000
MIN_WINDOW_WIDTH = 1600
MIN_WINDOW_HEIGHT = 800

# Performance thresholds
MIN_SHARPE_WARNING = 0.5
MAX_DRAWDOWN_WARNING = 0.2
MIN_TRADES_WARNING = 10