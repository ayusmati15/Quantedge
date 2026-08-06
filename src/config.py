"""
===========================================================
Quantedge Configuration File
===========================================================

Centralized configuration for the entire project.

Modify values here instead of changing code inside modules.

Author : Ayusmati Panda
Project : Quantedge
"""

# ==========================================================
# DATA SETTINGS
# ==========================================================

MARKET_SYMBOL = "^NSEI"
START_DATE = "2015-01-01"

END_DATE = None          # None -> today's date
TRAIN_RATIO = 0.80
VALIDATION_RATIO = 0.10
TEST_RATIO = 0.10
RANDOM_STATE = 42

# ==========================================================
# FEATURE ENGINEERING
# ==========================================================

MA_SHORT = 10
MA_LONG = 20
EMA_SHORT = 12
EMA_LONG = 26
RSI_PERIOD = 14
VOLATILITY_WINDOW = 20
MOMENTUM_WINDOW = 5
LAG_FEATURES = [1, 2, 3, 5]
ROLLING_WINDOW = 20
# ==========================================================
# SIGNAL GENERATION
# ==========================================================

BUY_THRESHOLD = 0.60
SELL_THRESHOLD = 0.40
CONFIDENCE_THRESHOLD = 0.70
NO_TRADE_ZONE = 0.05

# ==========================================================
# RANDOM FOREST
# ==========================================================

RF_N_ESTIMATORS = 300
RF_MAX_DEPTH = 8
RF_MIN_SAMPLES_SPLIT = 10
RF_MIN_SAMPLES_LEAF = 5

# ==========================================================
# ANN
# ==========================================================

ANN_EPOCHS = 30
ANN_BATCH_SIZE = 64
ANN_LEARNING_RATE = 0.001
ANN_DROPOUT = 0.30

# ==========================================================
# LSTM
# ==========================================================

LSTM_WINDOW = 20
LSTM_UNITS = 128
LSTM_DROPOUT = 0.30
LSTM_BATCH_SIZE = 64
LSTM_EPOCHS = 25

# ==========================================================
# TRANSFORMER
# ==========================================================

TRANSFORMER_SEQUENCE_LENGTH = 20

TRANSFORMER_EMBED_DIM = 64

TRANSFORMER_HEADS = 4

TRANSFORMER_FEED_FORWARD = 128

TRANSFORMER_LAYERS = 2

TRANSFORMER_DROPOUT = 0.20

TRANSFORMER_BATCH_SIZE = 64

TRANSFORMER_EPOCHS = 20

# ==========================================================
# WALK FORWARD VALIDATION
# ==========================================================

WF_TRAIN_SIZE = 500

WF_TEST_SIZE = 50

WF_STEP_SIZE = 50

# ==========================================================
# PORTFOLIO SETTINGS
# ==========================================================

INITIAL_CAPITAL = 1_000_000

RISK_FREE_RATE = 0.05

MAX_PORTFOLIOS = 10000

MAX_POSITION_SIZE = 0.25

TRANSACTION_COST = 0.0005

SLIPPAGE = 0.0002

# ==========================================================
# VALUE AT RISK
# ==========================================================

VAR_CONFIDENCE = 0.95

MONTE_CARLO_SIMULATIONS = 10000

GBM_PATHS = 500

GBM_DAYS = 252

# ==========================================================
# REGIME DETECTION
# ==========================================================

REGIME_WINDOW = 20

BULLISH = "bullish"

BEARISH = "bearish"

SIDEWAYS = "sideways"

UNKNOWN = "unknown"

# ==========================================================
# LOGGING
# ==========================================================

VERBOSE = True

SAVE_MODELS = True

SAVE_PLOTS = True

SAVE_REPORTS = True

# ==========================================================
# PATHS
# ==========================================================

MODEL_DIRECTORY = "models"

PLOT_DIRECTORY = "plots"

REPORT_DIRECTORY = "reports"

DATA_DIRECTORY = "data"

# ==========================================================
# MODEL NAMES
# ==========================================================

RF_MODEL_NAME = "random_forest.pkl"

ANN_MODEL_NAME = "ann.keras"

LSTM_MODEL_NAME = "lstm.keras"

TRANSFORMER_MODEL_NAME = "transformer.keras"

# ==========================================================
# PLOTTING
# ==========================================================

FIGURE_WIDTH = 12

FIGURE_HEIGHT = 6

DPI = 120

# ==========================================================
# COLORS
# ==========================================================

BUY_COLOR = "green"

SELL_COLOR = "red"

HOLD_COLOR = "gray"

PORTFOLIO_COLOR = "blue"

MARKET_COLOR = "orange"

# ==========================================================
# GLOBAL NUMPY SEED
# ==========================================================

SEED = 42
