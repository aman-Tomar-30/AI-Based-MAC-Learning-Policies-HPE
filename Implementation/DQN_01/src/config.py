# NETWORK ARCHITECTURE
# ─────────────────────────────────────────
INPUT_SIZE   = 3
HIDDEN_SIZE  = 64
OUTPUT_SIZE  = 7

# REPLAY BUFFER
# ─────────────────────────────────────────
BUFFER_CAPACITY = 10000
BATCH_SIZE      = 64
MIN_BUFFER_SIZE = 500

# TRAINING HYPERPARAMETERS
# ─────────────────────────────────────────
LEARNING_RATE   = 0.001
GAMMA           = 0.99
TARGET_UPDATE_N = 300

# EPSILON  
# ─────────────────────────────────────────
EPSILON_START = 1.0
EPSILON_END   = 0.05
EPSILON_DECAY = 0.997

# NORMALIZATION CONSTANTS
# ─────────────────────────────────────────
MAC_TABLE_CAPACITY = 20
MAX_FLOOD_RATE     = 1000
MAX_AGING_TIMER    = 300

# REWARD VALUES
# ─────────────────────────────────────────
REWARD_CORRECT_DELIVERY      = +1.0
REWARD_FLOODING_AVOIDED      = +0.8
REWARD_PREVENTIVE_EVICTION   = +0.3
REWARD_NECESSARY_FLOOD       = -0.1
REWARD_WRONG_PORT_STALE      = -0.5
REWARD_EARLY_ACTIVE_EVICTION = -0.3
REWARD_TABLE_OVERFLOW        = -0.7
REWARD_UNNECESSARY_FLOOD     = -1.0

# THRESHOLDS
# ─────────────────────────────────────────
TABLE_PREVENTIVE_THRESHOLD  = 0.80
TABLE_OVERFLOW_THRESHOLD    = 0.95
ENTRY_ACTIVE_AGE_THRESHOLD  = 0.20
AGING_INCREASE_STEP         = 0.10
AGING_DECREASE_STEP         = 0.10
MIN_AGING_TIMER_NORM        = 0.10
MAX_AGING_TIMER_NORM        = 1.0

# TRAINING EPISODES
# ─────────────────────────────────────────
MAX_EPISODES = 1000
MAX_STEPS    = 200