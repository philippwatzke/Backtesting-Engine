import numpy as np

# --- MNQ Instrument Constants ---
MNQ_TICK_SIZE = 0.25
MNQ_TICK_VALUE = 0.50
MNQ_COMMISSION_PER_SIDE = 0.54

# --- Session Constants ---
BARS_PER_RTH_SESSION = 390  # 09:30 to 16:00 ET = 390 one-minute bars
HARD_CLOSE_MINUTE = 389     # minute_of_day index for 15:59 (0-based from 09:30)

# --- Signal Codes ---
SIGNAL_LONG = 1
SIGNAL_SHORT = -1
SIGNAL_NONE = 0

# --- Exit Reason Codes ---
EXIT_TARGET = 0
EXIT_STOP = 1
EXIT_HARD_CLOSE = 2
EXIT_CIRCUIT_BREAKER = 3

# --- Trade Log Structured Array ---
TRADE_LOG_DTYPE = np.dtype([
    ("day_id", "i4"),
    ("phase_id", "i1"),
    ("payout_cycle_id", "i2"),
    ("entry_time", "i8"),
    ("exit_time", "i8"),
    ("entry_price", "f8"),
    ("exit_price", "f8"),
    ("entry_slippage", "f8"),
    ("exit_slippage", "f8"),
    ("entry_commission", "f8"),
    ("exit_commission", "f8"),
    ("contracts", "i4"),
    ("gross_pnl", "f8"),
    ("net_pnl", "f8"),
    ("signal_type", "i1"),
    ("exit_reason", "i1"),
])

# --- Daily Lifecycle Log Structured Array ---
DAILY_LOG_DTYPE = np.dtype([
    ("day_id", "i4"),
    ("phase_id", "i1"),
    ("payout_cycle_id", "i2"),
    ("had_trade", "i1"),
    ("n_trades", "i2"),
    ("day_pnl", "f8"),
    ("net_payout", "f8"),
])

# --- Slippage Floor ---
SLIPPAGE_FLOOR_POINTS = 0.25   # Minimum 1 tick slippage

# --- Params Array Index Constants ---
PARAMS_RANGE_MINUTES = 0
PARAMS_STOP_TICKS = 1
PARAMS_TARGET_TICKS = 2
PARAMS_CONTRACTS = 3
PARAMS_DAILY_STOP = 4
PARAMS_DAILY_TARGET = 5
PARAMS_MAX_TRADES = 6
PARAMS_BUFFER_TICKS = 7
PARAMS_VOL_THRESHOLD = 8
PARAMS_STOP_PENALTY = 9
PARAMS_COMMISSION = 10
PARAMS_ARRAY_LENGTH = 11
