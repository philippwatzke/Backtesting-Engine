import numpy as np

# --- MNQ Instrument Constants ---
MNQ_TICK_SIZE = 0.25
MNQ_TICK_VALUE = 0.50
MNQ_COMMISSION_PER_SIDE = 0.54

# --- Session Constants ---
BARS_PER_RTH_SESSION = 390  # 09:30 to 16:00 ET = 390 one-minute bars
HARD_CLOSE_MINUTE = 389     # minute_of_day index for 15:59 (0-based from 09:30)
MOC_ENTRY_MINUTE = 330      # 15:00 ET on the 5-minute resampled session grid

# --- Signal Codes ---
SIGNAL_LONG = 1
SIGNAL_SHORT = -1
SIGNAL_NONE = 0
SIGNAL_MOC_LONG = SIGNAL_LONG
SIGNAL_MOC_SHORT = SIGNAL_SHORT
SIGNAL_PULLBACK_LONG = SIGNAL_LONG
SIGNAL_PULLBACK_SHORT = SIGNAL_SHORT
SIGNAL_POC_BREAKOUT_LONG = 2
SIGNAL_POC_BREAKOUT_SHORT = -2

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
PARAMS_DISTANCE_TICKS = 11
PARAMS_SMA_PERIOD = 12
PARAMS_BREAKEVEN_TRIGGER_TICKS = 13
PARAMS_BAND_MULTIPLIER = 14
PARAMS_POC_LOOKBACK = 15
PARAMS_EXTRA_SLIPPAGE_TICKS = 16
PARAMS_ENTRY_MINUTE = 17
PARAMS_TREND_THRESHOLD_PCT = 18
PARAMS_TRIGGER_START_MINUTE = 19
PARAMS_TRIGGER_END_MINUTE = 20
PARAMS_TIME_STOP_MINUTE = 21
PARAMS_TICK_SIZE = 22
PARAMS_TICK_VALUE = 23
PARAMS_MIN_RVOL = 24
PARAMS_MAX_RVOL = 25
PARAMS_TRAIL_BAR_EXTREME = 26
PARAMS_BLOCKED_WEEKDAY = 27
PARAMS_ENTRY_ON_CLOSE = 28
PARAMS_ARRAY_LENGTH = 29

# --- Portfolio Strategy Profile Index Constants ---
PROFILE_RISK_PER_TRADE_USD = 0
PROFILE_STOP_ATR_MULTIPLIER = 1
PROFILE_TARGET_ATR_MULTIPLIER = 2
PROFILE_BREAKEVEN_TRIGGER_TICKS = 3
PROFILE_RISK_BUFFER_FRACTION = 4
PROFILE_MOC_FLOW = 0
PROFILE_VWAP_PULLBACK = 0
PROFILE_VWAP_POC_BREAKOUT = 1
PROFILE_ARRAY_LENGTH = 5
