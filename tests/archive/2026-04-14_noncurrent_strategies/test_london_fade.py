"""Tests for the London Fade strategy module.

Coverage
--------
TestHhmmToMinute          (3)  — time-string helper
TestComputeFadeSignals     (9)  — signal generation
TestRunLondonFadeSession   (9)  — per-session execution kernel
TestRunLondonFadeBacktest  (9)  — full multi-day backtest runner
"""

import numpy as np
import pytest

from propfirm.core.types import (
    DAILY_LOG_DTYPE,
    EXIT_HARD_CLOSE,
    EXIT_STOP,
    EXIT_TARGET,
    MNQ_TICK_SIZE,
    MNQ_TICK_VALUE,
    SIGNAL_LONG,
    SIGNAL_NONE,
    SIGNAL_SHORT,
    TRADE_LOG_DTYPE,
)
from propfirm.market.slippage import build_slippage_lookup
from propfirm.strategy.london_fade import (
    _hhmm_to_minute,
    _init_day_state,
    compute_fade_signals,
    run_london_fade_backtest,
    run_london_fade_session,
)

# ---------------------------------------------------------------------------
# Shared constants and helpers
# ---------------------------------------------------------------------------

BARS_PER_DAY = 32          # 15-min bars, 08:00–15:59
SESSION_MINUTES = 480      # minute count for 08:00–15:59
ATR = 5.0
BASE_PRICE = 20_000.0
TRIG_LOCAL = 14            # bar whose minute_of_day == 210 (11:30 ET offset)
ENTRY_LOCAL = 15           # bar after trigger → entry bar

_MOD_PER_DAY = np.arange(0, BARS_PER_DAY * 15, 15, dtype=np.int16)

BASE_CFG = {
    "session_start":    "08:00",
    "eval_start_time":  "08:00",
    "trigger_time":     "11:30",
    "min_trend_atr":    2.0,
    "stop_atr":         1.0,
    "rr_ratio":         1.5,
    "max_trades":       1,
    "time_stop_minute": 360,
    "daily_stop":       -500.0,
    "daily_target":     800.0,
    "contracts":        1,
    "commission_per_side": 0.54,
    "stop_penalty":     1.5,
    "extra_slippage_ticks": 0.0,
    "tick_size":        MNQ_TICK_SIZE,
    "tick_value":       MNQ_TICK_VALUE,
    "starting_equity":  50_000.0,
    "phase_id":         0,
    "payout_cycle_id":  -1,
}


def _make_slippage_lookup() -> np.ndarray:
    return build_slippage_lookup(None, session_minutes=SESSION_MINUTES)


def _make_state(n_cap: int = 20) -> tuple[dict, np.ndarray]:
    """Return (state_dict, trade_log) for single-day session tests."""
    slippage_lookup = _make_slippage_lookup()
    trade_log = np.zeros(n_cap, dtype=TRADE_LOG_DTYPE)
    state = {
        "equity":               50_000.0,
        "intraday_pnl":         0.0,
        "position":             0,
        "entry_price":          0.0,
        "stop_level":           0.0,
        "target_level":         0.0,
        "pending_signal":       SIGNAL_NONE,
        "pending_stop_dist":    np.nan,
        "daily_trade_count":    0,
        "open_trade_idx":       -1,
        "trade_idx":            0,
        "trade_log":            trade_log,
        "current_day_id":       0,
        "current_phase_id":     0,
        "current_payout_cycle_id": -1,
        "tick_size":            MNQ_TICK_SIZE,
        "tick_value":           MNQ_TICK_VALUE,
        "commission_per_side":  0.54,
        "stop_penalty":         1.5,
        "extra_slippage_points": 0.0,
        "minute_of_day":        None,
        "bar_atr":              None,
        "trailing_atr":         None,
        "slippage_lookup":      slippage_lookup,
    }
    return state, trade_log


def _make_session_arrays(
    open_price: float = BASE_PRICE,
    *,
    signal: int = SIGNAL_SHORT,
    stop_dist_value: float = ATR,  # stop_atr * ATR
    # override specific bars; key = bar index, value = (open, high, low, close)
    bar_overrides: dict | None = None,
) -> tuple:
    """Build one session's OHLC + signal arrays.

    Returns
    -------
    (opens, highs, lows, closes, timestamps, minute_of_day,
     bar_atr, trailing_atr, slippage_lookup, signals, stop_dists)
    """
    N = BARS_PER_DAY
    opens    = np.full(N, open_price, dtype=np.float64)
    highs    = np.full(N, open_price + 3.0, dtype=np.float64)
    lows     = np.full(N, open_price - 3.0, dtype=np.float64)
    closes   = np.full(N, open_price, dtype=np.float64)
    bar_atr  = np.full(N, ATR, dtype=np.float64)
    trailing_atr = np.full(N, ATR, dtype=np.float64)
    timestamps = np.arange(N, dtype=np.int64) * (15 * 60 * 1_000_000_000)
    mod = _MOD_PER_DAY.copy()

    if bar_overrides:
        for bar_idx, (o, h, l, c) in bar_overrides.items():
            opens[bar_idx]  = o
            highs[bar_idx]  = h
            lows[bar_idx]   = l
            closes[bar_idx] = c

    signals    = np.full(N, SIGNAL_NONE, dtype=np.int8)
    stop_dists = np.full(N, np.nan, dtype=np.float64)
    signals[TRIG_LOCAL]    = signal
    stop_dists[TRIG_LOCAL] = stop_dist_value

    slippage_lookup = _make_slippage_lookup()
    return (
        opens, highs, lows, closes, timestamps, mod,
        bar_atr, trailing_atr, slippage_lookup,
        signals, stop_dists,
    )


def _run_session(
    *,
    signal: int = SIGNAL_SHORT,
    bar_overrides: dict | None = None,
    stop_dist_value: float = ATR,
    cfg_overrides: dict | None = None,
) -> tuple[dict, np.ndarray]:
    """Run a complete single-session fade. Returns (state, trade_log)."""
    state, trade_log = _make_state()
    cfg = dict(BASE_CFG)
    if cfg_overrides:
        cfg.update(cfg_overrides)

    (opens, highs, lows, closes, timestamps, mod,
     bar_atr, trailing_atr, slippage_lookup, signals, stop_dists) = _make_session_arrays(
        signal=signal,
        stop_dist_value=stop_dist_value,
        bar_overrides=bar_overrides,
    )
    run_london_fade_session(
        opens, highs, lows, closes, timestamps, mod,
        bar_atr, trailing_atr, slippage_lookup,
        signals, stop_dists,
        state, cfg,
    )
    return state, trade_log


def _make_backtest_data(n_days: int = 5) -> dict:
    """Build a synthetic data_dict suitable for run_london_fade_backtest.

    Day 0           : stagnant (no signal)
    Days 1, 3       : up-move → SHORT  (target hit on TRIG_LOCAL+2)
    Days 2, 4       : down-move → LONG (target hit on TRIG_LOCAL+2)
    """
    N = n_days * BARS_PER_DAY

    opens    = np.full(N, BASE_PRICE, dtype=np.float64)
    closes   = np.full(N, BASE_PRICE, dtype=np.float64)
    highs    = np.full(N, BASE_PRICE + 3.0, dtype=np.float64)
    lows     = np.full(N, BASE_PRICE - 3.0, dtype=np.float64)
    bar_atr  = np.full(N, ATR, dtype=np.float64)
    trailing_atr = np.full(N, ATR, dtype=np.float64)
    timestamps   = np.arange(N, dtype=np.int64) * (15 * 60 * 1_000_000_000)
    minute_of_day = np.tile(_MOD_PER_DAY, n_days)

    day_boundaries = [
        (i * BARS_PER_DAY, (i + 1) * BARS_PER_DAY) for i in range(n_days)
    ]

    THRESHOLD = 2.0 * ATR  # = 10.0
    # Day 0: stagnant
    closes[0 * BARS_PER_DAY + TRIG_LOCAL] = BASE_PRICE + 5.0  # < 10

    for d_idx, sig in [(1, "short"), (2, "long"), (3, "short"), (4, "long")]:
        if d_idx >= n_days:
            continue
        d_start = d_idx * BARS_PER_DAY
        trig_abs   = d_start + TRIG_LOCAL
        entry_abs  = d_start + ENTRY_LOCAL       # bar 15
        exit_abs   = d_start + TRIG_LOCAL + 2    # bar 16

        if sig == "short":
            closes[trig_abs] = BASE_PRICE + 15.0  # +15 > +10 → SHORT
            # keep entry bar safe (no immediate stop/target)
            highs[entry_abs] = BASE_PRICE + 2.0
            lows[entry_abs]  = BASE_PRICE - 2.0
            # exit bar: low hits target_level (≈ BASE − 7.75)
            highs[exit_abs]  = BASE_PRICE + 2.0
            lows[exit_abs]   = BASE_PRICE - 10.0
        else:
            closes[trig_abs] = BASE_PRICE - 15.0  # −15 < −10 → LONG
            highs[entry_abs] = BASE_PRICE + 2.0
            lows[entry_abs]  = BASE_PRICE - 2.0
            # exit bar: high hits target_level (≈ BASE + 7.75)
            highs[exit_abs]  = BASE_PRICE + 10.0
            lows[exit_abs]   = BASE_PRICE - 2.0

    return {
        "open":               opens,
        "high":               highs,
        "low":                lows,
        "close":              closes,
        "bar_atr":            bar_atr,
        "timestamps":         timestamps,
        "minute_of_day":      minute_of_day,
        "trailing_median_atr": trailing_atr,
        "day_boundaries":     day_boundaries,
        "session_minutes":    SESSION_MINUTES,
        "timeframe_minutes":  15,
        "bars_per_session":   BARS_PER_DAY,
    }


# ---------------------------------------------------------------------------
# TestHhmmToMinute
# ---------------------------------------------------------------------------

class TestHhmmToMinute:
    def test_midnight(self):
        assert _hhmm_to_minute("00:00") == 0

    def test_session_open(self):
        assert _hhmm_to_minute("08:00") == 480

    def test_trigger_time(self):
        assert _hhmm_to_minute("11:30") == 690
        # Offset from 08:00 → 690 − 480 = 210
        offset = _hhmm_to_minute("11:30") - _hhmm_to_minute("08:00")
        assert offset == 210


# ---------------------------------------------------------------------------
# TestComputeFadeSignals
# ---------------------------------------------------------------------------

class TestComputeFadeSignals:
    """Signal generation: compute_fade_signals correctness."""

    def _make_one_day(
        self,
        trig_close: float,
        anchor_open: float = BASE_PRICE,
        atr: float = ATR,
    ) -> dict:
        """Single-day signal data."""
        N = BARS_PER_DAY
        opens  = np.full(N, anchor_open, dtype=np.float64)
        closes = np.full(N, anchor_open, dtype=np.float64)
        closes[TRIG_LOCAL] = trig_close
        bar_atr = np.full(N, atr, dtype=np.float64)
        mod = _MOD_PER_DAY.copy()
        return {
            "opens": opens,
            "closes": closes,
            "minute_of_day": mod,
            "bar_atr": bar_atr,
            "day_boundaries": [(0, N)],
        }

    def test_stagnant_day_no_signal(self):
        d = self._make_one_day(BASE_PRICE + 5.0)  # delta=5 < threshold=10
        result = compute_fade_signals(
            d["opens"], d["closes"], d["minute_of_day"], d["bar_atr"],
            d["day_boundaries"], BASE_CFG,
        )
        assert np.all(result["signal"] == SIGNAL_NONE)
        assert np.all(np.isnan(result["stop_dist"]))

    def test_up_move_short_signal(self):
        d = self._make_one_day(BASE_PRICE + 15.0)  # delta=15 > 10
        result = compute_fade_signals(
            d["opens"], d["closes"], d["minute_of_day"], d["bar_atr"],
            d["day_boundaries"], BASE_CFG,
        )
        assert result["signal"][TRIG_LOCAL] == SIGNAL_SHORT
        assert result["stop_dist"][TRIG_LOCAL] == pytest.approx(ATR * 1.0, rel=1e-9)

    def test_down_move_long_signal(self):
        d = self._make_one_day(BASE_PRICE - 15.0)  # delta=-15 < -10
        result = compute_fade_signals(
            d["opens"], d["closes"], d["minute_of_day"], d["bar_atr"],
            d["day_boundaries"], BASE_CFG,
        )
        assert result["signal"][TRIG_LOCAL] == SIGNAL_LONG
        assert result["stop_dist"][TRIG_LOCAL] == pytest.approx(ATR * 1.0, rel=1e-9)

    def test_exact_threshold_no_signal(self):
        # trend_distance == threshold is NOT > threshold → no signal
        threshold = 2.0 * ATR  # = 10.0
        d = self._make_one_day(BASE_PRICE + threshold)
        result = compute_fade_signals(
            d["opens"], d["closes"], d["minute_of_day"], d["bar_atr"],
            d["day_boundaries"], BASE_CFG,
        )
        assert result["signal"][TRIG_LOCAL] == SIGNAL_NONE

    def test_signal_only_at_trigger_bar(self):
        d = self._make_one_day(BASE_PRICE + 15.0)
        result = compute_fade_signals(
            d["opens"], d["closes"], d["minute_of_day"], d["bar_atr"],
            d["day_boundaries"], BASE_CFG,
        )
        non_trigger = np.arange(BARS_PER_DAY) != TRIG_LOCAL
        assert np.all(result["signal"][non_trigger] == SIGNAL_NONE)
        assert np.all(np.isnan(result["stop_dist"][non_trigger]))

    def test_zero_atr_no_signal(self):
        N = BARS_PER_DAY
        opens  = np.full(N, BASE_PRICE, dtype=np.float64)
        closes = np.full(N, BASE_PRICE + 15.0, dtype=np.float64)
        bar_atr = np.zeros(N, dtype=np.float64)  # ATR=0
        mod = _MOD_PER_DAY.copy()
        result = compute_fade_signals(
            opens, closes, mod, bar_atr, [(0, N)], BASE_CFG,
        )
        assert np.all(result["signal"] == SIGNAL_NONE)

    def test_multi_day_correct_bars(self):
        n_days = 3
        opens  = np.full(n_days * BARS_PER_DAY, BASE_PRICE, dtype=np.float64)
        closes = np.full(n_days * BARS_PER_DAY, BASE_PRICE, dtype=np.float64)
        bar_atr = np.full(n_days * BARS_PER_DAY, ATR, dtype=np.float64)
        mod = np.tile(_MOD_PER_DAY, n_days)
        day_bounds = [(i * BARS_PER_DAY, (i + 1) * BARS_PER_DAY) for i in range(n_days)]

        # Day 0: stagnant, Day 1: SHORT, Day 2: LONG
        closes[1 * BARS_PER_DAY + TRIG_LOCAL] = BASE_PRICE + 15.0
        closes[2 * BARS_PER_DAY + TRIG_LOCAL] = BASE_PRICE - 15.0

        result = compute_fade_signals(opens, closes, mod, bar_atr, day_bounds, BASE_CFG)

        assert result["signal"][0 * BARS_PER_DAY + TRIG_LOCAL] == SIGNAL_NONE
        assert result["signal"][1 * BARS_PER_DAY + TRIG_LOCAL] == SIGNAL_SHORT
        assert result["signal"][2 * BARS_PER_DAY + TRIG_LOCAL] == SIGNAL_LONG

    def test_stop_dist_equals_stop_atr_times_atr(self):
        cfg = dict(BASE_CFG, stop_atr=2.0)
        d = self._make_one_day(BASE_PRICE + 15.0)
        result = compute_fade_signals(
            d["opens"], d["closes"], d["minute_of_day"], d["bar_atr"],
            d["day_boundaries"], cfg,
        )
        assert result["stop_dist"][TRIG_LOCAL] == pytest.approx(2.0 * ATR, rel=1e-9)

    def test_higher_threshold_suppresses_signal(self):
        # min_trend_atr=4.0 → threshold = 20.0; delta=15 is insufficient
        cfg = dict(BASE_CFG, min_trend_atr=4.0)
        d = self._make_one_day(BASE_PRICE + 15.0)
        result = compute_fade_signals(
            d["opens"], d["closes"], d["minute_of_day"], d["bar_atr"],
            d["day_boundaries"], cfg,
        )
        assert result["signal"][TRIG_LOCAL] == SIGNAL_NONE


# ---------------------------------------------------------------------------
# TestRunLondonFadeSession
# ---------------------------------------------------------------------------

class TestRunLondonFadeSession:
    """Per-session execution kernel tests."""

    def _entry_slip(self) -> float:
        """Compute the expected entry slippage for mod=225 (entry bar)."""
        lookup = _make_slippage_lookup()
        # mod=225 is in the scaled (120,240,0.75) bucket → baseline~0.9375
        # raw = baseline * (ATR/ATR) * 1.0 * tick_size = 0.9375*0.25 ≈ 0.234 < floor(0.25)
        # → slippage = 0.25
        from propfirm.market.slippage import compute_slippage
        return compute_slippage(225, ATR, ATR, lookup, False, 1.5, MNQ_TICK_SIZE, 0.0)

    def test_short_entry_fill_price(self):
        state, trade_log = _run_session(signal=SIGNAL_SHORT)
        assert state["trade_idx"] == 1
        slip = self._entry_slip()
        expected_fill = BASE_PRICE - slip
        assert float(trade_log[0]["entry_price"]) == pytest.approx(expected_fill, rel=1e-9)
        assert float(trade_log[0]["entry_slippage"]) == pytest.approx(slip, rel=1e-9)

    def test_long_entry_fill_price(self):
        state, trade_log = _run_session(signal=SIGNAL_LONG)
        assert state["trade_idx"] == 1
        slip = self._entry_slip()
        expected_fill = BASE_PRICE + slip
        assert float(trade_log[0]["entry_price"]) == pytest.approx(expected_fill, rel=1e-9)
        assert float(trade_log[0]["entry_slippage"]) == pytest.approx(slip, rel=1e-9)

    def test_short_target_exit_rr(self):
        """SHORT EXIT_TARGET: exit_price − exit_slippage == target_level."""
        # Bar 16: low deep enough to hit target (BASE − 10 < target ≈ BASE − 7.75)
        overrides = {
            ENTRY_LOCAL:     (BASE_PRICE, BASE_PRICE + 2.0, BASE_PRICE - 2.0, BASE_PRICE),
            ENTRY_LOCAL + 1: (BASE_PRICE, BASE_PRICE + 2.0, BASE_PRICE - 10.0, BASE_PRICE - 10.0),
        }
        state, trade_log = _run_session(signal=SIGNAL_SHORT, bar_overrides=overrides)
        assert trade_log[0]["exit_reason"] == EXIT_TARGET
        fill = float(trade_log[0]["entry_price"])
        slip = self._entry_slip()
        stop_dist = 1.0 * ATR  # stop_atr * ATR
        expected_target = fill - 1.5 * stop_dist
        actual_target = float(trade_log[0]["exit_price"]) - float(trade_log[0]["exit_slippage"])
        assert actual_target == pytest.approx(expected_target, rel=1e-9)

    def test_long_target_exit_rr(self):
        """LONG EXIT_TARGET: exit_price + exit_slippage == target_level."""
        overrides = {
            ENTRY_LOCAL:     (BASE_PRICE, BASE_PRICE + 2.0, BASE_PRICE - 2.0, BASE_PRICE),
            ENTRY_LOCAL + 1: (BASE_PRICE, BASE_PRICE + 10.0, BASE_PRICE - 2.0, BASE_PRICE + 10.0),
        }
        state, trade_log = _run_session(signal=SIGNAL_LONG, bar_overrides=overrides)
        assert trade_log[0]["exit_reason"] == EXIT_TARGET
        fill = float(trade_log[0]["entry_price"])
        stop_dist = 1.0 * ATR
        expected_target = fill + 1.5 * stop_dist
        actual_target = float(trade_log[0]["exit_price"]) + float(trade_log[0]["exit_slippage"])
        assert actual_target == pytest.approx(expected_target, rel=1e-9)

    def test_short_stop_intrabar(self):
        """SHORT stop: bar high >= stop_level → EXIT_STOP."""
        # fill ≈ 20000 − 0.25 = 19999.75; stop = 19999.75 + 5 = 20004.75
        # Bar 16: high = 20010 ≥ 20004.75 → EXIT_STOP
        overrides = {
            ENTRY_LOCAL:     (BASE_PRICE, BASE_PRICE + 2.0, BASE_PRICE - 2.0, BASE_PRICE),
            ENTRY_LOCAL + 1: (BASE_PRICE, BASE_PRICE + 10.0, BASE_PRICE - 2.0, BASE_PRICE),
        }
        state, trade_log = _run_session(signal=SIGNAL_SHORT, bar_overrides=overrides)
        assert trade_log[0]["exit_reason"] == EXIT_STOP

    def test_long_stop_intrabar(self):
        """LONG stop: bar low <= stop_level → EXIT_STOP."""
        # fill ≈ 20000.25; stop = 20000.25 - 5 = 19995.25
        # Bar 16: low = 19990 ≤ 19995.25 → EXIT_STOP
        overrides = {
            ENTRY_LOCAL:     (BASE_PRICE, BASE_PRICE + 2.0, BASE_PRICE - 2.0, BASE_PRICE),
            ENTRY_LOCAL + 1: (BASE_PRICE, BASE_PRICE + 2.0, BASE_PRICE - 10.0, BASE_PRICE),
        }
        state, trade_log = _run_session(signal=SIGNAL_LONG, bar_overrides=overrides)
        assert trade_log[0]["exit_reason"] == EXIT_STOP

    def test_hard_close_last_bar(self):
        """No stop/target hit → hard-close on session's last bar."""
        # All bars safe: highs = BASE+3, lows = BASE-3 (no stop or target hit)
        state, trade_log = _run_session(
            signal=SIGNAL_SHORT,
            cfg_overrides={"time_stop_minute": 9999},  # disable time_stop
        )
        assert trade_log[0]["exit_reason"] == EXIT_HARD_CLOSE
        assert int(trade_log[0]["exit_time"]) == int(
            np.arange(BARS_PER_DAY, dtype=np.int64)[-1] * (15 * 60 * 1_000_000_000)
        )

    def test_time_stop_hard_close(self):
        """Position open at time_stop_minute offset → EXIT_HARD_CLOSE."""
        # time_stop = 240 (mod 240 = bar 16 in day); entry at bar 15
        state, trade_log = _run_session(
            signal=SIGNAL_SHORT,
            cfg_overrides={"time_stop_minute": 240},
        )
        assert trade_log[0]["exit_reason"] == EXIT_HARD_CLOSE

    def test_max_trades_respected(self):
        """After one trade, no further entries even if another signal arrives."""
        state, trade_log = _make_state()
        cfg = dict(BASE_CFG, max_trades=1)
        N = BARS_PER_DAY
        opens    = np.full(N, BASE_PRICE, dtype=np.float64)
        highs    = np.full(N, BASE_PRICE + 3.0, dtype=np.float64)
        lows     = np.full(N, BASE_PRICE - 3.0, dtype=np.float64)
        closes   = np.full(N, BASE_PRICE, dtype=np.float64)
        bar_atr  = np.full(N, ATR, dtype=np.float64)
        trailing_atr = np.full(N, ATR, dtype=np.float64)
        timestamps   = np.arange(N, dtype=np.int64) * (15 * 60 * 1_000_000_000)
        mod = _MOD_PER_DAY.copy()
        # Two signals: at bar 14 and bar 20
        signals    = np.full(N, SIGNAL_NONE, dtype=np.int8)
        stop_dists = np.full(N, np.nan, dtype=np.float64)
        signals[TRIG_LOCAL]  = SIGNAL_SHORT
        stop_dists[TRIG_LOCAL] = ATR
        signals[20]   = SIGNAL_SHORT
        stop_dists[20] = ATR
        run_london_fade_session(
            opens, highs, lows, closes, timestamps, mod,
            bar_atr, trailing_atr, state["slippage_lookup"],
            signals, stop_dists, state, cfg,
        )
        # Only one trade should exist (max_trades=1)
        assert state["trade_idx"] == 1


# ---------------------------------------------------------------------------
# TestRunLondonFadeBacktest
# ---------------------------------------------------------------------------

class TestRunLondonFadeBacktest:
    """Full multi-day backtest runner tests."""

    def test_returns_correct_dtypes(self):
        data_dict = _make_backtest_data()
        trade_log, daily_log = run_london_fade_backtest(data_dict, BASE_CFG)
        assert trade_log.dtype == TRADE_LOG_DTYPE
        assert daily_log.dtype == DAILY_LOG_DTYPE

    def test_daily_log_length(self):
        n_days = 5
        data_dict = _make_backtest_data(n_days)
        _, daily_log = run_london_fade_backtest(data_dict, BASE_CFG)
        assert len(daily_log) == n_days

    def test_sequential_day_ids(self):
        data_dict = _make_backtest_data(5)
        _, daily_log = run_london_fade_backtest(data_dict, BASE_CFG)
        for i in range(5):
            assert daily_log[i]["day_id"] == i

    def test_all_trades_closed(self):
        """Every trade must have a non-zero exit_time."""
        data_dict = _make_backtest_data(5)
        trade_log, _ = run_london_fade_backtest(data_dict, BASE_CFG)
        assert len(trade_log) > 0
        assert np.all(trade_log["exit_time"] != 0)

    def test_nonzero_entry_prices(self):
        data_dict = _make_backtest_data(5)
        trade_log, _ = run_london_fade_backtest(data_dict, BASE_CFG)
        assert len(trade_log) > 0
        assert np.all(trade_log["entry_price"] != 0.0)

    def test_net_pnl_equals_gross_minus_commissions(self):
        data_dict = _make_backtest_data(5)
        trade_log, _ = run_london_fade_backtest(data_dict, BASE_CFG)
        for t in trade_log:
            total_commission = float(t["entry_commission"]) + float(t["exit_commission"])
            expected_net = float(t["gross_pnl"]) - total_commission
            assert float(t["net_pnl"]) == pytest.approx(expected_net, abs=1e-9)

    def test_n_trades_sum_matches(self):
        data_dict = _make_backtest_data(5)
        trade_log, daily_log = run_london_fade_backtest(data_dict, BASE_CFG)
        assert int(np.sum(daily_log["n_trades"])) == len(trade_log)

    def test_max_trades_per_day_respected(self):
        data_dict = _make_backtest_data(5)
        trade_log, daily_log = run_london_fade_backtest(data_dict, BASE_CFG)
        for row in daily_log:
            assert int(row["n_trades"]) <= BASE_CFG["max_trades"]

    def test_stagnant_day_has_no_trade(self):
        """Day 0 is stagnant → no trade on that day."""
        data_dict = _make_backtest_data(5)
        _, daily_log = run_london_fade_backtest(data_dict, BASE_CFG)
        assert int(daily_log[0]["had_trade"]) == 0
        assert int(daily_log[0]["n_trades"]) == 0
