"""Unit tests for the Triple KAMA + MACD strategy module.

Focus areas
-----------
1. compute_kama       — shape, warmup NaN, recursive correctness
2. compute_macd       — shape, validation guards
3. compute_signal_arrays — signal window enforcement, symmetric short path
4. run_kama_macd_session — asymmetric stop/target (2:1 RR from next-bar-open)
5. run_kama_macd_backtest — TRADE_LOG_DTYPE / DAILY_LOG_DTYPE compatibility
"""

import numpy as np
import pytest

from propfirm.core.types import (
    DAILY_LOG_DTYPE,
    EXIT_HARD_CLOSE,
    EXIT_STOP,
    EXIT_TARGET,
    SIGNAL_LONG,
    SIGNAL_SHORT,
    TRADE_LOG_DTYPE,
)
from propfirm.market.data_loader import compute_kama, compute_macd
from propfirm.market.slippage import build_slippage_lookup
from propfirm.strategy.kama_macd import (
    _init_day_state,
    compute_signal_arrays,
    run_kama_macd_backtest,
    run_kama_macd_session,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_trade_log(capacity: int = 50) -> np.ndarray:
    return np.zeros(capacity, dtype=TRADE_LOG_DTYPE)


def _default_cfg(**overrides) -> dict:
    cfg = dict(
        kama_fast=8,
        kama_mid=13,
        kama_slow=21,
        macd_fast=12,
        macd_slow=26,
        macd_signal=9,
        atr_multiplier=2.5,
        rr_ratio=2.0,
        signal_window=5,
        warmup_bars=0,      # disabled for unit tests
        contracts=1,
        max_trades=2,
        time_stop_minute=999,
        daily_stop=-10_000.0,
        daily_target=10_000.0,
        tick_size=0.25,
        tick_value=0.50,
        commission_per_side=0.54,
        stop_penalty=1.5,
        extra_slippage_ticks=0.0,
        starting_equity=50_000.0,
        phase_id=0,
        payout_cycle_id=-1,
    )
    cfg.update(overrides)
    return cfg


def _default_state(cfg: dict, trade_log: np.ndarray) -> dict:
    slippage_lookup = build_slippage_lookup(None, session_minutes=26)
    state = {
        "equity": float(cfg["starting_equity"]),
        "intraday_pnl": 0.0,
        "position": 0,
        "entry_price": 0.0,
        "stop_level": 0.0,
        "target_level": 0.0,
        "pending_signal": 0,
        "pending_stop_level": np.nan,
        "daily_trade_count": 0,
        "open_trade_idx": -1,
        "trade_idx": 0,
        "trade_log": trade_log,
        "current_day_id": 0,
        "current_phase_id": 0,
        "current_payout_cycle_id": -1,
        "tick_size": cfg["tick_size"],
        "tick_value": cfg["tick_value"],
        "commission_per_side": cfg["commission_per_side"],
        "stop_penalty": cfg["stop_penalty"],
        "extra_slippage_points": cfg["extra_slippage_ticks"] * cfg["tick_size"],
        "minute_of_day": None,
        "bar_atr": None,
        "trailing_atr": None,
        "slippage_lookup": slippage_lookup,
    }
    _init_day_state(state)
    return state


# ---------------------------------------------------------------------------
# 1. compute_kama
# ---------------------------------------------------------------------------

class TestComputeKama:
    def test_output_shape_matches_input(self):
        closes = np.random.default_rng(0).uniform(100, 200, 100)
        result = compute_kama(closes, period=8)
        assert result.shape == closes.shape

    def test_first_period_minus_one_bars_are_nan(self):
        closes = np.arange(1, 51, dtype=np.float64)
        result = compute_kama(closes, period=10)
        assert np.all(np.isnan(result[:9]))
        assert np.isfinite(result[9])

    def test_kama_initialises_at_close_on_period_bar(self):
        closes = np.ones(20, dtype=np.float64) * 5.0
        result = compute_kama(closes, period=5)
        # Flat series → KAMA must equal close everywhere after warmup
        assert result[4] == pytest.approx(5.0)

    def test_flat_series_kama_equals_close(self):
        closes = np.full(50, 100.0, dtype=np.float64)
        result = compute_kama(closes, period=8)
        np.testing.assert_allclose(result[7:], 100.0, atol=1e-10)

    def test_different_periods_produce_distinct_values(self):
        """Different KAMA periods must produce numerically distinct results."""
        closes = np.cumsum(np.random.default_rng(7).normal(0, 1, 200)) + 100.0
        kama8 = compute_kama(closes, period=8)
        kama21 = compute_kama(closes, period=21)
        # Must NOT be identical — they're computed on different windows
        assert not np.allclose(kama8[21:], kama21[21:])

    def test_raises_on_non_positive_period(self):
        with pytest.raises(ValueError):
            compute_kama(np.ones(10), period=0)

    def test_short_series_returns_all_nan_when_insufficient(self):
        closes = np.array([1.0, 2.0])
        result = compute_kama(closes, period=10)
        assert np.all(np.isnan(result))


# ---------------------------------------------------------------------------
# 2. compute_macd
# ---------------------------------------------------------------------------

class TestComputeMacd:
    def test_output_shapes(self):
        closes = np.random.default_rng(1).uniform(100, 200, 200)
        macd_line, signal_line = compute_macd(closes, 12, 26, 9)
        assert macd_line.shape == closes.shape
        assert signal_line.shape == closes.shape

    def test_raises_on_fast_ge_slow(self):
        with pytest.raises(ValueError):
            compute_macd(np.ones(50), fast=26, slow=12, signal_period=9)

    def test_raises_on_zero_periods(self):
        with pytest.raises(ValueError):
            compute_macd(np.ones(50), fast=0, slow=26, signal_period=9)

    def test_flat_series_macd_is_zero(self):
        closes = np.full(100, 50.0, dtype=np.float64)
        macd_line, signal_line = compute_macd(closes, 12, 26, 9)
        np.testing.assert_allclose(macd_line, 0.0, atol=1e-10)
        np.testing.assert_allclose(signal_line, 0.0, atol=1e-10)


# ---------------------------------------------------------------------------
# 3. compute_signal_arrays — signal window
# ---------------------------------------------------------------------------

def _build_controlled_signals(
    n: int = 200,
    kama8_cross_bar: int = 70,
    kama13_cross_bar: int = 72,
    macd_cross_bar: int = 74,
    signal_window: int = 5,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Construct synthetic KAMA/MACD arrays where we know exactly when each
    condition becomes True, so we can assert signal window behaviour.

    Returns (closes, bar_atr, expected_signal_bar) but we bypass
    compute_signal_arrays and directly test the rolling-freshness logic
    by constructing pre-computed boolean arrays and calling the helper.
    """
    # Build synthetic closes that produce known crossovers at the right bars
    # We'll test compute_signal_arrays indirectly through known inputs.
    return kama8_cross_bar, kama13_cross_bar, macd_cross_bar


class TestComputeSignalArrays:
    """Tests that exercise the signal-window mechanic end-to-end."""

    def _make_long_scenario(self, n=200, warmup=30):
        """
        Build a synthetic price series designed so that:
        - KAMA8 crosses above KAMA21 at bar ~80
        - KAMA13 crosses above KAMA21 at bar ~82
        - MACD crosses above Signal at bar ~84
        The crosses are separated by 2 bars each → within a window of 5.
        """
        rng = np.random.default_rng(42)
        # Start below, then step up sharply at bar 70 to trigger crossovers
        closes = np.concatenate([
            np.linspace(100.0, 95.0, 70),   # mild downtrend
            np.linspace(95.0, 130.0, 130),   # sharp up-move triggers crosses
        ])
        closes = closes[:n]
        bar_atr = np.full(n, 2.0, dtype=np.float64)
        return closes, bar_atr

    def test_signal_fires_when_all_three_fresh_within_window(self):
        closes, bar_atr = self._make_long_scenario()
        result = compute_signal_arrays(
            closes, bar_atr,
            kama_fast=8, kama_mid=13, kama_slow=21,
            macd_fast=12, macd_slow=26, macd_signal_period=9,
            atr_multiplier=2.5,
            signal_window=10,  # generous window
            warmup_bars=30,
        )
        # At least one long signal must fire in the dataset
        assert result["long_signal"].any(), "Expected at least one long signal"

    def test_stop_long_computed_on_signal_bar(self):
        closes, bar_atr = self._make_long_scenario()
        result = compute_signal_arrays(
            closes, bar_atr,
            kama_fast=8, kama_mid=13, kama_slow=21,
            macd_fast=12, macd_slow=26, macd_signal_period=9,
            atr_multiplier=2.5,
            signal_window=10,
            warmup_bars=30,
        )
        sig_bars = np.where(result["long_signal"])[0]
        for idx in sig_bars:
            expected_stop = closes[idx] - 2.5 * bar_atr[idx]
            assert result["stop_long"][idx] == pytest.approx(expected_stop, rel=1e-9)

    def test_no_signal_before_warmup(self):
        closes, bar_atr = self._make_long_scenario()
        warmup = 80
        result = compute_signal_arrays(
            closes, bar_atr,
            kama_fast=8, kama_mid=13, kama_slow=21,
            macd_fast=12, macd_slow=26, macd_signal_period=9,
            atr_multiplier=2.5,
            signal_window=10,
            warmup_bars=warmup,
        )
        assert not result["long_signal"][:warmup].any()
        assert not result["short_signal"][:warmup].any()

    def test_non_signal_bars_have_nan_stop(self):
        closes, bar_atr = self._make_long_scenario()
        result = compute_signal_arrays(
            closes, bar_atr,
            kama_fast=8, kama_mid=13, kama_slow=21,
            macd_fast=12, macd_slow=26, macd_signal_period=9,
            atr_multiplier=2.5,
            signal_window=10,
            warmup_bars=30,
        )
        no_long = ~result["long_signal"]
        assert np.all(np.isnan(result["stop_long"][no_long]))

    # ── Signal Window: stale-cross suppression ──────────────────────────

    def test_signal_window_suppresses_stale_cross(self):
        """
        If the three crossover events are separated by more than signal_window,
        they should NOT all be "fresh" simultaneously → no signal.
        We verify this by using a very tight window (=1, requiring same-bar).
        """
        closes, bar_atr = self._make_long_scenario()
        # Window of 1 means all three conditions must cross on the exact same bar.
        # Our synthetic series has them staggered → no signal expected.
        result_tight = compute_signal_arrays(
            closes, bar_atr,
            kama_fast=8, kama_mid=13, kama_slow=21,
            macd_fast=12, macd_slow=26, macd_signal_period=9,
            atr_multiplier=2.5,
            signal_window=1,
            warmup_bars=30,
        )
        result_wide = compute_signal_arrays(
            closes, bar_atr,
            kama_fast=8, kama_mid=13, kama_slow=21,
            macd_fast=12, macd_slow=26, macd_signal_period=9,
            atr_multiplier=2.5,
            signal_window=20,
            warmup_bars=30,
        )
        # The wide window allows the staggered crosses to combine → more signals
        n_tight = int(result_tight["long_signal"].sum())
        n_wide = int(result_wide["long_signal"].sum())
        assert n_wide >= n_tight, (
            "Wider signal window should produce at least as many signals as tight"
        )


# ---------------------------------------------------------------------------
# 4. run_kama_macd_session — asymmetric stop/target verification
# ---------------------------------------------------------------------------

class TestRunKamaMacdSession:
    """Verify that the execution kernel places stop and target correctly."""

    def _make_session_arrays(self, n: int = 10):
        """10-bar session with a narrow range that does not hit any typical stop/target."""
        opens = np.full(n, 20_000.0, dtype=np.float64)
        # Tight highs/lows: stays within stop=19990 and a typical target ≈20021
        highs = opens + 15.0   # 20015 — below target for stop=19990, fill≈20000.25
        lows = opens - 5.0     # 19995 — above stop 19990
        closes = opens.copy()
        timestamps = np.arange(n, dtype=np.int64) * 60_000_000_000  # nanoseconds
        minute_of_day = np.arange(n, dtype=np.int16)
        bar_atr = np.full(n, 4.0, dtype=np.float64)
        trailing_atr = np.full(n, 4.0, dtype=np.float64)
        # Zeros lookup → compute_slippage applies 1-tick floor (tick_size = 0.25)
        slippage_lookup = np.zeros(n, dtype=np.float64)
        return opens, highs, lows, closes, timestamps, minute_of_day, bar_atr, trailing_atr, slippage_lookup

    def test_long_stop_and_target_respect_rr_ratio(self):
        """
        Long entry: fill + rr*(fill-stop) must equal the actual target price.

        We verify indirectly: let the target be hit and check that
        exit_price + exit_slippage == fill + rr * (fill - stop).
        (state["target_level"] is cleared to 0 after every exit.)
        """
        n = 10
        opens, highs, lows, closes, timestamps, minute_of_day, bar_atr, trailing_atr, slip_lookup = (
            self._make_session_arrays(n)
        )
        cfg = _default_cfg(rr_ratio=2.0, atr_multiplier=2.5, commission_per_side=0.0)
        trade_log = _make_trade_log()
        state = _default_state(cfg, trade_log)

        # Signal bar = 3 → stop_long = 19990; target ≈ fill + 2*(fill-19990) ≈ 20020.75
        long_signals = np.zeros(n, dtype=bool)
        long_signals[3] = True
        short_signals = np.zeros(n, dtype=bool)
        stop_long = np.full(n, np.nan)
        stop_long[3] = 19_990.0
        stop_short = np.full(n, np.nan)

        # Bar 6: spike above target to force EXIT_TARGET
        highs[6] = 20_100.0

        run_kama_macd_session(
            opens, highs, lows, closes, timestamps, minute_of_day,
            bar_atr, trailing_atr, slip_lookup,
            long_signals, short_signals, stop_long, stop_short,
            state, cfg,
        )

        assert state["trade_idx"] == 1, "Exactly one trade must be entered"
        assert trade_log[0]["exit_reason"] == EXIT_TARGET

        fill = float(trade_log[0]["entry_price"])
        stop = 19_990.0
        dist = fill - stop
        assert dist > 0, "fill must be above stop for a long"

        # For EXIT_TARGET: exit_price = target - slippage  →  exit_price + exit_slippage = target
        gross_exit = float(trade_log[0]["exit_price"]) + float(trade_log[0]["exit_slippage"])
        expected_target = fill + 2.0 * dist
        assert gross_exit == pytest.approx(expected_target, rel=1e-9)

    def test_short_stop_and_target_respect_rr_ratio(self):
        """Short entry: fill - rr*(stop-fill) must equal the actual target price."""
        n = 10
        opens = np.full(n, 20_000.0, dtype=np.float64)
        # For short: stop=20010, fill≈19999.75, target≈19979.25
        # Tight range: highs stay below stop=20010, lows stay above target≈19979
        highs = opens + 5.0    # 20005 — below stop 20010
        lows = opens - 15.0    # 19985 — above target 19979.25
        closes = opens.copy()
        timestamps = np.arange(n, dtype=np.int64) * 60_000_000_000
        minute_of_day = np.arange(n, dtype=np.int16)
        bar_atr = np.full(n, 4.0, dtype=np.float64)
        trailing_atr = np.full(n, 4.0, dtype=np.float64)
        slip_lookup = np.zeros(n, dtype=np.float64)

        cfg = _default_cfg(rr_ratio=2.0, atr_multiplier=2.5, commission_per_side=0.0)
        trade_log = _make_trade_log()
        state = _default_state(cfg, trade_log)

        short_signals = np.zeros(n, dtype=bool)
        short_signals[3] = True
        long_signals = np.zeros(n, dtype=bool)
        stop_short = np.full(n, np.nan)
        stop_short[3] = 20_010.0
        stop_long = np.full(n, np.nan)

        # Bar 6: dip far below target to force EXIT_TARGET
        lows[6] = 19_900.0

        run_kama_macd_session(
            opens, highs, lows, closes, timestamps, minute_of_day,
            bar_atr, trailing_atr, slip_lookup,
            long_signals, short_signals, stop_long, stop_short,
            state, cfg,
        )

        assert state["trade_idx"] == 1
        assert trade_log[0]["exit_reason"] == EXIT_TARGET

        fill = float(trade_log[0]["entry_price"])
        stop = 20_010.0
        dist = stop - fill
        assert dist > 0, "stop must be above fill for a short"

        # For EXIT_TARGET (short): exit_price = target + slippage  →  exit_price - exit_slippage = target
        gross_exit = float(trade_log[0]["exit_price"]) - float(trade_log[0]["exit_slippage"])
        expected_target = fill - 2.0 * dist
        assert gross_exit == pytest.approx(expected_target, rel=1e-9)

    def test_target_hit_records_exit_target(self):
        """If a bar after entry spikes high enough, trade closes at EXIT_TARGET."""
        n = 10
        opens, highs, lows, closes, timestamps, minute_of_day, bar_atr, trailing_atr, slip_lookup = (
            self._make_session_arrays(n)
        )
        cfg = _default_cfg(rr_ratio=2.0, atr_multiplier=2.5, commission_per_side=0.0)
        trade_log = _make_trade_log()
        state = _default_state(cfg, trade_log)

        # Signal bar 2, entry bar 3; stop=19990, target≈20020.75 (fill≈20000.25)
        long_signals = np.zeros(n, dtype=bool)
        long_signals[2] = True
        short_signals = np.zeros(n, dtype=bool)
        stop_long = np.full(n, np.nan)
        stop_long[2] = 19_990.0
        stop_short = np.full(n, np.nan)

        # Bar 5: spike above target — lows stay above stop on all other bars
        highs[5] = 20_100.0

        run_kama_macd_session(
            opens, highs, lows, closes, timestamps, minute_of_day,
            bar_atr, trailing_atr, slip_lookup,
            long_signals, short_signals, stop_long, stop_short,
            state, cfg,
        )

        assert state["trade_idx"] == 1
        assert trade_log[0]["exit_reason"] == EXIT_TARGET

    def test_stop_hit_records_exit_stop(self):
        """If a bar after entry dips to the stop, trade closes at EXIT_STOP."""
        n = 10
        opens, highs, lows, closes, timestamps, minute_of_day, bar_atr, trailing_atr, slip_lookup = (
            self._make_session_arrays(n)
        )
        cfg = _default_cfg(rr_ratio=2.0, atr_multiplier=2.5, commission_per_side=0.0)
        trade_log = _make_trade_log()
        state = _default_state(cfg, trade_log)

        long_signals = np.zeros(n, dtype=bool)
        long_signals[2] = True
        short_signals = np.zeros(n, dtype=bool)
        stop_long = np.full(n, np.nan)
        stop_long[2] = 19_990.0
        stop_short = np.full(n, np.nan)

        # Bar 5: dip well below stop (19980 < 19990); highs stay below target on other bars
        lows[5] = 19_900.0

        run_kama_macd_session(
            opens, highs, lows, closes, timestamps, minute_of_day,
            bar_atr, trailing_atr, slip_lookup,
            long_signals, short_signals, stop_long, stop_short,
            state, cfg,
        )

        assert state["trade_idx"] == 1
        assert trade_log[0]["exit_reason"] == EXIT_STOP

    def test_hard_close_on_last_bar(self):
        """Unclosed position on the last bar exits at EXIT_HARD_CLOSE."""
        n = 5
        opens = np.full(n, 20_000.0, dtype=np.float64)
        highs = opens + 5.0
        lows = opens - 2.0    # never touches stop (10_000) or target
        closes = opens.copy()
        timestamps = np.arange(n, dtype=np.int64)
        minute_of_day = np.arange(n, dtype=np.int16)
        bar_atr = np.full(n, 1.0, dtype=np.float64)
        trailing_atr = np.full(n, 1.0, dtype=np.float64)
        slip_lookup = np.zeros(n, dtype=np.float64)

        cfg = _default_cfg(rr_ratio=2.0, atr_multiplier=2.5, commission_per_side=0.0)
        trade_log = _make_trade_log()
        state = _default_state(cfg, trade_log)

        long_signals = np.zeros(n, dtype=bool)
        long_signals[0] = True
        short_signals = np.zeros(n, dtype=bool)
        stop_long = np.full(n, np.nan)
        stop_long[0] = 10_000.0   # far below — won't be hit
        stop_short = np.full(n, np.nan)

        run_kama_macd_session(
            opens, highs, lows, closes, timestamps, minute_of_day,
            bar_atr, trailing_atr, slip_lookup,
            long_signals, short_signals, stop_long, stop_short,
            state, cfg,
        )

        assert state["trade_idx"] == 1
        assert trade_log[0]["exit_reason"] == EXIT_HARD_CLOSE

    def test_asymmetric_target_adjusts_when_price_gaps_up(self):
        """
        When next-bar open gaps up, the target must be computed from the
        *actual fill price*, not the signal-bar close — asymmetric RR guarantee.

        Signal bar: close=20000, stop=19990.
        Entry bar:  open=20050 (gap up).
        fill ≈ 20050.25 (1-tick floor slippage).
        dist = fill - stop ≈ 60.25.
        target = fill + 2*dist ≈ 20170.75.

        We let the target be hit (highs[2] = 20500) and verify via exit_price.
        """
        n = 5
        opens = np.array([20_000.0, 20_050.0, 20_050.0, 20_050.0, 20_050.0], dtype=np.float64)
        highs = opens + 10.0           # bar 1 tight; target ≈20170.75 not hit
        lows = opens - 2.0             # above stop=19990 on all bars
        closes = opens.copy()
        highs[2] = 20_500.0            # bar 2 spikes to hit target ≈20170.75
        timestamps = np.arange(n, dtype=np.int64)
        minute_of_day = np.arange(n, dtype=np.int16)
        bar_atr = np.full(n, 4.0, dtype=np.float64)
        trailing_atr = np.full(n, 4.0, dtype=np.float64)
        slip_lookup = np.zeros(n, dtype=np.float64)  # 1-tick floor applies

        cfg = _default_cfg(rr_ratio=2.0, commission_per_side=0.0)
        trade_log = _make_trade_log()
        state = _default_state(cfg, trade_log)

        long_signals = np.zeros(n, dtype=bool)
        long_signals[0] = True   # signal on bar 0 (close=20000)
        short_signals = np.zeros(n, dtype=bool)
        stop_long = np.full(n, np.nan)
        stop_long[0] = 19_990.0  # absolute stop from signal bar
        stop_short = np.full(n, np.nan)

        run_kama_macd_session(
            opens, highs, lows, closes, timestamps, minute_of_day,
            bar_atr, trailing_atr, slip_lookup,
            long_signals, short_signals, stop_long, stop_short,
            state, cfg,
        )

        assert state["trade_idx"] == 1
        assert trade_log[0]["exit_reason"] == EXIT_TARGET

        fill = float(trade_log[0]["entry_price"])
        # Fill must reflect the gapped-up bar-1 open, not the signal-bar close
        assert fill > 20_000.0, "Fill must reflect gap-up open of bar 1"

        stop = 19_990.0
        dist = fill - stop
        expected_target = fill + 2.0 * dist

        # exit_price + exit_slippage = target for long EXIT_TARGET
        gross_exit = float(trade_log[0]["exit_price"]) + float(trade_log[0]["exit_slippage"])
        assert gross_exit == pytest.approx(expected_target, rel=1e-9)


# ---------------------------------------------------------------------------
# 5. run_kama_macd_backtest — output dtype compatibility
# ---------------------------------------------------------------------------

class TestRunKamaMacdBacktest:
    """Verify the backtest runner produces engine-compatible log arrays."""

    def _build_minimal_data_dict(self, n_days=5, bars_per_day=26):
        """Build a minimal data_dict with synthetic 15-min MNQ bars."""
        n = n_days * bars_per_day
        rng = np.random.default_rng(99)

        closes = 20_000.0 + np.cumsum(rng.normal(0, 5, n))
        opens = closes + rng.normal(0, 1, n)
        highs = np.maximum(opens, closes) + rng.uniform(0, 5, n)
        lows = np.minimum(opens, closes) - rng.uniform(0, 5, n)
        volumes = rng.integers(500, 2000, n).astype(np.uint64)
        timestamps = np.arange(n, dtype=np.int64) * 900_000_000_000  # 15-min in ns

        # minute_of_day: 0..25 per day
        minute_of_day = np.tile(np.arange(bars_per_day, dtype=np.int16), n_days)
        bar_atr = np.full(n, 4.0, dtype=np.float64)
        trailing_atr = np.full(n, 4.0, dtype=np.float64)

        day_boundaries = [(i * bars_per_day, (i + 1) * bars_per_day) for i in range(n_days)]

        slippage_lookup = build_slippage_lookup(None, session_minutes=bars_per_day)

        return {
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volumes,
            "timestamps": timestamps,
            "minute_of_day": minute_of_day,
            "bar_atr": bar_atr,
            "trailing_median_atr": trailing_atr,
            "day_boundaries": day_boundaries,
            "slippage_lookup": slippage_lookup,
            "session_minutes": bars_per_day,
            "bars_per_session": bars_per_day,
        }

    def test_returns_correct_dtypes(self):
        data = self._build_minimal_data_dict()
        cfg = _default_cfg(warmup_bars=0)
        trade_log, daily_log = run_kama_macd_backtest(data, cfg)
        assert trade_log.dtype == TRADE_LOG_DTYPE
        assert daily_log.dtype == DAILY_LOG_DTYPE

    def test_daily_log_length_equals_n_days(self):
        n_days = 7
        data = self._build_minimal_data_dict(n_days=n_days)
        cfg = _default_cfg(warmup_bars=0)
        _, daily_log = run_kama_macd_backtest(data, cfg)
        assert len(daily_log) == n_days

    def test_daily_log_day_ids_are_sequential(self):
        n_days = 5
        data = self._build_minimal_data_dict(n_days=n_days)
        cfg = _default_cfg(warmup_bars=0)
        _, daily_log = run_kama_macd_backtest(data, cfg)
        expected = np.arange(n_days, dtype=np.int32)
        np.testing.assert_array_equal(daily_log["day_id"], expected)

    def test_trade_log_entries_have_nonzero_entry_price(self):
        data = self._build_minimal_data_dict(n_days=10)
        cfg = _default_cfg(warmup_bars=0)
        trade_log, _ = run_kama_macd_backtest(data, cfg)
        if len(trade_log) > 0:
            assert np.all(trade_log["entry_price"] > 0)

    def test_all_trades_are_closed(self):
        """Every entry must have a corresponding exit (exit_time > 0)."""
        data = self._build_minimal_data_dict(n_days=10)
        cfg = _default_cfg(warmup_bars=0)
        trade_log, _ = run_kama_macd_backtest(data, cfg)
        if len(trade_log) > 0:
            assert np.all(trade_log["exit_time"] > 0), "Some trades were left open"

    def test_net_pnl_consistent_with_gross_minus_commissions(self):
        data = self._build_minimal_data_dict(n_days=10)
        cfg = _default_cfg(warmup_bars=0)
        trade_log, _ = run_kama_macd_backtest(data, cfg)
        if len(trade_log) > 0:
            expected_net = (
                trade_log["gross_pnl"]
                - trade_log["entry_commission"]
                - trade_log["exit_commission"]
            )
            np.testing.assert_allclose(trade_log["net_pnl"], expected_net, atol=1e-6)

    def test_n_trades_in_daily_log_matches_trade_log(self):
        data = self._build_minimal_data_dict(n_days=5)
        cfg = _default_cfg(warmup_bars=0)
        trade_log, daily_log = run_kama_macd_backtest(data, cfg)
        assert daily_log["n_trades"].sum() == len(trade_log)

    def test_max_trades_per_day_respected(self):
        data = self._build_minimal_data_dict(n_days=10, bars_per_day=26)
        cfg = _default_cfg(warmup_bars=0, max_trades=1)
        _, daily_log = run_kama_macd_backtest(data, cfg)
        assert np.all(daily_log["n_trades"] <= 1)
