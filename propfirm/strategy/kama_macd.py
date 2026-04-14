"""Triple KAMA + MACD strategy — isolated, engine-compatible module.

Signal logic
------------
Long: KAMA_fast > KAMA_slow  AND  KAMA_mid > KAMA_slow  AND  MACD > Signal
      Each of the three conditions must have crossed from False → True within
      the last ``signal_window`` bars and still be active (all three fresh).

Short: symmetric inverse conditions.

Execution
---------
- Next-bar-open after signal bar (identical to existing engine convention).
- Stop  : signal-bar close  ±  atr_multiplier × ATR(signal bar)
- Target: fill_price  ±  rr_ratio × |fill_price − stop_level|

Output
------
Produces TRADE_LOG_DTYPE and DAILY_LOG_DTYPE arrays — 100 % compatible with
the existing Monte-Carlo, stress-test, and MFF-evaluation pipeline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

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
from propfirm.market.data_loader import compute_kama, compute_macd
from propfirm.market.slippage import compute_slippage


# ---------------------------------------------------------------------------
# Signal array generation (vectorised, over the entire dataset at once)
# ---------------------------------------------------------------------------

def _rolling_any(arr: np.ndarray, window: int) -> np.ndarray:
    """True at position i if any element in arr[i-window+1 : i+1] is True."""
    return (
        pd.Series(arr.astype(np.float64))
        .rolling(window=window, min_periods=1)
        .max()
        .to_numpy(dtype=np.float64)
        > 0.5
    )


def compute_signal_arrays(
    closes: np.ndarray,
    bar_atr: np.ndarray,
    *,
    kama_fast: int = 8,
    kama_mid: int = 13,
    kama_slow: int = 21,
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal_period: int = 9,
    atr_multiplier: float = 2.5,
    signal_window: int = 5,
    warmup_bars: int = 60,
) -> dict:
    """Compute per-bar signal masks and pre-calculated stop levels.

    Returns
    -------
    dict with keys:
        long_signal   : bool ndarray — True on signal bar (long)
        short_signal  : bool ndarray — True on signal bar (short)
        stop_long     : float ndarray — stop price for long (NaN off-signal)
        stop_short    : float ndarray — stop price for short (NaN off-signal)
    """
    n = len(closes)

    kama8 = compute_kama(closes, kama_fast)
    kama13 = compute_kama(closes, kama_mid)
    kama21 = compute_kama(closes, kama_slow)
    macd_line, signal_line = compute_macd(closes, macd_fast, macd_slow, macd_signal_period)

    # Current-bar state conditions
    kama8_above = kama8 > kama21
    kama13_above = kama13 > kama21
    macd_above = macd_line > signal_line

    # Shift by 1 to find transition bars (cross-up / cross-down events)
    prev_kama8_above = np.empty(n, dtype=bool)
    prev_kama13_above = np.empty(n, dtype=bool)
    prev_macd_above = np.empty(n, dtype=bool)
    prev_kama8_above[0] = kama8_above[0]
    prev_kama13_above[0] = kama13_above[0]
    prev_macd_above[0] = macd_above[0]
    prev_kama8_above[1:] = kama8_above[:-1]
    prev_kama13_above[1:] = kama13_above[:-1]
    prev_macd_above[1:] = macd_above[:-1]

    kama8_cross_up = kama8_above & ~prev_kama8_above
    kama13_cross_up = kama13_above & ~prev_kama13_above
    macd_cross_up = macd_above & ~prev_macd_above

    kama8_cross_down = (~kama8_above) & prev_kama8_above
    kama13_cross_down = (~kama13_above) & prev_kama13_above
    macd_cross_down = (~macd_above) & prev_macd_above

    # Rolling freshness: was there a cross within the last signal_window bars?
    kama8_fresh_up = _rolling_any(kama8_cross_up, signal_window)
    kama13_fresh_up = _rolling_any(kama13_cross_up, signal_window)
    macd_fresh_up = _rolling_any(macd_cross_up, signal_window)

    kama8_fresh_down = _rolling_any(kama8_cross_down, signal_window)
    kama13_fresh_down = _rolling_any(kama13_cross_down, signal_window)
    macd_fresh_down = _rolling_any(macd_cross_down, signal_window)

    # Warmup guard — no signals before indicators are stable
    warmup = np.arange(n) >= warmup_bars

    # All indicator values must be finite
    valid = (
        np.isfinite(kama8)
        & np.isfinite(kama13)
        & np.isfinite(kama21)
        & np.isfinite(macd_line)
        & np.isfinite(signal_line)
        & np.isfinite(bar_atr)
        & (bar_atr > 0.0)
    )

    long_mask = (
        warmup
        & valid
        & kama8_above
        & kama13_above
        & macd_above
        & kama8_fresh_up
        & kama13_fresh_up
        & macd_fresh_up
    )
    short_mask = (
        warmup
        & valid
        & (~kama8_above)
        & (~kama13_above)
        & (~macd_above)
        & kama8_fresh_down
        & kama13_fresh_down
        & macd_fresh_down
    )

    stop_long = np.where(long_mask, closes - atr_multiplier * bar_atr, np.nan)
    stop_short = np.where(short_mask, closes + atr_multiplier * bar_atr, np.nan)

    return {
        "long_signal": long_mask,
        "short_signal": short_mask,
        "stop_long": stop_long,
        "stop_short": stop_short,
    }


# ---------------------------------------------------------------------------
# Execution kernel — single session
# ---------------------------------------------------------------------------

def _init_day_state(state: dict) -> None:
    """Reset per-day counters while preserving cross-day equity."""
    state["intraday_pnl"] = 0.0
    state["daily_trade_count"] = 0
    state["position"] = 0
    state["entry_price"] = 0.0
    state["stop_level"] = 0.0
    state["target_level"] = 0.0
    state["pending_signal"] = SIGNAL_NONE
    state["pending_stop_level"] = np.nan
    state["open_trade_idx"] = -1


def _slip(state: dict, bar_idx_local: int, is_stop: bool) -> float:
    mod = int(state["minute_of_day"][bar_idx_local])
    return compute_slippage(
        mod,
        float(state["bar_atr"][bar_idx_local]),
        float(state["trailing_atr"][bar_idx_local]),
        state["slippage_lookup"],
        is_stop,
        state["stop_penalty"],
        state["tick_size"],
        state["extra_slippage_points"],
    )


def run_kama_macd_session(
    opens: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    timestamps: np.ndarray,
    minute_of_day: np.ndarray,
    bar_atr: np.ndarray,
    trailing_atr: np.ndarray,
    slippage_lookup: np.ndarray,
    long_signals: np.ndarray,
    short_signals: np.ndarray,
    stop_long: np.ndarray,
    stop_short: np.ndarray,
    state: dict,
    cfg: dict,
) -> None:
    """Run one trading session. Mutates ``state`` in place.

    All array arguments are already sliced to the day's bar range.
    ``state`` carries cross-session equity; per-day counters are reset
    by the caller via ``_init_day_state`` before invoking this function.
    """
    n_bars = len(opens)
    max_trades: int = cfg["max_trades"]
    time_stop_minute: int = cfg["time_stop_minute"]
    daily_stop: float = cfg["daily_stop"]
    daily_target: float = cfg["daily_target"]
    contracts: int = cfg["contracts"]
    rr_ratio: float = cfg["rr_ratio"]
    tick_size: float = state["tick_size"]
    tick_value: float = state["tick_value"]

    # Inject per-day arrays into state so helper functions can reach them
    state["minute_of_day"] = minute_of_day
    state["bar_atr"] = bar_atr
    state["trailing_atr"] = trailing_atr
    state["slippage_lookup"] = slippage_lookup

    trade_log = state["trade_log"]
    day_id = state["current_day_id"]
    phase_id = state["current_phase_id"]
    payout_cycle_id = state["current_payout_cycle_id"]

    for bar_idx in range(n_bars):
        mod = int(minute_of_day[bar_idx])
        ts = int(timestamps[bar_idx])
        is_last = bar_idx == n_bars - 1
        bar_open = float(opens[bar_idx])
        bar_high = float(highs[bar_idx])
        bar_low = float(lows[bar_idx])
        bar_close = float(closes[bar_idx])
        exited = False

        # ── 1. Execute pending entry (next-bar-open) ──────────────────────
        if state["position"] == 0 and state["pending_signal"] != SIGNAL_NONE:
            halted = state["intraday_pnl"] <= daily_stop or state["intraday_pnl"] >= daily_target
            if halted or mod >= time_stop_minute:
                state["pending_signal"] = SIGNAL_NONE
                state["pending_stop_level"] = np.nan
            else:
                if state["trade_idx"] >= len(trade_log):
                    raise RuntimeError("trade_log capacity exceeded")

                slip = _slip(state, bar_idx, False)
                entry_commission = state["commission_per_side"] * contracts

                if state["pending_signal"] == SIGNAL_LONG:
                    fill_price = bar_open + slip
                    stop_level = state["pending_stop_level"]
                    distance = fill_price - stop_level
                    target_level = fill_price + rr_ratio * distance
                    state["position"] = contracts
                else:
                    fill_price = bar_open - slip
                    stop_level = state["pending_stop_level"]
                    distance = stop_level - fill_price
                    target_level = fill_price - rr_ratio * distance
                    state["position"] = -contracts

                state["entry_price"] = fill_price
                state["stop_level"] = stop_level
                state["target_level"] = target_level
                state["intraday_pnl"] -= entry_commission
                state["equity"] -= entry_commission
                state["daily_trade_count"] += 1

                open_trade_idx = state["trade_idx"]
                state["open_trade_idx"] = open_trade_idx
                trade_log[open_trade_idx]["day_id"] = day_id
                trade_log[open_trade_idx]["phase_id"] = phase_id
                trade_log[open_trade_idx]["payout_cycle_id"] = payout_cycle_id
                trade_log[open_trade_idx]["entry_time"] = ts
                trade_log[open_trade_idx]["entry_price"] = fill_price
                trade_log[open_trade_idx]["entry_slippage"] = slip
                trade_log[open_trade_idx]["entry_commission"] = entry_commission
                trade_log[open_trade_idx]["contracts"] = contracts
                trade_log[open_trade_idx]["signal_type"] = state["pending_signal"]
                state["trade_idx"] += 1
                state["pending_signal"] = SIGNAL_NONE
                state["pending_stop_level"] = np.nan

        # ── 2. Check exits ────────────────────────────────────────────────
        if state["position"] != 0:
            exit_price = 0.0
            exit_reason = -1
            slip = 0.0

            if mod >= time_stop_minute:
                slip = _slip(state, bar_idx, False)
                exit_price = bar_open - slip if state["position"] > 0 else bar_open + slip
                exit_reason = EXIT_HARD_CLOSE
            elif state["position"] > 0:
                if bar_open <= state["stop_level"]:
                    slip = _slip(state, bar_idx, True)
                    exit_price = bar_open - slip
                    exit_reason = EXIT_STOP
                elif bar_low <= state["stop_level"]:
                    slip = _slip(state, bar_idx, True)
                    exit_price = state["stop_level"] - slip
                    exit_reason = EXIT_STOP
                elif bar_high >= state["target_level"]:
                    slip = _slip(state, bar_idx, False)
                    exit_price = state["target_level"] - slip
                    exit_reason = EXIT_TARGET
            else:  # short
                if bar_open >= state["stop_level"]:
                    slip = _slip(state, bar_idx, True)
                    exit_price = bar_open + slip
                    exit_reason = EXIT_STOP
                elif bar_high >= state["stop_level"]:
                    slip = _slip(state, bar_idx, True)
                    exit_price = state["stop_level"] + slip
                    exit_reason = EXIT_STOP
                elif bar_low <= state["target_level"]:
                    slip = _slip(state, bar_idx, False)
                    exit_price = state["target_level"] + slip
                    exit_reason = EXIT_TARGET

            if exit_reason == -1 and is_last:
                slip = _slip(state, bar_idx, False)
                exit_price = bar_close - slip if state["position"] > 0 else bar_close + slip
                exit_reason = EXIT_HARD_CLOSE

            if exit_reason >= 0:
                abs_contracts = abs(state["position"])
                exit_commission = state["commission_per_side"] * abs_contracts

                if state["position"] > 0:
                    gross_pnl = (
                        (exit_price - state["entry_price"])
                        * abs_contracts
                        / tick_size
                        * tick_value
                    )
                else:
                    gross_pnl = (
                        (state["entry_price"] - exit_price)
                        * abs_contracts
                        / tick_size
                        * tick_value
                    )
                net_pnl = gross_pnl - float(trade_log[state["open_trade_idx"]]["entry_commission"]) - exit_commission

                oti = state["open_trade_idx"]
                trade_log[oti]["exit_time"] = ts
                trade_log[oti]["exit_price"] = exit_price
                trade_log[oti]["exit_slippage"] = slip
                trade_log[oti]["exit_commission"] = exit_commission
                trade_log[oti]["gross_pnl"] = gross_pnl
                trade_log[oti]["net_pnl"] = net_pnl
                trade_log[oti]["exit_reason"] = exit_reason

                state["intraday_pnl"] += gross_pnl - exit_commission
                state["equity"] += gross_pnl - exit_commission
                state["position"] = 0
                state["entry_price"] = 0.0
                state["stop_level"] = 0.0
                state["target_level"] = 0.0
                state["open_trade_idx"] = -1
                exited = True

        # ── 3. Generate signal for next bar ───────────────────────────────
        if not exited and state["position"] == 0 and state["pending_signal"] == SIGNAL_NONE:
            halted = state["intraday_pnl"] <= daily_stop or state["intraday_pnl"] >= daily_target
            if (
                not halted
                and state["daily_trade_count"] < max_trades
                and mod < time_stop_minute
                and not is_last
            ):
                if long_signals[bar_idx]:
                    state["pending_signal"] = SIGNAL_LONG
                    state["pending_stop_level"] = float(stop_long[bar_idx])
                elif short_signals[bar_idx]:
                    state["pending_signal"] = SIGNAL_SHORT
                    state["pending_stop_level"] = float(stop_short[bar_idx])


# ---------------------------------------------------------------------------
# Full backtest runner — iterates over all sessions, returns log arrays
# ---------------------------------------------------------------------------

def run_kama_macd_backtest(data_dict: dict, cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    """Run the full KAMA+MACD backtest over all sessions in ``data_dict``.

    Parameters
    ----------
    data_dict : dict
        Output of ``data_loader.load_session_data`` (15-min resampled MNQ).
    cfg : dict
        Strategy configuration.  Required keys:
            kama_fast, kama_mid, kama_slow,
            macd_fast, macd_slow, macd_signal,
            atr_period (for bar_atr — already in data_dict),
            atr_multiplier, rr_ratio, signal_window, warmup_bars,
            contracts, max_trades, daily_stop, daily_target,
            time_stop_minute,
            stop_penalty, commission_per_side, extra_slippage_ticks,
            tick_size, tick_value,
            starting_equity.

    Returns
    -------
    (trade_log, daily_log) — trimmed to actual trade / day counts.
    """
    closes = data_dict["close"]
    bar_atr = data_dict["bar_atr"]
    day_boundaries: list[tuple[int, int]] = data_dict["day_boundaries"]
    n_days = len(day_boundaries)

    # ── Pre-compute signal arrays (entire dataset at once) ─────────────────
    signals = compute_signal_arrays(
        closes,
        bar_atr,
        kama_fast=cfg["kama_fast"],
        kama_mid=cfg["kama_mid"],
        kama_slow=cfg["kama_slow"],
        macd_fast=cfg["macd_fast"],
        macd_slow=cfg["macd_slow"],
        macd_signal_period=cfg["macd_signal"],
        atr_multiplier=cfg["atr_multiplier"],
        signal_window=cfg["signal_window"],
        warmup_bars=cfg.get("warmup_bars", 60),
    )

    # ── Allocate log arrays (upper-bound capacity) ─────────────────────────
    max_trades_total = n_days * cfg["max_trades"] + 10
    trade_log = np.zeros(max_trades_total, dtype=TRADE_LOG_DTYPE)
    daily_log = np.zeros(n_days, dtype=DAILY_LOG_DTYPE)

    # ── Persistent cross-session state dict ───────────────────────────────
    tick_size: float = cfg.get("tick_size", MNQ_TICK_SIZE)
    tick_value: float = cfg.get("tick_value", MNQ_TICK_VALUE)
    extra_slippage_ticks: float = cfg.get("extra_slippage_ticks", 0.0)

    slippage_lookup = data_dict.get("slippage_lookup")
    if slippage_lookup is None:
        from propfirm.market.slippage import build_slippage_lookup
        # IMPORTANT: session_minutes = actual minute count of the session (e.g. 390
        # for RTH 09:30-15:59), NOT bars_per_session.  minute_of_day values for a
        # 15-min resampled session are 0, 15, 30, …, 375 — they are minute offsets
        # from session open, so the lookup array must be large enough to hold them.
        session_minutes = data_dict.get("session_minutes", 390)
        slippage_lookup = build_slippage_lookup(None, session_minutes=session_minutes)

    state: dict = {
        "equity": float(cfg.get("starting_equity", 50_000.0)),
        "intraday_pnl": 0.0,
        "position": 0,
        "entry_price": 0.0,
        "stop_level": 0.0,
        "target_level": 0.0,
        "pending_signal": SIGNAL_NONE,
        "pending_stop_level": np.nan,
        "daily_trade_count": 0,
        "open_trade_idx": -1,
        "trade_idx": 0,
        "trade_log": trade_log,
        "current_day_id": 0,
        "current_phase_id": int(cfg.get("phase_id", 0)),
        "current_payout_cycle_id": int(cfg.get("payout_cycle_id", -1)),
        "tick_size": tick_size,
        "tick_value": tick_value,
        "commission_per_side": float(cfg["commission_per_side"]),
        "stop_penalty": float(cfg.get("stop_penalty", 1.5)),
        "extra_slippage_points": extra_slippage_ticks * tick_size,
        # per-day arrays injected by run_kama_macd_session
        "minute_of_day": None,
        "bar_atr": None,
        "trailing_atr": None,
        "slippage_lookup": slippage_lookup,
    }

    opens = data_dict["open"]
    highs = data_dict["high"]
    lows = data_dict["low"]
    timestamps = data_dict["timestamps"]
    minute_of_day = data_dict["minute_of_day"]
    trailing_atr = data_dict["trailing_median_atr"]

    long_sig = signals["long_signal"]
    short_sig = signals["short_signal"]
    stop_long_arr = signals["stop_long"]
    stop_short_arr = signals["stop_short"]

    for day_idx, (d_start, d_end) in enumerate(day_boundaries):
        trades_before = state["trade_idx"]
        _init_day_state(state)
        state["current_day_id"] = day_idx

        run_kama_macd_session(
            opens=opens[d_start:d_end],
            highs=highs[d_start:d_end],
            lows=lows[d_start:d_end],
            closes=closes[d_start:d_end],
            timestamps=timestamps[d_start:d_end],
            minute_of_day=minute_of_day[d_start:d_end],
            bar_atr=bar_atr[d_start:d_end],
            trailing_atr=trailing_atr[d_start:d_end],
            slippage_lookup=slippage_lookup,
            long_signals=long_sig[d_start:d_end],
            short_signals=short_sig[d_start:d_end],
            stop_long=stop_long_arr[d_start:d_end],
            stop_short=stop_short_arr[d_start:d_end],
            state=state,
            cfg=cfg,
        )

        n_trades_today = state["trade_idx"] - trades_before
        daily_log[day_idx]["day_id"] = day_idx
        daily_log[day_idx]["phase_id"] = state["current_phase_id"]
        daily_log[day_idx]["payout_cycle_id"] = state["current_payout_cycle_id"]
        daily_log[day_idx]["had_trade"] = 1 if n_trades_today > 0 else 0
        daily_log[day_idx]["n_trades"] = n_trades_today
        daily_log[day_idx]["day_pnl"] = state["intraday_pnl"]
        daily_log[day_idx]["net_payout"] = 0.0  # MFF evaluation applied externally

    n_trades_total = state["trade_idx"]
    return trade_log[:n_trades_total], daily_log[:n_days]
