"""London Fade strategy — isolated, engine-compatible module.

Signal logic
------------
1. Anchor  : open price at ``eval_start_time`` (default 08:00 ET).
2. Trigger : 15-min bar whose minute_of_day offset == trigger offset (default 11:30 → offset 210).
3. Trend distance = close[trigger_bar] − anchor_price.
4. SHORT if trend_distance  >  min_trend_atr × ATR(trigger_bar)   (fade exhausted up-move)
   LONG  if trend_distance  < −min_trend_atr × ATR(trigger_bar)   (fade exhausted down-move)
   No signal on stagnant days.

Execution
---------
- Next-bar-open after trigger bar (identical engine convention).
- Stop  : fill_price ± stop_atr × ATR(trigger_bar)   (symmetric)
- Target: fill_price ± rr_ratio × stop_distance
- Hard-close at time_stop_minute offset (default 360 = 14:00 ET) or last bar.

Output
------
Produces TRADE_LOG_DTYPE and DAILY_LOG_DTYPE arrays — 100% compatible with the
existing Monte-Carlo, stress-test, and MFF-evaluation pipeline.
"""

from __future__ import annotations

import numpy as np

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
from propfirm.market.slippage import compute_slippage


# ---------------------------------------------------------------------------
# Time helpers
# ---------------------------------------------------------------------------

def _hhmm_to_minute(time_str: str) -> int:
    """Convert 'HH:MM' to minutes since midnight. E.g. '11:30' → 690."""
    h, m = time_str.split(":")
    return int(h) * 60 + int(m)


# ---------------------------------------------------------------------------
# Signal array generation — one pass over all day boundaries
# ---------------------------------------------------------------------------

def compute_fade_signals(
    opens: np.ndarray,
    closes: np.ndarray,
    minute_of_day: np.ndarray,
    bar_atr: np.ndarray,
    day_boundaries: list[tuple[int, int]],
    cfg: dict,
) -> dict:
    """Compute per-bar fade-signal and stop-distance arrays.

    Parameters
    ----------
    opens, closes, minute_of_day, bar_atr
        Full arrays over all loaded data (all days concatenated).
    day_boundaries
        List of (start_idx, end_idx) tuples — one per trading session.
    cfg
        Strategy config dict.  Required keys:
            session_start, eval_start_time, trigger_time,
            min_trend_atr, stop_atr.

    Returns
    -------
    dict with keys:
        signal    : int8 ndarray   — SIGNAL_LONG / SIGNAL_SHORT / SIGNAL_NONE per bar
        stop_dist : float64 ndarray — stop distance (ATR-based) on signal bars, NaN elsewhere
    """
    session_start   = cfg.get("session_start",   "08:00")
    eval_start_time = cfg.get("eval_start_time", "08:00")
    trigger_time    = cfg.get("trigger_time",    "11:30")
    min_trend_atr   = float(cfg.get("min_trend_atr", 2.0))
    stop_atr        = float(cfg.get("stop_atr",  1.0))

    session_open_min = _hhmm_to_minute(session_start)
    eval_offset      = _hhmm_to_minute(eval_start_time) - session_open_min   # e.g. 0
    trigger_offset   = _hhmm_to_minute(trigger_time)    - session_open_min   # e.g. 210

    n = len(opens)
    signal    = np.zeros(n, dtype=np.int8)
    stop_dist = np.full(n, np.nan, dtype=np.float64)

    for d_start, d_end in day_boundaries:
        day_mod    = minute_of_day[d_start:d_end]
        n_bars_day = d_end - d_start

        # ── Find eval bar (start anchor) ──────────────────────────────────
        eval_mask = np.where(day_mod == eval_offset)[0]
        if len(eval_mask) == 0:
            continue
        eval_bar_local = int(eval_mask[0])
        anchor_price   = float(opens[d_start + eval_bar_local])

        # ── Find trigger bar ──────────────────────────────────────────────
        trig_mask = np.where(day_mod == trigger_offset)[0]
        if len(trig_mask) == 0:
            continue
        trig_bar_local = int(trig_mask[0])

        # Need room for at least one entry bar after the trigger
        if trig_bar_local >= n_bars_day - 1:
            continue

        atr = float(bar_atr[d_start + trig_bar_local])
        if atr <= 0.0 or not np.isfinite(atr):
            continue

        trigger_close  = float(closes[d_start + trig_bar_local])
        trend_distance = trigger_close - anchor_price
        threshold      = min_trend_atr * atr
        dist           = stop_atr * atr

        abs_idx = d_start + trig_bar_local

        if trend_distance > threshold:
            signal[abs_idx]    = SIGNAL_SHORT
            stop_dist[abs_idx] = dist
        elif trend_distance < -threshold:
            signal[abs_idx]    = SIGNAL_LONG
            stop_dist[abs_idx] = dist

    return {"signal": signal, "stop_dist": stop_dist}


# ---------------------------------------------------------------------------
# Execution kernel — single session
# ---------------------------------------------------------------------------

def _init_day_state(state: dict) -> None:
    """Reset per-day counters while preserving cross-day equity."""
    state["intraday_pnl"]     = 0.0
    state["daily_trade_count"] = 0
    state["position"]         = 0
    state["entry_price"]      = 0.0
    state["stop_level"]       = 0.0
    state["target_level"]     = 0.0
    state["pending_signal"]   = SIGNAL_NONE
    state["pending_stop_dist"] = np.nan
    state["open_trade_idx"]   = -1


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


def run_london_fade_session(
    opens: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    timestamps: np.ndarray,
    minute_of_day: np.ndarray,
    bar_atr: np.ndarray,
    trailing_atr: np.ndarray,
    slippage_lookup: np.ndarray,
    signals: np.ndarray,
    stop_dists: np.ndarray,
    state: dict,
    cfg: dict,
) -> None:
    """Run one trading session.  Mutates ``state`` in place.

    All array arguments are already sliced to the day's bar range.
    ``state`` carries cross-session equity; per-day counters are reset
    by the caller via ``_init_day_state`` before invoking this function.
    """
    n_bars = len(opens)
    max_trades:      int   = cfg["max_trades"]
    time_stop_minute: int  = cfg["time_stop_minute"]
    daily_stop:      float = cfg["daily_stop"]
    daily_target:    float = cfg["daily_target"]
    contracts:       int   = cfg["contracts"]
    rr_ratio:        float = cfg["rr_ratio"]
    tick_size:       float = state["tick_size"]
    tick_value:      float = state["tick_value"]

    # Inject per-day arrays into state so _slip helper can reach them
    state["minute_of_day"]  = minute_of_day
    state["bar_atr"]        = bar_atr
    state["trailing_atr"]   = trailing_atr
    state["slippage_lookup"] = slippage_lookup

    trade_log         = state["trade_log"]
    day_id            = state["current_day_id"]
    phase_id          = state["current_phase_id"]
    payout_cycle_id   = state["current_payout_cycle_id"]

    for bar_idx in range(n_bars):
        mod      = int(minute_of_day[bar_idx])
        ts       = int(timestamps[bar_idx])
        is_last  = bar_idx == n_bars - 1
        bar_open  = float(opens[bar_idx])
        bar_high  = float(highs[bar_idx])
        bar_low   = float(lows[bar_idx])
        bar_close = float(closes[bar_idx])
        exited    = False

        # ── 1. Execute pending entry (next-bar-open) ──────────────────────
        if state["position"] == 0 and state["pending_signal"] != SIGNAL_NONE:
            halted = (
                state["intraday_pnl"] <= daily_stop
                or state["intraday_pnl"] >= daily_target
            )
            if halted or mod >= time_stop_minute:
                state["pending_signal"]   = SIGNAL_NONE
                state["pending_stop_dist"] = np.nan
            else:
                if state["trade_idx"] >= len(trade_log):
                    raise RuntimeError("trade_log capacity exceeded")

                slip             = _slip(state, bar_idx, False)
                entry_commission = state["commission_per_side"] * contracts
                stop_dist_val    = float(state["pending_stop_dist"])

                if state["pending_signal"] == SIGNAL_LONG:
                    fill_price   = bar_open + slip
                    stop_level   = fill_price - stop_dist_val
                    target_level = fill_price + rr_ratio * stop_dist_val
                    state["position"] = contracts
                else:  # SIGNAL_SHORT
                    fill_price   = bar_open - slip
                    stop_level   = fill_price + stop_dist_val
                    target_level = fill_price - rr_ratio * stop_dist_val
                    state["position"] = -contracts

                state["entry_price"]  = fill_price
                state["stop_level"]   = stop_level
                state["target_level"] = target_level
                state["intraday_pnl"] -= entry_commission
                state["equity"]       -= entry_commission
                state["daily_trade_count"] += 1

                open_trade_idx             = state["trade_idx"]
                state["open_trade_idx"]    = open_trade_idx
                trade_log[open_trade_idx]["day_id"]           = day_id
                trade_log[open_trade_idx]["phase_id"]         = phase_id
                trade_log[open_trade_idx]["payout_cycle_id"]  = payout_cycle_id
                trade_log[open_trade_idx]["entry_time"]       = ts
                trade_log[open_trade_idx]["entry_price"]      = fill_price
                trade_log[open_trade_idx]["entry_slippage"]   = slip
                trade_log[open_trade_idx]["entry_commission"] = entry_commission
                trade_log[open_trade_idx]["contracts"]        = contracts
                trade_log[open_trade_idx]["signal_type"]      = state["pending_signal"]
                state["trade_idx"]         += 1
                state["pending_signal"]     = SIGNAL_NONE
                state["pending_stop_dist"]  = np.nan

        # ── 2. Check exits ────────────────────────────────────────────────
        if state["position"] != 0:
            exit_price  = 0.0
            exit_reason = -1
            slip        = 0.0

            if mod >= time_stop_minute:
                slip        = _slip(state, bar_idx, False)
                exit_price  = bar_open - slip if state["position"] > 0 else bar_open + slip
                exit_reason = EXIT_HARD_CLOSE
            elif state["position"] > 0:
                if bar_open <= state["stop_level"]:
                    slip        = _slip(state, bar_idx, True)
                    exit_price  = bar_open - slip
                    exit_reason = EXIT_STOP
                elif bar_low <= state["stop_level"]:
                    slip        = _slip(state, bar_idx, True)
                    exit_price  = state["stop_level"] - slip
                    exit_reason = EXIT_STOP
                elif bar_high >= state["target_level"]:
                    slip        = _slip(state, bar_idx, False)
                    exit_price  = state["target_level"] - slip
                    exit_reason = EXIT_TARGET
            else:  # short
                if bar_open >= state["stop_level"]:
                    slip        = _slip(state, bar_idx, True)
                    exit_price  = bar_open + slip
                    exit_reason = EXIT_STOP
                elif bar_high >= state["stop_level"]:
                    slip        = _slip(state, bar_idx, True)
                    exit_price  = state["stop_level"] + slip
                    exit_reason = EXIT_STOP
                elif bar_low <= state["target_level"]:
                    slip        = _slip(state, bar_idx, False)
                    exit_price  = state["target_level"] + slip
                    exit_reason = EXIT_TARGET

            if exit_reason == -1 and is_last:
                slip        = _slip(state, bar_idx, False)
                exit_price  = bar_close - slip if state["position"] > 0 else bar_close + slip
                exit_reason = EXIT_HARD_CLOSE

            if exit_reason >= 0:
                abs_contracts    = abs(state["position"])
                exit_commission  = state["commission_per_side"] * abs_contracts

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

                oti = state["open_trade_idx"]
                entry_comm_paid     = float(trade_log[oti]["entry_commission"])
                net_pnl             = gross_pnl - entry_comm_paid - exit_commission

                trade_log[oti]["exit_time"]       = ts
                trade_log[oti]["exit_price"]      = exit_price
                trade_log[oti]["exit_slippage"]   = slip
                trade_log[oti]["exit_commission"] = exit_commission
                trade_log[oti]["gross_pnl"]       = gross_pnl
                trade_log[oti]["net_pnl"]         = net_pnl
                trade_log[oti]["exit_reason"]     = exit_reason

                state["intraday_pnl"] += gross_pnl - exit_commission
                state["equity"]       += gross_pnl - exit_commission
                state["position"]      = 0
                state["entry_price"]   = 0.0
                state["stop_level"]    = 0.0
                state["target_level"]  = 0.0
                state["open_trade_idx"] = -1
                exited = True

        # ── 3. Record signal for next bar ─────────────────────────────────
        if not exited and state["position"] == 0 and state["pending_signal"] == SIGNAL_NONE:
            halted = (
                state["intraday_pnl"] <= daily_stop
                or state["intraday_pnl"] >= daily_target
            )
            if (
                not halted
                and state["daily_trade_count"] < max_trades
                and mod < time_stop_minute
                and not is_last
            ):
                sig = int(signals[bar_idx])
                if sig != SIGNAL_NONE:
                    state["pending_signal"]    = sig
                    state["pending_stop_dist"] = float(stop_dists[bar_idx])


# ---------------------------------------------------------------------------
# Full backtest runner — iterates over all sessions, returns log arrays
# ---------------------------------------------------------------------------

def run_london_fade_backtest(data_dict: dict, cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    """Run the full London Fade backtest over all sessions in ``data_dict``.

    Parameters
    ----------
    data_dict : dict
        Output of ``data_loader.load_session_data`` with session_start='08:00'.
    cfg : dict
        Strategy configuration.  Required keys:
            session_start, eval_start_time, trigger_time,
            min_trend_atr, stop_atr, rr_ratio,
            contracts, max_trades, daily_stop, daily_target,
            time_stop_minute,
            stop_penalty, commission_per_side, extra_slippage_ticks,
            tick_size, tick_value,
            starting_equity.

    Returns
    -------
    (trade_log, daily_log) — trimmed to actual trade / day counts.
    """
    opens         = data_dict["open"]
    closes        = data_dict["close"]
    highs         = data_dict["high"]
    lows          = data_dict["low"]
    bar_atr       = data_dict["bar_atr"]
    timestamps    = data_dict["timestamps"]
    minute_of_day = data_dict["minute_of_day"]
    trailing_atr  = data_dict["trailing_median_atr"]
    day_boundaries: list[tuple[int, int]] = data_dict["day_boundaries"]
    n_days        = len(day_boundaries)

    # ── Pre-compute signal arrays (entire dataset at once) ─────────────────
    signals = compute_fade_signals(
        opens, closes, minute_of_day, bar_atr, day_boundaries, cfg
    )

    # ── Allocate log arrays (upper-bound capacity) ─────────────────────────
    max_trades_total = n_days * cfg["max_trades"] + 10
    trade_log  = np.zeros(max_trades_total, dtype=TRADE_LOG_DTYPE)
    daily_log  = np.zeros(n_days, dtype=DAILY_LOG_DTYPE)

    # ── Build slippage lookup ──────────────────────────────────────────────
    slippage_lookup = data_dict.get("slippage_lookup")
    if slippage_lookup is None:
        from propfirm.market.slippage import build_slippage_lookup
        # IMPORTANT: session_minutes = actual minute count of the session window
        # (e.g. 480 for 08:00–15:59), NOT bars_per_session (32 for 15-min bars).
        # minute_of_day values for 15-min bars are 0, 15, 30, …, 465 — minute
        # offsets from session open — so the lookup array must be at least 466
        # entries.  Using session_minutes=480 gives a safe margin.
        session_minutes = data_dict.get("session_minutes", 480)
        slippage_lookup = build_slippage_lookup(None, session_minutes=session_minutes)

    # ── Persistent cross-session state dict ───────────────────────────────
    tick_size:    float = cfg.get("tick_size",    MNQ_TICK_SIZE)
    tick_value:   float = cfg.get("tick_value",   MNQ_TICK_VALUE)
    extra_slip_ticks: float = cfg.get("extra_slippage_ticks", 0.0)

    state: dict = {
        "equity":              float(cfg.get("starting_equity", 50_000.0)),
        "intraday_pnl":        0.0,
        "position":            0,
        "entry_price":         0.0,
        "stop_level":          0.0,
        "target_level":        0.0,
        "pending_signal":      SIGNAL_NONE,
        "pending_stop_dist":   np.nan,
        "daily_trade_count":   0,
        "open_trade_idx":      -1,
        "trade_idx":           0,
        "trade_log":           trade_log,
        "current_day_id":      0,
        "current_phase_id":    int(cfg.get("phase_id", 0)),
        "current_payout_cycle_id": int(cfg.get("payout_cycle_id", -1)),
        "tick_size":           tick_size,
        "tick_value":          tick_value,
        "commission_per_side": float(cfg["commission_per_side"]),
        "stop_penalty":        float(cfg.get("stop_penalty", 1.5)),
        "extra_slippage_points": extra_slip_ticks * tick_size,
        # per-day arrays injected by run_london_fade_session
        "minute_of_day":       None,
        "bar_atr":             None,
        "trailing_atr":        None,
        "slippage_lookup":     slippage_lookup,
    }

    sig_arr  = signals["signal"]
    dist_arr = signals["stop_dist"]

    for day_idx, (d_start, d_end) in enumerate(day_boundaries):
        trades_before = state["trade_idx"]
        _init_day_state(state)
        state["current_day_id"] = day_idx

        run_london_fade_session(
            opens=opens[d_start:d_end],
            highs=highs[d_start:d_end],
            lows=lows[d_start:d_end],
            closes=closes[d_start:d_end],
            timestamps=timestamps[d_start:d_end],
            minute_of_day=minute_of_day[d_start:d_end],
            bar_atr=bar_atr[d_start:d_end],
            trailing_atr=trailing_atr[d_start:d_end],
            slippage_lookup=slippage_lookup,
            signals=sig_arr[d_start:d_end],
            stop_dists=dist_arr[d_start:d_end],
            state=state,
            cfg=cfg,
        )

        n_trades_today = state["trade_idx"] - trades_before
        daily_log[day_idx]["day_id"]          = day_idx
        daily_log[day_idx]["phase_id"]        = state["current_phase_id"]
        daily_log[day_idx]["payout_cycle_id"] = state["current_payout_cycle_id"]
        daily_log[day_idx]["had_trade"]       = 1 if n_trades_today > 0 else 0
        daily_log[day_idx]["n_trades"]        = n_trades_today
        daily_log[day_idx]["day_pnl"]         = state["intraday_pnl"]
        daily_log[day_idx]["net_payout"]      = 0.0  # MFF evaluation applied externally

    n_trades_total = state["trade_idx"]
    return trade_log[:n_trades_total], daily_log[:n_days]
