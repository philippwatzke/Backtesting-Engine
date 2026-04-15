#!/usr/bin/env python
"""Reproduce the archived 27-trade MNQ legacy run on the current workspace.

This runner freezes the legacy behavior that was inferred and verified against
the archived baseline:

- session-aware 30m MNQ path
- legacy simple rolling ATR for stop sizing
- legacy trailing ATR bootstrap
- next-bar-open entries (not entry-on-close)

The goal is not a new strategy variant. The goal is a reproducible recreation of
the archived `output/backtests_mnq_tf30_regime_test/latest_trade_log.npy` run.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from propfirm.core.engine import run_day_kernel_portfolio
from propfirm.core.types import (
    DAILY_LOG_DTYPE,
    PARAMS_CONTRACTS,
    PARAMS_ENTRY_ON_CLOSE,
    PARAMS_EXTRA_SLIPPAGE_TICKS,
    PARAMS_MAX_TRADES,
    PROFILE_ARRAY_LENGTH,
    PROFILE_BREAKEVEN_TRIGGER_TICKS,
    PROFILE_RISK_BUFFER_FRACTION,
    PROFILE_RISK_PER_TRADE_USD,
    PROFILE_STOP_ATR_MULTIPLIER,
    PROFILE_TARGET_ATR_MULTIPLIER,
    TRADE_LOG_DTYPE,
)
from propfirm.io.config import build_phase_params, load_mff_config, load_params_config
from propfirm.io.reporting import build_report, save_report
from propfirm.market.data_loader import load_session_data
from propfirm.market.slippage import build_slippage_lookup
from propfirm.rules.mff import MFFState
from propfirm.strategy.portfolio import combined_portfolio_signal


SESSION_TZ = "America/New_York"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/processed/MNQ_1m_full_test.parquet"))
    parser.add_argument("--mff-config", type=Path, default=Path("configs/mff_flex_50k_mnq.toml"))
    parser.add_argument(
        "--params-config",
        type=Path,
        default=Path("configs/default_params_mnq_legacy_frozen.toml"),
    )
    parser.add_argument(
        "--baseline-trade-log",
        type=Path,
        default=Path("output/backtests_mnq_tf30_regime_test/latest_trade_log.npy"),
    )
    parser.add_argument("--output", type=Path, default=Path("output/backtests_mnq_legacy_frozen"))
    parser.add_argument("--timeframe-minutes", type=int, default=30)
    return parser.parse_args()


def _legacy_bar_atr(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, atr_period: int) -> np.ndarray:
    n_bars = len(highs)
    tr = np.maximum(highs - lows, np.zeros(n_bars))
    if n_bars > 1:
        prev_close = np.roll(closes, 1)
        prev_close[0] = closes[0]
        tr = np.maximum(tr, np.abs(highs - prev_close))
        tr = np.maximum(tr, np.abs(lows - prev_close))

    bar_atr = np.zeros(n_bars, dtype=np.float64)
    if n_bars >= atr_period:
        kernel = np.ones(atr_period, dtype=np.float64) / atr_period
        convolved = np.convolve(tr, kernel, mode="full")[:n_bars]
        bar_atr[atr_period - 1 :] = convolved[atr_period - 1 :]
        bar_atr[: atr_period - 1] = convolved[: atr_period - 1]
    elif n_bars > 0:
        bar_atr[:] = np.mean(tr)
    return bar_atr


def _legacy_trailing_atr(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    day_boundaries: list[tuple[int, int]],
    atr_period: int,
    trailing_days: int,
) -> np.ndarray:
    n_bars = len(highs)
    result = np.zeros(n_bars, dtype=np.float64)

    tr = np.maximum(highs - lows, np.zeros(n_bars))
    if n_bars > 1:
        prev_close = np.roll(closes, 1)
        prev_close[0] = closes[0]
        tr = np.maximum(tr, np.abs(highs - prev_close))
        tr = np.maximum(tr, np.abs(lows - prev_close))

    session_atrs: list[float] = []
    for start, end in day_boundaries:
        session_tr = tr[start:end]
        if len(session_tr) >= atr_period:
            atr_values = np.convolve(
                session_tr,
                np.ones(atr_period, dtype=np.float64) / atr_period,
                mode="valid",
            )
            session_atrs.append(float(np.median(atr_values)))
        elif len(session_tr) > 0:
            session_atrs.append(float(np.mean(session_tr)))
        else:
            session_atrs.append(0.0)

    for day_idx, (start, end) in enumerate(day_boundaries):
        if day_idx == 0:
            trailing_val = session_atrs[0] if session_atrs[0] > 0 else 1.0
        else:
            lookback = session_atrs[max(0, day_idx - trailing_days) : day_idx]
            trailing_val = float(np.median(lookback)) if lookback else 1.0
        if trailing_val <= 0:
            trailing_val = 1.0
        result[start:end] = trailing_val

    return result


def _load_legacy_mnq_data(data_path: Path, params_cfg: dict) -> dict:
    shared_cfg = params_cfg["strategy"]["mgc_h1_trend"]["shared"]
    slip_cfg = params_cfg["slippage"]
    data = load_session_data(
        data_path,
        atr_period=slip_cfg["atr_period"],
        trailing_atr_days=slip_cfg["trailing_atr_days"],
        timeframe_minutes=30,
        session_start=shared_cfg["session_start"],
        session_end=shared_cfg["session_end"],
    )
    data["bar_atr"] = _legacy_bar_atr(
        data["high"], data["low"], data["close"], int(slip_cfg["atr_period"])
    )
    data["trailing_median_atr"] = _legacy_trailing_atr(
        data["high"],
        data["low"],
        data["close"],
        data["day_boundaries"],
        int(slip_cfg["atr_period"]),
        int(slip_cfg["trailing_atr_days"]),
    )
    return data


def _build_profiles(shared_cfg: dict, eval_cfg: dict, funded_cfg: dict, risk_buffer_fraction: float) -> tuple[np.ndarray, np.ndarray]:
    strategy_profiles_eval = np.zeros((1, PROFILE_ARRAY_LENGTH), dtype=np.float64)
    strategy_profiles_eval[0, PROFILE_RISK_PER_TRADE_USD] = float(eval_cfg["risk_per_trade_usd"])
    strategy_profiles_eval[0, PROFILE_STOP_ATR_MULTIPLIER] = float(eval_cfg["stop_atr_multiplier"])
    strategy_profiles_eval[0, PROFILE_TARGET_ATR_MULTIPLIER] = float(eval_cfg["target_atr_multiplier"])
    strategy_profiles_eval[0, PROFILE_BREAKEVEN_TRIGGER_TICKS] = float(shared_cfg["breakeven_trigger_ticks"])
    strategy_profiles_eval[0, PROFILE_RISK_BUFFER_FRACTION] = risk_buffer_fraction

    strategy_profiles_funded = np.zeros((1, PROFILE_ARRAY_LENGTH), dtype=np.float64)
    strategy_profiles_funded[0, PROFILE_RISK_PER_TRADE_USD] = float(funded_cfg["risk_per_trade_usd"])
    strategy_profiles_funded[0, PROFILE_STOP_ATR_MULTIPLIER] = float(funded_cfg["stop_atr_multiplier"])
    strategy_profiles_funded[0, PROFILE_TARGET_ATR_MULTIPLIER] = float(funded_cfg["target_atr_multiplier"])
    strategy_profiles_funded[0, PROFILE_BREAKEVEN_TRIGGER_TICKS] = float(shared_cfg["breakeven_trigger_ticks"])
    strategy_profiles_funded[0, PROFILE_RISK_BUFFER_FRACTION] = risk_buffer_fraction
    return strategy_profiles_eval, strategy_profiles_funded


def _compute_raw_metrics(trade_log: np.ndarray) -> dict:
    total_trades = int(len(trade_log))
    if total_trades == 0:
        return {
            "total_trades": 0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "net_profit": 0.0,
            "final_equity": 0.0,
        }

    net_pnl = trade_log["net_pnl"].astype(np.float64)
    gross_profit = float(net_pnl[net_pnl > 0.0].sum())
    gross_loss = float(-net_pnl[net_pnl < 0.0].sum())
    net_profit = float(net_pnl.sum())
    return {
        "total_trades": total_trades,
        "win_rate": float(np.mean(net_pnl > 0.0)),
        "profit_factor": float(gross_profit / gross_loss) if gross_loss > 0.0 else float("inf"),
        "net_profit": net_profit,
        "final_equity": net_profit,
    }


def _compare_against_baseline(baseline_path: Path, trade_log: np.ndarray) -> dict:
    if not baseline_path.exists():
        return {
            "baseline_found": False,
            "baseline_trade_log": str(baseline_path),
        }

    baseline = np.load(baseline_path, allow_pickle=False)
    old_df = pd.DataFrame(
        {
            "entry_time": pd.to_datetime(baseline["entry_time"].astype(np.int64), utc=True).tz_convert(SESSION_TZ),
            "exit_time": pd.to_datetime(baseline["exit_time"].astype(np.int64), utc=True).tz_convert(SESSION_TZ),
            "signal_type": baseline["signal_type"].astype(np.int64),
        }
    )
    new_df = pd.DataFrame(
        {
            "entry_time": pd.to_datetime(trade_log["entry_time"].astype(np.int64), utc=True).tz_convert(SESSION_TZ),
            "exit_time": pd.to_datetime(trade_log["exit_time"].astype(np.int64), utc=True).tz_convert(SESSION_TZ),
            "signal_type": trade_log["signal_type"].astype(np.int64),
        }
    )
    exact = old_df.merge(new_df, on=["entry_time", "exit_time", "signal_type"], how="outer", indicator=True)
    return {
        "baseline_found": True,
        "baseline_trade_log": str(baseline_path),
        "exact_match_counts": exact["_merge"].value_counts().to_dict(),
        "is_exact_match": bool(
            len(old_df) == len(new_df)
            and exact["_merge"].value_counts().to_dict().get("both", 0) == len(old_df)
        ),
    }


def main() -> None:
    args = parse_args()

    mff_cfg = load_mff_config(args.mff_config)
    params_cfg = load_params_config(args.params_config)
    shared_cfg = params_cfg["strategy"]["mgc_h1_trend"]["shared"]
    eval_cfg = params_cfg["strategy"]["mgc_h1_trend"]["eval"]
    funded_cfg = params_cfg["strategy"]["mgc_h1_trend"]["funded"]
    slip_cfg = params_cfg["slippage"]
    portfolio_shared = params_cfg.get("portfolio", {}).get("shared", {})
    risk_buffer_fraction = float(portfolio_shared.get("risk_buffer_fraction", 0.25))

    data = _load_legacy_mnq_data(args.data, params_cfg)
    slippage_lookup = build_slippage_lookup(None, require_file=False, session_minutes=int(data["session_minutes"]))

    params_eval = build_phase_params(
        shared_cfg,
        eval_cfg,
        slip_cfg,
        mff_cfg["instrument"]["commission_per_side"],
        strategy_name="mgc_h1_trend",
        instrument_cfg=mff_cfg["instrument"],
    )
    params_funded = build_phase_params(
        shared_cfg,
        funded_cfg,
        slip_cfg,
        mff_cfg["instrument"]["commission_per_side"],
        strategy_name="mgc_h1_trend",
        instrument_cfg=mff_cfg["instrument"],
    )
    for params in (params_eval, params_funded):
        params[PARAMS_EXTRA_SLIPPAGE_TICKS] = 0.0
        params[PARAMS_ENTRY_ON_CLOSE] = 0.0

    strategy_profiles_eval, strategy_profiles_funded = _build_profiles(
        shared_cfg,
        eval_cfg,
        funded_cfg,
        risk_buffer_fraction,
    )

    max_trades_per_day = int(max(params_eval[PARAMS_MAX_TRADES], params_funded[PARAMS_MAX_TRADES]))
    max_possible_trades = max(1, len(data["day_boundaries"]) * max_trades_per_day)
    all_trades = np.zeros(max_possible_trades, dtype=TRADE_LOG_DTYPE)
    daily_log = np.zeros(len(data["day_boundaries"]), dtype=DAILY_LOG_DTYPE)

    state = MFFState(mff_cfg)
    funded_payout_cycle_id = -1
    total_trade_count = 0
    total_day_count = 0

    for day_idx, (start, end) in enumerate(data["day_boundaries"]):
        session_date = data["session_dates"][day_idx]
        active_params = params_eval.copy() if state.phase == "eval" else params_funded.copy()
        strategy_profiles = (
            strategy_profiles_eval.copy()
            if state.phase == "eval"
            else strategy_profiles_funded.copy()
        )
        phase_id = 0 if state.phase == "eval" else 1
        payout_cycle_id = -1 if state.phase == "eval" else funded_payout_cycle_id
        active_params[PARAMS_CONTRACTS] = float(
            state.get_max_contracts() if state.phase == "funded" else mff_cfg["eval"]["max_contracts"]
        )

        n_trades, _, pnl = run_day_kernel_portfolio(
            data["open"][start:end],
            data["high"][start:end],
            data["low"][start:end],
            data["close"][start:end],
            data["volume"][start:end],
            data["timestamps"][start:end],
            data["minute_of_day"][start:end],
            data["bar_atr"][start:end],
            data["trailing_median_atr"][start:end],
            data["daily_atr_ratio"][start:end],
            data["rvol"][start:end],
            data["close_sma_50"][start:end],
            data["daily_regime_bias"][start:end],
            data["donchian_high_5"][start:end],
            data["donchian_low_5"][start:end],
            data["day_of_week"][start:end],
            slippage_lookup,
            day_idx,
            phase_id,
            payout_cycle_id,
            state.get_liquidation_floor_equity(),
            combined_portfolio_signal,
            all_trades[total_trade_count:],
            0,
            state.equity,
            0.0,
            active_params,
            strategy_profiles,
        )
        total_trade_count += n_trades
        result = state.update_eod(pnl, state.equity + pnl, had_trade=n_trades > 0, session_date=session_date)
        net_payout = 0.0

        if result == "passed":
            state.start_funded()
            funded_payout_cycle_id = 0
        elif state.phase == "funded" and state.payout_eligible:
            net_payout = state.process_payout()
            if net_payout > 0:
                funded_payout_cycle_id += 1

        daily_log[total_day_count]["day_id"] = day_idx
        daily_log[total_day_count]["phase_id"] = phase_id
        daily_log[total_day_count]["payout_cycle_id"] = payout_cycle_id
        daily_log[total_day_count]["had_trade"] = 1 if n_trades > 0 else 0
        daily_log[total_day_count]["n_trades"] = n_trades
        daily_log[total_day_count]["day_pnl"] = pnl
        daily_log[total_day_count]["net_payout"] = net_payout
        total_day_count += 1

        if result == "blown" or state.live_transition_ready:
            break

    trade_log = all_trades[:total_trade_count]
    daily_log = daily_log[:total_day_count]
    raw_metrics = _compute_raw_metrics(trade_log)
    legacy_recovery = _compare_against_baseline(args.baseline_trade_log, trade_log)

    args.output.mkdir(parents=True, exist_ok=True)
    trade_log_path = args.output / "latest_trade_log.npy"
    daily_log_path = args.output / "latest_daily_log.npy"
    np.save(trade_log_path, trade_log)
    np.save(daily_log_path, daily_log)

    report = build_report(
        params={
            "portfolio": {
                "shared": {
                    "strategies": ["mgc_h1_trend"],
                    "risk_buffer_fraction": risk_buffer_fraction,
                    "session_start": shared_cfg["session_start"],
                    "session_end": shared_cfg["session_end"],
                    "trigger_start_minute": int(shared_cfg["trigger_start_minute"]),
                    "trigger_end_minute": int(shared_cfg["trigger_end_minute"]),
                    "timeframe_minutes": int(args.timeframe_minutes),
                    "entry_on_close": False,
                    "atr_mode": "legacy_simple_rolling",
                },
                "profiles_eval": {
                    "mgc_h1_trend": {
                        "risk_per_trade_usd": float(eval_cfg["risk_per_trade_usd"]),
                        "stop_atr_multiplier": float(eval_cfg["stop_atr_multiplier"]),
                        "target_atr_multiplier": float(eval_cfg["target_atr_multiplier"]),
                        "donchian_lookback": int(shared_cfg["donchian_lookback"]),
                        "trigger_start_minute": int(shared_cfg["trigger_start_minute"]),
                        "trigger_end_minute": int(shared_cfg["trigger_end_minute"]),
                        "time_stop_minute": int(shared_cfg["time_stop_minute"]),
                        "breakeven_trigger_ticks": float(shared_cfg["breakeven_trigger_ticks"]),
                    }
                },
            }
        },
        mc_result=None,
        config_snapshot={"params_cfg": params_cfg, "mff_cfg": mff_cfg},
        data_split=args.data.stem,
        data_date_range=(data["session_dates"][0], data["session_dates"][-1]),
        seed=params_cfg["general"]["random_seed"],
        diagnostics={"legacy_recovery": legacy_recovery},
        stress_test={"extra_slippage_ticks": 0.0, "enabled": False},
    )
    report["artifacts"] = {
        "daily_log": str(daily_log_path),
        "trade_log": str(trade_log_path),
        "baseline_trade_log": str(args.baseline_trade_log),
    }
    report["runtime_meta"] = {
        "raw_metrics": {
            **raw_metrics,
            "ending_account_equity": float(state.equity),
        },
        "legacy_recovery": legacy_recovery,
        "payouts_completed": state.payouts_completed,
        "live_transition_ready": state.live_transition_ready,
        "live_transition_reason": state.live_transition_reason,
    }
    save_report(report, args.output / "latest_backtest.json")

    print(f"Trades: {raw_metrics['total_trades']}")
    print(f"Net profit: {raw_metrics['net_profit']:.6f}")
    print(f"Profit factor: {raw_metrics['profit_factor']:.6f}")
    print(f"Exact legacy match: {legacy_recovery['is_exact_match']}")
    print(f"Report: {args.output / 'latest_backtest.json'}")


if __name__ == "__main__":
    main()
