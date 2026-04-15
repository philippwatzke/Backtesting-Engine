#!/usr/bin/env python
"""Export exact Python-frozen signal-bar values for focused MNQ legacy dates."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from propfirm.core.types import (
    PARAMS_CONTRACTS,
    PARAMS_ENTRY_ON_CLOSE,
    PARAMS_EXTRA_SLIPPAGE_TICKS,
    PARAMS_TICK_SIZE,
    PARAMS_TICK_VALUE,
    PROFILE_BREAKEVEN_TRIGGER_TICKS,
    PROFILE_RISK_BUFFER_FRACTION,
    PROFILE_RISK_PER_TRADE_USD,
    PROFILE_STOP_ATR_MULTIPLIER,
    PROFILE_TARGET_ATR_MULTIPLIER,
)
from propfirm.io.config import build_phase_params, load_mff_config, load_params_config
from propfirm.rules.mff import MFFState
from propfirm.strategy.portfolio import combined_portfolio_signal
from run_mnq_legacy_frozen import _build_profiles, _load_legacy_mnq_data


FOCUS_DATES = [
    "2025-09-03",
    "2025-09-11",
    "2025-09-12",
    "2025-09-22",
    "2025-10-01",
    "2025-10-27",
    "2026-01-27",
]

SESSION_TZ = "America/New_York"
DATA_PATH = Path("data/processed/MNQ_1m_full_test.parquet")
MFF_CONFIG = Path("configs/mff_flex_50k_mnq.toml")
PARAMS_CONFIG = Path("configs/default_params_mnq_legacy_frozen.toml")
TRADE_LOG_PATH = Path("output/backtests_mnq_legacy_frozen/latest_trade_log.npy")
DAILY_LOG_PATH = Path("output/backtests_mnq_legacy_frozen/latest_daily_log.npy")
OUTPUT_DIR = Path("output/tradingview_forensics/mnq_legacy_python_focus")


def _load_trade_frame(path: Path) -> pd.DataFrame:
    arr = np.load(path, allow_pickle=False)
    df = pd.DataFrame(
        {
            "entry_time_et": pd.to_datetime(arr["entry_time"].astype(np.int64), utc=True).tz_convert(SESSION_TZ),
            "exit_time_et": pd.to_datetime(arr["exit_time"].astype(np.int64), utc=True).tz_convert(SESSION_TZ),
            "entry_price": arr["entry_price"].astype(np.float64),
            "exit_price": arr["exit_price"].astype(np.float64),
            "contracts": arr["contracts"].astype(np.int64),
            "net_pnl": arr["net_pnl"].astype(np.float64),
            "signal_type": arr["signal_type"].astype(np.int64),
            "exit_reason": arr["exit_reason"].astype(np.int64),
        }
    )
    df["entry_date_et"] = df["entry_time_et"].dt.strftime("%Y-%m-%d")
    df["signal_time_et"] = df["entry_time_et"] - pd.Timedelta(minutes=30)
    return df


def _load_daily_frame(path: Path, session_dates: list[str]) -> pd.DataFrame:
    arr = np.load(path, allow_pickle=False)
    df = pd.DataFrame(
        {
            "day_id": arr["day_id"].astype(np.int64),
            "phase_id": arr["phase_id"].astype(np.int64),
            "had_trade": arr["had_trade"].astype(np.int64),
            "n_trades": arr["n_trades"].astype(np.int64),
            "day_pnl": arr["day_pnl"].astype(np.float64),
            "net_payout": arr["net_payout"].astype(np.float64),
        }
    )
    df["session_date"] = session_dates[: len(df)]
    return df


def _signal_label(signal: int) -> str:
    if signal > 0:
        return "LONG"
    if signal < 0:
        return "SHORT"
    return "NONE"


def main() -> None:
    params_cfg = load_params_config(PARAMS_CONFIG)
    mff_cfg = load_mff_config(MFF_CONFIG)
    shared_cfg = params_cfg["strategy"]["mgc_h1_trend"]["shared"]
    eval_cfg = params_cfg["strategy"]["mgc_h1_trend"]["eval"]
    funded_cfg = params_cfg["strategy"]["mgc_h1_trend"]["funded"]
    slip_cfg = params_cfg["slippage"]
    risk_buffer_fraction = float(params_cfg.get("portfolio", {}).get("shared", {}).get("risk_buffer_fraction", 0.25))

    data = _load_legacy_mnq_data(DATA_PATH, params_cfg)
    trade_df = _load_trade_frame(TRADE_LOG_PATH)
    daily_df = _load_daily_frame(DAILY_LOG_PATH, data["session_dates"])

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

    state = MFFState(mff_cfg)
    rows: list[dict] = []

    for day_idx, (start, end) in enumerate(data["day_boundaries"]):
        session_date = data["session_dates"][day_idx]
        active_params = params_eval.copy() if state.phase == "eval" else params_funded.copy()
        strategy_profiles = strategy_profiles_eval.copy() if state.phase == "eval" else strategy_profiles_funded.copy()
        active_params[PARAMS_CONTRACTS] = float(
            state.get_max_contracts() if state.phase == "funded" else mff_cfg["eval"]["max_contracts"]
        )

        if session_date in FOCUS_DATES:
            day_trades = trade_df.loc[trade_df["entry_date_et"] == session_date].copy()
            actual_signal_times = set(day_trades["signal_time_et"].tolist())
            actual_entry_times = set(day_trades["entry_time_et"].tolist())

            for local_idx in range(end - start):
                global_idx = start + local_idx
                mod = int(data["minute_of_day"][global_idx])
                signal = combined_portfolio_signal(
                    local_idx,
                    data["open"][start:end],
                    data["high"][start:end],
                    data["low"][start:end],
                    data["close"][start:end],
                    data["volume"][start:end],
                    data["bar_atr"][start:end],
                    data["trailing_median_atr"][start:end],
                    data["daily_atr_ratio"][start:end],
                    data["rvol"][start:end],
                    data["close_sma_50"][start:end],
                    data["daily_regime_bias"][start:end],
                    data["donchian_high_5"][start:end],
                    data["donchian_low_5"][start:end],
                    data["minute_of_day"][start:end],
                    data["day_of_week"][start:end],
                    state.equity,
                    0.0,
                    0,
                    0.0,
                    False,
                    0,
                    active_params,
                )
                ts_et = pd.Timestamp(int(data["timestamps"][global_idx]), unit="ns", tz="UTC").tz_convert(SESSION_TZ)
                if signal == 0:
                    continue

                profile_idx = abs(int(signal)) - 1
                risk_per_trade_usd = float(strategy_profiles[profile_idx, PROFILE_RISK_PER_TRADE_USD])
                stop_atr_multiplier = float(strategy_profiles[profile_idx, PROFILE_STOP_ATR_MULTIPLIER])
                target_atr_multiplier = float(strategy_profiles[profile_idx, PROFILE_TARGET_ATR_MULTIPLIER])
                risk_buffer_fraction_bar = float(strategy_profiles[profile_idx, PROFILE_RISK_BUFFER_FRACTION])
                _ = float(strategy_profiles[profile_idx, PROFILE_BREAKEVEN_TRIGGER_TICKS])

                liquidation_floor_equity = float(state.get_liquidation_floor_equity())
                drawdown_buffer = float(state.equity - liquidation_floor_equity)
                risk_cap = drawdown_buffer * risk_buffer_fraction_bar
                effective_risk = min(risk_per_trade_usd, risk_cap)

                current_atr = float(data["bar_atr"][global_idx])
                tick_size = float(active_params[PARAMS_TICK_SIZE])
                tick_value = float(active_params[PARAMS_TICK_VALUE])
                dynamic_stop_ticks = (current_atr * stop_atr_multiplier) / tick_size
                dynamic_target_ticks = (current_atr * target_atr_multiplier) / tick_size
                risk_per_contract = dynamic_stop_ticks * tick_value
                contracts = int(np.floor(effective_risk / risk_per_contract)) if risk_per_contract > 0 else 0
                max_contracts = int(active_params[3])
                if max_contracts > 0 and contracts > max_contracts:
                    contracts = max_contracts

                rows.append(
                    {
                        "session_date": session_date,
                        "phase": state.phase,
                        "equity_before_day": float(state.equity),
                        "liquidation_floor_equity": liquidation_floor_equity,
                        "drawdown_buffer": drawdown_buffer,
                        "risk_cap": effective_risk,
                        "signal_time_et": str(ts_et),
                        "entry_time_et": str(ts_et + pd.Timedelta(minutes=30)),
                        "minute_of_day": mod,
                        "signal": int(signal),
                        "signal_label": _signal_label(int(signal)),
                        "close": float(data["close"][global_idx]),
                        "high": float(data["high"][global_idx]),
                        "low": float(data["low"][global_idx]),
                        "sma50": float(data["close_sma_50"][global_idx]),
                        "atr14_legacy": current_atr,
                        "donchian_high_5": float(data["donchian_high_5"][global_idx]),
                        "donchian_low_5": float(data["donchian_low_5"][global_idx]),
                        "daily_regime_bias": float(data["daily_regime_bias"][global_idx]),
                        "stop_atr_multiplier": stop_atr_multiplier,
                        "target_atr_multiplier": target_atr_multiplier,
                        "stop_ticks": dynamic_stop_ticks,
                        "target_ticks": dynamic_target_ticks,
                        "contracts_python": contracts,
                        "is_actual_trade_signal_bar": ts_et in actual_signal_times,
                        "is_actual_trade_entry_plus_30m": (ts_et + pd.Timedelta(minutes=30)) in actual_entry_times,
                    }
                )

        if day_idx < len(daily_df):
            day_row = daily_df.iloc[day_idx]
            result = state.update_eod(
                float(day_row["day_pnl"]),
                float(state.equity + day_row["day_pnl"]),
                had_trade=bool(day_row["had_trade"]),
                session_date=session_date,
            )
            if result == "passed":
                state.start_funded()
            elif state.phase == "funded" and state.payout_eligible:
                state.process_payout()
            if result == "blown" or state.live_transition_ready:
                break

    result_df = pd.DataFrame(rows).sort_values(["session_date", "signal_time_et"], kind="stable")
    focus_actual_df = result_df.loc[result_df["is_actual_trade_signal_bar"]].copy()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(OUTPUT_DIR / "focus_signal_candidates.csv", index=False)
    focus_actual_df.to_csv(OUTPUT_DIR / "focus_actual_trade_signals.csv", index=False)

    summary = {
        "focus_dates": FOCUS_DATES,
        "artifacts": {
            "focus_signal_candidates": str(OUTPUT_DIR / "focus_signal_candidates.csv"),
            "focus_actual_trade_signals": str(OUTPUT_DIR / "focus_actual_trade_signals.csv"),
        },
        "candidate_rows": int(len(result_df)),
        "actual_trade_signal_rows": int(len(focus_actual_df)),
    }
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("MNQ legacy focus signal export")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Candidate rows: {len(result_df)}")
    print(f"Actual trade signal rows: {len(focus_actual_df)}")


if __name__ == "__main__":
    main()
