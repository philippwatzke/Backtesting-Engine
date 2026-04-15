from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from propfirm.io.config import build_phase_params


SESSION_ET = "America/New_York"


@dataclass(frozen=True)
class BacktestMetrics:
    trade_count: int
    win_rate: float
    profit_factor: float
    net_profit: float
    max_drawdown: float


def time_string_to_minutes(value: str) -> int:
    ts = pd.Timestamp(f"2000-01-01 {value}")
    return int(ts.hour * 60 + ts.minute)


def _ensure_utc_ns(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, utc=True).astype("datetime64[ns, UTC]")


def load_nt8_signal_feed(csv_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["timestamp_utc"] = _ensure_utc_ns(df["Timestamp_UTC"])
    df = df.rename(
        columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
            "SMA_50": "sma_50",
            "ATR_14_Wilder": "atr_14_wilder",
            "DonchianHigh_5": "donchian_high_5",
            "DonchianLow_5": "donchian_low_5",
            "SignalDirection": "signal_direction",
            "SignalStopDistance": "signal_stop_distance",
            "SignalTargetDistance": "signal_target_distance",
        }
    )
    expected_numeric = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "sma_50",
        "atr_14_wilder",
        "donchian_high_5",
        "donchian_low_5",
        "signal_direction",
        "signal_stop_distance",
        "signal_target_distance",
    ]
    for col in expected_numeric:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df[
        [
            "timestamp_utc",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "sma_50",
            "atr_14_wilder",
            "donchian_high_5",
            "donchian_low_5",
            "signal_direction",
            "signal_stop_distance",
            "signal_target_distance",
        ]
    ]
    return df.sort_values("timestamp_utc", kind="stable").reset_index(drop=True)


def load_nt8_raw_feed(csv_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["timestamp_utc"] = _ensure_utc_ns(df["Timestamp_UTC"])
    df = df.rename(columns={"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"})
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df[["timestamp_utc", "open", "high", "low", "close", "volume"]].dropna(subset=["open", "high", "low", "close"])
    return df.sort_values("timestamp_utc", kind="stable").reset_index(drop=True)


def add_nt8_session_columns(df: pd.DataFrame, session_start: str = "08:00", session_end: str = "15:59") -> pd.DataFrame:
    out = df.copy()
    ts_et = out["timestamp_utc"].dt.tz_convert(SESSION_ET)
    session_start_minute = time_string_to_minutes(session_start)
    session_end_minute = time_string_to_minutes(session_end)
    total_minutes = ts_et.dt.hour * 60 + ts_et.dt.minute
    out["timestamp_et"] = ts_et
    out["session_date_et"] = ts_et.dt.strftime("%Y-%m-%d")
    out["minute_total_et"] = total_minutes.astype(np.int32)
    out["minute_of_day"] = (total_minutes - session_start_minute).astype(np.int32)
    out["day_of_week"] = ts_et.dt.weekday.astype(np.int8)
    out["inside_research_session"] = (total_minutes >= session_start_minute) & (total_minutes <= session_end_minute)
    return out


def compute_performance_metrics(trades_df: pd.DataFrame) -> BacktestMetrics:
    if trades_df.empty:
        return BacktestMetrics(0, 0.0, 0.0, 0.0, 0.0)

    ordered = trades_df.sort_values(["exit_time", "entry_time"], kind="stable").reset_index(drop=True)
    net = ordered["net_pnl"].to_numpy(dtype=np.float64)
    equity = np.cumsum(net)
    peaks = np.maximum.accumulate(equity)
    drawdown = peaks - equity
    gross_profit = float(net[net > 0.0].sum())
    gross_loss = float(-net[net < 0.0].sum())
    profit_factor = float(gross_profit / gross_loss) if gross_loss > 0.0 else (float("inf") if gross_profit > 0.0 else 0.0)
    return BacktestMetrics(
        trade_count=int(len(ordered)),
        win_rate=float(np.mean(net > 0.0)),
        profit_factor=profit_factor,
        net_profit=float(net.sum()),
        max_drawdown=float(drawdown.max()) if len(drawdown) else 0.0,
    )


def split_is_oos_metrics(
    trades_df: pd.DataFrame,
    is_end_date: str = "2024-12-31",
    oos_start_date: str = "2025-01-01",
) -> dict[str, BacktestMetrics]:
    if trades_df.empty:
        empty = compute_performance_metrics(trades_df)
        return {"is": empty, "oos": empty}

    entry_dates = pd.to_datetime(trades_df["entry_session_date"])
    is_mask = entry_dates <= pd.Timestamp(is_end_date)
    oos_mask = entry_dates >= pd.Timestamp(oos_start_date)
    return {
        "is": compute_performance_metrics(trades_df.loc[is_mask].reset_index(drop=True)),
        "oos": compute_performance_metrics(trades_df.loc[oos_mask].reset_index(drop=True)),
    }


def prepare_signal_feed(signal_df: pd.DataFrame, shared_cfg: dict) -> pd.DataFrame:
    return add_nt8_session_columns(
        signal_df,
        session_start=shared_cfg["session_start"],
        session_end=shared_cfg["session_end"],
    )


def generate_signals(
    signal_df: pd.DataFrame,
    shared_cfg: dict,
    phase_cfg: dict | None = None,
    params: np.ndarray | None = None,
    use_daily_regime_filter: bool = True,
) -> pd.DataFrame:
    """Template hook for strategy-specific signal generation.

    The dual-feed executor treats Feed A as the sole HTF truth and Feed B as the
    sole 1m execution truth. This function is intentionally strategy-empty after
    retiring the Donchian/SMA breakout logic. A future hypothesis should
    populate at least:

    - `signal_direction`: `1` for long, `-1` for short, `0` for no trade
    - optionally `signal_stop_distance` / `signal_target_distance` in price units

    If these columns are absent, the engine stays flat by design.
    """

    prepared = prepare_signal_feed(signal_df, shared_cfg).copy()
    if "signal_direction" not in prepared.columns:
        prepared["signal_direction"] = 0
    prepared["signal_direction"] = pd.to_numeric(prepared["signal_direction"], errors="coerce").fillna(0.0).astype(np.int8)

    for col in ["signal_stop_distance", "signal_target_distance"]:
        if col not in prepared.columns:
            prepared[col] = np.nan
        prepared[col] = pd.to_numeric(prepared[col], errors="coerce")
    return prepared


def _prepare_raw_execution_feed(raw_df: pd.DataFrame, shared_cfg: dict) -> tuple[pd.DataFrame, np.ndarray]:
    prepared_input = raw_df.copy()
    prepared_input["timestamp_utc"] = _ensure_utc_ns(prepared_input["timestamp_utc"])
    prepared = add_nt8_session_columns(
        prepared_input,
        session_start=shared_cfg["session_start"],
        session_end=shared_cfg["session_end"],
    )
    raw_ns = prepared["timestamp_utc"].array.asi8.astype(np.int64, copy=False)
    return prepared, raw_ns


def _calculate_contracts(
    stop_ticks: float,
    phase_cfg: dict,
    instrument_cfg: dict,
    max_contracts: int,
) -> int:
    risk_per_trade = float(phase_cfg["risk_per_trade_usd"])
    tick_value = float(instrument_cfg["tick_value"])
    if stop_ticks <= 0.0 or tick_value <= 0.0:
        return 0
    contracts = int(np.floor(risk_per_trade / (stop_ticks * tick_value)))
    if contracts < 1:
        return 0
    return int(min(max_contracts, contracts))


def _find_entry_index(
    raw_df: pd.DataFrame,
    raw_ns: np.ndarray,
    signal_timestamp: pd.Timestamp,
    session_date_et: str,
) -> int:
    signal_ns = signal_timestamp.value
    idx = int(np.searchsorted(raw_ns, signal_ns, side="right"))
    while idx < len(raw_df):
        row = raw_df.iloc[idx]
        if row["session_date_et"] != session_date_et:
            return -1
        if bool(row["inside_research_session"]):
            return idx
        idx += 1
    return -1


def _route_trade_on_raw(
    raw_df: pd.DataFrame,
    entry_idx: int,
    direction: int,
    stop_distance: float,
    target_distance: float,
    tick_size: float,
    tick_value: float,
    commission_per_side: float,
    contracts: int,
) -> dict:
    entry_row = raw_df.iloc[entry_idx]
    entry_time = entry_row["timestamp_utc"]
    entry_session_date = str(entry_row["session_date_et"])
    entry_price = float(entry_row["open"])

    if direction > 0:
        stop_price = entry_price - stop_distance
        target_price = entry_price + target_distance
    else:
        stop_price = entry_price + stop_distance
        target_price = entry_price - target_distance

    exit_time = entry_time
    exit_price = entry_price
    exit_reason = "hard_close"

    for raw_idx in range(entry_idx, len(raw_df)):
        row = raw_df.iloc[raw_idx]
        if str(row["session_date_et"]) != entry_session_date:
            break
        if not bool(row["inside_research_session"]):
            continue

        bar_open = float(row["open"])
        bar_high = float(row["high"])
        bar_low = float(row["low"])
        bar_close = float(row["close"])

        if raw_idx > entry_idx:
            if direction > 0:
                if bar_open <= stop_price:
                    exit_time = row["timestamp_utc"]
                    exit_price = bar_open
                    exit_reason = "stop_gap"
                    break
                if bar_open >= target_price:
                    exit_time = row["timestamp_utc"]
                    exit_price = bar_open
                    exit_reason = "target_gap"
                    break
            else:
                if bar_open >= stop_price:
                    exit_time = row["timestamp_utc"]
                    exit_price = bar_open
                    exit_reason = "stop_gap"
                    break
                if bar_open <= target_price:
                    exit_time = row["timestamp_utc"]
                    exit_price = bar_open
                    exit_reason = "target_gap"
                    break

        if direction > 0:
            stop_hit = bar_low <= stop_price
            target_hit = bar_high >= target_price
            if stop_hit and target_hit:
                exit_time = row["timestamp_utc"]
                exit_price = stop_price
                exit_reason = "stop_same_bar_priority"
                break
            if stop_hit:
                exit_time = row["timestamp_utc"]
                exit_price = stop_price
                exit_reason = "stop"
                break
            if target_hit:
                exit_time = row["timestamp_utc"]
                exit_price = target_price
                exit_reason = "target"
                break
        else:
            stop_hit = bar_high >= stop_price
            target_hit = bar_low <= target_price
            if stop_hit and target_hit:
                exit_time = row["timestamp_utc"]
                exit_price = stop_price
                exit_reason = "stop_same_bar_priority"
                break
            if stop_hit:
                exit_time = row["timestamp_utc"]
                exit_price = stop_price
                exit_reason = "stop"
                break
            if target_hit:
                exit_time = row["timestamp_utc"]
                exit_price = target_price
                exit_reason = "target"
                break

        if int(row["minute_total_et"]) == time_string_to_minutes("15:59"):
            exit_time = row["timestamp_utc"]
            exit_price = bar_close
            exit_reason = "hard_close"
            break

    commission = float(commission_per_side) * float(contracts)
    gross_pnl = (exit_price - entry_price) / tick_size * tick_value * contracts if direction > 0 else (entry_price - exit_price) / tick_size * tick_value * contracts
    net_pnl = gross_pnl - commission - commission

    return {
        "entry_time": entry_time,
        "exit_time": exit_time,
        "entry_price": entry_price,
        "exit_price": exit_price,
        "stop_price": stop_price,
        "target_price": target_price,
        "gross_pnl": gross_pnl,
        "net_pnl": net_pnl,
        "contracts": contracts,
        "exit_reason": exit_reason,
    }


def run_nt8_dual_feed_backtest(
    signal_df: pd.DataFrame,
    raw_df: pd.DataFrame,
    params_cfg: dict,
    mff_cfg: dict,
    phase: str = "eval",
    use_daily_regime_filter: bool = True,
) -> pd.DataFrame:
    if phase not in {"eval", "funded"}:
        raise ValueError("phase must be 'eval' or 'funded'")

    shared_cfg = params_cfg["strategy"]["mgc_h1_trend"]["shared"]
    phase_cfg = params_cfg["strategy"]["mgc_h1_trend"][phase]
    slip_cfg = params_cfg["slippage"]
    instrument_cfg = mff_cfg["instrument"]
    max_contracts = int(mff_cfg[phase]["max_contracts"]) if phase == "eval" else int(mff_cfg["funded"]["scaling"]["tiers"][-1]["max_contracts"])
    params = build_phase_params(
        shared_cfg,
        phase_cfg,
        slip_cfg,
        instrument_cfg["commission_per_side"],
        strategy_name="mgc_h1_trend",
        instrument_cfg=instrument_cfg,
    )

    signal_prepared = generate_signals(
        signal_df,
        shared_cfg,
        phase_cfg=phase_cfg,
        params=params,
        use_daily_regime_filter=use_daily_regime_filter,
    )
    raw_prepared, raw_ns = _prepare_raw_execution_feed(raw_df, shared_cfg)

    trades: list[dict] = []
    current_session_date: str | None = None
    daily_trade_count = 0
    daily_realized_pnl = 0.0

    for bar_idx in range(len(signal_prepared)):
        row = signal_prepared.iloc[bar_idx]
        session_date = str(row["session_date_et"])
        if session_date != current_session_date:
            current_session_date = session_date
            daily_trade_count = 0
            daily_realized_pnl = 0.0

        if not bool(row["inside_research_session"]):
            continue
        if daily_trade_count >= int(shared_cfg["max_trades_day"]):
            continue
        if daily_realized_pnl <= float(phase_cfg["daily_stop"]) or daily_realized_pnl >= float(phase_cfg["daily_target"]):
            continue

        signal = int(row["signal_direction"])
        if signal == 0:
            continue

        stop_distance = float(row["signal_stop_distance"]) if np.isfinite(row["signal_stop_distance"]) else np.nan
        target_distance = float(row["signal_target_distance"]) if np.isfinite(row["signal_target_distance"]) else np.nan
        if not np.isfinite(stop_distance) or not np.isfinite(target_distance):
            atr_value = float(row["atr_14_wilder"])
            if not np.isfinite(atr_value):
                continue
            stop_distance = atr_value * float(phase_cfg["stop_atr_multiplier"])
            target_distance = atr_value * float(phase_cfg["target_atr_multiplier"])

        stop_ticks = max(1.0, stop_distance / float(instrument_cfg["tick_size"]))
        target_ticks = max(1.0, target_distance / float(instrument_cfg["tick_size"]))
        contracts = _calculate_contracts(stop_ticks, phase_cfg, instrument_cfg, max_contracts)
        if contracts < 1:
            continue

        entry_idx = _find_entry_index(raw_prepared, raw_ns, row["timestamp_utc"], session_date)
        if entry_idx < 0:
            continue

        routed = _route_trade_on_raw(
            raw_prepared,
            entry_idx,
            signal,
            stop_distance,
            target_distance,
            float(instrument_cfg["tick_size"]),
            float(instrument_cfg["tick_value"]),
            float(instrument_cfg["commission_per_side"]),
            contracts,
        )
        routed["signal_time"] = row["timestamp_utc"]
        routed["signal_session_date"] = session_date
        routed["entry_session_date"] = session_date
        routed["direction"] = "LONG" if signal > 0 else "SHORT"
        routed["stop_ticks"] = stop_ticks
        routed["target_ticks"] = target_ticks
        trades.append(routed)
        daily_trade_count += 1
        daily_realized_pnl += float(routed["net_pnl"])

    if not trades:
        return pd.DataFrame(
            columns=[
                "signal_time",
                "entry_time",
                "exit_time",
                "signal_session_date",
                "entry_session_date",
                "direction",
                "contracts",
                "entry_price",
                "exit_price",
                "stop_price",
                "target_price",
                "stop_ticks",
                "target_ticks",
                "gross_pnl",
                "net_pnl",
                "exit_reason",
            ]
        )
    result = pd.DataFrame(trades)
    return result.sort_values(["entry_time", "exit_time"], kind="stable").reset_index(drop=True)
