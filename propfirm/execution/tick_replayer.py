from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from numba import njit

from propfirm.core.types import (
    EXIT_HARD_CLOSE,
    EXIT_STOP,
    EXIT_TARGET,
    MNQ_TICK_SIZE,
    MNQ_TICK_VALUE,
)
from propfirm.market.data_loader import SESSION_TZ


BUY_ORDER_SIDE = np.int8(1)
SELL_ORDER_SIDE = np.int8(-1)
UNKNOWN_ORDER_SIDE = np.int8(0)
DEFAULT_HARD_CLOSE_LOOKBACK_SECONDS = 10


def _coerce_timestamp_bound(value: Any) -> pd.Timestamp | None:
    if value is None:
        return None
    if isinstance(value, pd.Timestamp):
        ts = value
    elif isinstance(value, np.datetime64):
        ts = pd.Timestamp(value)
    elif isinstance(value, (np.integer, int)):
        ts = pd.to_datetime(int(value), utc=True, unit="ns")
    else:
        ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def _schema_columns(parquet_path: Path) -> set[str]:
    return set(pq.read_schema(parquet_path).names)


def _required_columns(columns: set[str]) -> tuple[list[str], str]:
    has_price_size_side = {"price", "size", "side"}.issubset(columns)
    has_book_snapshot = {"bid_px_00", "ask_px_00", "bid_sz_00", "ask_sz_00"}.issubset(columns)
    has_book_hints = bool({"action", "depth"} & columns)

    if has_price_size_side and has_book_hints:
        return ["ts_event", "price", "size", "side", "action"], "book_event"
    if has_price_size_side:
        return ["ts_event", "price", "size", "side"], "trade"
    if has_book_snapshot:
        return ["ts_event", "bid_px_00", "ask_px_00", "bid_sz_00", "ask_sz_00"], "book_snapshot"
    raise ValueError(
        "Unsupported Databento parquet schema; expected trade columns "
        "('ts_event', 'price', 'size', 'side') or MBP/BBO columns "
        "('ts_event', 'bid_px_00', 'ask_px_00', 'bid_sz_00', 'ask_sz_00')."
    )


def _load_event_frame(
    parquet_path: Path,
    columns: list[str],
    start_time: pd.Timestamp | None,
    end_time: pd.Timestamp | None,
) -> pd.DataFrame:
    filters: list[tuple[str, str, Any]] = []
    if start_time is not None:
        filters.append(("ts_event", ">=", start_time))
    if end_time is not None:
        filters.append(("ts_event", "<=", end_time))
    frame = pd.read_parquet(
        parquet_path,
        columns=columns,
        filters=filters or None,
        engine="pyarrow",
    )
    if "ts_event" not in frame.columns:
        raise ValueError("Databento parquet must expose 'ts_event' as a column")
    if frame.empty:
        return frame
    frame = frame.sort_values("ts_event", kind="stable").reset_index(drop=True)
    return frame


def _normalize_times(ts_event: pd.Series, timezone: str) -> np.ndarray:
    event_times = pd.to_datetime(ts_event, utc=True)
    if getattr(event_times.dt, "tz", None) is None:
        event_times = event_times.dt.tz_localize("UTC")
    local_times = event_times.dt.tz_convert(timezone)
    ns_dtype = f"datetime64[ns, {timezone}]"
    return local_times.astype(ns_dtype).astype("int64").to_numpy(dtype=np.int64, copy=False)


def _encode_trade_sides(side_values: pd.Series) -> np.ndarray:
    side = side_values.fillna("").astype(str).str.upper().str[0]
    encoded = np.zeros(len(side), dtype=np.int8)
    buy_mask = side.eq("B")
    sell_mask = side.eq("A") | side.eq("S")
    encoded[buy_mask.to_numpy()] = BUY_ORDER_SIDE
    encoded[sell_mask.to_numpy()] = SELL_ORDER_SIDE
    return encoded


def _encode_book_sides(side_values: pd.Series) -> np.ndarray:
    side = side_values.fillna("").astype(str).str.upper().str[0]
    encoded = np.zeros(len(side), dtype=np.int8)
    bid_mask = side.eq("B")
    ask_mask = side.eq("A")
    encoded[ask_mask.to_numpy()] = BUY_ORDER_SIDE
    encoded[bid_mask.to_numpy()] = SELL_ORDER_SIDE
    return encoded


def _collapse_event_schema(
    frame: pd.DataFrame,
    source_type: str,
    timezone: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    tick_times = _normalize_times(frame["ts_event"], timezone=timezone)
    tick_prices = frame["price"].to_numpy(dtype=np.float64, copy=True)
    tick_volumes = frame["size"].fillna(0).to_numpy(dtype=np.int32, copy=True)
    if source_type == "trade":
        tick_sides = _encode_trade_sides(frame["side"])
    else:
        tick_sides = _encode_book_sides(frame["side"])
    valid = (
        np.isfinite(tick_prices)
        & (tick_volumes > 0)
        & (tick_sides != UNKNOWN_ORDER_SIDE)
    )
    return (
        tick_times[valid],
        tick_prices[valid],
        tick_volumes[valid],
        tick_sides[valid],
    )


def _collapse_snapshot_schema(
    frame: pd.DataFrame,
    timezone: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    tick_times = _normalize_times(frame["ts_event"], timezone=timezone)
    bid_prices = frame["bid_px_00"].to_numpy(dtype=np.float64, copy=True)
    ask_prices = frame["ask_px_00"].to_numpy(dtype=np.float64, copy=True)
    bid_sizes = frame["bid_sz_00"].fillna(0).to_numpy(dtype=np.int32, copy=True)
    ask_sizes = frame["ask_sz_00"].fillna(0).to_numpy(dtype=np.int32, copy=True)

    bid_valid = np.isfinite(bid_prices) & (bid_sizes > 0)
    ask_valid = np.isfinite(ask_prices) & (ask_sizes > 0)
    total = int(bid_valid.sum() + ask_valid.sum())

    out_times = np.empty(total, dtype=np.int64)
    out_prices = np.empty(total, dtype=np.float64)
    out_volumes = np.empty(total, dtype=np.int32)
    out_sides = np.empty(total, dtype=np.int8)

    write_idx = 0
    for idx in range(len(frame)):
        if ask_valid[idx]:
            out_times[write_idx] = tick_times[idx]
            out_prices[write_idx] = ask_prices[idx]
            out_volumes[write_idx] = ask_sizes[idx]
            out_sides[write_idx] = BUY_ORDER_SIDE
            write_idx += 1
        if bid_valid[idx]:
            out_times[write_idx] = tick_times[idx]
            out_prices[write_idx] = bid_prices[idx]
            out_volumes[write_idx] = bid_sizes[idx]
            out_sides[write_idx] = SELL_ORDER_SIDE
            write_idx += 1

    order = np.argsort(out_times, kind="stable")
    return (
        out_times[order],
        out_prices[order],
        out_volumes[order],
        out_sides[order],
    )


def load_databento_ticks(
    parquet_path: str | Path,
    start_time: Any = None,
    end_time: Any = None,
    timezone: str = SESSION_TZ,
) -> dict[str, Any]:
    """Load Databento trades or top-of-book data into replay arrays.

    The returned ``tick_times`` array is compatible with the existing session
    data and trade logs: it is stored as UTC nanoseconds after normalizing the
    source timestamps through the session timezone.
    """
    parquet_path = Path(parquet_path)
    if not parquet_path.exists():
        raise FileNotFoundError(f"Databento parquet not found: {parquet_path}")

    start_ts = _coerce_timestamp_bound(start_time)
    end_ts = _coerce_timestamp_bound(end_time)
    if start_ts is not None and end_ts is not None and end_ts < start_ts:
        raise ValueError("end_time must be greater than or equal to start_time")

    columns = _schema_columns(parquet_path)
    required_cols, source_type = _required_columns(columns)
    frame = _load_event_frame(
        parquet_path=parquet_path,
        columns=required_cols,
        start_time=start_ts,
        end_time=end_ts,
    )
    if frame.empty:
        return {
            "tick_times": np.empty(0, dtype=np.int64),
            "tick_prices": np.empty(0, dtype=np.float64),
            "tick_volumes": np.empty(0, dtype=np.int32),
            "tick_sides": np.empty(0, dtype=np.int8),
            "source_type": source_type,
            "timezone": timezone,
        }

    if source_type in {"trade", "book_event"}:
        tick_times, tick_prices, tick_volumes, tick_sides = _collapse_event_schema(frame, source_type, timezone)
    else:
        tick_times, tick_prices, tick_volumes, tick_sides = _collapse_snapshot_schema(frame, timezone)

    return {
        "tick_times": tick_times,
        "tick_prices": tick_prices,
        "tick_volumes": tick_volumes,
        "tick_sides": tick_sides,
        "source_type": source_type,
        "timezone": timezone,
    }


@njit(cache=True)
def _lower_bound(times: np.ndarray, value: np.int64) -> int:
    lo = 0
    hi = len(times)
    while lo < hi:
        mid = (lo + hi) // 2
        if times[mid] < value:
            lo = mid + 1
        else:
            hi = mid
    return lo


@njit(cache=True)
def _upper_bound(times: np.ndarray, value: np.int64) -> int:
    lo = 0
    hi = len(times)
    while lo < hi:
        mid = (lo + hi) // 2
        if times[mid] <= value:
            lo = mid + 1
        else:
            hi = mid
    return lo


@njit(cache=True)
def _signed_order_side(signal_type: int) -> np.int8:
    if signal_type > 0:
        return BUY_ORDER_SIDE
    if signal_type < 0:
        return SELL_ORDER_SIDE
    return UNKNOWN_ORDER_SIDE


@njit(cache=True)
def _simulate_market_fill_from_index(
    start_idx: int,
    order_side: np.int8,
    quantity: int,
    tick_prices: np.ndarray,
    tick_volumes: np.ndarray,
    tick_sides: np.ndarray,
) -> float:
    if quantity <= 0 or order_side == UNKNOWN_ORDER_SIDE or start_idx >= len(tick_prices):
        return np.nan

    remaining = quantity
    cumulative_notional = 0.0
    filled = 0
    last_price = np.nan

    for tick_idx in range(start_idx, len(tick_prices)):
        if tick_sides[tick_idx] != order_side:
            continue
        price = tick_prices[tick_idx]
        volume = int(tick_volumes[tick_idx])
        if volume <= 0 or not np.isfinite(price):
            continue

        take_qty = volume
        if take_qty > remaining:
            take_qty = remaining
        cumulative_notional += price * take_qty
        filled += take_qty
        remaining -= take_qty
        last_price = price

        if remaining == 0:
            return cumulative_notional / filled

    if filled == 0 or not np.isfinite(last_price):
        return np.nan

    cumulative_notional += last_price * remaining
    return cumulative_notional / quantity


@njit(cache=True)
def _simulate_market_fill(
    start_time: np.int64,
    order_side: np.int8,
    quantity: int,
    tick_times: np.ndarray,
    tick_prices: np.ndarray,
    tick_volumes: np.ndarray,
    tick_sides: np.ndarray,
) -> float:
    start_idx = _lower_bound(tick_times, start_time)
    return _simulate_market_fill_from_index(
        start_idx,
        order_side,
        quantity,
        tick_prices,
        tick_volumes,
        tick_sides,
    )


@njit(cache=True)
def _simulate_market_fill_backward(
    end_time: np.int64,
    order_side: np.int8,
    quantity: int,
    tick_times: np.ndarray,
    tick_prices: np.ndarray,
    tick_volumes: np.ndarray,
    tick_sides: np.ndarray,
    lookback_ns: np.int64,
) -> float:
    if quantity <= 0 or order_side == UNKNOWN_ORDER_SIDE:
        return np.nan

    start_time = end_time - lookback_ns
    start_idx = _lower_bound(tick_times, start_time)
    end_idx = _lower_bound(tick_times, end_time)
    remaining = quantity
    cumulative_notional = 0.0
    filled = 0
    oldest_price = np.nan

    for tick_idx in range(end_idx - 1, start_idx - 1, -1):
        if tick_sides[tick_idx] != order_side:
            continue
        price = tick_prices[tick_idx]
        volume = int(tick_volumes[tick_idx])
        if volume <= 0 or not np.isfinite(price):
            continue

        take_qty = volume
        if take_qty > remaining:
            take_qty = remaining
        cumulative_notional += price * take_qty
        filled += take_qty
        remaining -= take_qty
        oldest_price = price

        if remaining == 0:
            return cumulative_notional / filled

    if filled == 0 or not np.isfinite(oldest_price):
        return np.nan

    cumulative_notional += oldest_price * remaining
    return cumulative_notional / quantity


@njit(cache=True)
def _trigger_touched(
    price: float,
    trigger_price: float,
    entry_side: np.int8,
    exit_reason: int,
) -> bool:
    if not np.isfinite(price) or not np.isfinite(trigger_price):
        return False
    if exit_reason == EXIT_STOP:
        if entry_side == BUY_ORDER_SIDE:
            return price <= trigger_price
        return price >= trigger_price
    if exit_reason == EXIT_TARGET:
        if entry_side == BUY_ORDER_SIDE:
            return price >= trigger_price
        return price <= trigger_price
    return False


@njit(cache=True)
def _find_trigger_touch_index(
    start_time: np.int64,
    order_side: np.int8,
    entry_side: np.int8,
    exit_reason: int,
    trigger_price: float,
    tick_times: np.ndarray,
    tick_prices: np.ndarray,
    tick_sides: np.ndarray,
) -> int:
    start_idx = _lower_bound(tick_times, start_time)
    for tick_idx in range(start_idx, len(tick_times)):
        if tick_sides[tick_idx] != order_side:
            continue
        if _trigger_touched(tick_prices[tick_idx], trigger_price, entry_side, exit_reason):
            return tick_idx
    return -1


@njit(cache=True)
def simulate_tick_execution(
    trade_log: np.ndarray,
    tick_times: np.ndarray,
    tick_prices: np.ndarray,
    tick_volumes: np.ndarray,
    tick_sides: np.ndarray,
    tick_size: float = MNQ_TICK_SIZE,
    tick_value: float = MNQ_TICK_VALUE,
    hard_close_lookback_seconds: int = DEFAULT_HARD_CLOSE_LOOKBACK_SECONDS,
    hard_close_bar_seconds: int = 0,
) -> np.ndarray:
    """Replay theoretical fills against tick liquidity and return a new log."""
    result = trade_log.copy()
    lookback_ns = np.int64(hard_close_lookback_seconds) * np.int64(1_000_000_000)
    hard_close_bar_ns = np.int64(hard_close_bar_seconds) * np.int64(1_000_000_000)

    for trade_idx in range(len(result)):
        contracts = int(result[trade_idx]["contracts"])
        entry_time = np.int64(result[trade_idx]["entry_time"])
        exit_time = np.int64(result[trade_idx]["exit_time"])
        signal_type = int(result[trade_idx]["signal_type"])
        exit_reason = int(result[trade_idx]["exit_reason"])

        if contracts <= 0 or entry_time <= 0 or exit_time <= 0:
            continue

        entry_side = _signed_order_side(signal_type)
        if entry_side == UNKNOWN_ORDER_SIDE:
            continue
        exit_side = np.int8(-entry_side)

        theoretical_entry = float(result[trade_idx]["entry_price"])
        theoretical_exit = float(result[trade_idx]["exit_price"])
        theoretical_exit_slippage = float(result[trade_idx]["exit_slippage"])
        simulated_entry = _simulate_market_fill(
            entry_time,
            entry_side,
            contracts,
            tick_times,
            tick_prices,
            tick_volumes,
            tick_sides,
        )

        simulated_exit = np.nan
        if exit_reason == EXIT_HARD_CLOSE:
            hard_close_anchor = exit_time
            if hard_close_bar_ns > 0:
                hard_close_anchor = exit_time + hard_close_bar_ns
            simulated_exit = _simulate_market_fill_backward(
                hard_close_anchor,
                exit_side,
                contracts,
                tick_times,
                tick_prices,
                tick_volumes,
                tick_sides,
                lookback_ns,
            )
        elif exit_reason == EXIT_STOP or exit_reason == EXIT_TARGET:
            trigger_price = theoretical_exit - exit_side * theoretical_exit_slippage
            trigger_idx = _find_trigger_touch_index(
                exit_time,
                exit_side,
                entry_side,
                exit_reason,
                trigger_price,
                tick_times,
                tick_prices,
                tick_sides,
            )
            if trigger_idx >= 0:
                simulated_exit = _simulate_market_fill_from_index(
                    trigger_idx,
                    exit_side,
                    contracts,
                    tick_prices,
                    tick_volumes,
                    tick_sides,
                )
        else:
            simulated_exit = _simulate_market_fill(
                exit_time,
                exit_side,
                contracts,
                tick_times,
                tick_prices,
                tick_volumes,
                tick_sides,
            )

        if not np.isfinite(simulated_entry):
            simulated_entry = theoretical_entry
        if not np.isfinite(simulated_exit):
            simulated_exit = theoretical_exit

        entry_slippage = (simulated_entry - theoretical_entry) * entry_side
        exit_slippage = (simulated_exit - theoretical_exit) * exit_side

        if entry_side == BUY_ORDER_SIDE:
            gross_pnl = (simulated_exit - simulated_entry) * contracts / tick_size * tick_value
        else:
            gross_pnl = (simulated_entry - simulated_exit) * contracts / tick_size * tick_value
        net_pnl = gross_pnl - float(result[trade_idx]["entry_commission"]) - float(result[trade_idx]["exit_commission"])

        result[trade_idx]["entry_price"] = simulated_entry
        result[trade_idx]["exit_price"] = simulated_exit
        result[trade_idx]["entry_slippage"] = entry_slippage
        result[trade_idx]["exit_slippage"] = exit_slippage
        result[trade_idx]["gross_pnl"] = gross_pnl
        result[trade_idx]["net_pnl"] = net_pnl

    return result


def infer_trade_bar_seconds(trade_log: np.ndarray) -> int:
    """Infer the dominant bar interval from observed trade timestamps."""
    timestamps = np.concatenate(
        [
            trade_log["entry_time"][trade_log["entry_time"] > 0],
            trade_log["exit_time"][trade_log["exit_time"] > 0],
        ]
    )
    if timestamps.size < 2:
        return 0

    unique_times = np.unique(np.sort(timestamps.astype(np.int64, copy=False)))
    diffs = np.diff(unique_times)
    diffs = diffs[diffs > 0]
    if diffs.size == 0:
        return 0

    diff_seconds = np.rint(diffs.astype(np.float64) / 1_000_000_000.0).astype(np.int64)
    diff_seconds = diff_seconds[diff_seconds > 0]
    if diff_seconds.size == 0:
        return 0

    return int(diff_seconds.min())


def _average_or_nan(values: np.ndarray) -> float:
    if values.size == 0:
        return float("nan")
    return float(values.mean())


def _std_or_nan(values: np.ndarray) -> float:
    if values.size == 0:
        return float("nan")
    return float(values.std())


def _format_ticks(value: float) -> str:
    if not np.isfinite(value):
        return "n/a"
    return f"{value:.4f}"


def _isoformat_ns(timestamp_ns: int) -> str:
    return pd.to_datetime(int(timestamp_ns), utc=True, unit="ns").isoformat()


def _trigger_touched_py(
    price: float,
    trigger_price: float,
    entry_side: int,
    exit_reason: int,
) -> bool:
    if not np.isfinite(price) or not np.isfinite(trigger_price):
        return False
    if exit_reason == EXIT_STOP:
        if entry_side == BUY_ORDER_SIDE:
            return price <= trigger_price
        return price >= trigger_price
    if exit_reason == EXIT_TARGET:
        if entry_side == BUY_ORDER_SIDE:
            return price >= trigger_price
        return price <= trigger_price
    return False


def _fill_completion_time_from_index(
    start_idx: int,
    order_side: int,
    quantity: int,
    tick_times: np.ndarray,
    tick_volumes: np.ndarray,
    tick_sides: np.ndarray,
) -> int | None:
    if quantity <= 0 or start_idx >= len(tick_times):
        return None
    remaining = int(quantity)
    last_time: int | None = None
    for tick_idx in range(start_idx, len(tick_times)):
        if int(tick_sides[tick_idx]) != int(order_side):
            continue
        volume = int(tick_volumes[tick_idx])
        if volume <= 0:
            continue
        remaining -= min(volume, remaining)
        last_time = int(tick_times[tick_idx])
        if remaining == 0:
            return last_time
    return last_time


def _build_trade_diagnostic(
    theoretical_trade: np.void,
    simulated_trade: np.void,
    tick_times: np.ndarray,
    tick_prices: np.ndarray,
    tick_volumes: np.ndarray,
    tick_sides: np.ndarray,
    tick_size: float,
    hard_close_bar_seconds: int,
) -> dict[str, float | int | str]:
    signal_sign = 1.0 if int(theoretical_trade["signal_type"]) > 0 else -1.0
    entry_side = BUY_ORDER_SIDE if signal_sign > 0 else SELL_ORDER_SIDE
    exit_side = -entry_side
    entry_slip_ticks = (
        (float(simulated_trade["entry_price"]) - float(theoretical_trade["entry_price"])) * signal_sign
    ) / tick_size
    exit_slip_ticks = (
        (float(simulated_trade["exit_price"]) - float(theoretical_trade["exit_price"])) * (-signal_sign)
    ) / tick_size
    total_slip_ticks = entry_slip_ticks + exit_slip_ticks

    exit_time = int(theoretical_trade["exit_time"])
    exit_fill_time = exit_time
    exit_reason = int(theoretical_trade["exit_reason"])
    contracts = int(theoretical_trade["contracts"])

    time_to_fill_ms = 0.0
    if exit_reason == EXIT_HARD_CLOSE:
        exit_fill_time = exit_time
    elif exit_reason == EXIT_STOP or exit_reason == EXIT_TARGET:
        trigger_price = float(theoretical_trade["exit_price"]) - float(exit_side) * float(theoretical_trade["exit_slippage"])
        start_idx = int(np.searchsorted(tick_times, exit_time, side="left"))
        trigger_idx = -1
        for tick_idx in range(start_idx, len(tick_times)):
            if int(tick_sides[tick_idx]) != int(exit_side):
                continue
            if _trigger_touched_py(float(tick_prices[tick_idx]), trigger_price, int(entry_side), exit_reason):
                trigger_idx = tick_idx
                break
        if trigger_idx >= 0:
            fill_time = _fill_completion_time_from_index(
                trigger_idx,
                int(exit_side),
                contracts,
                tick_times,
                tick_volumes,
                tick_sides,
            )
            if fill_time is not None:
                exit_fill_time = fill_time
    else:
        start_idx = int(np.searchsorted(tick_times, exit_time, side="left"))
        fill_time = _fill_completion_time_from_index(
            start_idx,
            int(exit_side),
            contracts,
            tick_times,
            tick_volumes,
            tick_sides,
        )
        if fill_time is not None:
            exit_fill_time = fill_time

    if exit_fill_time >= exit_time and exit_reason != EXIT_HARD_CLOSE:
        time_to_fill_ms = float((exit_fill_time - exit_time) / 1_000_000.0)

    return {
        "timestamp": _isoformat_ns(int(theoretical_trade["entry_time"])),
        "entry_slippage_ticks": float(entry_slip_ticks),
        "exit_slippage_ticks": float(exit_slip_ticks),
        "total_slippage_ticks": float(total_slip_ticks),
        "time_to_fill_ms": float(time_to_fill_ms),
    }


def compare_trade_logs(
    theoretical_trade_log: np.ndarray,
    simulated_trade_log: np.ndarray,
    tick_size: float = MNQ_TICK_SIZE,
    tick_value: float = MNQ_TICK_VALUE,
) -> dict[str, float | int | list[dict[str, float | int | str]]]:
    """Compare theoretical and replayed fills at the trade-log level."""
    if len(theoretical_trade_log) != len(simulated_trade_log):
        raise ValueError("Trade logs must have the same length")
    if tick_size <= 0.0:
        raise ValueError("tick_size must be positive")

    valid = (
        (theoretical_trade_log["contracts"] > 0)
        & (theoretical_trade_log["entry_time"] > 0)
        & (theoretical_trade_log["exit_time"] > 0)
        & (theoretical_trade_log["signal_type"] != 0)
    )
    if not np.any(valid):
        return {
            "n_trades": 0,
            "average_entry_slippage_ticks": 0.0,
            "average_exit_slippage_ticks": 0.0,
            "average_exit_slippage_ticks_target": float("nan"),
            "average_exit_slippage_ticks_stop": float("nan"),
            "average_exit_slippage_ticks_hard_close": float("nan"),
            "stop_exit_slippage_std_ticks": float("nan"),
            "stop_exit_slippage_3sigma_usd_per_contract": float("nan"),
            "theoretical_net_pnl": 0.0,
            "simulated_tick_pnl": 0.0,
            "toxic_trades": [],
        }

    theoretical = theoretical_trade_log[valid]
    simulated = simulated_trade_log[valid]
    signal_sign = np.sign(theoretical["signal_type"]).astype(np.float64)
    exit_side = -signal_sign

    entry_slippage_ticks = ((simulated["entry_price"] - theoretical["entry_price"]) * signal_sign) / tick_size
    exit_slippage_ticks = ((simulated["exit_price"] - theoretical["exit_price"]) * exit_side) / tick_size
    target_mask = theoretical["exit_reason"] == EXIT_TARGET
    stop_mask = theoretical["exit_reason"] == EXIT_STOP
    hard_close_mask = theoretical["exit_reason"] == EXIT_HARD_CLOSE
    stop_std_ticks = _std_or_nan(exit_slippage_ticks[stop_mask])

    return {
        "n_trades": int(valid.sum()),
        "average_entry_slippage_ticks": float(entry_slippage_ticks.mean()),
        "average_exit_slippage_ticks": float(exit_slippage_ticks.mean()),
        "average_exit_slippage_ticks_target": _average_or_nan(exit_slippage_ticks[target_mask]),
        "average_exit_slippage_ticks_stop": _average_or_nan(exit_slippage_ticks[stop_mask]),
        "average_exit_slippage_ticks_hard_close": _average_or_nan(exit_slippage_ticks[hard_close_mask]),
        "stop_exit_slippage_std_ticks": stop_std_ticks,
        "stop_exit_slippage_3sigma_usd_per_contract": (
            float(3.0 * stop_std_ticks * tick_value) if np.isfinite(stop_std_ticks) else float("nan")
        ),
        "theoretical_net_pnl": float(theoretical["net_pnl"].sum()),
        "simulated_tick_pnl": float(simulated["net_pnl"].sum()),
        "toxic_trades": [],
    }


def format_reality_report(metrics: dict[str, float | int | list[dict[str, float | int | str]]]) -> str:
    toxic_trades = metrics.get("toxic_trades", [])
    lines = [
        f"Trades: {int(metrics['n_trades'])}",
        f"Average Entry Slippage (Ticks): {float(metrics['average_entry_slippage_ticks']):.4f}",
        f"Average Exit Slippage (Ticks): {float(metrics['average_exit_slippage_ticks']):.4f}",
        f"Theoretischer Net PnL: {float(metrics['theoretical_net_pnl']):,.2f}",
        f"Simulierter Tick PnL: {float(metrics['simulated_tick_pnl']):,.2f}",
        "",
        "Slippage by Exit Reason (Durchschnitt in Ticks):",
        f"- TARGET Exits: {_format_ticks(float(metrics.get('average_exit_slippage_ticks_target', float('nan'))))} Ticks",
        f"- STOP Exits: {_format_ticks(float(metrics.get('average_exit_slippage_ticks_stop', float('nan'))))} Ticks",
        f"- HARD CLOSE Exits: {_format_ticks(float(metrics.get('average_exit_slippage_ticks_hard_close', float('nan'))))} Ticks",
        "",
        'The "Toxic Trades" Top 3:',
    ]
    if toxic_trades:
        for idx, trade in enumerate(toxic_trades[:3], start=1):
            lines.append(
                f"- #{idx} {trade['timestamp']} | "
                f"Entry-Slippage: {float(trade['entry_slippage_ticks']):.2f} Ticks | "
                f"Exit-Slippage: {float(trade['exit_slippage_ticks']):.2f} Ticks | "
                f"Time-to-Fill: {float(trade['time_to_fill_ms']):.3f} ms"
            )
    else:
        lines.append("- n/a")

    lines.extend(["", "Limit-Breaker Warnung:"])
    stop_std_ticks = float(metrics.get("stop_exit_slippage_std_ticks", float("nan")))
    sigma_cost = float(metrics.get("stop_exit_slippage_3sigma_usd_per_contract", float("nan")))
    if np.isfinite(stop_std_ticks) and np.isfinite(sigma_cost):
        lines.append(
            f"- Ein 3-Sigma Slippage Event bei einem Stop-Loss kostet uns theoretisch "
            f"{sigma_cost:.2f} USD pro Kontrakt."
        )
    else:
        lines.append("- Nicht genug STOP-Exits fuer eine stabile 3-Sigma-Schaetzung.")
    return "\n".join(lines)


def run_tick_replay_report(
    theoretical_trade_log: np.ndarray,
    parquet_path: str | Path,
    timezone: str = SESSION_TZ,
    tick_size: float = MNQ_TICK_SIZE,
    tick_value: float = MNQ_TICK_VALUE,
    hard_close_lookback_seconds: int = DEFAULT_HARD_CLOSE_LOOKBACK_SECONDS,
    hard_close_bar_seconds: int | None = None,
) -> tuple[np.ndarray, dict[str, float | int]]:
    """Load ticks, replay fills, and return the simulated log plus summary metrics."""
    tail_buffer_ns = int(pd.Timedelta(minutes=5).value)
    valid_mask = (
        (theoretical_trade_log["contracts"] > 0)
        & (theoretical_trade_log["entry_time"] > 0)
        & (theoretical_trade_log["exit_time"] > 0)
    )
    simulated_trade_log = theoretical_trade_log.copy()
    trade_diagnostics: list[dict[str, float | int | str]] = []
    if hard_close_bar_seconds is None:
        hard_close_bar_seconds = infer_trade_bar_seconds(theoretical_trade_log)

    for trade_idx in np.flatnonzero(valid_mask):
        trade_slice = theoretical_trade_log[trade_idx : trade_idx + 1]
        start_time = int(trade_slice["entry_time"][0])
        end_time = int(trade_slice["exit_time"][0]) + tail_buffer_ns
        tick_data = load_databento_ticks(
            parquet_path=parquet_path,
            start_time=start_time,
            end_time=end_time,
            timezone=timezone,
        )
        simulated_slice = simulate_tick_execution(
            trade_slice,
            tick_data["tick_times"],
            tick_data["tick_prices"],
            tick_data["tick_volumes"],
            tick_data["tick_sides"],
            tick_size=tick_size,
            tick_value=tick_value,
            hard_close_lookback_seconds=hard_close_lookback_seconds,
            hard_close_bar_seconds=hard_close_bar_seconds,
        )
        simulated_trade_log[trade_idx] = simulated_slice[0]
        trade_diagnostics.append(
            _build_trade_diagnostic(
                trade_slice[0],
                simulated_slice[0],
                tick_data["tick_times"],
                tick_data["tick_prices"],
                tick_data["tick_volumes"],
                tick_data["tick_sides"],
                tick_size=tick_size,
                hard_close_bar_seconds=hard_close_bar_seconds,
            )
        )

    metrics = compare_trade_logs(
        theoretical_trade_log=theoretical_trade_log,
        simulated_trade_log=simulated_trade_log,
        tick_size=tick_size,
        tick_value=tick_value,
    )
    metrics["toxic_trades"] = sorted(
        trade_diagnostics,
        key=lambda row: float(row["total_slippage_ticks"]),
        reverse=True,
    )[:3]
    return simulated_trade_log, metrics
