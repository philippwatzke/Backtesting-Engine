import numpy as np
import pytest

from propfirm.core.multi_engine import run_multi_asset_day_kernel
from propfirm.core.types import (
    EXIT_CIRCUIT_BREAKER,
    PARAMS_ARRAY_LENGTH,
    PARAMS_COMMISSION,
    PARAMS_CONTRACTS,
    PARAMS_DAILY_STOP,
    PARAMS_EXTRA_SLIPPAGE_TICKS,
    PARAMS_MAX_TRADES,
    PARAMS_STOP_PENALTY,
    PARAMS_TICK_SIZE,
    PARAMS_TICK_VALUE,
    PARAMS_TIME_STOP_MINUTE,
    PROFILE_ARRAY_LENGTH,
    PROFILE_BREAKEVEN_TRIGGER_TICKS,
    PROFILE_RISK_BUFFER_FRACTION,
    PROFILE_RISK_PER_TRADE_USD,
    PROFILE_STOP_ATR_MULTIPLIER,
    PROFILE_TARGET_ATR_MULTIPLIER,
    SIGNAL_LONG,
    SIGNAL_NONE,
    TRADE_LOG_DTYPE,
)


def first_bar_long_strategy(
    bar_idx,
    opens,
    highs,
    lows,
    closes,
    volumes,
    bar_atr,
    trailing_atr,
    daily_atr_ratio,
    rvol,
    close_sma_50,
    daily_regime_bias,
    donchian_high_5,
    donchian_low_5,
    minute_of_day,
    day_of_week,
    equity,
    intraday_pnl,
    position,
    entry_price,
    halted,
    daily_trade_count,
    params,
):
    if bar_idx == 0 and position == 0 and not halted:
        return SIGNAL_LONG
    return SIGNAL_NONE


def _make_params(tick_value: float) -> np.ndarray:
    params = np.zeros(PARAMS_ARRAY_LENGTH, dtype=np.float64)
    params[PARAMS_CONTRACTS] = 1.0
    params[PARAMS_DAILY_STOP] = -10_000.0
    params[PARAMS_MAX_TRADES] = 1.0
    params[PARAMS_STOP_PENALTY] = 1.0
    params[PARAMS_COMMISSION] = 0.0
    params[PARAMS_EXTRA_SLIPPAGE_TICKS] = 0.0
    params[PARAMS_TIME_STOP_MINUTE] = 999.0
    params[PARAMS_TICK_SIZE] = 1.0
    params[PARAMS_TICK_VALUE] = tick_value
    return params


def _make_profiles() -> np.ndarray:
    profiles = np.zeros((2, PROFILE_ARRAY_LENGTH), dtype=np.float64)
    profiles[0, PROFILE_RISK_PER_TRADE_USD] = 2_000.0
    profiles[0, PROFILE_STOP_ATR_MULTIPLIER] = 20.0
    profiles[0, PROFILE_TARGET_ATR_MULTIPLIER] = 1_000.0
    profiles[0, PROFILE_BREAKEVEN_TRIGGER_TICKS] = 0.0
    profiles[0, PROFILE_RISK_BUFFER_FRACTION] = 1.0
    return profiles


def _make_asset(name: str, timestamps: np.ndarray, closes: np.ndarray, tick_value: float, final_open: float) -> dict:
    opens = np.array([100.0, 100.0, 100.0, final_open], dtype=np.float64)
    highs = np.maximum(opens, closes) + 1.0
    lows = np.minimum(opens, closes) - 1.0
    volumes = np.full(len(opens), 100, dtype=np.uint64)
    minute_of_day = np.array([0, 30, 60, 90], dtype=np.int16)
    bar_atr = np.full(len(opens), 10.0, dtype=np.float64)
    trailing_atr = np.full(len(opens), 10.0, dtype=np.float64)
    slippage_lookup = np.ones(600, dtype=np.float64)

    return {
        "name": name,
        "opens": opens,
        "highs": highs,
        "lows": lows,
        "closes": closes,
        "volumes": volumes,
        "timestamps": timestamps,
        "minute_of_day": minute_of_day,
        "bar_atr": bar_atr,
        "trailing_atr": trailing_atr,
        "slippage_lookup": slippage_lookup,
        "strategy_fn": first_bar_long_strategy,
        "strategy_profiles": _make_profiles(),
        "trade_log": np.zeros(4, dtype=TRADE_LOG_DTYPE),
        "params": _make_params(tick_value),
        "starting_equity": 10_000.0,
        "starting_pnl": 0.0,
        "liquidation_floor_equity": 0.0,
    }


def test_global_circuit_breaker_closes_both_assets_on_next_open():
    mgc = _make_asset(
        "MGC",
        np.array([100, 200, 300, 400], dtype=np.int64),
        np.array([100.0, 100.0, 1.0, 0.0], dtype=np.float64),
        tick_value=5.0,
        final_open=0.0,
    )
    mnq = _make_asset(
        "MNQ",
        np.array([150, 250, 350, 450], dtype=np.int64),
        np.array([100.0, 100.0, 58.0, 50.0], dtype=np.float64),
        tick_value=7.0,
        final_open=50.0,
    )

    result = run_multi_asset_day_kernel([mgc, mnq], circuit_breaker_threshold=-800.0)

    mgc_log = mgc["trade_log"][: result["assets"]["MGC"]["n_trades"]]
    mnq_log = mnq["trade_log"][: result["assets"]["MNQ"]["n_trades"]]

    assert result["global_halt"] is True
    assert result["trigger_timestamp"] == 350
    assert result["trigger_global_pnl"] == pytest.approx(-801.0)
    assert result["trigger_snapshot"]["MGC"]["position"] == 1
    assert result["trigger_snapshot"]["MNQ"]["position"] == 1
    assert result["trigger_snapshot"]["MGC"]["unrealized_pnl"] == pytest.approx(-500.0)
    assert result["trigger_snapshot"]["MNQ"]["unrealized_pnl"] == pytest.approx(-301.0)

    assert len(mgc_log) == 1
    assert len(mnq_log) == 1
    assert mgc_log[0]["exit_reason"] == EXIT_CIRCUIT_BREAKER
    assert mnq_log[0]["exit_reason"] == EXIT_CIRCUIT_BREAKER
    assert mgc_log[0]["exit_time"] == 400
    assert mnq_log[0]["exit_time"] == 450
    assert result["assets"]["MGC"]["forced_exit_count"] == 1
    assert result["assets"]["MNQ"]["forced_exit_count"] == 1


def test_empty_asset_day_is_ignored_while_other_asset_runs():
    mgc = {
        "name": "MGC",
        "opens": np.empty(0, dtype=np.float64),
        "highs": np.empty(0, dtype=np.float64),
        "lows": np.empty(0, dtype=np.float64),
        "closes": np.empty(0, dtype=np.float64),
        "volumes": np.empty(0, dtype=np.uint64),
        "timestamps": np.empty(0, dtype=np.int64),
        "minute_of_day": np.empty(0, dtype=np.int16),
        "bar_atr": np.empty(0, dtype=np.float64),
        "trailing_atr": np.empty(0, dtype=np.float64),
        "daily_atr_ratio": np.empty(0, dtype=np.float64),
        "rvol": np.empty(0, dtype=np.float64),
        "close_sma_50": np.empty(0, dtype=np.float64),
        "daily_regime_bias": np.empty(0, dtype=np.float64),
        "donchian_high_5": np.empty(0, dtype=np.float64),
        "donchian_low_5": np.empty(0, dtype=np.float64),
        "day_of_week": np.empty(0, dtype=np.int8),
        "slippage_lookup": np.ones(600, dtype=np.float64),
        "strategy_fn": first_bar_long_strategy,
        "strategy_profiles": _make_profiles(),
        "trade_log": np.zeros(1, dtype=TRADE_LOG_DTYPE),
        "params": _make_params(5.0),
        "starting_equity": 10_000.0,
        "starting_pnl": 0.0,
        "liquidation_floor_equity": 0.0,
    }
    mnq = _make_asset(
        "MNQ",
        np.array([150, 250, 350, 450], dtype=np.int64),
        np.array([100.0, 120.0, 130.0, 140.0], dtype=np.float64),
        tick_value=7.0,
        final_open=140.0,
    )

    result = run_multi_asset_day_kernel([mgc, mnq], circuit_breaker_threshold=-800.0)

    assert result["global_halt"] is False
    assert result["assets"]["MGC"]["n_trades"] == 0
    assert result["assets"]["MNQ"]["n_trades"] == 1
