import json
import numpy as np
import pytest
from types import SimpleNamespace
from unittest.mock import patch
from propfirm.optim.walk_forward import (
    _backtest_param_set,
    _build_params_array,
    _serialize_param_overrides,
    run_walk_forward,
)
from propfirm.core.types import (
    TRADE_LOG_DTYPE, DAILY_LOG_DTYPE, PARAMS_ARRAY_LENGTH,
    PARAMS_STOP_TICKS, PARAMS_TARGET_TICKS, PARAMS_CONTRACTS,
    PARAMS_DAILY_STOP, PARAMS_DAILY_TARGET, PARAMS_RANGE_MINUTES,
    PARAMS_MAX_TRADES, PARAMS_BUFFER_TICKS, PARAMS_VOL_THRESHOLD,
    PARAMS_STOP_PENALTY, PARAMS_COMMISSION,
)
from propfirm.market.slippage import build_slippage_lookup


def make_mff_config():
    return {
        "eval": {
            "profit_target": 3000.0,
            "max_loss_limit": 2000.0,
            "consistency_max_pct": 0.50,
            "min_trading_days": 2,
            "max_contracts": 50,
        },
        "funded": {
            "max_loss_limit": 2000.0,
            "mll_frozen_value": 100.0,
            "winning_day_threshold": 150.0,
            "payout_winning_days_required": 5,
            "payout_max_pct": 0.50,
            "payout_cap": 5000.0,
            "payout_min_gross": 250.0,
            "profit_split_trader": 0.80,
            "eval_cost": 107.0,
            "scaling": {
                "tiers": [
                    {"min_profit": -1e9, "max_profit": 1500.0, "max_contracts": 20},
                    {"min_profit": 1500.0, "max_profit": 2000.0, "max_contracts": 30},
                    {"min_profit": 2000.0, "max_profit": 1e9, "max_contracts": 50},
                ],
            },
        },
    }


def make_base_params():
    params = np.zeros(PARAMS_ARRAY_LENGTH, dtype=np.float64)
    params[PARAMS_RANGE_MINUTES] = 15.0
    params[PARAMS_STOP_TICKS] = 40.0
    params[PARAMS_TARGET_TICKS] = 60.0
    params[PARAMS_CONTRACTS] = 10.0
    params[PARAMS_DAILY_STOP] = -750.0
    params[PARAMS_DAILY_TARGET] = 600.0
    params[PARAMS_MAX_TRADES] = 2.0
    params[PARAMS_BUFFER_TICKS] = 2.0
    params[PARAMS_VOL_THRESHOLD] = 0.0
    params[PARAMS_STOP_PENALTY] = 1.5
    params[PARAMS_COMMISSION] = 0.54
    return params


class TestBuildParamsArray:
    def test_eval_overrides_apply_to_eval_only(self):
        base = make_base_params()
        overrides = {
            ("eval", PARAMS_STOP_TICKS): 30.0,
            ("funded", PARAMS_TARGET_TICKS): 100.0,
        }
        result = _build_params_array(base, overrides, "eval")
        assert result[PARAMS_STOP_TICKS] == 30.0
        assert result[PARAMS_TARGET_TICKS] == 60.0

    def test_funded_overrides_apply_to_funded_only(self):
        base = make_base_params()
        overrides = {
            ("eval", PARAMS_STOP_TICKS): 30.0,
            ("funded", PARAMS_TARGET_TICKS): 100.0,
        }
        result = _build_params_array(base, overrides, "funded")
        assert result[PARAMS_STOP_TICKS] == 40.0
        assert result[PARAMS_TARGET_TICKS] == 100.0

    def test_does_not_mutate_base(self):
        base = make_base_params()
        original_stop = base[PARAMS_STOP_TICKS]
        overrides = {("eval", PARAMS_STOP_TICKS): 999.0}
        _build_params_array(base, overrides, "eval")
        assert base[PARAMS_STOP_TICKS] == original_stop


class TestSerializeParamOverrides:
    def test_returns_json_safe_nested_dict(self):
        result = _serialize_param_overrides({
            ("eval", PARAMS_STOP_TICKS): 30.0,
            ("funded", PARAMS_CONTRACTS): 20.0,
        })
        assert result == {
            "eval": {"stop_ticks": 30.0},
            "funded": {"contracts": 20.0},
        }
        json.dumps(result)


class TestBacktestParamSet:
    @pytest.fixture
    def synthetic_session(self):
        bars_per_day = 390
        n_days = 10
        n_bars = bars_per_day * n_days
        base = 20000.0
        data = {
            "open": np.full(n_bars, base, dtype=np.float64),
            "high": np.full(n_bars, base + 5.0, dtype=np.float64),
            "low": np.full(n_bars, base - 5.0, dtype=np.float64),
            "close": np.full(n_bars, base, dtype=np.float64),
            "volume": np.full(n_bars, 1000, dtype=np.uint64),
            "timestamps": np.arange(n_bars, dtype=np.int64) + 1_640_000_000_000_000_000,
            "minute_of_day": np.tile(np.arange(bars_per_day, dtype=np.int16), n_days),
            "bar_atr": np.full(n_bars, 10.0, dtype=np.float64),
            "trailing_median_atr": np.full(n_bars, 10.0, dtype=np.float64),
            "day_boundaries": [(i * bars_per_day, (i + 1) * bars_per_day) for i in range(n_days)],
            "session_dates": [f"2022-01-{3 + i:02d}" for i in range(n_days)],
        }
        return data

    def test_daily_log_has_lifecycle_fields(self, synthetic_session):
        params = make_base_params()
        slippage_lookup = build_slippage_lookup(None)
        result = _backtest_param_set(
            synthetic_session, (0, 5), params, params, slippage_lookup, make_mff_config()
        )
        dl = result["daily_log"]
        assert "phase_id" in dl.dtype.names
        assert "payout_cycle_id" in dl.dtype.names
        assert "day_pnl" in dl.dtype.names
        assert "had_trade" in dl.dtype.names

    def test_eval_days_have_negative_payout_cycle_id(self, synthetic_session):
        params = make_base_params()
        slippage_lookup = build_slippage_lookup(None)
        result = _backtest_param_set(
            synthetic_session, (0, 5), params, params, slippage_lookup, make_mff_config()
        )
        dl = result["daily_log"]
        eval_days = dl[dl["phase_id"] == 0]
        if len(eval_days) > 0:
            assert np.all(eval_days["payout_cycle_id"] == -1)

    def test_funded_days_have_nonneg_payout_cycle_id(self, synthetic_session):
        params = make_base_params()
        slippage_lookup = build_slippage_lookup(None)
        result = _backtest_param_set(
            synthetic_session, (0, 10), params, params, slippage_lookup, make_mff_config()
        )
        dl = result["daily_log"]
        funded_days = dl[dl["phase_id"] == 1]
        if len(funded_days) > 0:
            assert np.all(funded_days["payout_cycle_id"] >= 0)


class TestRunWalkForward:
    @pytest.fixture
    def synthetic_session(self):
        bars_per_day = 390
        n_days = 30
        n_bars = bars_per_day * n_days
        base = 20000.0
        data = {
            "open": np.full(n_bars, base, dtype=np.float64),
            "high": np.full(n_bars, base + 5.0, dtype=np.float64),
            "low": np.full(n_bars, base - 5.0, dtype=np.float64),
            "close": np.full(n_bars, base, dtype=np.float64),
            "volume": np.full(n_bars, 1000, dtype=np.uint64),
            "timestamps": np.arange(n_bars, dtype=np.int64) + 1_640_000_000_000_000_000,
            "minute_of_day": np.tile(np.arange(bars_per_day, dtype=np.int16), n_days),
            "bar_atr": np.full(n_bars, 10.0, dtype=np.float64),
            "trailing_median_atr": np.full(n_bars, 10.0, dtype=np.float64),
            "day_boundaries": [(i * bars_per_day, (i + 1) * bars_per_day) for i in range(n_days)],
            "session_dates": [f"2022-01-{3 + i:02d}" for i in range(n_days)],
        }
        return data

    def test_returns_list_of_window_results(self, synthetic_session):
        base_params = make_base_params()
        slippage_lookup = build_slippage_lookup(None)
        param_grid = {("eval", PARAMS_STOP_TICKS): [30.0, 40.0]}
        results = run_walk_forward(
            synthetic_session, slippage_lookup,
            base_params, base_params, param_grid, make_mff_config(),
            window_train_days=10, window_test_days=5, step_days=5,
            n_mc_sims=20, seed=42, n_workers=1,
        )
        assert isinstance(results, list)
        for r in results:
            assert "window" in r
            assert "in_sample_nve" in r
            assert "oos_nve" in r
            assert "status" in r
            assert "oos_status" in r

    def test_not_scored_windows_are_none_not_zero(self, synthetic_session):
        base_params = make_base_params()
        slippage_lookup = build_slippage_lookup(None)
        param_grid = {("eval", PARAMS_STOP_TICKS): [40.0]}
        results = run_walk_forward(
            synthetic_session, slippage_lookup,
            base_params, base_params, param_grid, make_mff_config(),
            window_train_days=5, window_test_days=3, step_days=3,
            n_mc_sims=10, seed=42, n_workers=1,
        )
        for r in results:
            if r["status"] == "not_scored":
                assert r["in_sample_nve"] is None
            if r["oos_status"] == "not_scored":
                assert r["oos_nve"] is None

    def test_does_not_import_synthetic_grid_search(self):
        import inspect
        from propfirm.optim import walk_forward
        source = inspect.getsource(walk_forward)
        assert "run_synthetic_grid_search" not in source
        assert "generate_synthetic_trades" not in source

    def test_each_result_has_best_params(self, synthetic_session):
        base_params = make_base_params()
        slippage_lookup = build_slippage_lookup(None)
        param_grid = {("eval", PARAMS_STOP_TICKS): [30.0, 40.0]}
        results = run_walk_forward(
            synthetic_session, slippage_lookup,
            base_params, base_params, param_grid, make_mff_config(),
            window_train_days=10, window_test_days=5, step_days=5,
            n_mc_sims=20, seed=42, n_workers=1,
        )
        for r in results:
            if r["status"] == "ok":
                assert r["best_params"] is not None

    def test_window_ranges_use_session_dates(self, synthetic_session):
        base_params = make_base_params()
        slippage_lookup = build_slippage_lookup(None)
        results = run_walk_forward(
            synthetic_session, slippage_lookup,
            base_params, base_params, {("eval", PARAMS_STOP_TICKS): [40.0]}, make_mff_config(),
            window_train_days=10, window_test_days=5, step_days=5,
            n_mc_sims=10, seed=42, n_workers=1,
        )
        assert results[0]["train_date_range"] == (
            synthetic_session["session_dates"][0],
            synthetic_session["session_dates"][9],
        )
        assert results[0]["test_date_range"] == (
            synthetic_session["session_dates"][10],
            synthetic_session["session_dates"][14],
        )

    def test_forwards_mc_block_config_to_monte_carlo(self, synthetic_session):
        base_params = make_base_params()
        slippage_lookup = build_slippage_lookup(None)
        daily_log = np.zeros(12, dtype=DAILY_LOG_DTYPE)
        daily_log["day_id"] = np.arange(12, dtype=np.int32)
        daily_log["phase_id"] = np.array([0] * 6 + [1] * 6, dtype=np.int8)
        daily_log["payout_cycle_id"] = np.array([-1] * 6 + [0] * 6, dtype=np.int16)
        daily_log["had_trade"] = 1
        daily_log["n_trades"] = 1
        daily_log["day_pnl"] = np.linspace(50.0, 160.0, 12)

        with patch(
            "propfirm.optim.walk_forward._backtest_param_set",
            return_value={"trade_log": np.zeros(0, dtype=TRADE_LOG_DTYPE), "daily_log": daily_log},
        ), patch(
            "propfirm.optim.walk_forward.split_daily_log_for_mc",
            return_value={
                "eval_day_pnls": np.array([100.0, 0.0, 120.0], dtype=np.float64),
                "funded_day_pnls": np.array([90.0, 0.0, 110.0], dtype=np.float64),
            },
        ), patch(
            "propfirm.optim.walk_forward.run_monte_carlo",
            return_value=SimpleNamespace(nve=1.0, payout_rate=0.1),
        ) as mock_mc:
            run_walk_forward(
                synthetic_session, slippage_lookup,
                base_params, base_params, {("eval", PARAMS_STOP_TICKS): [40.0]}, make_mff_config(),
                window_train_days=10, window_test_days=5, step_days=5,
                n_mc_sims=11, mc_block_min=7, mc_block_max=9, seed=42, n_workers=1,
            )

        assert mock_mc.call_count >= 1
        for call in mock_mc.call_args_list:
            assert call.kwargs["block_mode"] == "daily"
            assert call.kwargs["block_min"] == 7
            assert call.kwargs["block_max"] == 9
            assert call.kwargs["n_sims"] == 11
