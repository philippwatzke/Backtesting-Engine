import numpy as np
import pytest
from propfirm.monte_carlo.bootstrap import (
    block_bootstrap_single,
    split_daily_log_for_mc,
    run_monte_carlo,
    MCResult,
    _simulate_single_path,
)


def make_winning_trades(n=50):
    return np.array([100.0] * n, dtype=np.float64)

def make_losing_trades(n=50):
    return np.array([-200.0] * n, dtype=np.float64)

def make_mixed_trades(n=100):
    rng = np.random.RandomState(42)
    trades = np.where(rng.rand(n) < 0.6, 150.0, -100.0)
    return trades

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


def make_structured_daily_log():
    log = np.zeros(6, dtype=[
        ("day_id", "i4"), ("phase_id", "i1"), ("payout_cycle_id", "i2"),
        ("had_trade", "i1"), ("n_trades", "i2"), ("day_pnl", "f8"), ("net_payout", "f8")
    ])
    log["day_id"] = np.array([0, 1, 2, 3, 4, 5], dtype=np.int32)
    log["phase_id"] = np.array([0, 0, 0, 1, 1, 1], dtype=np.int8)
    log["payout_cycle_id"] = np.array([-1, -1, -1, 0, 0, 1], dtype=np.int16)
    log["had_trade"] = np.array([1, 0, 1, 1, 0, 1], dtype=np.int8)
    log["n_trades"] = np.array([2, 0, 1, 2, 0, 1], dtype=np.int16)
    log["day_pnl"] = np.array([50.0, 0.0, 120.0, 90.0, 0.0, 60.0])
    log["net_payout"] = np.array([0.0, 0.0, 0.0, 0.0, 200.0, 0.0])
    return log


class TestBlockBootstrapSingle:
    def test_returns_sequence_of_correct_length(self):
        trades = make_mixed_trades(100)
        seq = block_bootstrap_single(trades, target_length=50, seed=42, block_min=5, block_max=10)
        assert len(seq) >= 50

    def test_deterministic_with_same_seed(self):
        trades = make_mixed_trades(100)
        r1 = block_bootstrap_single(trades, 50, seed=42, block_min=5, block_max=10)
        r2 = block_bootstrap_single(trades, 50, seed=42, block_min=5, block_max=10)
        np.testing.assert_array_equal(r1, r2)

    def test_different_with_different_seed(self):
        trades = make_mixed_trades(100)
        r1 = block_bootstrap_single(trades, 50, seed=42, block_min=5, block_max=10)
        r2 = block_bootstrap_single(trades, 50, seed=99, block_min=5, block_max=10)
        assert not np.array_equal(r1, r2)

    def test_daily_block_mode_bootstraps_day_level_series(self):
        daily_values = np.array([50.0, 0.0, 120.0, -30.0, 90.0], dtype=np.float64)
        seq = block_bootstrap_single(daily_values, 8, seed=42, block_min=2, block_max=3)
        assert len(seq) >= 8
        assert 0.0 in seq

    def test_split_daily_log_for_mc_keeps_zero_trade_days(self):
        pools = split_daily_log_for_mc(make_structured_daily_log())
        np.testing.assert_array_equal(pools["eval_day_pnls"], np.array([50.0, 0.0, 120.0]))
        np.testing.assert_array_equal(pools["funded_day_pnls"], np.array([90.0, 0.0]))
        assert 60.0 not in pools["funded_day_pnls"]


class TestRunMonteCarlo:
    def test_returns_mc_result_with_disaggregated_rates(self):
        trades = make_mixed_trades(100)
        result = run_monte_carlo(trades, make_mff_config(), funded_pnls=trades,
                                 n_sims=100, seed=42, n_workers=1, block_mode="fixed",
                                 block_min=5, block_max=10)
        assert isinstance(result, MCResult)
        assert 0.0 <= result.eval_pass_rate <= 1.0
        assert 0.0 <= result.funded_survival_rate <= 1.0
        assert 0.0 <= result.payout_rate <= 1.0
        assert result.payout_rate <= result.eval_pass_rate + 1e-9

    def test_winning_trades_high_eval_pass_rate(self):
        trades = make_winning_trades(100)
        result = run_monte_carlo(trades, make_mff_config(), funded_pnls=trades,
                                 n_sims=200, seed=42, n_workers=1, block_mode="fixed",
                                 block_min=5, block_max=10)
        assert result.eval_pass_rate > 0.5

    def test_losing_trades_zero_rates(self):
        trades = make_losing_trades(50)
        result = run_monte_carlo(trades, make_mff_config(), funded_pnls=trades,
                                 n_sims=100, seed=42, n_workers=1, block_mode="fixed",
                                 block_min=5, block_max=10)
        assert result.eval_pass_rate == 0.0
        assert result.payout_rate == 0.0

    def test_reproducible(self):
        trades = make_mixed_trades(100)
        r1 = run_monte_carlo(trades, make_mff_config(), funded_pnls=trades,
                             n_sims=100, block_min=5, block_max=10,
                             seed=42, n_workers=1, block_mode="fixed")
        r2 = run_monte_carlo(trades, make_mff_config(), funded_pnls=trades,
                             n_sims=100, block_min=5, block_max=10,
                             seed=42, n_workers=1, block_mode="fixed")
        assert r1.eval_pass_rate == r2.eval_pass_rate
        assert r1.payout_rate == r2.payout_rate

    def test_ci_brackets_eval_pass_rate(self):
        trades = make_mixed_trades(200)
        result = run_monte_carlo(trades, make_mff_config(), funded_pnls=trades,
                                 n_sims=500, seed=42, n_workers=1, block_mode="fixed",
                                 block_min=5, block_max=10)
        if 0.1 < result.eval_pass_rate < 0.9:
            assert result.eval_pass_rate_ci_5 < result.eval_pass_rate
            assert result.eval_pass_rate_ci_95 > result.eval_pass_rate
            assert result.eval_pass_rate_ci_5 > 0.0
            assert result.eval_pass_rate_ci_95 < 1.0

    def test_multiprocessing(self):
        trades = make_mixed_trades(100)
        result = run_monte_carlo(trades, make_mff_config(), funded_pnls=trades,
                                 n_sims=200, seed=42, n_workers=2, block_mode="fixed",
                                 block_min=5, block_max=10)
        assert isinstance(result, MCResult)
        assert 0.0 <= result.eval_pass_rate <= 1.0

    def test_daily_mode_requires_explicit_funded_day_inputs(self):
        day_pnls = np.array([50.0, 0.0, 120.0], dtype=np.float64)
        with pytest.raises(ValueError):
            run_monte_carlo(day_pnls, make_mff_config(), n_sims=10, seed=42,
                            n_workers=1, block_mode="daily")

    def test_empty_inputs_rejected(self):
        empty = np.array([], dtype=np.float64)
        with pytest.raises(ValueError):
            run_monte_carlo(empty, make_mff_config(), funded_pnls=empty,
                            n_sims=10, seed=42, n_workers=1, block_mode="fixed")

    def test_drawdown_uses_full_lifecycle_max(self):
        eval_seq = np.array([2500.0, -400.0, 1000.0], dtype=np.float64)
        funded_seq = np.array([200.0, 200.0, 200.0, 200.0, 200.0], dtype=np.float64)
        result = _simulate_single_path(eval_seq, funded_seq, make_mff_config(),
                                        eval_trades_per_day=1, funded_trades_per_day=1)
        assert result["eval_passed"] == True
        assert result["payout_net"] > 0.0
        assert result["drawdown"] >= 400.0
