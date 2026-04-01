import pytest
from propfirm.optim.grid_search import run_synthetic_grid_search, generate_synthetic_trades


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


class TestGenerateSyntheticTrades:
    def test_returns_array(self):
        trades = generate_synthetic_trades(
            win_rate=0.60, reward_ticks=60.0, risk_ticks=40.0,
            contracts=10, n_trades=100, seed=42,
        )
        assert len(trades) == 100

    def test_win_rate_approximate(self):
        trades = generate_synthetic_trades(
            win_rate=0.60, reward_ticks=60.0, risk_ticks=40.0,
            contracts=10, n_trades=10000, seed=42,
        )
        actual_wr = (trades > 0).sum() / len(trades)
        assert abs(actual_wr - 0.60) < 0.05

    def test_deterministic(self):
        t1 = generate_synthetic_trades(0.6, 60.0, 40.0, 10, 100, 42)
        t2 = generate_synthetic_trades(0.6, 60.0, 40.0, 10, 100, 42)
        assert (t1 == t2).all()


class TestRunSyntheticGridSearch:
    def test_returns_ranked_results(self):
        param_grid = {
            "win_rate": [0.55, 0.65],
            "risk_reward": [1.0, 1.5],
            "contracts": [10],
        }
        results = run_synthetic_grid_search(
            param_grid, make_mff_config(),
            n_mc_sims=50, seed=42, n_workers=1,
        )
        assert len(results) > 0
        nves = [r["nve"] for r in results]
        assert nves == sorted(nves, reverse=True)

    def test_each_result_has_required_keys(self):
        param_grid = {
            "win_rate": [0.60],
            "risk_reward": [1.5],
            "contracts": [10],
        }
        results = run_synthetic_grid_search(
            param_grid, make_mff_config(),
            n_mc_sims=50, seed=42, n_workers=1,
        )
        r = results[0]
        assert "params" in r
        assert "nve" in r
        assert "eval_pass_rate" in r
        assert "payout_rate" in r
        assert "funded_survival_rate" in r
        assert "daily_stop" not in r["params"]
