import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "fleet_simulator.py"
SPEC = importlib.util.spec_from_file_location("fleet_simulator", SCRIPT_PATH)
fleet_simulator = importlib.util.module_from_spec(SPEC)
sys.modules["fleet_simulator"] = fleet_simulator
assert SPEC.loader is not None
SPEC.loader.exec_module(fleet_simulator)


Account = fleet_simulator.Account
DEFAULT_FIRST_PAYOUT_MLL = fleet_simulator.DEFAULT_FIRST_PAYOUT_MLL
DEFAULT_ACCOUNT_COST = fleet_simulator.DEFAULT_ACCOUNT_COST
DONCHIAN_PORTFOLIO_OOS_PNLS_PATH = fleet_simulator.DONCHIAN_PORTFOLIO_OOS_PNLS_PATH
load_trade_pool = fleet_simulator.load_trade_pool
simulate_single_path = fleet_simulator.simulate_single_path
run_fleet_simulation = fleet_simulator.run_fleet_simulation
build_yearly_summary_rows = fleet_simulator.build_yearly_summary_rows
format_yearly_summary_table = fleet_simulator.format_yearly_summary_table


def test_first_payout_shifts_mll_and_returns_net_cash():
    account = Account(account_id=0, equity=51_000.0)

    net_payout = account.process_payout()

    assert net_payout == 400.0
    assert account.equity == 50_500.0
    assert account.payout_count == 1
    assert account.mll == DEFAULT_FIRST_PAYOUT_MLL
    assert account.alive is True


def test_blowup_stops_future_pnl_updates():
    account = Account(account_id=0, equity=48_050.0)

    account.apply_trade(-100.0)
    equity_after_breach = account.equity
    account.apply_trade(1_000.0)

    assert account.alive is False
    assert account.equity == equity_after_breach


def test_trade_pool_loader_requires_fixed_artifact_path():
    with pytest.raises(ValueError):
        load_trade_pool(Path("output/other_file.npy"))


def test_trade_pool_loader_reads_fixed_artifact(tmp_path):
    fixed_path = tmp_path / DONCHIAN_PORTFOLIO_OOS_PNLS_PATH.name
    np.save(fixed_path, np.array([1.25, -2.5, 3.75], dtype=np.float64))

    original = fleet_simulator.DONCHIAN_PORTFOLIO_OOS_PNLS_PATH
    fleet_simulator.DONCHIAN_PORTFOLIO_OOS_PNLS_PATH = fixed_path
    try:
        trade_pool, source = load_trade_pool(fixed_path)
    finally:
        fleet_simulator.DONCHIAN_PORTFOLIO_OOS_PNLS_PATH = original

    np.testing.assert_allclose(trade_pool, np.array([1.25, -2.5, 3.75]))
    assert source.startswith("npy:")


def test_simulation_is_reproducible_and_reinvests():
    trade_pool = np.full(200, 400.0, dtype=np.float64)

    result_a = run_fleet_simulation(
        trade_pool=trade_pool,
        n_sims=8,
        months=6,
        trades_per_month=20,
        initial_accounts=1,
        max_accounts=4,
        seed=123,
    )
    result_b = run_fleet_simulation(
        trade_pool=trade_pool,
        n_sims=8,
        months=6,
        trades_per_month=20,
        initial_accounts=1,
        max_accounts=4,
        seed=123,
    )

    np.testing.assert_allclose(result_a.liquid_wealth_paths, result_b.liquid_wealth_paths)
    np.testing.assert_array_equal(result_a.active_account_paths, result_b.active_account_paths)
    assert np.max(result_a.active_account_paths) == 4
    assert np.all(result_a.liquid_wealth_paths >= -DEFAULT_ACCOUNT_COST)


def test_single_path_survival_metric_uses_initial_accounts():
    trade_pool = np.full(50, -2_500.0, dtype=np.float64)
    result = simulate_single_path(
        trade_pool=trade_pool,
        months=3,
        trades_per_month=2,
        initial_accounts=2,
        max_accounts=2,
        rng=np.random.default_rng(7),
    )

    assert result.initial_account_survival_after_12m == 0.0
    assert np.all(result.active_accounts_path == 0)


def test_yearly_summary_rows_and_formatting():
    trade_pool = np.full(200, 300.0, dtype=np.float64)
    result = run_fleet_simulation(
        trade_pool=trade_pool,
        n_sims=4,
        months=24,
        trades_per_month=20,
        initial_accounts=1,
        max_accounts=3,
        seed=9,
    )

    rows = build_yearly_summary_rows(result)
    table = format_yearly_summary_table(rows)

    assert len(rows) == 2
    assert rows[0]["year"] == 1.0
    assert rows[1]["year"] == 2.0
    assert "Year-End Total Capital" in table
