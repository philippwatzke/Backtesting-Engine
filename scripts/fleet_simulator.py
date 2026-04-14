#!/usr/bin/env python
"""Simulate a 60-month MyFundedFutures Flex fleet scaling journey.

The simulator models a simplified cash-flow layer for a fleet of accounts:
- 20 bootstrapped trades per month
- 2 payout checks per month
- reinvestment into new accounts once liquid wealth covers account cost

The simulator loads a single deterministic OOS trade-PnL artifact:
`output/donchian_portfolio_oos_pnls.npy`.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DEFAULT_EQUITY = 50_000.0
DEFAULT_MLL = 48_000.0
DEFAULT_FIRST_PAYOUT_MLL = 50_100.0
DEFAULT_ACCOUNT_COST = 150.0
DEFAULT_MONTHS = 60
DEFAULT_TRADES_PER_MONTH = 20
DEFAULT_N_SIMS = 1_000
DEFAULT_MAX_ACCOUNTS = 15
DEFAULT_IMPACT_PER_EXTRA_ACCOUNT = 0.15
DEFAULT_PAYOUT_CHECKS_PER_MONTH = 2
DONCHIAN_PORTFOLIO_OOS_PNLS_PATH = Path("output/donchian_portfolio_oos_pnls.npy")


@dataclass
class Account:
    """Single-account state for the fleet simulator."""

    account_id: int
    equity: float = DEFAULT_EQUITY
    mll: float = DEFAULT_MLL
    payout_count: int = 0
    cost: float = DEFAULT_ACCOUNT_COST
    last_equity_after_payout: float = DEFAULT_EQUITY
    alive: bool = True

    def apply_trade(self, trade_pnl: float) -> None:
        if not self.alive:
            return
        self.equity += float(trade_pnl)
        if self.equity <= self.mll:
            self.alive = False

    def process_payout(self) -> float:
        if not self.alive or self.equity <= DEFAULT_EQUITY:
            return 0.0

        available_profit = self.equity - max(DEFAULT_EQUITY, self.last_equity_after_payout)
        gross_payout = min(5_000.0, available_profit * 0.5)
        if gross_payout <= 0.0:
            return 0.0

        self.equity -= gross_payout
        self.payout_count += 1
        self.last_equity_after_payout = self.equity

        if self.payout_count == 1:
            self.mll = DEFAULT_FIRST_PAYOUT_MLL
            if self.equity <= self.mll:
                self.alive = False

        return gross_payout * 0.8


@dataclass
class FleetPathResult:
    liquid_wealth_path: np.ndarray
    active_accounts_path: np.ndarray
    deployed_equity_path: np.ndarray
    total_fleet_capital_path: np.ndarray
    monthly_net_payout_path: np.ndarray
    initial_account_survival_after_12m: float


@dataclass
class FleetSimulationResult:
    liquid_wealth_paths: np.ndarray
    active_account_paths: np.ndarray
    deployed_equity_paths: np.ndarray
    total_fleet_capital_paths: np.ndarray
    monthly_net_payout_paths: np.ndarray
    median_final_liquid_wealth: float
    median_final_deployed_equity: float
    median_final_total_fleet_capital: float
    survival_rate_initial_after_12m: float
    p90_final_liquid_wealth: float
    median_final_year_avg_monthly_payout: float
    median_final_year_avg_annual_payout: float


def build_yearly_summary_rows(result: FleetSimulationResult) -> list[dict[str, float]]:
    months = result.liquid_wealth_paths.shape[1]
    n_years = months // 12
    rows: list[dict[str, float]] = []
    if n_years == 0:
        return rows

    for year_idx in range(n_years):
        month_start = year_idx * 12
        month_end = month_start + 12
        rows.append(
            {
                "year": float(year_idx + 1),
                "median_avg_monthly_payout": float(
                    np.median(np.mean(result.monthly_net_payout_paths[:, month_start:month_end], axis=1))
                ),
                "median_year_end_liquid_wealth": float(
                    np.median(result.liquid_wealth_paths[:, month_end - 1])
                ),
                "median_year_end_deployed_equity": float(
                    np.median(result.deployed_equity_paths[:, month_end - 1])
                ),
                "median_year_end_total_capital": float(
                    np.median(result.total_fleet_capital_paths[:, month_end - 1])
                ),
            }
        )
    return rows


def format_yearly_summary_table(rows: list[dict[str, float]]) -> str:
    if not rows:
        return "No yearly summary available."

    header = (
        "Year | Median Avg Monthly Payout | Year-End Liquid Wealth | "
        "Year-End Deployed Equity | Year-End Total Capital"
    )
    separator = "-" * len(header)
    body = [
        (
            f"{int(row['year']):>4} | "
            f"${row['median_avg_monthly_payout']:>24,.2f} | "
            f"${row['median_year_end_liquid_wealth']:>21,.2f} | "
            f"${row['median_year_end_deployed_equity']:>24,.2f} | "
            f"${row['median_year_end_total_capital']:>21,.2f}"
        )
        for row in rows
    ]
    return "\n".join([header, separator, *body])


def load_trade_pool(
    trade_pool_path: Path = DONCHIAN_PORTFOLIO_OOS_PNLS_PATH,
) -> tuple[np.ndarray, str]:
    if trade_pool_path != DONCHIAN_PORTFOLIO_OOS_PNLS_PATH:
        raise ValueError(
            "fleet_simulator is locked to output/donchian_portfolio_oos_pnls.npy"
        )
    if not trade_pool_path.exists():
        raise FileNotFoundError(
            f"Required OOS trade artifact not found: {trade_pool_path}"
        )
    trade_pool = np.load(trade_pool_path, allow_pickle=False).astype(np.float64)
    if trade_pool.ndim != 1:
        trade_pool = trade_pool.reshape(-1)
    if len(trade_pool) == 0:
        raise ValueError(f"No values found in {trade_pool_path}")
    return trade_pool, f"npy:{trade_pool_path}"


def bootstrap_month_trades(
    trade_pool: np.ndarray,
    trades_per_month: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if trades_per_month <= 0:
        raise ValueError("trades_per_month must be positive")
    if len(trade_pool) == 0:
        raise ValueError("trade_pool must be non-empty")
    return rng.choice(trade_pool, size=trades_per_month, replace=True).astype(np.float64)


def _reinvest_accounts(
    accounts: list[Account],
    liquid_wealth: float,
    next_account_id: int,
    max_accounts: int,
    account_cost: float,
) -> tuple[float, int]:
    while liquid_wealth >= account_cost and sum(account.alive for account in accounts) < max_accounts:
        liquid_wealth -= account_cost
        accounts.append(Account(account_id=next_account_id, cost=account_cost))
        next_account_id += 1
    return liquid_wealth, next_account_id


def simulate_single_path(
    trade_pool: np.ndarray,
    months: int = DEFAULT_MONTHS,
    trades_per_month: int = DEFAULT_TRADES_PER_MONTH,
    initial_accounts: int = 1,
    max_accounts: int = DEFAULT_MAX_ACCOUNTS,
    payout_checks_per_month: int = DEFAULT_PAYOUT_CHECKS_PER_MONTH,
    extra_account_trade_penalty: float = DEFAULT_IMPACT_PER_EXTRA_ACCOUNT,
    rng: np.random.Generator | None = None,
) -> FleetPathResult:
    if months <= 0:
        raise ValueError("months must be positive")
    if initial_accounts <= 0:
        raise ValueError("initial_accounts must be positive")
    if max_accounts < initial_accounts:
        raise ValueError("max_accounts must be >= initial_accounts")
    if payout_checks_per_month <= 0:
        raise ValueError("payout_checks_per_month must be positive")

    if rng is None:
        rng = np.random.default_rng()

    accounts = [Account(account_id=i) for i in range(initial_accounts)]
    initial_account_ids = set(range(initial_accounts))
    next_account_id = initial_accounts
    liquid_wealth = 0.0
    liquid_wealth_path = np.zeros(months, dtype=np.float64)
    active_accounts_path = np.zeros(months, dtype=np.int32)
    deployed_equity_path = np.zeros(months, dtype=np.float64)
    total_fleet_capital_path = np.zeros(months, dtype=np.float64)
    monthly_net_payout_path = np.zeros(months, dtype=np.float64)
    survival_after_12m = 0.0

    for month_idx in range(months):
        month_trades = bootstrap_month_trades(trade_pool, trades_per_month, rng)
        trade_chunks = np.array_split(month_trades, payout_checks_per_month)
        month_net_payout = 0.0

        for chunk in trade_chunks:
            for trade_pnl in chunk:
                alive_accounts = [account for account in accounts if account.alive]
                alive_count = len(alive_accounts)
                if alive_count == 0:
                    break

                impact_penalty = max(0, alive_count - 1) * extra_account_trade_penalty
                adjusted_trade_pnl = float(trade_pnl) - impact_penalty
                for account in alive_accounts:
                    account.apply_trade(adjusted_trade_pnl)

            for account in accounts:
                net_payout = account.process_payout()
                liquid_wealth += net_payout
                month_net_payout += net_payout

            liquid_wealth, next_account_id = _reinvest_accounts(
                accounts=accounts,
                liquid_wealth=liquid_wealth,
                next_account_id=next_account_id,
                max_accounts=max_accounts,
                account_cost=DEFAULT_ACCOUNT_COST,
            )

        deployed_equity = float(sum(account.equity for account in accounts if account.alive))
        liquid_wealth_path[month_idx] = liquid_wealth
        active_accounts_path[month_idx] = sum(account.alive for account in accounts)
        deployed_equity_path[month_idx] = deployed_equity
        total_fleet_capital_path[month_idx] = liquid_wealth + deployed_equity
        monthly_net_payout_path[month_idx] = month_net_payout

        if month_idx == 11:
            initial_alive = [
                account.alive for account in accounts
                if account.account_id in initial_account_ids
            ]
            survival_after_12m = float(np.mean(initial_alive)) if initial_alive else 0.0

    if months < 12:
        initial_alive = [
            account.alive for account in accounts
            if account.account_id in initial_account_ids
        ]
        survival_after_12m = float(np.mean(initial_alive)) if initial_alive else 0.0

    return FleetPathResult(
        liquid_wealth_path=liquid_wealth_path,
        active_accounts_path=active_accounts_path,
        deployed_equity_path=deployed_equity_path,
        total_fleet_capital_path=total_fleet_capital_path,
        monthly_net_payout_path=monthly_net_payout_path,
        initial_account_survival_after_12m=survival_after_12m,
    )


def run_fleet_simulation(
    trade_pool: np.ndarray,
    n_sims: int = DEFAULT_N_SIMS,
    months: int = DEFAULT_MONTHS,
    trades_per_month: int = DEFAULT_TRADES_PER_MONTH,
    initial_accounts: int = 1,
    max_accounts: int = DEFAULT_MAX_ACCOUNTS,
    payout_checks_per_month: int = DEFAULT_PAYOUT_CHECKS_PER_MONTH,
    extra_account_trade_penalty: float = DEFAULT_IMPACT_PER_EXTRA_ACCOUNT,
    seed: int = 42,
) -> FleetSimulationResult:
    if n_sims <= 0:
        raise ValueError("n_sims must be positive")

    master_rng = np.random.default_rng(seed)
    liquid_wealth_paths = np.zeros((n_sims, months), dtype=np.float64)
    active_account_paths = np.zeros((n_sims, months), dtype=np.int32)
    deployed_equity_paths = np.zeros((n_sims, months), dtype=np.float64)
    total_fleet_capital_paths = np.zeros((n_sims, months), dtype=np.float64)
    monthly_net_payout_paths = np.zeros((n_sims, months), dtype=np.float64)
    survival_after_12m = np.zeros(n_sims, dtype=np.float64)

    for sim_idx in range(n_sims):
        path_rng = np.random.default_rng(master_rng.integers(0, np.iinfo(np.int64).max))
        path_result = simulate_single_path(
            trade_pool=trade_pool,
            months=months,
            trades_per_month=trades_per_month,
            initial_accounts=initial_accounts,
            max_accounts=max_accounts,
            payout_checks_per_month=payout_checks_per_month,
            extra_account_trade_penalty=extra_account_trade_penalty,
            rng=path_rng,
        )
        liquid_wealth_paths[sim_idx] = path_result.liquid_wealth_path
        active_account_paths[sim_idx] = path_result.active_accounts_path
        deployed_equity_paths[sim_idx] = path_result.deployed_equity_path
        total_fleet_capital_paths[sim_idx] = path_result.total_fleet_capital_path
        monthly_net_payout_paths[sim_idx] = path_result.monthly_net_payout_path
        survival_after_12m[sim_idx] = path_result.initial_account_survival_after_12m

    final_wealth = liquid_wealth_paths[:, -1]
    final_deployed_equity = deployed_equity_paths[:, -1]
    final_total_fleet_capital = total_fleet_capital_paths[:, -1]
    trailing_window = min(12, months)
    final_year_avg_monthly_payout = np.mean(monthly_net_payout_paths[:, -trailing_window:], axis=1)
    return FleetSimulationResult(
        liquid_wealth_paths=liquid_wealth_paths,
        active_account_paths=active_account_paths,
        deployed_equity_paths=deployed_equity_paths,
        total_fleet_capital_paths=total_fleet_capital_paths,
        monthly_net_payout_paths=monthly_net_payout_paths,
        median_final_liquid_wealth=float(np.median(final_wealth)),
        median_final_deployed_equity=float(np.median(final_deployed_equity)),
        median_final_total_fleet_capital=float(np.median(final_total_fleet_capital)),
        survival_rate_initial_after_12m=float(np.mean(survival_after_12m)),
        p90_final_liquid_wealth=float(np.percentile(final_wealth, 90)),
        median_final_year_avg_monthly_payout=float(np.median(final_year_avg_monthly_payout)),
        median_final_year_avg_annual_payout=float(np.median(final_year_avg_monthly_payout) * 12.0),
    )


def plot_simulation(
    result: FleetSimulationResult,
    output_path: Path,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "matplotlib is required for plotting. Install it or run with --skip-plot."
        ) from exc

    output_path.parent.mkdir(parents=True, exist_ok=True)
    months = np.arange(1, result.liquid_wealth_paths.shape[1] + 1)
    median_path = np.median(result.liquid_wealth_paths, axis=0)
    avg_active_accounts = np.mean(result.active_account_paths, axis=0)

    fig, axes = plt.subplots(2, 1, figsize=(12, 9), constrained_layout=True)

    for path in result.liquid_wealth_paths:
        axes[0].plot(months, path, color="steelblue", alpha=0.04, linewidth=0.8)
    axes[0].plot(months, median_path, color="black", linewidth=2.5, label="Median")
    axes[0].set_title("Liquid Wealth Paths")
    axes[0].set_xlabel("Month")
    axes[0].set_ylabel("Liquid Wealth ($)")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].plot(months, avg_active_accounts, color="darkred", linewidth=2.5)
    axes[1].set_title("Average Active Accounts")
    axes[1].set_xlabel("Month")
    axes[1].set_ylabel("Active Accounts")
    axes[1].grid(alpha=0.25)

    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("output/fleet_simulator"))
    parser.add_argument("--n-sims", type=int, default=DEFAULT_N_SIMS)
    parser.add_argument("--months", type=int, default=DEFAULT_MONTHS)
    parser.add_argument("--trades-per-month", type=int, default=DEFAULT_TRADES_PER_MONTH)
    parser.add_argument("--initial-accounts", type=int, default=1)
    parser.add_argument("--max-accounts", type=int, default=DEFAULT_MAX_ACCOUNTS)
    parser.add_argument("--payout-checks-per-month", type=int, default=DEFAULT_PAYOUT_CHECKS_PER_MONTH)
    parser.add_argument("--impact-per-extra-account", type=float, default=DEFAULT_IMPACT_PER_EXTRA_ACCOUNT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--print-yearly-summary", action="store_true")
    parser.add_argument("--skip-plot", action="store_true")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    trade_pool, trade_source = load_trade_pool()
    result = run_fleet_simulation(
        trade_pool=trade_pool,
        n_sims=args.n_sims,
        months=args.months,
        trades_per_month=args.trades_per_month,
        initial_accounts=args.initial_accounts,
        max_accounts=args.max_accounts,
        payout_checks_per_month=args.payout_checks_per_month,
        extra_account_trade_penalty=args.impact_per_extra_account,
        seed=args.seed,
    )

    args.output.mkdir(parents=True, exist_ok=True)
    if not args.skip_plot:
        plot_simulation(result, args.output / "fleet_paths.png")

    print(f"Trade source: {trade_source}")
    print(f"Median Liquid Wealth nach {args.months} Monaten: ${result.median_final_liquid_wealth:,.2f}")
    print(f"Median Deployed Equity nach {args.months} Monaten: ${result.median_final_deployed_equity:,.2f}")
    print(
        f"Median Total Fleet Capital nach {args.months} Monaten: "
        f"${result.median_final_total_fleet_capital:,.2f}"
    )
    print(
        "Survival Rate der ersten Konten nach 12 Monaten: "
        f"{result.survival_rate_initial_after_12m:.2%}"
    )
    print(f"90th Percentile Wealth (Das Lucky-Szenario): ${result.p90_final_liquid_wealth:,.2f}")
    print(
        "Median Net Payout Run-Rate im letzten Jahr: "
        f"${result.median_final_year_avg_monthly_payout:,.2f}/Monat | "
        f"${result.median_final_year_avg_annual_payout:,.2f}/Jahr"
    )
    if args.print_yearly_summary:
        print()
        print("Yearly Summary")
        print(format_yearly_summary_table(build_yearly_summary_rows(result)))
    if not args.skip_plot:
        print(f"Plot saved to: {args.output / 'fleet_paths.png'}")


if __name__ == "__main__":
    main()
