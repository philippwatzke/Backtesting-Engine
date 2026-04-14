#!/usr/bin/env python
"""Analyze portfolio-level performance and drawdown risk for the MFF fleet."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


ACCOUNT_SIZE = 50_000.0
MFF_DRAWDOWN_LIMIT = 2_000.0
SESSION_TZ = "America/New_York"

DEFAULT_DONCHIAN_PORTFOLIO_LOG_PATH = Path("output/donchian_portfolio_oos_trade_log.npy")
DEFAULT_MCL_LOG_PATH = Path("output/vwap_mcl_trade_log_oos.npy")
DEFAULT_RB_LOG_PATH = Path("output/vwap_rb_trade_log.npy")
DEFAULT_EQUITY_OUTPUT_PATH = Path("output/portfolio_equity_curve.csv")


def _infer_epoch_unit(values: np.ndarray) -> str:
    max_abs = int(np.max(np.abs(values))) if len(values) else 0
    if max_abs >= 10**17:
        return "ns"
    if max_abs >= 10**14:
        return "us"
    if max_abs >= 10**11:
        return "ms"
    return "s"


def _to_ny_timestamp(values: np.ndarray) -> pd.DatetimeIndex:
    unit = _infer_epoch_unit(values.astype(np.int64))
    return pd.to_datetime(values.astype(np.int64), unit=unit, utc=True).tz_convert(SESSION_TZ)


def _load_trade_log(path: Path, asset_name: str | None = None) -> pd.DataFrame:
    arr = np.load(path, allow_pickle=False)
    if arr.dtype.names is None:
        raise TypeError(f"{path} is not a structured trade log")
    df = pd.DataFrame.from_records(arr)
    if asset_name is not None and "asset" not in df.columns:
        df["asset"] = asset_name
    df["exit_dt"] = _to_ny_timestamp(df["exit_time"].to_numpy(dtype=np.int64))
    df["exit_date"] = df["exit_dt"].dt.normalize()
    df["net_pnl"] = df["net_pnl"].astype(float)
    return df


def _load_donchian_logs(portfolio_log_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    if portfolio_log_path.exists():
        df = _load_trade_log(portfolio_log_path)
        if "asset" not in df.columns:
            raise ValueError("Combined Donchian trade log is missing asset labels")
        mnq = df[df["asset"] == "MNQ"].copy()
        mgc = df[df["asset"] == "MGC"].copy()
        if mnq.empty or mgc.empty:
            raise ValueError("Combined Donchian trade log does not contain both MNQ and MGC")
        return mnq, mgc
    raise FileNotFoundError(
        f"Missing Donchian trade log: {portfolio_log_path}"
    )


def _daily_pnl(log_df: pd.DataFrame) -> pd.Series:
    daily = log_df.groupby("exit_date", sort=True)["net_pnl"].sum().sort_index()
    daily.index = pd.DatetimeIndex(daily.index)
    return daily.astype(float)


def _profit_factor(pnl: pd.Series) -> float:
    gross_profit = float(pnl[pnl > 0.0].sum())
    gross_loss = float(-pnl[pnl < 0.0].sum())
    return float(gross_profit / gross_loss) if gross_loss > 0.0 else float("inf")


def _rolling_start_breach_probability(daily_pnl: pd.Series, start_equity: float, limit: float) -> float:
    if daily_pnl.empty:
        return 0.0
    breaches = 0
    total = len(daily_pnl)
    values = daily_pnl.to_numpy(dtype=np.float64)
    for start_idx in range(total):
        curve = start_equity + np.cumsum(values[start_idx:])
        running_peak = np.maximum.accumulate(curve)
        max_drawdown = float(np.max(running_peak - curve)) if len(curve) else 0.0
        if max_drawdown >= limit:
            breaches += 1
    return breaches / total


def _format_pct(value: float) -> str:
    return f"{value:.2%}"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--donchian-log", type=Path, default=DEFAULT_DONCHIAN_PORTFOLIO_LOG_PATH)
    parser.add_argument("--mcl-log", type=Path, default=DEFAULT_MCL_LOG_PATH)
    parser.add_argument("--rb-log", type=Path, default=DEFAULT_RB_LOG_PATH)
    parser.add_argument("--equity-output", type=Path, default=DEFAULT_EQUITY_OUTPUT_PATH)
    parser.add_argument("--account-size", type=float, default=ACCOUNT_SIZE)
    parser.add_argument("--drawdown-limit", type=float, default=MFF_DRAWDOWN_LIMIT)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    mnq_log, mgc_log = _load_donchian_logs(args.donchian_log)
    mcl_log = _load_trade_log(args.mcl_log, asset_name="MCL")
    rb_log = _load_trade_log(args.rb_log, asset_name="RB")

    mnq_daily = _daily_pnl(mnq_log).rename("mnq")
    mgc_daily = _daily_pnl(mgc_log).rename("mgc")
    mcl_daily = _daily_pnl(mcl_log).rename("mcl")
    rb_daily = _daily_pnl(rb_log).rename("rb")

    portfolio_daily = pd.concat([mnq_daily, mgc_daily, mcl_daily], axis=1, sort=True).fillna(0.0)
    portfolio_daily["trend"] = portfolio_daily["mnq"] + portfolio_daily["mgc"]
    portfolio_daily["portfolio"] = portfolio_daily["trend"] + portfolio_daily["mcl"]

    observation_daily = pd.concat([portfolio_daily[["trend", "mcl"]], rb_daily], axis=1, sort=True).fillna(0.0)

    equity_curve = pd.DataFrame(index=portfolio_daily.index)
    equity_curve["daily_pnl"] = portfolio_daily["portfolio"]
    equity_curve["equity"] = ACCOUNT_SIZE + equity_curve["daily_pnl"].cumsum()
    equity_curve["running_peak"] = equity_curve["equity"].cummax()
    equity_curve["drawdown"] = equity_curve["running_peak"] - equity_curve["equity"]

    args.equity_output.parent.mkdir(parents=True, exist_ok=True)
    equity_curve.to_csv(args.equity_output, index_label="date")

    max_drawdown = float(equity_curve["drawdown"].max()) if not equity_curve.empty else 0.0
    actual_breach = max_drawdown >= args.drawdown_limit
    breach_probability = _rolling_start_breach_probability(
        portfolio_daily["portfolio"], args.account_size, args.drawdown_limit
    )

    trend_vs_mcl_corr = float(observation_daily["trend"].corr(observation_daily["mcl"]))
    mcl_vs_rb_corr = float(observation_daily["mcl"].corr(observation_daily["rb"]))

    trend_negative_days = portfolio_daily["trend"] < 0.0
    mcl_positive_when_trend_negative = int((trend_negative_days & (portfolio_daily["mcl"] > 0.0)).sum())
    total_trend_negative_days = int(trend_negative_days.sum())
    diversification_rate = (
        mcl_positive_when_trend_negative / total_trend_negative_days if total_trend_negative_days else 0.0
    )

    mnq_mcl_loss_days = (portfolio_daily["mnq"] < 0.0) & (portfolio_daily["mcl"] < 0.0)
    combined_loss_series = (portfolio_daily["mnq"] + portfolio_daily["mcl"]).loc[mnq_mcl_loss_days]
    same_day_loss_count = int(mnq_mcl_loss_days.sum())
    avg_combined_loss = float(combined_loss_series.mean()) if same_day_loss_count else 0.0
    worst_combined_loss = float(combined_loss_series.min()) if same_day_loss_count else 0.0

    print("Portfolio Performance Analysis")
    print(f"History window: {portfolio_daily.index.min().date()} -> {portfolio_daily.index.max().date()}")
    print(f"Included equity pillars: MNQ + MGC + MCL")
    print(f"RB observation only: {args.rb_log}")
    print(f"Combined portfolio net PnL: ${portfolio_daily['portfolio'].sum():,.2f}")
    print(f"Ending equity: ${args.account_size + portfolio_daily['portfolio'].sum():,.2f}")
    print(f"Maximum Peak-to-Valley Drawdown: ${max_drawdown:,.2f}")
    print(
        f"MFF {args.drawdown_limit:,.0f} USD drawdown breach on actual path: {'YES' if actual_breach else 'NO'}"
    )
    print(
        f"Historical breach probability across rolling start dates: {_format_pct(breach_probability)}"
    )
    print(f"Daily return correlation (MNQ+MGC vs MCL): {trend_vs_mcl_corr:.4f}")
    print(f"Daily return correlation (MCL vs RB): {mcl_vs_rb_corr:.4f}")
    print(
        "Diversification proof: "
        f"MCL was positive on {mcl_positive_when_trend_negative} of {total_trend_negative_days} "
        f"Donchian-negative days ({_format_pct(diversification_rate)})"
    )
    print(
        "MNQ+MCL same-day loss overlap: "
        f"{same_day_loss_count} days | avg combined loss ${avg_combined_loss:,.2f} | "
        f"worst combined loss ${worst_combined_loss:,.2f}"
    )
    print(f"Portfolio equity curve saved to: {args.equity_output}")


if __name__ == "__main__":
    main()
