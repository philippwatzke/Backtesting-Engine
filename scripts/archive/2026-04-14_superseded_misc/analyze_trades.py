#!/usr/bin/env python
"""Summarize the latest backtest trade log."""
import argparse
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--trade-log",
        type=Path,
        default=Path("output/backtests/latest_trade_log.npy"),
    )
    parser.add_argument(
        "--daily-log",
        type=Path,
        default=Path("output/backtests/latest_daily_log.npy"),
    )
    args = parser.parse_args()

    trades = np.load(args.trade_log, allow_pickle=True)
    daily = np.load(args.daily_log, allow_pickle=True)

    wins = trades[trades["net_pnl"] > 0]
    losses = trades[trades["net_pnl"] < 0]
    win_rate = len(wins) / len(trades) if len(trades) else 0.0

    print(f"Total trades: {len(trades)}")
    print(f"Wins: {len(wins)}")
    print(f"Losses: {len(losses)}")
    print(f"Win rate: {win_rate:.2%}")
    if len(wins):
        print(f"Average win: ${wins['net_pnl'].mean():.2f}")
    if len(losses):
        print(f"Average loss: ${losses['net_pnl'].mean():.2f}")
    if len(daily):
        print(f"Trading days: {len(daily)}")
        print(f"Average day PnL: ${daily['day_pnl'].mean():.2f}")
        print(f"Worst day PnL: ${daily['day_pnl'].min():.2f}")


if __name__ == "__main__":
    main()
