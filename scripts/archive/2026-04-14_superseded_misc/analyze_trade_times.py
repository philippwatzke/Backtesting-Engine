#!/usr/bin/env python
"""Compare train and OOS trade performance by New York entry-time buckets."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


SESSION_TZ = "America/New_York"
TIME_BUCKETS = (
    ("09:30", "10:30", "AM Session 1"),
    ("10:30", "11:30", "AM Session 2"),
    ("11:30", "13:30", "Midday Chop"),
    ("13:30", "16:00", "PM Session"),
)


def _load_trade_log(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Trade log not found: {path}")
    return np.load(path, allow_pickle=False)


def _bucket_stats(trade_log: np.ndarray) -> list[dict]:
    if len(trade_log) == 0:
        return [
            {"label": label, "trades": 0, "win_rate": 0.0, "net_pnl": 0.0}
            for _, _, label in TIME_BUCKETS
        ]

    entry_times = pd.to_datetime(trade_log["entry_time"], utc=True).tz_convert(SESSION_TZ)
    entry_minutes = entry_times.hour * 60 + entry_times.minute
    net_pnl = trade_log["net_pnl"]

    rows = []
    for start_str, end_str, label in TIME_BUCKETS:
        start_hour, start_minute = map(int, start_str.split(":"))
        end_hour, end_minute = map(int, end_str.split(":"))
        start_total = start_hour * 60 + start_minute
        end_total = end_hour * 60 + end_minute
        mask = (entry_minutes >= start_total) & (entry_minutes < end_total)
        bucket_pnl = net_pnl[mask]
        trades = int(bucket_pnl.size)
        win_rate = float((bucket_pnl > 0).mean() * 100.0) if trades else 0.0
        rows.append(
            {
                "label": label,
                "trades": trades,
                "win_rate": win_rate,
                "net_pnl": float(bucket_pnl.sum()) if trades else 0.0,
            }
        )
    return rows


def _format_table(train_rows: list[dict], oos_rows: list[dict]) -> str:
    headers = (
        ("label", "Time Block"),
        ("train_trades", "Train Trades"),
        ("train_win_rate", "Train WR%"),
        ("train_net_pnl", "Train Net PnL"),
        ("oos_trades", "OOS Trades"),
        ("oos_win_rate", "OOS WR%"),
        ("oos_net_pnl", "OOS Net PnL"),
    )

    merged = []
    for train_row, oos_row in zip(train_rows, oos_rows):
        merged.append(
            {
                "label": train_row["label"],
                "train_trades": train_row["trades"],
                "train_win_rate": train_row["win_rate"],
                "train_net_pnl": train_row["net_pnl"],
                "oos_trades": oos_row["trades"],
                "oos_win_rate": oos_row["win_rate"],
                "oos_net_pnl": oos_row["net_pnl"],
            }
        )

    def fmt(key: str, value: object) -> str:
        if key in {"train_win_rate", "oos_win_rate"}:
            return f"{float(value):.2f}"
        if key in {"train_net_pnl", "oos_net_pnl"}:
            return f"{float(value):,.2f}"
        return str(value)

    widths = []
    for key, label in headers:
        width = len(label)
        for row in merged:
            width = max(width, len(fmt(key, row[key])))
        widths.append(width)

    lines = []
    lines.append("  ".join(label.ljust(widths[idx]) for idx, (_, label) in enumerate(headers)))
    lines.append("  ".join("-" * widths[idx] for idx in range(len(headers))))
    for row in merged:
        lines.append(
            "  ".join(fmt(key, row[key]).ljust(widths[idx]) for idx, (key, _) in enumerate(headers))
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train",
        type=Path,
        default=Path("output/backtests/latest_trade_log.npy"),
    )
    parser.add_argument(
        "--oos",
        type=Path,
        default=Path("output/backtests_oos/latest_trade_log.npy"),
    )
    args = parser.parse_args()

    train_rows = _bucket_stats(_load_trade_log(args.train))
    oos_rows = _bucket_stats(_load_trade_log(args.oos))

    print(f"Train log: {args.train}")
    print(f"OOS log:   {args.oos}\n")
    print(_format_table(train_rows, oos_rows))


if __name__ == "__main__":
    main()
