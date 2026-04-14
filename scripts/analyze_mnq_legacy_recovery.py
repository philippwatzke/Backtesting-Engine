#!/usr/bin/env python
"""Assess how close the current MNQ legacy rerun is to the archived legacy trade log."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


SESSION_TZ = "America/New_York"
OLD_LOG = Path("output/backtests_mnq_tf30_regime_test/latest_trade_log.npy")
CURRENT_LOG = Path("output/backtests_mnq_legacy_rerun/latest_trade_log.npy")
OUTPUT_PATH = Path("output/mnq_legacy_recovery_report.json")


def _to_frame(trade_log: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "entry_time": pd.to_datetime(trade_log["entry_time"].astype(np.int64), utc=True).tz_convert(SESSION_TZ),
            "exit_time": pd.to_datetime(trade_log["exit_time"].astype(np.int64), utc=True).tz_convert(SESSION_TZ),
            "signal_type": trade_log["signal_type"].astype(np.int64),
            "exit_reason": trade_log["exit_reason"].astype(np.int64),
            "net_pnl": trade_log["net_pnl"].astype(np.float64),
        }
    )


def _match_report(old_df: pd.DataFrame, new_df: pd.DataFrame) -> dict:
    exact = old_df.merge(
        new_df,
        on=["entry_time", "exit_time", "signal_type"],
        how="outer",
        indicator=True,
        suffixes=("_old", "_new"),
    )
    by_entry = old_df.merge(
        new_df,
        on=["entry_time", "signal_type"],
        how="outer",
        indicator=True,
        suffixes=("_old", "_new"),
    )
    by_date = (
        old_df.assign(entry_date=old_df["entry_time"].dt.date)
        .merge(
            new_df.assign(entry_date=new_df["entry_time"].dt.date),
            on=["entry_date", "signal_type"],
            how="outer",
            indicator=True,
            suffixes=("_old", "_new"),
        )
    )
    both_entry = by_entry[by_entry["_merge"] == "both"].copy()
    both_entry["same_exit_time"] = both_entry["exit_time_old"] == both_entry["exit_time_new"]
    both_entry["same_exit_reason"] = both_entry["exit_reason_old"] == both_entry["exit_reason_new"]

    missing_dates = (
        by_date.loc[by_date["_merge"] == "left_only", ["entry_date", "signal_type", "net_pnl_old"]]
        .sort_values(["entry_date", "signal_type"])
        .assign(entry_date=lambda df: df["entry_date"].astype(str))
        .to_dict(orient="records")
    )

    return {
        "exact_match_counts": exact["_merge"].value_counts().to_dict(),
        "entry_match_counts": by_entry["_merge"].value_counts().to_dict(),
        "date_direction_match_counts": by_date["_merge"].value_counts().to_dict(),
        "same_exit_time_on_entry_matches": int(both_entry["same_exit_time"].sum()),
        "same_exit_reason_on_entry_matches": int(both_entry["same_exit_reason"].sum()),
        "entry_matches": int(len(both_entry)),
        "missing_old_dates": missing_dates,
        "old_trade_count": int(len(old_df)),
        "new_trade_count": int(len(new_df)),
        "old_net_pnl": float(old_df["net_pnl"].sum()),
        "new_net_pnl": float(new_df["net_pnl"].sum()),
    }


def main() -> None:
    old_df = _to_frame(np.load(OLD_LOG, allow_pickle=False))
    current_df = _to_frame(np.load(CURRENT_LOG, allow_pickle=False))

    shifted_entry_df = current_df.copy()
    shifted_entry_df["entry_time"] = shifted_entry_df["entry_time"] + pd.Timedelta(minutes=30)

    report = {
        "artifacts": {
            "old_trade_log": str(OLD_LOG),
            "current_trade_log": str(CURRENT_LOG),
        },
        "direct_current_vs_old": _match_report(old_df, current_df),
        "entry_plus_30m_current_vs_old": _match_report(old_df, shifted_entry_df),
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote recovery report to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
