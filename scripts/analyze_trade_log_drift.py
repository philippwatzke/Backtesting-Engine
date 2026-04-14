"""Compare two structured trade logs and summarize drift drivers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


SESSION_TZ = "America/New_York"


def _load_trade_log(path: Path) -> pd.DataFrame:
    arr = np.load(path, allow_pickle=False)
    if arr.dtype.names is None:
        raise TypeError(f"{path} is not a structured trade log")
    df = pd.DataFrame.from_records(arr)
    df["asset"] = df["asset"].astype(str)
    for col in ("entry_time", "exit_time"):
        df[col] = pd.to_datetime(df[col].astype("int64"), unit="ns", utc=True).dt.tz_convert(SESSION_TZ)
    for col in ("net_pnl", "gross_pnl", "entry_price", "exit_price", "contracts"):
        df[col] = df[col].astype(float)
    return df


def _summary(df: pd.DataFrame) -> dict[str, object]:
    by_asset = (
        df.groupby("asset", sort=True)["net_pnl"]
        .agg(["count", "sum"])
        .reset_index()
        .to_dict(orient="records")
    )
    return {
        "count": int(len(df)),
        "net_pnl_sum": float(df["net_pnl"].sum()) if len(df) else 0.0,
        "first_entry": df["entry_time"].min().isoformat() if len(df) else None,
        "last_entry": df["entry_time"].max().isoformat() if len(df) else None,
        "by_asset": by_asset,
    }


def _compare(old_df: pd.DataFrame, new_df: pd.DataFrame) -> dict[str, object]:
    keys = ["asset", "entry_time", "exit_time", "signal_type"]
    merged = old_df.merge(new_df, on=keys, how="outer", suffixes=("_old", "_new"), indicator=True)

    both = merged[merged["_merge"] == "both"].copy()
    left_only = merged[merged["_merge"] == "left_only"].copy()
    right_only = merged[merged["_merge"] == "right_only"].copy()

    if len(both):
        both["net_pnl_delta"] = both["net_pnl_new"] - both["net_pnl_old"]
    else:
        both["net_pnl_delta"] = pd.Series(dtype=float)

    asset_rows = []
    for asset in sorted(set(old_df["asset"]) | set(new_df["asset"])):
        old_asset = old_df[old_df["asset"] == asset]
        new_asset = new_df[new_df["asset"] == asset]
        asset_match = both[both["asset"] == asset]
        asset_left = left_only[left_only["asset"] == asset]
        asset_right = right_only[right_only["asset"] == asset]
        asset_rows.append(
            {
                "asset": asset,
                "old_count": int(len(old_asset)),
                "new_count": int(len(new_asset)),
                "matched_count": int(len(asset_match)),
                "old_only_count": int(len(asset_left)),
                "new_only_count": int(len(asset_right)),
                "old_net_pnl_sum": float(old_asset["net_pnl"].sum()) if len(old_asset) else 0.0,
                "new_net_pnl_sum": float(new_asset["net_pnl"].sum()) if len(new_asset) else 0.0,
                "matched_net_pnl_delta_sum": float(asset_match["net_pnl_delta"].sum()) if len(asset_match) else 0.0,
                "old_first_entry": old_asset["entry_time"].min().isoformat() if len(old_asset) else None,
                "old_last_entry": old_asset["entry_time"].max().isoformat() if len(old_asset) else None,
                "new_first_entry": new_asset["entry_time"].min().isoformat() if len(new_asset) else None,
                "new_last_entry": new_asset["entry_time"].max().isoformat() if len(new_asset) else None,
            }
        )

    return {
        "merge_counts": merged["_merge"].value_counts().to_dict(),
        "matched_trade_count": int(len(both)),
        "matched_trade_pnl_delta_sum": float(both["net_pnl_delta"].sum()) if len(both) else 0.0,
        "matched_trade_pnl_changed_count": int((both["net_pnl_delta"].abs() > 1e-9).sum()) if len(both) else 0,
        "old_only_examples": left_only[["asset", "entry_time", "exit_time", "net_pnl_old"]]
        .head(10)
        .assign(
            entry_time=lambda x: x["entry_time"].astype(str),
            exit_time=lambda x: x["exit_time"].astype(str),
        )
        .to_dict(orient="records"),
        "new_only_examples": right_only[["asset", "entry_time", "exit_time", "net_pnl_new"]]
        .head(10)
        .assign(
            entry_time=lambda x: x["entry_time"].astype(str),
            exit_time=lambda x: x["exit_time"].astype(str),
        )
        .to_dict(orient="records"),
        "largest_negative_matched_pnl_deltas": both[
            ["asset", "entry_time", "exit_time", "net_pnl_old", "net_pnl_new", "net_pnl_delta"]
        ]
        .sort_values("net_pnl_delta")
        .head(10)
        .assign(
            entry_time=lambda x: x["entry_time"].astype(str),
            exit_time=lambda x: x["exit_time"].astype(str),
        )
        .to_dict(orient="records"),
        "by_asset": asset_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-log", type=Path, required=True)
    parser.add_argument("--new-log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    old_df = _load_trade_log(args.old_log)
    new_df = _load_trade_log(args.new_log)
    report = {
        "old_log": str(args.old_log),
        "new_log": str(args.new_log),
        "old_summary": _summary(old_df),
        "new_summary": _summary(new_df),
        "drift": _compare(old_df, new_df),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved drift report: {args.output}")


if __name__ == "__main__":
    main()
