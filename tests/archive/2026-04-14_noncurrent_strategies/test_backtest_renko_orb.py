import importlib.util
import sys
from pathlib import Path

import numpy as np


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "backtest_renko_orb.py"
SPEC = importlib.util.spec_from_file_location("backtest_renko_orb", SCRIPT_PATH)
module = importlib.util.module_from_spec(SPEC)
sys.modules["backtest_renko_orb"] = module
assert SPEC.loader is not None
SPEC.loader.exec_module(module)


AssetSpec = module.AssetSpec
_find_first_renko_signal = module._find_first_renko_signal
_simulate_trade = module._simulate_trade
SIGNAL_LONG = module.SIGNAL_LONG
EXIT_TARGET = module.EXIT_TARGET


def make_spec() -> AssetSpec:
    return AssetSpec(
        name="MNQ",
        data_path=Path("unused.parquet"),
        brick_size=10.0,
        opening_range_minutes=15,
        tick_size=0.25,
        tick_value=0.50,
        commission_per_side=0.54,
    )


def test_find_first_renko_signal_returns_long_breakout():
    closes = np.array([100.0, 100.0, 109.0, 121.0, 132.0], dtype=np.float64)
    ema_200 = np.array([95.0, 95.0, 95.0, 95.0, 95.0], dtype=np.float64)

    result = _find_first_renko_signal(
        closes=closes,
        ema_200=ema_200,
        start_idx=2,
        end_idx=len(closes),
        or_high=118.0,
        or_low=90.0,
        brick_size=10.0,
    )

    assert result == (3, SIGNAL_LONG)


def test_simulate_trade_hits_target():
    spec = make_spec()
    timestamps = np.arange(6, dtype=np.int64) * 60_000_000_000
    minute_of_day = np.array([15, 16, 17, 18, 19, 20], dtype=np.int16)
    opens = np.array([120.0, 121.0, 122.0, 123.0, 124.0, 125.0], dtype=np.float64)
    highs = np.array([121.0, 182.0, 125.0, 126.0, 127.0, 128.0], dtype=np.float64)
    lows = np.array([119.0, 120.0, 121.0, 122.0, 123.0, 124.0], dtype=np.float64)

    trade = _simulate_trade(
        spec=spec,
        day_id=0,
        timestamps=timestamps,
        minute_of_day=minute_of_day,
        opens=opens,
        highs=highs,
        lows=lows,
        signal_idx=0,
        signal=SIGNAL_LONG,
    )

    assert trade is not None
    assert int(trade["exit_reason"]) == EXIT_TARGET
    assert float(trade["net_pnl"]) > 0.0
