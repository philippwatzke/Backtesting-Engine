import importlib.util
import sys
import uuid
from pathlib import Path

import numpy as np
import pandas as pd

from propfirm.core.types import EXIT_TARGET, SIGNAL_LONG


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "tick_renko_orb_mgc.py"
SPEC = importlib.util.spec_from_file_location("tick_renko_orb_mgc", SCRIPT_PATH)
module = importlib.util.module_from_spec(SPEC)
sys.modules["tick_renko_orb_mgc"] = module
assert SPEC.loader is not None
SPEC.loader.exec_module(module)


load_mgc_ticks = module.load_mgc_ticks
_simulate_day = module._simulate_day
BRICK_SIZE = module.BRICK_SIZE


def _ts(value: str) -> pd.Timestamp:
    return pd.Timestamp(value, tz="UTC")


def _temp_parquet_path(name: str) -> Path:
    return Path.cwd() / f"{name}_{uuid.uuid4().hex}.parquet"


def test_load_mgc_ticks_prefers_midquote_schema():
    path = _temp_parquet_path("mgc_tick_midquote")
    frame = pd.DataFrame(
        {
            "ts_event": [
                _ts("2026-01-05 14:30:00"),
                _ts("2026-01-05 14:30:01"),
            ],
            "bid_px_00": [3000.0, 3000.2],
            "ask_px_00": [3000.2, 3000.4],
            "bid_sz_00": [10, 11],
            "ask_sz_00": [12, 13],
        }
    )
    frame.to_parquet(path, index=False)

    loaded = load_mgc_ticks(path)

    assert loaded["source_type"] == "midquote"
    np.testing.assert_allclose(loaded["tick_prices"], np.array([3000.1, 3000.3]))
    np.testing.assert_array_equal(loaded["tick_volumes"], np.array([22, 24], dtype=np.int32))


def test_simulate_day_hits_target_from_tick_stream():
    local_times = pd.to_datetime(
        [
            "2026-01-05 09:30:00",
            "2026-01-05 09:40:00",
            "2026-01-05 10:00:00",
            "2026-01-05 10:00:01",
            "2026-01-05 10:00:02",
            "2026-01-05 10:00:03",
        ]
    ).astype("datetime64[ns]").asi8
    utc_times = pd.to_datetime(
        [
            "2026-01-05 14:30:00+00:00",
            "2026-01-05 14:40:00+00:00",
            "2026-01-05 15:00:00+00:00",
            "2026-01-05 15:00:01+00:00",
            "2026-01-05 15:00:02+00:00",
            "2026-01-05 15:00:03+00:00",
        ]
    ).asi8
    prices = np.array([3000.0, 3000.2, 3000.1, 3000.5, 3002.4, 3002.5], dtype=np.float64)
    session_label_ns = pd.Timestamp("2026-01-05 00:00:00").value

    entry_idx, exit_idx, entry_price, exit_price, signal, exit_reason = _simulate_day(
        utc_times,
        local_times,
        prices,
        0,
        len(prices),
        session_label_ns,
    )

    assert entry_idx == 3
    assert exit_idx == 5
    assert signal == SIGNAL_LONG
    assert exit_reason == EXIT_TARGET
    assert entry_price > prices[entry_idx]
    assert exit_price < prices[exit_idx]
    assert BRICK_SIZE == 0.30
