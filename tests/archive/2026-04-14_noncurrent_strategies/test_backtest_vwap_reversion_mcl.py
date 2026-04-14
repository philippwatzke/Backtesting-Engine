import importlib.util
import sys
from pathlib import Path

import pandas as pd


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "backtest_vwap_reversion_mcl.py"
SPEC = importlib.util.spec_from_file_location("backtest_vwap_reversion_mcl", SCRIPT_PATH)
module = importlib.util.module_from_spec(SPEC)
sys.modules["backtest_vwap_reversion_mcl"] = module
assert SPEC.loader is not None
SPEC.loader.exec_module(module)


_session_ids = module._session_ids
_is_signal_window = module._is_signal_window


def test_session_ids_reset_at_1800_est():
    idx = pd.DatetimeIndex(
        [
            pd.Timestamp("2026-01-05 17:59:00", tz="America/New_York"),
            pd.Timestamp("2026-01-05 18:00:00", tz="America/New_York"),
            pd.Timestamp("2026-01-06 09:00:00", tz="America/New_York"),
        ]
    )

    session_ids = _session_ids(idx)

    assert session_ids.iloc[0].date().isoformat() == "2026-01-05"
    assert session_ids.iloc[1].date().isoformat() == "2026-01-06"
    assert session_ids.iloc[2].date().isoformat() == "2026-01-06"


def test_signal_window_only_allows_monday_tuesday_thursday_1200_to_1259_est():
    assert _is_signal_window(weekday=0, minute_total=12 * 60) is True
    assert _is_signal_window(weekday=1, minute_total=12 * 60 + 30) is True
    assert _is_signal_window(weekday=3, minute_total=12 * 60 + 59) is True
    assert _is_signal_window(weekday=0, minute_total=11 * 60 + 59) is False
    assert _is_signal_window(weekday=3, minute_total=13 * 60) is False
    assert _is_signal_window(weekday=2, minute_total=12 * 60 + 15) is False
    assert _is_signal_window(weekday=4, minute_total=12 * 60) is False
