import pandas as pd

from propfirm.execution.nt8_dual_feed import (
    _find_entry_index,
    _prepare_raw_execution_feed,
    _route_trade_on_raw,
    compute_performance_metrics,
    split_is_oos_metrics,
)


def test_route_trade_on_raw_prioritizes_stop_when_both_hit_same_bar():
    raw = pd.DataFrame(
        {
            "timestamp_utc": pd.to_datetime(
                [
                    "2026-03-18 14:31:00+00:00",
                    "2026-03-18 14:32:00+00:00",
                ],
                utc=True,
            ),
            "session_date_et": ["2026-03-18", "2026-03-18"],
            "inside_research_session": [True, True],
            "minute_total_et": [631, 632],
            "open": [100.0, 100.0],
            "high": [111.0, 101.0],
            "low": [89.0, 99.0],
            "close": [100.5, 100.0],
        }
    )

    trade = _route_trade_on_raw(
        raw_df=raw,
        entry_idx=0,
        direction=1,
        stop_distance=10.0,
        target_distance=10.0,
        tick_size=1.0,
        tick_value=1.0,
        commission_per_side=0.0,
        contracts=1,
    )

    assert trade["entry_price"] == 100.0
    assert trade["exit_price"] == 90.0
    assert trade["exit_reason"] == "stop_same_bar_priority"
    assert trade["gross_pnl"] == -10.0


def test_split_is_oos_metrics_isolates_trade_sets():
    trades = pd.DataFrame(
        {
            "entry_time": pd.to_datetime(
                [
                    "2024-12-20 15:00:00+00:00",
                    "2025-01-10 15:00:00+00:00",
                ],
                utc=True,
            ),
            "exit_time": pd.to_datetime(
                [
                    "2024-12-20 16:00:00+00:00",
                    "2025-01-10 16:00:00+00:00",
                ],
                utc=True,
            ),
            "entry_session_date": ["2024-12-20", "2025-01-10"],
            "net_pnl": [100.0, -40.0],
        }
    )

    split = split_is_oos_metrics(trades, is_end_date="2024-12-31", oos_start_date="2025-01-01")

    assert split["is"].trade_count == 1
    assert split["is"].net_profit == 100.0
    assert split["oos"].trade_count == 1
    assert split["oos"].net_profit == -40.0


def test_compute_performance_metrics_max_drawdown_uses_split_local_equity():
    trades = pd.DataFrame(
        {
            "entry_time": pd.to_datetime(
                [
                    "2025-01-01 15:00:00+00:00",
                    "2025-01-02 15:00:00+00:00",
                    "2025-01-03 15:00:00+00:00",
                ],
                utc=True,
            ),
            "exit_time": pd.to_datetime(
                [
                    "2025-01-01 16:00:00+00:00",
                    "2025-01-02 16:00:00+00:00",
                    "2025-01-03 16:00:00+00:00",
                ],
                utc=True,
            ),
            "net_pnl": [100.0, -60.0, -50.0],
        }
    )

    metrics = compute_performance_metrics(trades)
    assert metrics.trade_count == 3
    assert metrics.net_profit == -10.0
    assert metrics.max_drawdown == 110.0


def test_prepare_raw_execution_feed_uses_nanosecond_timestamps_for_search():
    raw = pd.DataFrame(
        {
            "timestamp_utc": pd.to_datetime(
                [
                    "2026-03-18 14:30:00+00:00",
                    "2026-03-18 14:31:00+00:00",
                    "2026-03-18 14:32:00+00:00",
                ],
                utc=True,
            ),
            "open": [100.0, 101.0, 102.0],
            "high": [100.0, 101.0, 102.0],
            "low": [100.0, 101.0, 102.0],
            "close": [100.0, 101.0, 102.0],
            "volume": [1.0, 1.0, 1.0],
        }
    )
    prepared, raw_ns = _prepare_raw_execution_feed(raw, {"session_start": "08:00", "session_end": "15:59"})
    signal_time = pd.Timestamp("2026-03-18 14:30:00+00:00")
    assert raw_ns[0] == signal_time.value
    idx = _find_entry_index(prepared, raw_ns, signal_time, "2026-03-18")
    assert idx == 1
