from pathlib import Path
import uuid

import numpy as np
import pandas as pd
import pytest

from propfirm.core.types import (
    EXIT_HARD_CLOSE,
    EXIT_STOP,
    EXIT_TARGET,
    MNQ_TICK_SIZE,
    MNQ_TICK_VALUE,
    TRADE_LOG_DTYPE,
)
from propfirm.execution.tick_replayer import (
    BUY_ORDER_SIDE,
    SELL_ORDER_SIDE,
    compare_trade_logs,
    format_reality_report,
    load_databento_ticks,
    run_tick_replay_report,
    simulate_tick_execution,
)
from propfirm.market.data_loader import SESSION_TZ


def _ts(value: str) -> pd.Timestamp:
    return pd.Timestamp(value, tz="UTC")


def _ns(value: str) -> int:
    return _ts(value).value


def _make_trade_log(
    *,
    entry_time: str = "2026-01-05 14:30:00",
    exit_time: str = "2026-01-05 14:31:00",
    entry_price: float = 100.0,
    exit_price: float = 101.0,
    contracts: int = 3,
    signal_type: int = 1,
    exit_reason: int = EXIT_TARGET,
    entry_slippage: float = 0.0,
    exit_slippage: float = 0.0,
    entry_commission: float = 0.0,
    exit_commission: float = 0.0,
) -> np.ndarray:
    trade_log = np.zeros(1, dtype=TRADE_LOG_DTYPE)
    trade_log[0]["entry_time"] = _ns(entry_time)
    trade_log[0]["exit_time"] = _ns(exit_time)
    trade_log[0]["entry_price"] = entry_price
    trade_log[0]["exit_price"] = exit_price
    trade_log[0]["contracts"] = contracts
    trade_log[0]["signal_type"] = signal_type
    trade_log[0]["entry_slippage"] = entry_slippage
    trade_log[0]["exit_slippage"] = exit_slippage
    trade_log[0]["entry_commission"] = entry_commission
    trade_log[0]["exit_commission"] = exit_commission
    trade_log[0]["exit_reason"] = exit_reason
    if signal_type > 0:
        theoretical_gross = (exit_price - entry_price) * contracts / MNQ_TICK_SIZE * MNQ_TICK_VALUE
    else:
        theoretical_gross = (entry_price - exit_price) * contracts / MNQ_TICK_SIZE * MNQ_TICK_VALUE
    trade_log[0]["gross_pnl"] = theoretical_gross
    trade_log[0]["net_pnl"] = theoretical_gross - entry_commission - exit_commission
    return trade_log


def _temp_parquet_path(name: str) -> Path:
    return Path.cwd() / f"{name}_{uuid.uuid4().hex}.parquet"


def test_load_databento_ticks_trade_schema_normalizes_to_session_timezone():
    path = _temp_parquet_path("trade_ticks")
    frame = pd.DataFrame(
        {
            "ts_event": [
                _ts("2026-01-05 14:29:59"),
                _ts("2026-01-05 14:30:00"),
                _ts("2026-01-05 14:30:01"),
                _ts("2026-01-05 14:30:02"),
            ],
            "price": [99.5, 100.25, 100.5, 100.75],
            "size": [9, 1, 2, 3],
            "side": ["B", "B", "A", "N"],
        }
    )
    frame.to_parquet(path, index=False)

    loaded = load_databento_ticks(
        path,
        start_time=_ns("2026-01-05 14:30:00"),
        end_time=_ns("2026-01-05 14:30:02"),
    )

    assert loaded["source_type"] == "trade"
    assert np.array_equal(loaded["tick_prices"], np.array([100.25, 100.5]))
    assert np.array_equal(loaded["tick_volumes"], np.array([1, 2], dtype=np.int32))
    assert np.array_equal(loaded["tick_sides"], np.array([BUY_ORDER_SIDE, SELL_ORDER_SIDE], dtype=np.int8))

    localized = pd.to_datetime(loaded["tick_times"], utc=True).tz_convert(SESSION_TZ)
    assert list(localized.strftime("%Y-%m-%d %H:%M:%S")) == [
        "2026-01-05 09:30:00",
        "2026-01-05 09:30:01",
    ]


def test_load_databento_ticks_book_event_schema_maps_book_side_to_order_side():
    path = _temp_parquet_path("book_event_ticks")
    frame = pd.DataFrame(
        {
            "ts_event": [
                _ts("2026-01-05 14:30:00"),
                _ts("2026-01-05 14:30:01"),
            ],
            "price": [100.0, 100.25],
            "size": [4, 5],
            "side": ["B", "A"],
            "action": ["A", "A"],
        }
    )
    frame.to_parquet(path, index=False)

    loaded = load_databento_ticks(path)

    assert loaded["source_type"] == "book_event"
    assert np.array_equal(loaded["tick_prices"], np.array([100.0, 100.25]))
    assert np.array_equal(loaded["tick_sides"], np.array([SELL_ORDER_SIDE, BUY_ORDER_SIDE], dtype=np.int8))


def test_load_databento_ticks_book_snapshot_schema_expands_bid_and_ask():
    path = _temp_parquet_path("bbo_ticks")
    frame = pd.DataFrame(
        {
            "ts_event": [
                _ts("2026-01-05 14:30:00"),
                _ts("2026-01-05 14:30:01"),
            ],
            "bid_px_00": [100.0, 100.25],
            "ask_px_00": [100.25, 100.5],
            "bid_sz_00": [5, 6],
            "ask_sz_00": [3, 4],
        }
    )
    frame.to_parquet(path, index=False)

    loaded = load_databento_ticks(path)

    assert loaded["source_type"] == "book_snapshot"
    assert np.array_equal(
        loaded["tick_prices"],
        np.array([100.25, 100.0, 100.5, 100.25]),
    )
    assert np.array_equal(
        loaded["tick_volumes"],
        np.array([3, 5, 4, 6], dtype=np.int32),
    )
    assert np.array_equal(
        loaded["tick_sides"],
        np.array([BUY_ORDER_SIDE, SELL_ORDER_SIDE, BUY_ORDER_SIDE, SELL_ORDER_SIDE], dtype=np.int8),
    )


def test_simulate_tick_execution_and_report_metrics():
    path = _temp_parquet_path("replay_ticks")
    frame = pd.DataFrame(
        {
            "ts_event": [
                _ts("2026-01-05 14:29:59"),
                _ts("2026-01-05 14:30:00"),
                _ts("2026-01-05 14:30:00.000000100"),
                _ts("2026-01-05 14:30:00.000000200"),
                _ts("2026-01-05 14:31:00"),
                _ts("2026-01-05 14:31:00.000000100"),
                _ts("2026-01-05 14:36:00"),
            ],
            "price": [99.0, 100.25, 100.5, 99.75, 101.0, 100.75, 100.5],
            "size": [10, 1, 2, 8, 1, 2, 50],
            "side": ["B", "B", "B", "A", "A", "A", "A"],
        }
    )
    frame.to_parquet(path, index=False)

    theoretical = _make_trade_log()
    loaded = load_databento_ticks(path)
    simulated = simulate_tick_execution(
        theoretical,
        loaded["tick_times"],
        loaded["tick_prices"],
        loaded["tick_volumes"],
        loaded["tick_sides"],
    )
    metrics = compare_trade_logs(theoretical, simulated)

    assert simulated[0]["entry_price"] == pytest.approx((100.25 + 2 * 100.5) / 3.0)
    assert simulated[0]["exit_price"] == pytest.approx((101.0 + 2 * 100.75) / 3.0)
    assert simulated[0]["entry_slippage"] == pytest.approx(((100.25 + 2 * 100.5) / 3.0) - 100.0)
    assert simulated[0]["exit_slippage"] == pytest.approx(101.0 - ((101.0 + 2 * 100.75) / 3.0))
    assert simulated[0]["gross_pnl"] == pytest.approx(2.5)
    assert simulated[0]["net_pnl"] == pytest.approx(2.5)

    assert metrics["n_trades"] == 1
    assert metrics["average_entry_slippage_ticks"] == pytest.approx(5.0 / 3.0)
    assert metrics["average_exit_slippage_ticks"] == pytest.approx(2.0 / 3.0)
    assert metrics["average_exit_slippage_ticks_target"] == pytest.approx(2.0 / 3.0)
    assert np.isnan(metrics["average_exit_slippage_ticks_stop"])
    assert metrics["theoretical_net_pnl"] == pytest.approx(6.0)
    assert metrics["simulated_tick_pnl"] == pytest.approx(2.5)

    report = format_reality_report(metrics)
    assert "Average Entry Slippage (Ticks): 1.6667" in report
    assert "Slippage by Exit Reason (Durchschnitt in Ticks):" in report
    assert "TARGET Exits: 0.6667 Ticks" in report
    assert 'The "Toxic Trades" Top 3:' in report
    assert "Nicht genug STOP-Exits" in report
    assert "Simulierter Tick PnL: 2.50" in report

    rerun_simulated, rerun_metrics = run_tick_replay_report(theoretical, path)
    assert np.array_equal(rerun_simulated, simulated)
    assert rerun_metrics["n_trades"] == pytest.approx(metrics["n_trades"])
    assert rerun_metrics["average_entry_slippage_ticks"] == pytest.approx(metrics["average_entry_slippage_ticks"])
    assert rerun_metrics["average_exit_slippage_ticks"] == pytest.approx(metrics["average_exit_slippage_ticks"])
    assert len(rerun_metrics["toxic_trades"]) == 1
    assert rerun_metrics["toxic_trades"][0]["time_to_fill_ms"] == pytest.approx(0.0001, rel=0.0, abs=1e-6)


def test_simulate_tick_execution_uses_preclose_window_for_hard_close():
    path = _temp_parquet_path("hard_close_ticks")
    frame = pd.DataFrame(
        {
            "ts_event": [
                _ts("2026-01-05 14:30:00"),
                _ts("2026-01-05 14:30:00.000000100"),
                _ts("2026-01-05 14:30:55"),
                _ts("2026-01-05 14:30:59"),
                _ts("2026-01-05 14:31:00"),
                _ts("2026-01-05 14:31:00.000000100"),
            ],
            "price": [100.25, 100.5, 101.0, 100.75, 90.0, 89.5],
            "size": [1, 2, 1, 2, 10, 10],
            "side": ["B", "B", "A", "A", "A", "A"],
        }
    )
    frame.to_parquet(path, index=False)

    theoretical = _make_trade_log(exit_reason=EXIT_HARD_CLOSE)
    loaded = load_databento_ticks(path)
    simulated = simulate_tick_execution(
        theoretical,
        loaded["tick_times"],
        loaded["tick_prices"],
        loaded["tick_volumes"],
        loaded["tick_sides"],
    )

    assert simulated[0]["exit_price"] == pytest.approx((101.0 + 2 * 100.75) / 3.0)
    assert simulated[0]["exit_price"] > 90.0


def test_simulate_tick_execution_waits_for_stop_trigger_touch():
    path = _temp_parquet_path("stop_ticks")
    frame = pd.DataFrame(
        {
            "ts_event": [
                _ts("2026-01-05 14:30:00"),
                _ts("2026-01-05 14:30:00.000000100"),
                _ts("2026-01-05 14:31:00"),
                _ts("2026-01-05 14:31:00.000000100"),
                _ts("2026-01-05 14:31:00.000000200"),
                _ts("2026-01-05 14:31:00.000000300"),
            ],
            "price": [100.25, 100.5, 100.0, 99.9, 99.75, 99.5],
            "size": [1, 2, 1, 1, 1, 2],
            "side": ["B", "B", "A", "A", "A", "A"],
        }
    )
    frame.to_parquet(path, index=False)

    theoretical = _make_trade_log(
        exit_price=99.5,
        exit_reason=EXIT_STOP,
        exit_slippage=0.25,
    )
    loaded = load_databento_ticks(path)
    simulated = simulate_tick_execution(
        theoretical,
        loaded["tick_times"],
        loaded["tick_prices"],
        loaded["tick_volumes"],
        loaded["tick_sides"],
    )

    assert simulated[0]["exit_price"] == pytest.approx((99.75 + 2 * 99.5) / 3.0)
    assert simulated[0]["exit_price"] < 99.75 + 0.2


def test_format_reality_report_includes_toxic_trade_and_sigma_warning():
    metrics = {
        "n_trades": 3,
        "average_entry_slippage_ticks": 1.25,
        "average_exit_slippage_ticks": 2.5,
        "average_exit_slippage_ticks_target": 0.5,
        "average_exit_slippage_ticks_stop": 3.0,
        "average_exit_slippage_ticks_hard_close": 1.0,
        "stop_exit_slippage_std_ticks": 2.0,
        "stop_exit_slippage_3sigma_usd_per_contract": 6.0,
        "theoretical_net_pnl": 100.0,
        "simulated_tick_pnl": 85.0,
        "toxic_trades": [
            {
                "timestamp": "2026-01-05T14:30:00+00:00",
                "entry_slippage_ticks": 4.0,
                "exit_slippage_ticks": 7.5,
                "total_slippage_ticks": 11.5,
                "time_to_fill_ms": 25.0,
            }
        ],
    }

    report = format_reality_report(metrics)

    assert "TARGET Exits: 0.5000 Ticks" in report
    assert "STOP Exits: 3.0000 Ticks" in report
    assert "HARD CLOSE Exits: 1.0000 Ticks" in report
    assert "#1 2026-01-05T14:30:00+00:00" in report
    assert "Entry-Slippage: 4.00 Ticks" in report
    assert "Exit-Slippage: 7.50 Ticks" in report
    assert "Time-to-Fill: 25.000 ms" in report
    assert "kostet uns theoretisch 6.00 USD pro Kontrakt" in report
