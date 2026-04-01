import numpy as np
from propfirm.core.types import (
    TRADE_LOG_DTYPE,
    DAILY_LOG_DTYPE,
    MNQ_TICK_SIZE,
    MNQ_TICK_VALUE,
    MNQ_COMMISSION_PER_SIDE,
    BARS_PER_RTH_SESSION,
    EXIT_TARGET,
    EXIT_STOP,
    EXIT_HARD_CLOSE,
    EXIT_CIRCUIT_BREAKER,
    SIGNAL_LONG,
    SIGNAL_SHORT,
    SIGNAL_NONE,
    PARAMS_RANGE_MINUTES,
    PARAMS_STOP_TICKS,
    PARAMS_TARGET_TICKS,
    PARAMS_CONTRACTS,
    PARAMS_DAILY_STOP,
    PARAMS_DAILY_TARGET,
    PARAMS_MAX_TRADES,
    PARAMS_BUFFER_TICKS,
    PARAMS_VOL_THRESHOLD,
    PARAMS_STOP_PENALTY,
    PARAMS_COMMISSION,
    PARAMS_ARRAY_LENGTH,
)


class TestTradeLogDtype:
    def test_has_required_fields(self):
        names = TRADE_LOG_DTYPE.names
        assert "day_id" in names
        assert "phase_id" in names
        assert "payout_cycle_id" in names
        assert "entry_time" in names
        assert "exit_time" in names
        assert "entry_price" in names
        assert "exit_price" in names
        assert "entry_slippage" in names
        assert "exit_slippage" in names
        assert "entry_commission" in names
        assert "exit_commission" in names
        assert "contracts" in names
        assert "gross_pnl" in names
        assert "net_pnl" in names
        assert "signal_type" in names
        assert "exit_reason" in names

    def test_can_create_empty_array(self):
        arr = np.zeros(100, dtype=TRADE_LOG_DTYPE)
        assert arr.shape == (100,)
        assert arr["net_pnl"][0] == 0.0

    def test_field_types(self):
        assert TRADE_LOG_DTYPE["day_id"] == np.dtype("i4")
        assert TRADE_LOG_DTYPE["phase_id"] == np.dtype("i1")
        assert TRADE_LOG_DTYPE["payout_cycle_id"] == np.dtype("i2")
        assert TRADE_LOG_DTYPE["entry_time"] == np.dtype("i8")
        assert TRADE_LOG_DTYPE["net_pnl"] == np.dtype("f8")
        assert TRADE_LOG_DTYPE["gross_pnl"] == np.dtype("f8")
        assert TRADE_LOG_DTYPE["contracts"] == np.dtype("i4")
        assert TRADE_LOG_DTYPE["signal_type"] == np.dtype("i1")


class TestDailyLogDtype:
    def test_has_required_fields(self):
        names = DAILY_LOG_DTYPE.names
        assert "day_id" in names
        assert "phase_id" in names
        assert "payout_cycle_id" in names
        assert "had_trade" in names
        assert "n_trades" in names
        assert "day_pnl" in names
        assert "net_payout" in names

    def test_field_types(self):
        assert DAILY_LOG_DTYPE["day_id"] == np.dtype("i4")
        assert DAILY_LOG_DTYPE["phase_id"] == np.dtype("i1")
        assert DAILY_LOG_DTYPE["payout_cycle_id"] == np.dtype("i2")
        assert DAILY_LOG_DTYPE["had_trade"] == np.dtype("i1")
        assert DAILY_LOG_DTYPE["n_trades"] == np.dtype("i2")
        assert DAILY_LOG_DTYPE["day_pnl"] == np.dtype("f8")
        assert DAILY_LOG_DTYPE["net_payout"] == np.dtype("f8")


class TestConstants:
    def test_mnq_constants(self):
        assert MNQ_TICK_SIZE == 0.25
        assert MNQ_TICK_VALUE == 0.50
        assert MNQ_COMMISSION_PER_SIDE == 0.54

    def test_session_bars(self):
        assert BARS_PER_RTH_SESSION == 390  # 09:30-16:00 = 390 minutes

    def test_exit_reasons(self):
        assert EXIT_TARGET == 0
        assert EXIT_STOP == 1
        assert EXIT_HARD_CLOSE == 2
        assert EXIT_CIRCUIT_BREAKER == 3

    def test_signal_codes(self):
        assert SIGNAL_LONG == 1
        assert SIGNAL_SHORT == -1
        assert SIGNAL_NONE == 0


class TestParamsIndexConstants:
    def test_indices_are_sequential(self):
        indices = [
            PARAMS_RANGE_MINUTES, PARAMS_STOP_TICKS, PARAMS_TARGET_TICKS,
            PARAMS_CONTRACTS, PARAMS_DAILY_STOP, PARAMS_DAILY_TARGET,
            PARAMS_MAX_TRADES, PARAMS_BUFFER_TICKS, PARAMS_VOL_THRESHOLD,
            PARAMS_STOP_PENALTY, PARAMS_COMMISSION,
        ]
        assert indices == list(range(11))

    def test_array_length_matches(self):
        assert PARAMS_ARRAY_LENGTH == 11

    def test_no_duplicate_indices(self):
        indices = [
            PARAMS_RANGE_MINUTES, PARAMS_STOP_TICKS, PARAMS_TARGET_TICKS,
            PARAMS_CONTRACTS, PARAMS_DAILY_STOP, PARAMS_DAILY_TARGET,
            PARAMS_MAX_TRADES, PARAMS_BUFFER_TICKS, PARAMS_VOL_THRESHOLD,
            PARAMS_STOP_PENALTY, PARAMS_COMMISSION,
        ]
        assert len(indices) == len(set(indices))
