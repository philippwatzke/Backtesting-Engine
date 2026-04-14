import numpy as np

from propfirm.core.types import (
    PARAMS_ARRAY_LENGTH,
    PARAMS_DAILY_TARGET,
    PARAMS_MAX_RVOL,
    PARAMS_MAX_TRADES,
    PARAMS_MIN_RVOL,
    PARAMS_RANGE_MINUTES,
    PARAMS_TRIGGER_END_MINUTE,
    PARAMS_TRIGGER_START_MINUTE,
)
from propfirm.strategy.mgc_macro_orb_strategy import mgc_macro_orb_signal


class TestMGCMacroORBSignal:
    def _make_params(self):
        params = np.zeros(PARAMS_ARRAY_LENGTH, dtype=np.float64)
        params[PARAMS_DAILY_TARGET] = 600.0
        params[PARAMS_MAX_TRADES] = 1.0
        params[PARAMS_RANGE_MINUTES] = 30.0
        params[PARAMS_TRIGGER_START_MINUTE] = 30.0
        params[PARAMS_TRIGGER_END_MINUTE] = 180.0
        params[PARAMS_MIN_RVOL] = 1.2
        params[PARAMS_MAX_RVOL] = 2.5
        return params

    def test_long_signal_when_price_breaks_above_opening_range_with_valid_rvol(self):
        n = 60
        o = np.full(n, 2400.0, dtype=np.float64)
        h = np.full(n, 2400.5, dtype=np.float64)
        l = np.full(n, 2399.5, dtype=np.float64)
        c = np.full(n, 2400.0, dtype=np.float64)
        v = np.full(n, 1000, dtype=np.uint64)
        mod = np.arange(n, dtype=np.int16)
        rvol = np.ones(n, dtype=np.float64)

        h[35] = 2402.0
        c[35] = 2401.8
        rvol[35] = 1.5

        atr = np.ones_like(c)
        trail = np.ones_like(c)
        daily_ratio = np.ones_like(c)
        sig = mgc_macro_orb_signal(
            35, o, h, l, c, v, atr, trail, daily_ratio, rvol, mod,
            0.0, 0.0, 0, 0.0, False, 0, self._make_params()
        )
        assert sig == 1

    def test_short_signal_when_price_breaks_below_opening_range_with_valid_rvol(self):
        n = 60
        o = np.full(n, 2400.0, dtype=np.float64)
        h = np.full(n, 2400.5, dtype=np.float64)
        l = np.full(n, 2399.5, dtype=np.float64)
        c = np.full(n, 2400.0, dtype=np.float64)
        v = np.full(n, 1000, dtype=np.uint64)
        mod = np.arange(n, dtype=np.int16)
        rvol = np.ones(n, dtype=np.float64)

        l[35] = 2398.0
        c[35] = 2398.2
        rvol[35] = 1.5

        atr = np.ones_like(c)
        trail = np.ones_like(c)
        daily_ratio = np.ones_like(c)
        sig = mgc_macro_orb_signal(
            35, o, h, l, c, v, atr, trail, daily_ratio, rvol, mod,
            0.0, 0.0, 0, 0.0, False, 0, self._make_params()
        )
        assert sig == -1

    def test_blocks_inside_range_building_window(self):
        n = 60
        o = np.full(n, 2400.0, dtype=np.float64)
        h = np.full(n, 2400.5, dtype=np.float64)
        l = np.full(n, 2399.5, dtype=np.float64)
        c = np.full(n, 2400.0, dtype=np.float64)
        v = np.full(n, 1000, dtype=np.uint64)
        mod = np.arange(n, dtype=np.int16)
        rvol = np.ones(n, dtype=np.float64)

        h[10] = 2402.0
        atr = np.ones_like(c)
        trail = np.ones_like(c)
        daily_ratio = np.ones_like(c)
        sig = mgc_macro_orb_signal(
            10, o, h, l, c, v, atr, trail, daily_ratio, rvol, mod,
            0.0, 0.0, 0, 0.0, False, 0, self._make_params()
        )
        assert sig == 0

    def test_blocks_breakout_when_rvol_is_too_high(self):
        n = 60
        o = np.full(n, 2400.0, dtype=np.float64)
        h = np.full(n, 2400.5, dtype=np.float64)
        l = np.full(n, 2399.5, dtype=np.float64)
        c = np.full(n, 2400.0, dtype=np.float64)
        v = np.full(n, 1000, dtype=np.uint64)
        mod = np.arange(n, dtype=np.int16)
        rvol = np.ones(n, dtype=np.float64)
        h[35] = 2402.0
        c[35] = 2401.8
        rvol[35] = 3.0

        atr = np.ones_like(c)
        trail = np.ones_like(c)
        daily_ratio = np.ones_like(c)
        sig = mgc_macro_orb_signal(
            35, o, h, l, c, v, atr, trail, daily_ratio, rvol, mod,
            0.0, 0.0, 0, 0.0, False, 0, self._make_params()
        )
        assert sig == 0
