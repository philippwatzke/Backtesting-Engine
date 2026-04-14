import numpy as np

from propfirm.core.types import (
    PARAMS_ARRAY_LENGTH,
    PARAMS_BAND_MULTIPLIER,
    PARAMS_DAILY_TARGET,
    PARAMS_MAX_TRADES,
    PARAMS_SMA_PERIOD,
)
from propfirm.strategy.m6a_fade_strategy import m6a_fade_signal


class TestM6AFadeSignal:
    def _make_params(self):
        params = np.zeros(PARAMS_ARRAY_LENGTH, dtype=np.float64)
        params[PARAMS_DAILY_TARGET] = 600.0
        params[PARAMS_MAX_TRADES] = 1.0
        params[PARAMS_SMA_PERIOD] = 20.0
        params[PARAMS_BAND_MULTIPLIER] = 2.0
        return params

    def test_long_signal_when_close_below_lower_band(self):
        n = 40
        o = np.full(n, 0.7200, dtype=np.float64)
        h = np.full(n, 0.7201, dtype=np.float64)
        l = np.full(n, 0.7199, dtype=np.float64)
        c = np.full(n, 0.7200, dtype=np.float64)
        v = np.full(n, 1000, dtype=np.uint64)
        mod = np.arange(0, n * 5, 5, dtype=np.int16)
        c[39] = 0.7185

        atr = np.ones_like(c)
        trail = np.ones_like(c)
        daily_ratio = np.ones_like(c)
        sig = m6a_fade_signal(
            39, o, h, l, c, v, atr, trail, daily_ratio, mod,
            0.0, 0.0, 0, 0.0, False, 0, self._make_params()
        )
        assert sig == 1

    def test_short_signal_when_close_above_upper_band(self):
        n = 40
        o = np.full(n, 0.7200, dtype=np.float64)
        h = np.full(n, 0.7201, dtype=np.float64)
        l = np.full(n, 0.7199, dtype=np.float64)
        c = np.full(n, 0.7200, dtype=np.float64)
        v = np.full(n, 1000, dtype=np.uint64)
        mod = np.arange(0, n * 5, 5, dtype=np.int16)
        c[39] = 0.7215

        atr = np.ones_like(c)
        trail = np.ones_like(c)
        daily_ratio = np.ones_like(c)
        sig = m6a_fade_signal(
            39, o, h, l, c, v, atr, trail, daily_ratio, mod,
            0.0, 0.0, 0, 0.0, False, 0, self._make_params()
        )
        assert sig == -1
