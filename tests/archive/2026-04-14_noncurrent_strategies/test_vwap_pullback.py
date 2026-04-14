import numpy as np

from propfirm.core.types import PARAMS_ARRAY_LENGTH, PARAMS_TICK_SIZE
from propfirm.strategy.vwap_pullback import vwap_pullback_signal


class TestVWAPPullbackSignal:
    def _make_arrays(self, n, base=20000.0):
        opens = np.full(n, base, dtype=np.float64)
        highs = np.full(n, base + 1.0, dtype=np.float64)
        lows = np.full(n, base - 1.0, dtype=np.float64)
        closes = np.full(n, base, dtype=np.float64)
        volumes = np.full(n, 1000, dtype=np.uint64)
        mod = np.arange(n, dtype=np.int16)
        params = np.zeros(PARAMS_ARRAY_LENGTH, dtype=np.float64)
        params[5] = 600.0
        params[6] = 2.0
        params[11] = 10.0
        params[12] = 20.0
        params[PARAMS_TICK_SIZE] = 0.25
        return opens, highs, lows, closes, volumes, mod, params

    def test_no_signal_before_0945(self):
        o, h, l, c, v, mod, p = self._make_arrays(390)
        atr = np.ones_like(c)
        trail = np.ones_like(c)
        sig = vwap_pullback_signal(10, o, h, l, c, v, atr, trail, mod, 0.0, 0.0, 0, 0.0, False, 0, p)
        assert sig == 0

    def test_long_signal_when_vwap_holds(self):
        o, h, l, c, v, mod, p = self._make_arrays(390)
        for i in range(25):
            o[i] = 20000.0 + 0.2 * i
            c[i] = 20000.1 + 0.2 * i
            h[i] = c[i] + 0.3
            l[i] = o[i] - 0.2
        idx = 25
        o[idx] = 20003.0
        l[idx] = 20002.2
        h[idx] = 20005.0
        c[idx] = 20004.4
        atr = np.ones_like(c)
        trail = np.ones_like(c)
        sig = vwap_pullback_signal(idx, o, h, l, c, v, atr, trail, mod, 0.0, 0.0, 0, 0.0, False, 0, p)
        assert sig == 1

    def test_short_signal_when_vwap_rejects(self):
        o, h, l, c, v, mod, p = self._make_arrays(390)
        for i in range(25):
            o[i] = 20000.0 - 0.2 * i
            c[i] = 19999.9 - 0.2 * i
            h[i] = o[i] + 0.2
            l[i] = c[i] - 0.3
        idx = 25
        o[idx] = 19997.0
        h[idx] = 19997.8
        l[idx] = 19995.0
        c[idx] = 19995.6
        atr = np.ones_like(c)
        trail = np.ones_like(c)
        sig = vwap_pullback_signal(idx, o, h, l, c, v, atr, trail, mod, 0.0, 0.0, 0, 0.0, False, 0, p)
        assert sig == -1

    def test_no_signal_after_1530(self):
        o, h, l, c, v, mod, p = self._make_arrays(390)
        atr = np.ones_like(c)
        trail = np.ones_like(c)
        sig = vwap_pullback_signal(361, o, h, l, c, v, atr, trail, mod, 0.0, 0.0, 0, 0.0, False, 0, p)
        assert sig == 0
