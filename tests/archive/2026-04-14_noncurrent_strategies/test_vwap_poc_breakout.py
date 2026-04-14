import numpy as np

from propfirm.strategy.vwap_poc_breakout import vwap_poc_breakout_signal


class TestVWAPPOCBreakoutSignal:
    def _make_arrays(self, n, base=20000.0):
        opens = np.full(n, base, dtype=np.float64)
        highs = np.full(n, base + 1.0, dtype=np.float64)
        lows = np.full(n, base - 1.0, dtype=np.float64)
        closes = np.full(n, base, dtype=np.float64)
        volumes = np.full(n, 1000, dtype=np.uint64)
        mod = np.arange(n, dtype=np.int16)
        params = np.zeros(16, dtype=np.float64)
        params[5] = 600.0
        params[6] = 2.0
        params[12] = 20.0
        params[14] = 1.0
        params[15] = 30.0
        return opens, highs, lows, closes, volumes, mod, params

    def test_no_signal_before_0945(self):
        o, h, l, c, v, mod, p = self._make_arrays(390)
        atr = np.ones_like(c)
        trail = np.ones_like(c)
        sig = vwap_poc_breakout_signal(10, o, h, l, c, v, atr, trail, mod, 0.0, 0.0, 0, 0.0, False, 0, p)
        assert sig == 0

    def test_long_breakout_signal(self):
        o, h, l, c, v, mod, p = self._make_arrays(390)
        for i in range(40):
            o[i] = 20000.0 + 0.01 * i
            c[i] = 20000.1 + 0.01 * i
            h[i] = c[i] + 0.2
            l[i] = o[i] - 0.2
            v[i] = 1000 + i
        v[39] = 5000
        idx = 40
        o[idx] = 20000.0
        h[idx] = 20003.0
        l[idx] = 19999.9
        c[idx] = 20002.8
        v[idx] = 1200
        atr = np.ones_like(c)
        trail = np.ones_like(c)
        sig = vwap_poc_breakout_signal(idx, o, h, l, c, v, atr, trail, mod, 0.0, 0.0, 0, 0.0, False, 0, p)
        assert sig == 1

    def test_no_signal_after_1530(self):
        o, h, l, c, v, mod, p = self._make_arrays(390)
        atr = np.ones_like(c)
        trail = np.ones_like(c)
        sig = vwap_poc_breakout_signal(361, o, h, l, c, v, atr, trail, mod, 0.0, 0.0, 0, 0.0, False, 0, p)
        assert sig == 0
