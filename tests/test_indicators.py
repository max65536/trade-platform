import numpy as np
import pandas as pd

from trade_platform import indicators as ta


def test_sma_basic():
    s = pd.Series([1, 2, 3, 4, 5])
    out = ta.sma(s, 3)
    assert np.isnan(out.iloc[1])
    assert out.iloc[4] == (3 + 4 + 5) / 3


def test_rsi_bounds():
    up = pd.Series(np.arange(1, 50))
    rsi_up = ta.rsi(up, 14)
    assert rsi_up.iloc[-1] > 70

    down = pd.Series(np.arange(50, 1, -1))
    rsi_down = ta.rsi(down, 14)
    assert rsi_down.iloc[-1] < 30


def test_macd_shapes():
    s = pd.Series(np.linspace(1, 2, 100))
    macd_line, signal, hist = ta.macd(s)
    assert len(macd_line) == 100
    assert len(signal) == 100
    assert len(hist) == 100

