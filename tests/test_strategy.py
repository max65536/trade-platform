import numpy as np
import pandas as pd

from trade_platform import strategy


def make_trend_df(n=60, base=100.0, step=0.5, hl_spread=1.0):
    ts = pd.date_range("2023-01-01", periods=n, freq="h")
    close = base + np.arange(n) * step
    open_ = close - 0.1
    high = np.maximum(open_, close) + hl_spread / 2
    low = np.minimum(open_, close) - hl_spread / 2
    df = pd.DataFrame({
        "timestamp": (ts.view("int64") // 10**6),
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": 1.0,
        "datetime": ts,
    })
    return df


def test_rsi_filter_keeps_high_momentum():
    df = make_trend_df()
    df = strategy.ensure_indicators(df)
    # signals around later indices where RSI should be higher
    signals = pd.DataFrame({
        "index": [30, 40, 50],
        "signal": ["buy", "buy", "buy"],
        "price": [df.loc[30, "close"], df.loc[40, "close"], df.loc[50, "close"]],
    })
    out = strategy.apply_signal_filters(signals, df, rsi_min=55)
    assert len(out) == len(signals)  # strong uptrend should pass


def test_rsi_filter_drops_on_max():
    df = make_trend_df()
    df = strategy.ensure_indicators(df)
    signals = pd.DataFrame({
        "index": [10, 20, 30],
        "signal": ["buy", "buy", "buy"],
        "price": [df.loc[10, "close"], df.loc[20, "close"], df.loc[30, "close"]],
    })
    out = strategy.apply_signal_filters(signals, df, rsi_max=40)
    assert len(out) == 0


def test_atr_filter_min_drops_all():
    df = make_trend_df(hl_spread=0.5)
    df = strategy.ensure_indicators(df)
    signals = pd.DataFrame({
        "index": [10, 20, 30],
        "signal": ["buy", "buy", "buy"],
        "price": [df.loc[10, "close"], df.loc[20, "close"], df.loc[30, "close"]],
    })
    # Demand unrealistically high volatility so none pass
    out = strategy.apply_signal_filters(signals, df, min_atr_pct=0.05)
    assert len(out) == 0

