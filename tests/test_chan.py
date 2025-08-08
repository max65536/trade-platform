import numpy as np
import pandas as pd

from trade_platform import chan


def make_wave_df(n=120, amp=10.0, noise=0.0):
    # Build a synthetic oscillating close series to yield fractals/pens/segments
    x = np.linspace(0, 6 * np.pi, n)
    close = 100 + amp * np.sin(x)
    if noise > 0:
        rng = np.random.default_rng(42)
        close = close + rng.normal(0, noise, size=n)
    open_ = close + 0.1
    high = np.maximum(open_, close) + 0.5
    low = np.minimum(open_, close) - 0.5
    ts = pd.date_range("2023-01-01", periods=n, freq="H")
    df = pd.DataFrame({
        "timestamp": (ts.view('int64') // 10**6),
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": 1.0,
        "datetime": ts,
    })
    return df


def test_chan_pipeline_outputs():
    df = make_wave_df()
    out = chan.analyze(df)
    assert isinstance(out["fractals"], list)
    assert isinstance(out["pens"], list)
    assert isinstance(out["segments"], list)
    assert "signals" in out
    bands = out.get("bands")
    assert bands is not None
    assert len(bands) == len(df)
    assert set(["pivot_low", "pivot_high"]).issubset(set(bands.columns))

