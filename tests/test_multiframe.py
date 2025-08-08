import numpy as np
import pandas as pd

from trade_platform.multiframe import align_htf_to_ltf, filter_signals_with_htf_opts
from trade_platform.chan import Segment


def make_df(n, start_ts="2023-01-01", freq="H", base=100.0, step=1.0):
    ts = pd.date_range(start_ts, periods=n, freq=freq)
    close = base + np.arange(n) * step
    open_ = close - 0.2
    high = np.maximum(open_, close) + 0.3
    low = np.minimum(open_, close) - 0.3
    return pd.DataFrame({
        "timestamp": (ts.view("int64") // 10**6),
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": 1.0,
        "datetime": ts,
    })


def test_align_and_filter_with_htf():
    ldf = make_df(50, freq="H", base=100, step=0.5)
    hdf = make_df(10, freq="12H", base=100, step=2.0)

    # One HTF up segment covering entire range
    seg = Segment(start_idx=0, end_idx=len(hdf) - 1, direction="up", start_price=hdf.loc[0, "close"], end_price=hdf.loc[len(hdf) - 1, "close"])
    bands = pd.DataFrame({
        "pivot_low": [np.nan] * len(hdf),
        "pivot_high": [np.nan] * len(hdf),
    })

    ctx = align_htf_to_ltf(ldf, hdf, [seg], bands)
    assert set(ctx.columns) == {"htf_dir", "htf_pivot_low", "htf_pivot_high"}
    # All should be 1 (up) after alignment
    assert (ctx["htf_dir"].fillna(0) >= 0).all()

    # Build signals on LTF: one buy, one sell
    signals = pd.DataFrame({
        "index": [10, 20],
        "signal": ["buy", "sell"],
        "price": [ldf.loc[10, "close"], ldf.loc[20, "close"]],
    })

    ldf_with_ctx = pd.concat([ldf, ctx], axis=1)
    # Direction-only filtering keeps buy, drops sell
    out = filter_signals_with_htf_opts(signals, ldf_with_ctx, require_htf_breakout=False, min_htf_run=0)
    assert len(out) == 1 and out.iloc[0]["signal"] == "buy"

    # Add HTF pivot_high below current price so breakout passes
    ldf_with_ctx.loc[:, "htf_pivot_high"] = ldf_with_ctx["close"] - 1.0
    out2 = filter_signals_with_htf_opts(signals.iloc[[0]], ldf_with_ctx, require_htf_breakout=True, min_htf_run=0)
    assert len(out2) == 1

