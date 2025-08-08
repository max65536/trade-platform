import numpy as np
import pandas as pd

from trade_platform.backtest import simple_execute


def test_simple_execute_roundtrip():
    # Build simple series: open increases by 1 each bar
    n = 10
    open_ = np.arange(10, 10 + n)
    close = open_ + 0.5
    df = pd.DataFrame({
        "open": open_,
        "high": close + 0.2,
        "low": open_ - 0.2,
        "close": close,
    })

    # Buy at 0 -> executed at open[1]; Sell at 2 -> executed at open[3]
    signals = pd.DataFrame({
        "index": [0, 2],
        "signal": ["buy", "sell"],
        "price": [float(close[0]), float(close[2])],
    })
    fee = 0.001
    res = simple_execute(df, signals, fee_rate=fee)

    assert len(res.trades) == 1
    entry_px = open_[1] * (1 + fee)
    exit_px = open_[3] * (1 - fee)
    exp_ret = (exit_px - entry_px) / entry_px
    assert abs(res.trades.iloc[0]["ret"] - exp_ret) < 1e-9

