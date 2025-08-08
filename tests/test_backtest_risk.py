import numpy as np
import pandas as pd

from trade_platform.backtest import execute_with_risk


def test_execute_with_risk_tp_hits_first():
    # Build bars so that TP is hit on the next bar's high
    df = pd.DataFrame({
        "open": [100, 101, 102],
        "high": [101, 105, 103],  # next bar spikes to 105
        "low": [99, 100, 101],
        "close": [101, 104, 102],
    })
    # Buy at index 0 -> executed at open[1]
    signals = pd.DataFrame({"index": [0], "signal": ["buy"], "price": [100.0]})
    fee = 0.0
    tp = 0.03  # 3%
    res = execute_with_risk(df, signals, fee_rate=fee, take_profit_pct=tp, stop_loss_pct=0.02)
    assert len(res.trades) == 1
    entry_px = df.loc[1, "open"]
    exp_tp = entry_px * (1 + tp)
    assert abs(res.trades.iloc[0]["exit_px"] - exp_tp) < 1e-9


def test_execute_with_risk_sl_hits():
    # Build bars where SL hits before any opposite signal
    df = pd.DataFrame({
        "open": [100, 99, 98],
        "high": [101, 100, 99],
        "low": [99, 90, 97],  # second bar low breaches SL
        "close": [100, 95, 98],
    })
    signals = pd.DataFrame({"index": [0], "signal": ["buy"], "price": [100.0]})
    fee = 0.0
    sl = 0.05
    res = execute_with_risk(df, signals, fee_rate=fee, take_profit_pct=None, stop_loss_pct=sl)
    assert len(res.trades) == 1
    entry_px = df.loc[1, "open"]
    exp_sl = entry_px * (1 - sl)
    assert abs(res.trades.iloc[0]["exit_px"] - exp_sl) < 1e-9

