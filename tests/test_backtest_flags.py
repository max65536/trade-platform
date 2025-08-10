import math
import pandas as pd

from trade_platform.backtest import simple_execute, execute_with_risk


def _mk_df(n=6, start=100.0, step=1.0, freq='H'):
    # Build a simple OHLCV with deterministic progression
    opens = [start + i * step for i in range(n)]
    highs = [o + step for o in opens]
    lows = [o - step for o in opens]
    closes = [o for o in opens]
    ts = pd.date_range('2023-01-01', periods=n, freq=freq)
    return pd.DataFrame({
        'datetime': ts,
        'open': opens,
        'high': highs,
        'low': lows,
        'close': closes,
        'volume': [1.0] * n,
    })


def test_simple_execute_ppyear_and_volatility_ann():
    df = _mk_df(n=6)
    # Signals: two round-trips (buy0->sell1, buy2->sell3)
    sigs = pd.DataFrame({
        'index': [0, 1, 2, 3],
        'signal': ['buy', 'sell', 'buy', 'sell'],
        'price': [df.loc[i, 'close'] for i in [0, 1, 2, 3]],
    })
    ppy = 100.0
    res = simple_execute(
        df,
        sigs,
        fee_rate=0.0,
        initial_capital=1.0,
        position_size=1.0,
        slippage_bps=0.0,
        mode='long',
        close_at_end=False,
        periods_per_year=ppy,
    )
    # Two closed trades
    assert len(res.trades) == 2
    # Volatility annualization check (sigma * sqrt(ppy)) on equity bar returns
    r = res.equity_curve.pct_change().dropna()
    sigma = float(r.std(ddof=0)) if len(r) else 0.0
    expected_vol = sigma * math.sqrt(ppy)
    assert math.isfinite(res.stats['volatility_ann'])
    assert abs(res.stats['volatility_ann'] - expected_vol) < 1e-9


def test_simple_execute_long_short_and_close_at_end():
    df = _mk_df(n=6)
    # One long round-trip, then enter short via sell when flat and close at end
    sigs = pd.DataFrame({
        'index': [0, 1, 4],  # buy@0->exit on next, then sell@4 enters short at 5
        'signal': ['buy', 'sell', 'sell'],
        'price': [df.loc[i, 'close'] for i in [0, 1, 4]],
    })
    res = simple_execute(
        df,
        sigs,
        fee_rate=0.0,
        initial_capital=1.0,
        position_size=1.0,
        slippage_bps=0.0,
        mode='long_short',
        close_at_end=True,
        periods_per_year=252.0,
    )
    # Expect two trades: one long closed, one short closed at end
    assert len(res.trades) == 2
    last = res.trades.iloc[-1]
    assert last['side'] == 'short'
    assert int(last['exit_idx']) == len(df) - 1  # closed at final bar


def test_execute_with_risk_intrabar_tp_long_and_short():
    # Construct bars so that TP is hit intrabar on entry bar for both long and short
    ts = pd.date_range('2023-01-01', periods=3, freq='H')
    df = pd.DataFrame({
        'datetime': ts,
        'open': [100.0, 100.0, 100.0],
        'high': [100.0, 106.0, 100.0],  # bar1 high big enough
        'low': [100.0, 94.0, 100.0],    # bar1 low small enough
        'close': [100.0, 100.0, 100.0],
        'volume': [1.0, 1.0, 1.0],
    })

    # Long: buy at 0 -> entry at 1; tp 5% => 105, bar1 high=106 triggers intrabar exit
    sig_long = pd.DataFrame({'index': [0], 'signal': ['buy'], 'price': [100.0]})
    res_long = execute_with_risk(
        df,
        sig_long,
        fee_rate=0.0,
        stop_loss_pct=None,
        take_profit_pct=0.05,
        initial_capital=1.0,
        position_size=1.0,
        slippage_bps=0.0,
        mode='long',
        close_at_end=False,
        periods_per_year=252.0,
    )
    assert len(res_long.trades) == 1
    t = res_long.trades.iloc[0]
    assert int(t['entry_idx']) == 1 and int(t['exit_idx']) == 1
    assert t['side'] == 'long'
    assert abs(t['ret'] - 0.05) < 1e-9

    # Short: sell at 0 -> entry at 1; tp 5% => entry*(1-0.05)=95, bar1 low=94 triggers intrabar exit
    sig_short = pd.DataFrame({'index': [0], 'signal': ['sell'], 'price': [100.0]})
    res_short = execute_with_risk(
        df,
        sig_short,
        fee_rate=0.0,
        stop_loss_pct=None,
        take_profit_pct=0.05,
        initial_capital=1.0,
        position_size=1.0,
        slippage_bps=0.0,
        mode='long_short',
        close_at_end=False,
        periods_per_year=252.0,
    )
    assert len(res_short.trades) == 1
    t2 = res_short.trades.iloc[0]
    assert int(t2['entry_idx']) == 1 and int(t2['exit_idx']) == 1
    assert t2['side'] == 'short'
    # Short 5% profit
    assert abs(t2['ret'] - 0.05) < 1e-9

