from __future__ import annotations

from pathlib import Path
import pandas as pd

from trade_platform.dataio import CandleFrame
from trade_platform import chan, strategy, plotting
from trade_platform.backtest import execute_with_risk


def ensure_data(csv_path: Path):
    if not csv_path.exists():
        from examples.generate_synthetic import main as gen
        gen()


def main():
    data_path = Path("examples/data/BTCUSDT-4h.csv")
    ensure_data(data_path)
    df = CandleFrame.read_csv(str(data_path)).df.copy().reset_index(drop=True)

    # Analyze to get pivots/segments/signals
    out = chan.analyze(df)
    bands = out.get("bands")
    pens = out.get("pens")
    segments = out.get("segments")
    signals = out.get("signals")

    # Compute a quick backtest to overlay trades (optional)
    df_ind = strategy.ensure_indicators(df)
    filt = strategy.apply_signal_filters(signals, df_ind, rsi_min=55, min_atr_pct=0.004)
    res = execute_with_risk(df, filt, fee_rate=0.0005, stop_loss_pct=0.02, take_profit_pct=0.04)

    out_dir = Path("examples/out")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_png = out_dir / "plot.png"

    plotting.plot_kline(
        df,
        out=str(out_png),
        title="Synthetic 4h Plot",
        pivot_low=bands["pivot_low"] if bands is not None and not bands.empty else None,
        pivot_high=bands["pivot_high"] if bands is not None and not bands.empty else None,
        pens=pens,
        segments=segments,
        signals=signals,
        trades=res.trades,
        theme="dark",
        show_macd=True,
        show_rsi=True,
        show_atr=True,
    )
    print(f"Saved plot to {out_png}")


if __name__ == "__main__":
    main()

