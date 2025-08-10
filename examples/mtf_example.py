from __future__ import annotations

from pathlib import Path
import pandas as pd

from trade_platform.dataio import CandleFrame
from trade_platform import chan, multiframe as mtf, strategy
from trade_platform.backtest import simple_execute, execute_with_risk


def ensure_data(lp: Path, hp: Path):
    if not (lp.exists() and hp.exists()):
        from examples.generate_synthetic import main as gen
        gen()


def main():
    lower_path = Path("examples/data/BTCUSDT-4h.csv")
    higher_path = Path("examples/data/BTCUSDT-1d.csv")
    ensure_data(lower_path, higher_path)

    ldf = CandleFrame.read_csv(str(lower_path)).df.copy().reset_index(drop=True)
    hdf = CandleFrame.read_csv(str(higher_path)).df.copy().reset_index(drop=True)

    lo = chan.analyze(ldf)
    ho = chan.analyze(hdf)

    htf_ctx = mtf.align_htf_to_ltf(ldf, hdf, ho["segments"], ho["bands"])
    # Attach context to lower frame
    for col in htf_ctx.columns:
        ldf[col] = htf_ctx[col].values

    base_sigs = lo["signals"]
    filt_sigs = mtf.filter_signals_with_htf_opts(
        base_sigs,
        ldf,
        require_htf_breakout=True,
        min_htf_run=2,
    )

    # Optional strategy filters
    ldf_ind = strategy.ensure_indicators(ldf)
    strat_sigs = strategy.apply_signal_filters(
        filt_sigs,
        ldf_ind,
        rsi_min=55,
        rsi_max=None,
        min_atr_pct=0.004,
        max_atr_pct=None,
    )

    # Backtest MTF-filtered signals (risk-aware)
    res = execute_with_risk(
        ldf,
        strat_sigs,
        fee_rate=0.0005,
        stop_loss_pct=0.02,
        take_profit_pct=0.04,
        initial_capital=1.0,
        position_size=1.0,
        slippage_bps=0.0,
        mode="long",
        close_at_end=True,
        periods_per_year=365.0 * 24.0 / 4.0,  # 4h bars per year
    )

    print("MTF Backtest stats (4h + 1d synthetic):")
    for k, v in res.stats.items():
        print(f"- {k}: {v}")


if __name__ == "__main__":
    main()

