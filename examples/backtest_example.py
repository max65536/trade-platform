from __future__ import annotations

from pathlib import Path
import json
import pandas as pd

from trade_platform.dataio import CandleFrame
from trade_platform import chan, strategy
from trade_platform.backtest import execute_with_risk, simple_execute


def ensure_data(csv_path: Path):
    if not csv_path.exists():
        # Generate synthetic if missing
        from examples.generate_synthetic import main as gen
        gen()


def main():
    data_path = Path("examples/data/BTCUSDT-1h.csv")
    ensure_data(data_path)
    cf = CandleFrame.read_csv(str(data_path))
    df = cf.df.copy().reset_index(drop=True)

    # Analyze with Chan and derive base signals
    out = chan.analyze(df)
    signals = out["signals"]

    # Optional strategy filters (RSI/ATR thresholds)
    df_ind = strategy.ensure_indicators(df)
    signals = strategy.apply_signal_filters(
        signals,
        df_ind,
        rsi_min=55,
        rsi_max=None,
        min_atr_pct=0.004,
        max_atr_pct=None,
    )

    # Run risk-aware executor (TP/SL intrabar), long-only
    res = execute_with_risk(
        df,
        signals,
        fee_rate=0.0005,
        stop_loss_pct=0.02,
        take_profit_pct=0.04,
        initial_capital=1.0,
        position_size=1.0,
        slippage_bps=0.0,
        mode="long",
        close_at_end=True,
        periods_per_year=8760.0,  # hourly
    )

    print("Backtest stats (1h synthetic):")
    for k, v in res.stats.items():
        print(f"- {k}: {v}")

    out_dir = Path("examples/out")
    out_dir.mkdir(parents=True, exist_ok=True)
    res.trades.to_csv(out_dir / "trades.csv", index=False)
    with open(out_dir / "stats.json", "w", encoding="utf-8") as f:
        json.dump(res.stats, f, indent=2)
    print(f"Saved trades/stats under {out_dir}")


if __name__ == "__main__":
    main()

