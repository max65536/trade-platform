from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from datetime import datetime, timezone
from typing import List, Optional

import pandas as pd

from trade_platform.exchanges import ExchangeClient
from trade_platform.dataio import CandleFrame
from trade_platform import chan
from trade_platform import multiframe as mtf
from trade_platform import strategy
from trade_platform.backtest import simple_execute, execute_with_risk
from trade_platform import plotting


def parse_date_ms(s: Optional[str]) -> Optional[int]:
    if not s:
        return None
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def fetch_csv(exchange: str, symbol: str, timeframe: str, out_path: str, since: Optional[str], limit: int, max_bars: Optional[int]):
    ex = ExchangeClient(exchange)
    ex.load_markets()
    candles = ex.fetch_ohlcv_all(symbol=symbol, timeframe=timeframe, since=parse_date_ms(since), limit=limit, max_bars=max_bars)
    cf = CandleFrame.from_ohlcv(candles)
    ensure_dir(os.path.dirname(out_path))
    cf.to_csv(out_path)
    return cf.df


def run_pipeline(
    exchange: str,
    symbols: List[str],
    lower_tf: str,
    higher_tf: str,
    out_dir: str,
    since: Optional[str] = None,
    limit: int = 500,
    max_bars: Optional[int] = 5000,
    # MTF filters
    require_htf_breakout: bool = False,
    min_htf_run: int = 0,
    # Strategy filters
    rsi_min: Optional[float] = None,
    rsi_max: Optional[float] = None,
    min_atr_pct: Optional[float] = None,
    max_atr_pct: Optional[float] = None,
    # Risk
    fee: float = 0.0005,
    stop_pct: Optional[float] = None,
    tp_pct: Optional[float] = None,
    # Plotting
    theme: str = "light",
    # Divergence tuning
    div_min_price_ext_pct: float = 0.0,
    div_min_hist_delta: float = 0.0,
    div_require_hist_sign_consistency: bool = False,
) -> None:
    raw_dir = os.path.join(out_dir, "raw")
    ann_dir = os.path.join(out_dir, "annotated")
    plot_dir = os.path.join(out_dir, "plots")
    trades_dir = os.path.join(out_dir, "trades")
    stats_dir = os.path.join(out_dir, "stats")
    for d in (raw_dir, ann_dir, plot_dir, trades_dir, stats_dir):
        ensure_dir(d)

    for sym in symbols:
        sym_noslash = sym.replace("/", "")
        # 1) Fetch LTF and HTF CSVs
        ltf_csv = os.path.join(raw_dir, f"{sym_noslash}-{lower_tf}.csv")
        htf_csv = os.path.join(raw_dir, f"{sym_noslash}-{higher_tf}.csv")
        print(f"Fetching {sym} {lower_tf} -> {ltf_csv}")
        ldf = fetch_csv(exchange, sym, lower_tf, ltf_csv, since, limit, max_bars)
        print(f"Fetching {sym} {higher_tf} -> {htf_csv}")
        hdf = fetch_csv(exchange, sym, higher_tf, htf_csv, since, limit, max_bars)

        # 2) Analyze both frames
        lo = chan.analyze(
            ldf,
            div_min_price_ext_pct=div_min_price_ext_pct,
            div_min_hist_delta=div_min_hist_delta,
            div_require_hist_sign_consistency=div_require_hist_sign_consistency,
        )
        ho = chan.analyze(
            hdf,
            div_min_price_ext_pct=div_min_price_ext_pct,
            div_min_hist_delta=div_min_hist_delta,
            div_require_hist_sign_consistency=div_require_hist_sign_consistency,
        )

        # 3) Align HTF context to LTF
        htf_ctx = mtf.align_htf_to_ltf(ldf, hdf, ho["segments"], ho["bands"])
        ldf2 = ldf.reset_index(drop=True).copy()
        for col in htf_ctx.columns:
            ldf2[col] = htf_ctx[col].values

        # 4) Build LTF base signals and filter by HTF
        base_sigs = lo["signals"]
        mtf_sigs = mtf.filter_signals_with_htf_opts(
            base_sigs,
            ldf2,
            require_htf_breakout=require_htf_breakout,
            min_htf_run=min_htf_run,
        )

        # 5) Apply strategy filters (RSI/ATR)
        ldf2 = strategy.ensure_indicators(ldf2)
        filt_sigs = strategy.apply_signal_filters(
            mtf_sigs,
            ldf2,
            rsi_min=rsi_min,
            rsi_max=rsi_max,
            min_atr_pct=min_atr_pct,
            max_atr_pct=max_atr_pct,
        )

        # 6) Execute trades
        if stop_pct is not None or tp_pct is not None:
            res = execute_with_risk(ldf2, filt_sigs, fee_rate=fee, stop_loss_pct=stop_pct, take_profit_pct=tp_pct)
        else:
            res = simple_execute(ldf2, filt_sigs, fee_rate=fee)

        # 7) Save annotated LTF CSV
        ann_path = os.path.join(ann_dir, f"{sym_noslash}-{lower_tf}-mtf.csv")
        ldf2.loc[:, "signal_base"] = None
        ldf2.loc[:, "signal_mtf"] = None
        if not base_sigs.empty:
            ldf2.loc[base_sigs["index"], "signal_base"] = base_sigs["signal"].values
        if not filt_sigs.empty:
            ldf2.loc[filt_sigs["index"], "signal_mtf"] = filt_sigs["signal"].values
        ldf2.to_csv(ann_path, index=False)
        print(f"Annotated saved: {ann_path}")

        # 8) Save trades and stats
        trades_path = os.path.join(trades_dir, f"{sym_noslash}-{lower_tf}-trades.csv")
        stats_path = os.path.join(stats_dir, f"{sym_noslash}-{lower_tf}-stats.json")
        if not res.trades.empty:
            res.trades.to_csv(trades_path, index=False)
            print(f"Trades saved: {trades_path}")
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(res.stats, f, indent=2)
        print(f"Stats saved: {stats_path}")

        # 9) Plot
        plot_path = os.path.join(plot_dir, f"{sym_noslash}-{lower_tf}-mtf.png")
        plotting.plot_kline(
            ldf2,
            out=plot_path,
            title=f"{sym} {lower_tf} (HTF: {higher_tf})",
            pivot_low=ldf2.get("htf_pivot_low"),
            pivot_high=ldf2.get("htf_pivot_high"),
            signals=filt_sigs,
            theme=theme,
            trades=res.trades,
        )
        print(f"Plot saved: {plot_path}")


def build_parser():
    p = argparse.ArgumentParser(description="End-to-end pipeline: fetch -> analyze -> MTF -> filter -> backtest -> plot")
    p.add_argument("--exchange", required=True)
    p.add_argument("--symbols", nargs="+", required=True)
    p.add_argument("--lower-tf", required=True)
    p.add_argument("--higher-tf", required=True)
    p.add_argument("--out-dir", default="runs")
    p.add_argument("--since", default=None)
    p.add_argument("--limit", type=int, default=500)
    p.add_argument("--max-bars", type=int, default=5000)
    # MTF
    p.add_argument("--require-htf-breakout", action="store_true")
    p.add_argument("--min-htf-run", type=int, default=0)
    # Strategy filters
    p.add_argument("--rsi-min", type=float, default=None)
    p.add_argument("--rsi-max", type=float, default=None)
    p.add_argument("--min-atr-pct", type=float, default=None)
    p.add_argument("--max-atr-pct", type=float, default=None)
    # Risk
    p.add_argument("--fee", type=float, default=0.0005)
    p.add_argument("--stop-pct", type=float, default=None)
    p.add_argument("--tp-pct", type=float, default=None)
    # Plot
    p.add_argument("--theme", choices=["light", "dark", "minimal"], default="light")
    # Divergence tuning
    p.add_argument("--div-min-price-ext-pct", type=float, default=0.0, help="Min price extension pct for divergence (e.g., 0.003)")
    p.add_argument("--div-min-hist-delta", type=float, default=0.0, help="Min MACD histogram delta for divergence")
    p.add_argument("--div-require-hist-sign-consistency", action="store_true", help="Require MACD hist signs consistent (buy3<=0, sell3>=0)")
    return p


def main():
    args = build_parser().parse_args()
    run_pipeline(
        exchange=args.exchange,
        symbols=args.symbols,
        lower_tf=args.lower_tf,
        higher_tf=args.higher_tf,
        out_dir=args.out_dir,
        since=args.since,
        limit=args.limit,
        max_bars=args.max_bars,
        require_htf_breakout=args.require_htf_breakout,
        min_htf_run=args.min_htf_run,
        rsi_min=args.rsi_min,
        rsi_max=args.rsi_max,
        min_atr_pct=args.min_atr_pct,
        max_atr_pct=args.max_atr_pct,
        fee=args.fee,
        stop_pct=args.stop_pct,
        tp_pct=args.tp_pct,
        theme=args.theme,
        div_min_price_ext_pct=args.div_min_price_ext_pct,
        div_min_hist_delta=args.div_min_hist_delta,
        div_require_hist_sign_consistency=args.div_require_hist_sign_consistency,
    )


if __name__ == "__main__":
    main()
