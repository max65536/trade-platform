from __future__ import annotations

import argparse
import os
from datetime import datetime, timezone

import pandas as pd

from .exchanges import ExchangeClient
from .dataio import CandleFrame
from . import indicators as ta
from . import chan
from .backtest import simple_execute
from . import multiframe as mtf


def _parse_date(s: str) -> int:
    # return ms timestamp
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def cmd_fetch(args: argparse.Namespace):
    ex = ExchangeClient(args.exchange)
    ex.load_markets()
    since = _parse_date(args.since) if args.since else None
    candles = ex.fetch_ohlcv_all(
        symbol=args.symbol,
        timeframe=args.timeframe,
        since=since,
        limit=args.limit,
        max_bars=args.max_bars,
    )
    cf = CandleFrame.from_ohlcv(candles)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    cf.to_csv(args.output)
    print(f"Saved {len(cf.df)} candles to {args.output}")


def cmd_analyze(args: argparse.Namespace):
    cf = CandleFrame.read_csv(args.input)
    df = cf.df.copy()
    # attach basic indicators for reference
    df["sma20"] = ta.sma(df["close"], 20)
    df["ema50"] = ta.ema(df["close"], 50)
    df["rsi14"] = ta.rsi(df["close"], 14)
    df["atr14"] = ta.atr(df, 14)
    out = chan.analyze(df)
    sigs = out["signals"]
    # annotate pivot bands into output
    bands = out.get("bands")
    if bands is not None and not bands.empty:
        for col in bands.columns:
            df[col] = bands[col].values
    if not sigs.empty:
        df.loc[sigs["index"], "signal"] = sigs["signal"].values
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        df.to_csv(args.out, index=False)
        print(f"Annotated data saved to {args.out}")
    print(
        f"Fractals: {len(out['fractals'])}, Pens: {len(out['pens'])}, Segments: {len(out['segments'])}, Pivots: {len(out['pivots'])}, Signals: {len(sigs)}"
    )


def cmd_backtest(args: argparse.Namespace):
    cf = CandleFrame.read_csv(args.input)
    df = cf.df.copy()
    # time slicing
    if args.start:
        start = pd.Timestamp(args.start)
        df = df[df["datetime"] >= start]
    if args.end:
        end = pd.Timestamp(args.end)
        df = df[df["datetime"] <= end]
    df = df.reset_index(drop=True)

    out = chan.analyze(df)
    res = simple_execute(df, out["signals"], fee_rate=args.fee)
    print("Backtest stats:")
    for k, v in res.stats.items():
        print(f"- {k}: {v}")


def cmd_mtf(args: argparse.Namespace):
    # load lower/higher
    lcf = CandleFrame.read_csv(args.lower_input)
    hcf = CandleFrame.read_csv(args.higher_input)
    ldf = lcf.df.copy().reset_index(drop=True)
    hdf = hcf.df.copy().reset_index(drop=True)

    # analyze both frames
    lo = chan.analyze(ldf)
    ho = chan.analyze(hdf)

    # align HTF context onto LTF
    htf_ctx = mtf.align_htf_to_ltf(ldf, hdf, ho["segments"], ho["bands"])
    # add columns to lower df
    for col in htf_ctx.columns:
        ldf[col] = htf_ctx[col].values

    # base signals from LTF
    base_sigs = lo["signals"]
    filt_sigs = mtf.filter_signals_with_htf(base_sigs, ldf)

    # annotate both sets of signals into df for inspection
    if not base_sigs.empty:
        ldf.loc[base_sigs["index"], "signal_base"] = base_sigs["signal"].values
    if not filt_sigs.empty:
        ldf.loc[filt_sigs["index"], "signal_mtf"] = filt_sigs["signal"].values

    print(
        f"LTF: Fractals {len(lo['fractals'])}, Pens {len(lo['pens'])}, Segments {len(lo['segments'])}, Pivots {len(lo['pivots'])}, Signals {len(base_sigs)}"
    )
    print(
        f"HTF: Fractals {len(ho['fractals'])}, Pens {len(ho['pens'])}, Segments {len(ho['segments'])}, Pivots {len(ho['pivots'])}"
    )
    print(f"MTF-filtered signals: {len(filt_sigs)}")

    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        ldf.to_csv(args.out, index=False)
        print(f"Annotated LTF saved to {args.out}")

    if args.run_backtest:
        res = simple_execute(ldf, filt_sigs, fee_rate=args.fee)
        print("Backtest (MTF) stats:")
        for k, v in res.stats.items():
            print(f"- {k}: {v}")

def build_parser():
    p = argparse.ArgumentParser(description="Trade platform CLI (ccxt + TA + Chan)")
    sub = p.add_subparsers(dest="cmd")

    f = sub.add_parser("fetch", help="Fetch OHLCV via ccxt")
    f.add_argument("--exchange", required=True, help="ccxt exchange id, e.g., binance")
    f.add_argument("--symbol", required=True, help="e.g., BTC/USDT")
    f.add_argument("--timeframe", default="1h")
    f.add_argument("--since", default=None, help="ISO date, e.g., 2023-01-01")
    f.add_argument("--limit", type=int, default=500)
    f.add_argument("--max-bars", type=int, default=None)
    f.add_argument("--output", required=True)
    f.set_defaults(func=cmd_fetch)

    a = sub.add_parser("analyze", help="Run indicators + Chan analysis")
    a.add_argument("--input", required=True)
    a.add_argument("--out", default=None)
    a.set_defaults(func=cmd_analyze)

    b = sub.add_parser("backtest", help="Backtest simplified Chan strategy")
    b.add_argument("--input", required=True)
    b.add_argument("--start", default=None)
    b.add_argument("--end", default=None)
    b.add_argument("--fee", type=float, default=0.0005)
    b.set_defaults(func=cmd_backtest)

    m = sub.add_parser("mtf", help="Multi-timeframe: align HTF context to LTF and optional backtest")
    m.add_argument("--lower-input", required=True, help="Lower timeframe CSV path")
    m.add_argument("--higher-input", required=True, help="Higher timeframe CSV path")
    m.add_argument("--out", default=None, help="Write annotated LTF CSV")
    m.add_argument("--run-backtest", action="store_true", help="Run backtest on filtered LTF signals")
    m.add_argument("--fee", type=float, default=0.0005)
    m.set_defaults(func=cmd_mtf)

    return p


def main():
    parser = build_parser()
    args = parser.parse_args()
    if not hasattr(args, "func"):
        parser.print_help()
        return
    args.func(args)


if __name__ == "__main__":
    main()
