from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class BacktestResult:
    trades: pd.DataFrame
    equity_curve: pd.Series
    stats: dict


def simple_execute(df: pd.DataFrame, signals: pd.DataFrame, fee_rate: float = 0.0005) -> BacktestResult:
    """Execute next-bar at open after signal index. Long-only, flip on sell.

    df: must include columns [open, close]
    signals: columns [index, signal, price]
    """
    df = df.reset_index(drop=True)
    sig = signals.sort_values("index").reset_index(drop=True)
    positions = []
    in_pos = False
    entry_idx = None
    entry_px = None

    for _, s in sig.iterrows():
        idx = int(s["index"]) + 1  # next bar
        if idx >= len(df):
            break
        if s["signal"] == "buy" and not in_pos:
            entry_idx = idx
            entry_px = df.loc[idx, "open"] * (1 + fee_rate)
            in_pos = True
        elif s["signal"] == "sell" and in_pos:
            exit_idx = idx
            exit_px = df.loc[idx, "open"] * (1 - fee_rate)
            ret = (exit_px - entry_px) / entry_px
            positions.append({
                "entry_idx": entry_idx,
                "entry_px": float(entry_px),
                "exit_idx": exit_idx,
                "exit_px": float(exit_px),
                "ret": float(ret),
            })
            in_pos = False
            entry_idx = None
            entry_px = None

    trades = pd.DataFrame(positions)
    equity = (1 + trades["ret"]).cumprod() if not trades.empty else pd.Series([1.0])
    stats = {}
    if not trades.empty:
        stats = {
            "trades": len(trades),
            "win_rate": float((trades["ret"] > 0).mean()),
            "avg_ret": float(trades["ret"].mean()),
            "cum_return": float(equity.iloc[-1] - 1),
            "profit_factor": float(trades.loc[trades.ret>0, "ret"].sum() / abs(trades.loc[trades.ret<0, "ret"].sum())
                                   ) if (trades.ret<0).any() else np.inf,
        }

    return BacktestResult(trades=trades, equity_curve=equity, stats=stats)

