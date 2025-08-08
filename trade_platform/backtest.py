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


def execute_with_risk(
    df: pd.DataFrame,
    signals: pd.DataFrame,
    fee_rate: float = 0.0005,
    stop_loss_pct: Optional[float] = None,
    take_profit_pct: Optional[float] = None,
) -> BacktestResult:
    """Bar-by-bar executor with next-bar entries, TP/SL intrabar checks, and opposite-signal exits.

    Execution rules:
    - Enter long at next bar open when encountering a buy signal.
    - While in position, on each subsequent bar check TP (first) then SL against high/low.
    - If no TP/SL hit and a sell signal occurs, exit at the next bar open.
    """
    df = df.reset_index(drop=True)
    n = len(df)
    # Map signals by bar index
    sig_map = {}
    for _, s in signals.sort_values("index").iterrows():
        i = int(s["index"])
        if i not in sig_map:
            sig_map[i] = []
        sig_map[i].append(s["signal"])

    trades = []
    in_pos = False
    entry_exec_idx: Optional[int] = None
    entry_px: Optional[float] = None
    stop_px: Optional[float] = None
    tp_px: Optional[float] = None

    i = 0
    while i < n:
        # Check TP/SL on current bar if in position (including entry bar)
        if in_pos and entry_exec_idx is not None and i >= entry_exec_idx:
            bar_low = float(df.loc[i, "low"]) if "low" in df.columns else float(df.loc[i, "open"])  # fallback
            bar_high = float(df.loc[i, "high"]) if "high" in df.columns else float(df.loc[i, "open"])  # fallback
            # TP first
            if tp_px is not None and bar_high >= tp_px:
                exit_px = tp_px * (1 - fee_rate)
                trades.append({
                    "entry_idx": entry_exec_idx,
                    "entry_px": float(entry_px),
                    "exit_idx": i,
                    "exit_px": float(exit_px),
                    "ret": float((exit_px - entry_px) / entry_px),
                })
                in_pos = False
                entry_exec_idx = None
                entry_px = None
                stop_px = None
                tp_px = None
                i += 1
                continue
            if stop_px is not None and bar_low <= stop_px:
                exit_px = stop_px * (1 - fee_rate)
                trades.append({
                    "entry_idx": entry_exec_idx,
                    "entry_px": float(entry_px),
                    "exit_idx": i,
                    "exit_px": float(exit_px),
                    "ret": float((exit_px - entry_px) / entry_px),
                })
                in_pos = False
                entry_exec_idx = None
                entry_px = None
                stop_px = None
                tp_px = None
                i += 1
                continue

        # Process signals on this bar
        sigs_here = sig_map.get(i, [])
        if not in_pos and "buy" in sigs_here:
            if i + 1 < n:
                entry_exec_idx = i + 1
                entry_px = float(df.loc[entry_exec_idx, "open"]) * (1 + fee_rate)
                stop_px = entry_px * (1 - stop_loss_pct) if stop_loss_pct is not None else None
                tp_px = entry_px * (1 + take_profit_pct) if take_profit_pct is not None else None
                in_pos = True
        elif in_pos and "sell" in sigs_here:
            # Exit at next-bar open if exists and no TP/SL already hit
            if i + 1 < n:
                exit_idx = i + 1
                exit_px = float(df.loc[exit_idx, "open"]) * (1 - fee_rate)
                trades.append({
                    "entry_idx": entry_exec_idx,
                    "entry_px": float(entry_px),
                    "exit_idx": exit_idx,
                    "exit_px": float(exit_px),
                    "ret": float((exit_px - entry_px) / entry_px),
                })
                in_pos = False
                entry_exec_idx = None
                entry_px = None
                stop_px = None
                tp_px = None

        i += 1

    trades = pd.DataFrame(trades)
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
