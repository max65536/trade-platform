from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, List

import numpy as np
import pandas as pd


@dataclass
class BacktestResult:
    trades: pd.DataFrame
    equity_curve: pd.Series  # aligned to input df rows (forward-filled)
    stats: dict


def _build_equity_curve(
    n_bars: int,
    trades: pd.DataFrame,
    initial_capital: float = 1.0,
    position_size: float = 1.0,
) -> pd.Series:
    """Construct equity curve aligned to bars, updating at each trade exit.

    position_size is the fraction of equity allocated to each trade (0..1),
    applied multiplicatively on trade returns.
    """
    if n_bars <= 0:
        return pd.Series([], dtype=float)
    eq = np.full(n_bars, float(initial_capital), dtype=float)
    if trades is None or trades.empty:
        return pd.Series(eq)
    # Update equity at each trade exit index
    for _, row in trades.iterrows():
        exit_idx = int(row["exit_idx"]) if not pd.isna(row.get("exit_idx", np.nan)) else None
        if exit_idx is None or exit_idx >= n_bars:
            continue
        ret = float(row["ret"]) if "ret" in row else 0.0
        # equity changes only at exit; forward-filled afterwards
        prev = eq[exit_idx - 1] if exit_idx > 0 else eq[0]
        eq[exit_idx] = prev * (1.0 + position_size * ret)
        # forward fill next bars until next update
        if exit_idx + 1 < n_bars:
            eq[exit_idx + 1 :] = eq[exit_idx]
    # Ensure prefix prior to first exit is initialized
    for i in range(1, n_bars):
        if eq[i] == 0.0:
            eq[i] = eq[i - 1]
    return pd.Series(eq)


def _compute_drawdown_stats(equity: pd.Series) -> Dict[str, float]:
    if equity is None or len(equity) == 0:
        return {"max_drawdown": 0.0}
    peaks = equity.cummax()
    dd = (equity / peaks) - 1.0
    max_dd = float(dd.min())
    # duration (bars) for max drawdown
    end_idx = int(dd.idxmin()) if not dd.empty else 0
    start_idx = int((equity[: end_idx + 1]).idxmax()) if end_idx >= 0 else 0
    return {
        "max_drawdown": max_dd,
        "max_dd_start_idx": start_idx,
        "max_dd_end_idx": end_idx,
    }


def _basic_stats(
    trades: pd.DataFrame,
    equity: pd.Series,
    exposure_bars: int,
) -> Dict[str, float]:
    stats: Dict[str, float] = {}
    if trades is None or trades.empty:
        stats.update({
            "trades": 0,
            "win_rate": 0.0,
            "avg_ret": 0.0,
            "cum_return": 0.0,
            "profit_factor": np.inf,
            "exposure_bars": exposure_bars,
            "exposure_pct": float(exposure_bars / max(len(equity), 1)) if equity is not None and len(equity) > 0 else 0.0,
        })
        stats.update(_compute_drawdown_stats(equity if equity is not None else pd.Series([1.0])))
        return stats

    wins = trades.loc[trades["ret"] > 0, "ret"]
    losses = trades.loc[trades["ret"] < 0, "ret"]
    gross_profit = float(wins.sum()) if not wins.empty else 0.0
    gross_loss = float(losses.sum()) if not losses.empty else 0.0
    profit_factor = (gross_profit / abs(gross_loss)) if gross_loss != 0 else np.inf

    # holding periods
    if "entry_idx" in trades.columns and "exit_idx" in trades.columns:
        holding_bars = (trades["exit_idx"] - trades["entry_idx"]).astype(int)
        avg_hold_bars = float(holding_bars.mean()) if not holding_bars.empty else 0.0
    else:
        avg_hold_bars = 0.0

    stats.update({
        "trades": int(len(trades)),
        "win_rate": float((trades["ret"] > 0).mean()),
        "avg_ret": float(trades["ret"].mean()),
        "cum_return": float((equity.iloc[-1] / equity.iloc[0]) - 1.0) if equity is not None and len(equity) > 0 else float((1 + trades["ret"]).prod() - 1),
        "profit_factor": float(profit_factor),
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
        "best_trade": float(trades["ret"].max()),
        "worst_trade": float(trades["ret"].min()),
        "avg_hold_bars": avg_hold_bars,
        "exposure_bars": exposure_bars,
        "exposure_pct": float(exposure_bars / max(len(equity), 1)) if equity is not None and len(equity) > 0 else 0.0,
    })
    stats.update(_compute_drawdown_stats(equity))
    return stats


def simple_execute(
    df: pd.DataFrame,
    signals: pd.DataFrame,
    fee_rate: float = 0.0005,
    *,
    initial_capital: float = 1.0,
    position_size: float = 1.0,
    slippage_bps: float = 0.0,
) -> BacktestResult:
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
    slip = slippage_bps / 10000.0 if slippage_bps else 0.0

    for _, s in sig.iterrows():
        idx = int(s["index"]) + 1  # next bar
        if idx >= len(df):
            break
        if s["signal"] == "buy" and not in_pos:
            entry_idx = idx
            entry_px = df.loc[idx, "open"] * (1 + fee_rate + slip)
            in_pos = True
        elif s["signal"] == "sell" and in_pos:
            exit_idx = idx
            exit_px = df.loc[idx, "open"] * (1 - fee_rate - slip)
            ret = (exit_px - entry_px) / entry_px
            positions.append({
                "entry_idx": entry_idx,
                "entry_px": float(entry_px),
                "exit_idx": exit_idx,
                "exit_px": float(exit_px),
                "ret": float(ret),
                "size": float(position_size),
                "side": "long",
            })
            in_pos = False
            entry_idx = None
            entry_px = None

    trades = pd.DataFrame(positions)
    equity = _build_equity_curve(len(df), trades, initial_capital=initial_capital, position_size=position_size)
    # exposure: time in market (bars)
    exposure_bars = 0
    if not trades.empty:
        exposure_bars = int((trades["exit_idx"] - trades["entry_idx"]).clip(lower=0).sum())
    stats = _basic_stats(trades, equity, exposure_bars)
    return BacktestResult(trades=trades, equity_curve=equity, stats=stats)


def execute_with_risk(
    df: pd.DataFrame,
    signals: pd.DataFrame,
    fee_rate: float = 0.0005,
    stop_loss_pct: Optional[float] = None,
    take_profit_pct: Optional[float] = None,
    *,
    initial_capital: float = 1.0,
    position_size: float = 1.0,
    slippage_bps: float = 0.0,
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
    slip = slippage_bps / 10000.0 if slippage_bps else 0.0

    i = 0
    while i < n:
        # Check TP/SL on current bar if in position (including entry bar)
        if in_pos and entry_exec_idx is not None and i >= entry_exec_idx:
            bar_low = float(df.loc[i, "low"]) if "low" in df.columns else float(df.loc[i, "open"])  # fallback
            bar_high = float(df.loc[i, "high"]) if "high" in df.columns else float(df.loc[i, "open"])  # fallback
            # TP first
            if tp_px is not None and bar_high >= tp_px:
                exit_px = tp_px * (1 - fee_rate - slip)
                trades.append({
                    "entry_idx": entry_exec_idx,
                    "entry_px": float(entry_px),
                    "exit_idx": i,
                    "exit_px": float(exit_px),
                    "ret": float((exit_px - entry_px) / entry_px),
                    "size": float(position_size),
                    "side": "long",
                })
                in_pos = False
                entry_exec_idx = None
                entry_px = None
                stop_px = None
                tp_px = None
                i += 1
                continue
            if stop_px is not None and bar_low <= stop_px:
                exit_px = stop_px * (1 - fee_rate - slip)
                trades.append({
                    "entry_idx": entry_exec_idx,
                    "entry_px": float(entry_px),
                    "exit_idx": i,
                    "exit_px": float(exit_px),
                    "ret": float((exit_px - entry_px) / entry_px),
                    "size": float(position_size),
                    "side": "long",
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
                entry_px = float(df.loc[entry_exec_idx, "open"]) * (1 + fee_rate + slip)
                stop_px = entry_px * (1 - stop_loss_pct) if stop_loss_pct is not None else None
                tp_px = entry_px * (1 + take_profit_pct) if take_profit_pct is not None else None
                in_pos = True
        elif in_pos and "sell" in sigs_here:
            # Exit at next-bar open if exists and no TP/SL already hit
            if i + 1 < n:
                exit_idx = i + 1
                exit_px = float(df.loc[exit_idx, "open"]) * (1 - fee_rate - slip)
                trades.append({
                    "entry_idx": entry_exec_idx,
                    "entry_px": float(entry_px),
                    "exit_idx": exit_idx,
                    "exit_px": float(exit_px),
                    "ret": float((exit_px - entry_px) / entry_px),
                    "size": float(position_size),
                    "side": "long",
                })
                in_pos = False
                entry_exec_idx = None
                entry_px = None
                stop_px = None
                tp_px = None

        i += 1

    trades = pd.DataFrame(trades)
    equity = _build_equity_curve(len(df), trades, initial_capital=initial_capital, position_size=position_size)
    exposure_bars = 0
    if not trades.empty:
        exposure_bars = int((trades["exit_idx"] - trades["entry_idx"]).clip(lower=0).sum())
    stats = _basic_stats(trades, equity, exposure_bars)
    return BacktestResult(trades=trades, equity_curve=equity, stats=stats)
