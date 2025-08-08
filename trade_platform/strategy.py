from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from . import indicators as ta


def ensure_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "rsi14" not in out.columns:
        out["rsi14"] = ta.rsi(out["close"], 14)
    if "atr14" not in out.columns:
        out["atr14"] = ta.atr(out, 14)
    return out


def filter_signals_by_rsi(
    signals: pd.DataFrame,
    df: pd.DataFrame,
    rsi_min: Optional[float] = None,
    rsi_max: Optional[float] = None,
) -> pd.DataFrame:
    if signals is None or signals.empty:
        return signals
    if "rsi14" not in df.columns:
        raise ValueError("RSI column missing; call ensure_indicators first")
    s = signals.copy().reset_index(drop=True)
    keep = []
    for _, row in s.iterrows():
        idx = int(row["index"])
        val = float(df.at[idx, "rsi14"]) if idx in df.index else None
        if val is None:
            keep.append(False)
            continue
        ok = True
        if rsi_min is not None:
            ok = ok and (val >= rsi_min)
        if rsi_max is not None:
            ok = ok and (val <= rsi_max)
        keep.append(ok)
    return s[pd.Series(keep)].reset_index(drop=True)


def filter_signals_by_atr_pct(
    signals: pd.DataFrame,
    df: pd.DataFrame,
    min_atr_pct: Optional[float] = None,
    max_atr_pct: Optional[float] = None,
) -> pd.DataFrame:
    if signals is None or signals.empty:
        return signals
    if "atr14" not in df.columns:
        raise ValueError("ATR column missing; call ensure_indicators first")
    s = signals.copy().reset_index(drop=True)
    keep = []
    for _, row in s.iterrows():
        idx = int(row["index"])
        atr = float(df.at[idx, "atr14"]) if idx in df.index else None
        close = float(df.at[idx, "close"]) if idx in df.index else None
        if atr is None or close is None or close == 0:
            keep.append(False)
            continue
        pct = atr / close
        ok = True
        if min_atr_pct is not None:
            ok = ok and (pct >= min_atr_pct)
        if max_atr_pct is not None:
            ok = ok and (pct <= max_atr_pct)
        keep.append(ok)
    return s[pd.Series(keep)].reset_index(drop=True)


def apply_signal_filters(
    signals: pd.DataFrame,
    df: pd.DataFrame,
    rsi_min: Optional[float] = None,
    rsi_max: Optional[float] = None,
    min_atr_pct: Optional[float] = None,
    max_atr_pct: Optional[float] = None,
) -> pd.DataFrame:
    if signals is None or signals.empty:
        return signals
    df2 = ensure_indicators(df)
    out = filter_signals_by_rsi(signals, df2, rsi_min=rsi_min, rsi_max=rsi_max)
    out = filter_signals_by_atr_pct(out, df2, min_atr_pct=min_atr_pct, max_atr_pct=max_atr_pct)
    return out

