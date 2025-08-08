from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd

from .chan import Segment, PivotZone


def segment_direction_series(n_rows: int, segments: List[Segment]) -> pd.Series:
    """Create a per-index direction series: 1 for up, -1 for down, 0 otherwise."""
    arr = np.zeros(n_rows, dtype=int)
    for seg in segments:
        a = max(0, seg.start_idx)
        b = max(a, min(n_rows - 1, seg.end_idx))
        val = 1 if seg.direction == "up" else -1
        arr[a : b + 1] = val
    return pd.Series(arr)


def align_htf_to_ltf(
    ltf_df: pd.DataFrame,
    htf_df: pd.DataFrame,
    htf_segments: List[Segment],
    htf_bands: pd.DataFrame,
) -> pd.DataFrame:
    """Align higher-timeframe (HTF) context to lower-timeframe (LTF) rows using merge_asof.

    Returns a DataFrame with columns: htf_dir, htf_pivot_low, htf_pivot_high aligned on datetime.
    """
    if "datetime" not in ltf_df.columns or "datetime" not in htf_df.columns:
        raise ValueError("Both dataframes must include 'datetime' column")

    htf_dir = segment_direction_series(len(htf_df), htf_segments)
    htf_ctx = pd.concat([
        htf_df[["datetime"]].reset_index(drop=True),
        pd.DataFrame({
            "htf_dir": htf_dir,
            "htf_pivot_low": htf_bands.get("pivot_low", pd.Series([np.nan] * len(htf_df))),
            "htf_pivot_high": htf_bands.get("pivot_high", pd.Series([np.nan] * len(htf_df))),
        }).reset_index(drop=True),
    ], axis=1).sort_values("datetime")

    ltf_sorted = ltf_df[["datetime"]].copy()
    ltf_sorted["__orig_idx__"] = ltf_sorted.index
    ltf_sorted = ltf_sorted.sort_values("datetime")
    merged = pd.merge_asof(
        ltf_sorted,
        htf_ctx,
        on="datetime",
        direction="backward",
    )
    # restore original order
    merged = merged.sort_values("__orig_idx__").set_index("__orig_idx__")
    merged = merged.reindex(ltf_df.index)
    return merged[["htf_dir", "htf_pivot_low", "htf_pivot_high"]]


def filter_signals_with_htf(signals: pd.DataFrame, ltf_with_htf: pd.DataFrame) -> pd.DataFrame:
    """Filter LTF signals using HTF direction, optional pivot breakout, and min run length.

    Parameters (via ltf_with_htf columns):
      - htf_dir: 1 for up, -1 for down, 0/NaN otherwise
      - htf_pivot_low/high: optional pivot band reference from HTF

    This is a thin wrapper; see `filter_signals_with_htf_opts` for options.
    """
    return filter_signals_with_htf_opts(signals, ltf_with_htf)


def _compute_run_length(htf_dir: pd.Series) -> pd.Series:
    """Compute run length of consecutive non-zero direction per index."""
    arr = htf_dir.fillna(0).astype(int).values
    run = np.zeros_like(arr)
    curr = 0
    last_sign = 0
    for i, v in enumerate(arr):
        sign = int(np.sign(v))
        if sign == 0:
            curr = 0
            last_sign = 0
            run[i] = 0
        else:
            if sign == last_sign:
                curr += 1
            else:
                curr = 1
                last_sign = sign
            run[i] = curr
    return pd.Series(run, index=htf_dir.index)


def filter_signals_with_htf_opts(
    signals: pd.DataFrame,
    ltf_with_htf: pd.DataFrame,
    require_htf_breakout: bool = False,
    min_htf_run: int = 0,
) -> pd.DataFrame:
    """Filter signals by HTF direction, optional HTF pivot breakout, and minimum run length.

    - keep buy if: htf_dir>0 and (if require_breakout: close>htf_pivot_high) and run_len>=min_htf_run
    - keep sell if: htf_dir<0 and (if require_breakout: close<htf_pivot_low) and run_len>=min_htf_run
    """
    if signals is None or signals.empty:
        return signals

    df = ltf_with_htf.copy()
    if "close" not in df.columns:
        raise ValueError("ltf_with_htf must include 'close' column for breakout filtering")

    run_len = _compute_run_length(df["htf_dir"]) if min_htf_run > 0 else None

    sigs = signals.copy().reset_index(drop=True)
    keep = []
    for _, row in sigs.iterrows():
        idx = int(row["index"])
        if idx not in df.index:
            keep.append(False)
            continue
        hdir = int(df.at[idx, "htf_dir"]) if not pd.isna(df.at[idx, "htf_dir"]) else 0
        ok_dir = (row["signal"] == "buy" and hdir > 0) or (row["signal"] == "sell" and hdir < 0)
        if not ok_dir:
            keep.append(False)
            continue
        ok_break = True
        if require_htf_breakout:
            c = float(df.at[idx, "close"]) if not pd.isna(df.at[idx, "close"]) else np.nan
            if row["signal"] == "buy":
                h = df.at[idx, "htf_pivot_high"] if "htf_pivot_high" in df.columns else np.nan
                ok_break = pd.notna(h) and c > float(h)
            else:
                l = df.at[idx, "htf_pivot_low"] if "htf_pivot_low" in df.columns else np.nan
                ok_break = pd.notna(l) and c < float(l)
        if not ok_break:
            keep.append(False)
            continue
        ok_run = True
        if min_htf_run > 0:
            r = int(run_len.at[idx]) if not pd.isna(run_len.at[idx]) else 0
            ok_run = r >= min_htf_run
        keep.append(ok_run)

    return sigs[pd.Series(keep)].reset_index(drop=True)
