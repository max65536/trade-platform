from __future__ import annotations

from dataclasses import dataclass
from typing import List

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
    """Filter LTF signals using HTF direction alignment.

    - keep buy only if htf_dir > 0
    - keep sell only if htf_dir < 0
    """
    if signals is None or signals.empty:
        return signals
    sigs = signals.copy().reset_index(drop=True)
    keep = []
    for i, row in sigs.iterrows():
        idx = int(row["index"])
        if idx not in ltf_with_htf.index:
            continue
        hdir = int(ltf_with_htf.loc[idx, "htf_dir"]) if not pd.isna(ltf_with_htf.loc[idx, "htf_dir"]) else 0
        if row["signal"] == "buy" and hdir > 0:
            keep.append(True)
        elif row["signal"] == "sell" and hdir < 0:
            keep.append(True)
        else:
            keep.append(False)
    return sigs[pd.Series(keep)].reset_index(drop=True)
