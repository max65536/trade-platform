from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd


DEFAULT_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]


@dataclass
class CandleFrame:
    df: pd.DataFrame

    @classmethod
    def from_ohlcv(cls, ohlcv: list[list[float]]):
        df = pd.DataFrame(ohlcv, columns=DEFAULT_COLUMNS)
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True).dt.tz_convert(None)
        return cls(df)

    @classmethod
    def read_csv(cls, path: str):
        df = pd.read_csv(path)
        # best-effort normalize
        if "datetime" not in df.columns and "timestamp" in df.columns:
            df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True).dt.tz_convert(None)
        return cls(df)

    def to_csv(self, path: str):
        self.df.to_csv(path, index=False)


def merge_ohlcv_frames(df_old: pd.DataFrame, df_new: pd.DataFrame, *, key: str = "timestamp") -> pd.DataFrame:
    """Merge two OHLCV DataFrames and de-duplicate by the key column (default: timestamp).

    - Keeps the last occurrence on duplicate keys (favor newer fetch).
    - Sorts ascending by key.
    - Returns a new DataFrame.
    """
    if df_old is None or df_old.empty:
        out = df_new.copy().reset_index(drop=True)
    elif df_new is None or df_new.empty:
        out = df_old.copy().reset_index(drop=True)
    else:
        out = pd.concat([df_old, df_new], ignore_index=True)
        out = out.drop_duplicates(subset=[key], keep="last").sort_values(key).reset_index(drop=True)
    return out


def write_csv_merged(path: str, df_new: pd.DataFrame, *, key: str = "timestamp") -> int:
    """Append-deduplicate-save helper. Returns total rows written.

    If path exists, merges with existing content and de-duplicates by key.
    Otherwise writes df_new as-is.
    """
    try:
        existing = pd.read_csv(path)
    except Exception:
        existing = pd.DataFrame(columns=df_new.columns)
    merged = merge_ohlcv_frames(existing, df_new, key=key)
    merged.to_csv(path, index=False)
    return len(merged)
