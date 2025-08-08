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

