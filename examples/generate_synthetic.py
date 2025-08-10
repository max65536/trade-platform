from __future__ import annotations

import os
from pathlib import Path
import math
import numpy as np
import pandas as pd

from trade_platform.dataio import CandleFrame


def _make_synth(df_len: int, freq: str, start_ts: str = "2023-01-01", seed: int = 7):
    rng = np.random.default_rng(seed)
    ts = pd.date_range(start_ts, periods=df_len, freq=freq)
    t = np.arange(df_len)
    base = 100 + 0.02 * t  # mild drift
    cyc = 3.0 * np.sin(t / 12.0)  # slow cycle
    noise = rng.normal(0, 0.4, size=df_len)
    close = base + cyc + noise
    open_ = np.r_[close[0], close[:-1]]
    high = np.maximum(open_, close) + np.abs(rng.normal(0.0, 0.5, size=df_len))
    low = np.minimum(open_, close) - np.abs(rng.normal(0.0, 0.5, size=df_len))
    vol = rng.uniform(10, 100, size=df_len)
    df = pd.DataFrame({
        "timestamp": (ts.view('int64') // 1_000_000).astype(np.int64),  # ms
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": vol,
    })
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True).dt.tz_convert(None)
    return df


def main():
    out_dir = Path("examples/data")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1h synthetic
    df_1h = _make_synth(1000, "1h")
    path_1h = out_dir / "BTCUSDT-1h.csv"
    df_1h.to_csv(path_1h, index=False)

    # 4h synthetic
    df_4h = _make_synth(800, "4h")
    path_4h = out_dir / "BTCUSDT-4h.csv"
    df_4h.to_csv(path_4h, index=False)

    # 1d synthetic
    df_1d = _make_synth(400, "1d")
    path_1d = out_dir / "BTCUSDT-1d.csv"
    df_1d.to_csv(path_1d, index=False)

    print("Synthetic CSVs written:")
    print(f"- {path_1h}")
    print(f"- {path_4h}")
    print(f"- {path_1d}")


if __name__ == "__main__":
    main()

