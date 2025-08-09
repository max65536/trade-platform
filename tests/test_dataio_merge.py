import pandas as pd

from trade_platform.dataio import merge_ohlcv_frames


def test_merge_ohlcv_frames_dedup_and_sort():
    old = pd.DataFrame({
        "timestamp": [1000, 2000, 3000],
        "open": [1, 2, 3],
        "high": [1, 2, 3],
        "low": [1, 2, 3],
        "close": [1, 2, 3],
        "volume": [1, 1, 1],
    })
    new = pd.DataFrame({
        "timestamp": [3000, 4000, 5000],  # overlap at 3000
        "open": [33, 4, 5],
        "high": [33, 4, 5],
        "low": [33, 4, 5],
        "close": [33, 4, 5],
        "volume": [1, 1, 1],
    })

    merged = merge_ohlcv_frames(old, new)
    # Expect timestamps: 1000,2000,3000,4000,5000 -> len 5
    assert list(merged["timestamp"]) == [1000, 2000, 3000, 4000, 5000]
    # Overlap keeps new row for 3000 (close=33)
    assert merged.loc[merged["timestamp"] == 3000, "close"].iloc[0] == 33

