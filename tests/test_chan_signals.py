import pandas as pd
import numpy as np

from trade_platform.chan import pivot_retest_signals, divergence_signals, Segment


def test_pivot_retest_buy2_and_sell2():
    # Construct a small df with constant pivot bands and a price path
    n = 12
    ph = 105.0  # pivot_high
    pl = 95.0   # pivot_low

    # Sequence:
    # 0-2 below ph; 3 breakout above ph; 4 dip back below; 5 reclaim above -> buy2 at 5
    # 6-7 above pl; 8 breakdown below pl; 9 rally above; 10 lose it again -> sell2 at 10
    close = [100, 102, 104, 106, 104, 106, 100, 99, 94, 96, 94, 96]
    df = pd.DataFrame({
        "close": close,
        "pivot_high": [ph] * n,
        "pivot_low": [pl] * n,
    })

    sigs = pivot_retest_signals(df)
    # Expect two signals: buy2 at index 5 and sell2 at index 10
    assert not sigs.empty
    kinds = list(sigs.get("kind", []))
    # Ensure kinds exist and contain expected entries
    assert "buy2" in kinds and "sell2" in kinds
    # Validate positions
    idx_by_kind = {row["kind"]: int(row["index"]) for _, row in sigs.iterrows()}
    assert idx_by_kind["buy2"] == 5
    assert idx_by_kind["sell2"] == 10


def test_divergence_signals_basic_buy3_and_sell3():
    # Build a minimal price series; MACD histogram is not constrained because we set a very
    # permissive threshold (min_hist_delta very negative). We focus on price extension logic.
    close = [10, 9, 8, 9, 7, 8, 9, 10, 11, 10, 12, 13]
    df = pd.DataFrame({"close": close})

    # Construct segments: down1 -> up -> down2 should yield buy3 at end of down2
    segs = [
        Segment(start_idx=0, end_idx=2, direction="down", start_price=close[0], end_price=close[2]),
        Segment(start_idx=2, end_idx=3, direction="up", start_price=close[2], end_price=close[3]),
        Segment(start_idx=3, end_idx=4, direction="down", start_price=close[3], end_price=close[4]),
        # Then up1 -> down -> up2 should yield sell3 at end of up2
        Segment(start_idx=4, end_idx=6, direction="up", start_price=close[4], end_price=close[6]),
        Segment(start_idx=6, end_idx=7, direction="down", start_price=close[6], end_price=close[7]),
        Segment(start_idx=7, end_idx=11, direction="up", start_price=close[7], end_price=close[11]),
    ]

    sigs = divergence_signals(
        df,
        segs,
        min_price_ext_pct=0.0,
        min_hist_delta=-1e6,  # extremely permissive to avoid dependency on MACD sign
        require_hist_sign_consistency=False,
    )
    assert not sigs.empty
    kinds = set(sigs.get("kind", []))
    assert "buy3" in kinds and "sell3" in kinds
    # Expected indices: end of second down (4) and end of second up (11)
    idxs = {int(r["index"]): r["kind"] for _, r in sigs.iterrows()}
    assert idxs.get(4) == "buy3"
    assert idxs.get(11) == "sell3"

