from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Optional

import pandas as pd


FractalType = Literal["top", "bottom"]


@dataclass
class Fractal:
    index: int
    kind: FractalType
    price: float


@dataclass
class Pen:
    start: Fractal
    end: Fractal
    direction: Literal["up", "down"]


@dataclass
class Segment:
    start_idx: int
    end_idx: int
    direction: Literal["up", "down"]
    start_price: float
    end_price: float


@dataclass
class PivotZone:
    """Zhongshu (中枢) simplified representation as overlapping price range.

    Built from consecutive pens whose price ranges overlap for >= min_pens.
    """
    start_idx: int
    end_idx: int
    low: float
    high: float
    pens: int


def find_fractals(df: pd.DataFrame, left: int = 2, right: int = 2) -> List[Fractal]:
    """Simplified fractal detection: high/low vs neighbors.

    A 'top' if high[i] is the maximum in [i-left, i+right].
    A 'bottom' if low[i] is the minimum in [i-left, i+right].
    """
    highs = df["high"].values
    lows = df["low"].values
    n = len(df)
    frs: List[Fractal] = []
    for i in range(left, n - right):
        window_h = highs[i - left : i + right + 1]
        window_l = lows[i - left : i + right + 1]
        if highs[i] == window_h.max() and (window_h == highs[i]).sum() == 1:
            frs.append(Fractal(index=i, kind="top", price=highs[i]))
        if lows[i] == window_l.min() and (window_l == lows[i]).sum() == 1:
            frs.append(Fractal(index=i, kind="bottom", price=lows[i]))
    frs.sort(key=lambda f: f.index)
    return frs


def build_pens(
    fractals: List[Fractal],
    min_separation: int = 3,
    min_price_move: float = 0.002,
) -> List[Pen]:
    """Connect alternating fractals to form 'pens' (简化笔识别)。

    - 邻近同类分型会被弱化（择更极端者）
    - 需要满足最小间隔和最小百分比波动
    """
    if not fractals:
        return []

    # compress adjacent same-type fractals by extremity
    compressed: List[Fractal] = []
    for f in fractals:
        if not compressed:
            compressed.append(f)
            continue
        last = compressed[-1]
        if f.kind == last.kind:
            if f.kind == "top":
                if f.price >= last.price:
                    compressed[-1] = f
            else:  # bottom
                if f.price <= last.price:
                    compressed[-1] = f
        else:
            compressed.append(f)

    pens: List[Pen] = []
    for a, b in zip(compressed, compressed[1:]):
        if b.index - a.index < min_separation:
            continue
        pct = abs(b.price - a.price) / a.price
        if pct < min_price_move:
            continue
        direction = "up" if b.price > a.price else "down"
        pens.append(Pen(start=a, end=b, direction=direction))
    return pens


def build_segments(pens: List[Pen], min_pens: int = 3) -> List[Segment]:
    """Group pens into segments when direction persists for >= min_pens.

    线段由至少 min_pens 根同向笔组成（简化）。
    """
    if not pens:
        return []
    segments: List[Segment] = []
    run: List[Pen] = [pens[0]]
    for p in pens[1:]:
        if p.direction == run[-1].direction:
            run.append(p)
        else:
            if len(run) >= min_pens:
                start = run[0].start
                end = run[-1].end
                segments.append(Segment(
                    start_idx=start.index,
                    end_idx=end.index,
                    direction=run[0].direction,
                    start_price=start.price,
                    end_price=end.price,
                ))
            run = [p]
    # flush last run
    if len(run) >= min_pens:
        start = run[0].start
        end = run[-1].end
        segments.append(Segment(
            start_idx=start.index,
            end_idx=end.index,
            direction=run[0].direction,
            start_price=start.price,
            end_price=end.price,
        ))
    return segments


def build_pivots(pens: List[Pen], min_pens: int = 3, eps: float = 1e-9) -> List[PivotZone]:
    """Detect Zhongshu (中枢) as the running intersection of pen price ranges.

    Logic (simplified, engineering-friendly):
    - Represent each pen as [low, high] based on its endpoints.
    - Maintain running intersection with consecutive pens; when intersection exists
      across >= min_pens pens, a PivotZone is active.
    - Extend the active zone while new pens still intersect the current [low, high].
    - When intersection breaks, finalize the zone if it met min_pens; start a new candidate.
    """
    if not pens:
        return []

    pivots: List[PivotZone] = []
    curr_low: Optional[float] = None
    curr_high: Optional[float] = None
    count = 0
    start_idx: Optional[int] = None
    last_idx: Optional[int] = None

    for p in pens:
        lo = min(p.start.price, p.end.price)
        hi = max(p.start.price, p.end.price)
        if curr_low is None:
            curr_low, curr_high = lo, hi
            count = 1
            start_idx = p.start.index
            last_idx = p.end.index
            continue

        # update running intersection
        new_low = max(curr_low, lo)
        new_high = min(curr_high, hi)
        if new_low <= new_high + eps:
            curr_low, curr_high = new_low, new_high
            count += 1
            last_idx = p.end.index
        else:
            # finalize previous zone if valid
            if count >= min_pens and curr_high - curr_low > eps and start_idx is not None and last_idx is not None:
                pivots.append(PivotZone(start_idx=start_idx, end_idx=last_idx, low=float(curr_low), high=float(curr_high), pens=count))
            # reset with current pen
            curr_low, curr_high = lo, hi
            count = 1
            start_idx = p.start.index
            last_idx = p.end.index

    # flush tail
    if count >= min_pens and curr_high is not None and curr_low is not None and start_idx is not None and last_idx is not None:
        if curr_high - curr_low > eps:
            pivots.append(PivotZone(start_idx=start_idx, end_idx=last_idx, low=float(curr_low), high=float(curr_high), pens=count))

    return pivots


def segment_signals(segments: List[Segment]) -> pd.DataFrame:
    """Generate entry/exit signals at segment turns (简化示例)."""
    rows = []
    for i in range(1, len(segments)):
        prev = segments[i - 1]
        curr = segments[i]
        if prev.direction == "down" and curr.direction == "up":
            rows.append({
                "index": curr.start_idx,
                "signal": "buy",
                "price": curr.start_price,
            })
        elif prev.direction == "up" and curr.direction == "down":
            rows.append({
                "index": curr.start_idx,
                "signal": "sell",
                "price": curr.start_price,
            })
    return pd.DataFrame(rows)


def annotate_pivot_bands(n_rows: int, pivots: List[PivotZone]) -> pd.DataFrame:
    """Create per-index pivot bands as two columns: pivot_low/pivot_high.

    Only filled within each pivot's [start_idx, end_idx] inclusive.
    """
    import numpy as np

    low = np.full(n_rows, np.nan, dtype=float)
    high = np.full(n_rows, np.nan, dtype=float)
    for z in pivots:
        a = max(0, z.start_idx)
        b = min(n_rows - 1, z.end_idx)
        if a <= b:
            low[a : b + 1] = z.low
            high[a : b + 1] = z.high
    return pd.DataFrame({"pivot_low": low, "pivot_high": high})


def pivot_breakout_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Generate signals when close crosses pivot bands.

    Assumes df contains columns: close, pivot_low, pivot_high.
    - Cross above pivot_high -> buy
    - Cross below pivot_low -> sell
    Signals are placed at the bar of the cross; execution happens next bar in backtest.
    """
    rows: List[dict] = []
    close = df["close"].values
    pl = df["pivot_low"].values
    ph = df["pivot_high"].values
    n = len(df)
    for i in range(1, n):
        # breakout up
        if pd.notna(ph[i]) or pd.notna(ph[i - 1]):
            h_prev = ph[i - 1] if pd.notna(ph[i - 1]) else ph[i]
            if pd.notna(h_prev) and close[i - 1] <= h_prev and close[i] > h_prev:
                rows.append({"index": i, "signal": "buy", "price": float(close[i])})
        # breakout down
        if pd.notna(pl[i]) or pd.notna(pl[i - 1]):
            l_prev = pl[i - 1] if pd.notna(pl[i - 1]) else pl[i]
            if pd.notna(l_prev) and close[i - 1] >= l_prev and close[i] < l_prev:
                rows.append({"index": i, "signal": "sell", "price": float(close[i])})
    if not rows:
        return pd.DataFrame(columns=["index", "signal", "price"])
    return pd.DataFrame(rows).drop_duplicates(subset=["index", "signal"]).sort_values("index").reset_index(drop=True)


def analyze(df: pd.DataFrame):
    """Convenience pipeline: fractals -> pens -> segments -> pivots -> signals."""
    fr = find_fractals(df)
    pens = build_pens(fr)
    segs = build_segments(pens)
    pivots = build_pivots(pens)
    # annotate pivot bands for downstream usage
    bands = annotate_pivot_bands(len(df), pivots)
    # merge segment turn signals and pivot breakout signals
    seg_sigs = segment_signals(segs)
    df_local = df.reset_index(drop=True).copy()
    df_local = pd.concat([df_local, bands], axis=1)
    piv_sigs = pivot_breakout_signals(df_local)
    sigs = pd.concat([seg_sigs, piv_sigs], ignore_index=True)
    if not sigs.empty:
        sigs = sigs.sort_values(["index", "signal"]).drop_duplicates(subset=["index"], keep="first").reset_index(drop=True)
    return {
        "fractals": fr,
        "pens": pens,
        "segments": segs,
        "pivots": pivots,
        "signals": sigs,
        "bands": bands,
    }
