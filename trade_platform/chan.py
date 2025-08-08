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


def analyze(df: pd.DataFrame):
    """Convenience pipeline: fractals -> pens -> segments -> signals."""
    fr = find_fractals(df)
    pens = build_pens(fr)
    segs = build_segments(pens)
    sigs = segment_signals(segs)
    return {
        "fractals": fr,
        "pens": pens,
        "segments": segs,
        "signals": sigs,
    }

