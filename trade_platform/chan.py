from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Optional, Tuple

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
                rows.append({"index": i, "signal": "buy", "price": float(close[i]), "kind": "buy1"})
        # breakout down
        if pd.notna(pl[i]) or pd.notna(pl[i - 1]):
            l_prev = pl[i - 1] if pd.notna(pl[i - 1]) else pl[i]
            if pd.notna(l_prev) and close[i - 1] >= l_prev and close[i] < l_prev:
                rows.append({"index": i, "signal": "sell", "price": float(close[i]), "kind": "sell1"})
    if not rows:
        return pd.DataFrame(columns=["index", "signal", "price"])
    return pd.DataFrame(rows).drop_duplicates(subset=["index", "signal"]).sort_values("index").reset_index(drop=True)


def analyze(df: pd.DataFrame):
    """Convenience pipeline: fractals -> pens -> segments -> pivots -> signals.

    Note: This is the simplified pipeline retained for backward-compatibility.
    For a fuller Chan workflow with K-line inclusion (包含关系) and extended signals,
    call analyze_full(df) or analyze(df, mode="full").
    """
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


# =============================
# Full Chan workflow (工程化完整版)
# =============================

@dataclass
class MergedBar:
    index: int
    high: float
    low: float
    close: float


def _bars_contained(h1: float, l1: float, h2: float, l2: float) -> bool:
    """Return True if bar2 is contained by bar1 or vice-versa (包含关系)."""
    return (h2 <= h1 and l2 >= l1) or (h2 >= h1 and l2 <= l1)


def merge_kbars_inclusion(df: pd.DataFrame) -> List[MergedBar]:
    """Merge K-lines by Chan inclusion rules (包含关系合并).

    Heuristic: maintain current trend direction; when containment occurs, collapse
    both highs and lows toward the trend side (up: take max; down: take min).
    Returns a list of merged bars carrying their last original index.
    """
    merged: List[MergedBar] = []
    direction = 0  # -1 down, +1 up
    highs = df["high"].values
    lows = df["low"].values
    closes = df["close"].values
    for i in range(len(df)):
        hi, lo, cl = float(highs[i]), float(lows[i]), float(closes[i])
        if not merged:
            merged.append(MergedBar(index=i, high=hi, low=lo, close=cl))
            continue
        last = merged[-1]
        if _bars_contained(last.high, last.low, hi, lo):
            # collapse toward trend side
            if direction >= 0:
                # up: keep larger high and higher low
                last.high = max(last.high, hi)
                last.low = max(last.low, lo)
            else:
                # down: keep lower high and lower low
                last.high = min(last.high, hi)
                last.low = min(last.low, lo)
            last.index = i
            last.close = cl
            continue
        # not contained: add new and update direction
        if hi > last.high and lo > last.low:
            direction = +1
        elif hi < last.high and lo < last.low:
            direction = -1
        # else ambiguous, keep previous direction
        merged.append(MergedBar(index=i, high=hi, low=lo, close=cl))
    return merged


def find_fractals_on_merged(bars: List[MergedBar], left: int = 2, right: int = 2) -> List[Fractal]:
    """Fractals computed on inclusion-merged bars, mapped back to original indices."""
    frs: List[Fractal] = []
    n = len(bars)
    if n == 0:
        return frs
    for i in range(left, n - right):
        window = bars[i - left : i + right + 1]
        hs = [b.high for b in window]
        ls = [b.low for b in window]
        center = bars[i]
        if center.high == max(hs) and hs.count(center.high) == 1:
            frs.append(Fractal(index=center.index, kind="top", price=center.high))
        if center.low == min(ls) and ls.count(center.low) == 1:
            frs.append(Fractal(index=center.index, kind="bottom", price=center.low))
    frs.sort(key=lambda f: f.index)
    return frs


def build_pens_full(
    fractals: List[Fractal],
    min_price_move: float = 0.0,
) -> List[Pen]:
    """Build pens from alternating fractals with inclusion-aware indices.

    Differences vs simplified:
    - requires strict alternation top/bottom
    - optional minimal price move
    - no hard min_separation; separation implied by merged bars
    """
    pens: List[Pen] = []
    if not fractals:
        return pens

    # deduplicate adjacent same-type by extremity
    fs: List[Fractal] = []
    for f in fractals:
        if not fs:
            fs.append(f)
            continue
        last = fs[-1]
        if f.kind == last.kind:
            if f.kind == "top":
                if f.price >= last.price:
                    fs[-1] = f
            else:
                if f.price <= last.price:
                    fs[-1] = f
        else:
            fs.append(f)

    for a, b in zip(fs, fs[1:]):
        if a.kind == b.kind:
            continue
        pct = abs(b.price - a.price) / max(1e-12, a.price)
        if pct < min_price_move:
            continue
        direction = "up" if b.price > a.price else "down"
        pens.append(Pen(start=a, end=b, direction=direction))
    return pens


def build_segments_full(pens: List[Pen]) -> List[Segment]:
    """Three-pens make a segment (三笔成段) with persistent direction."""
    segs: List[Segment] = []
    if len(pens) < 3:
        return segs
    i = 0
    while i + 2 < len(pens):
        p0, p1, p2 = pens[i], pens[i + 1], pens[i + 2]
        # segment direction follows p2
        if p0.direction == p1.direction == p2.direction:
            start = p0.start
            end = p2.end
            segs.append(
                Segment(
                    start_idx=start.index,
                    end_idx=end.index,
                    direction=p2.direction,
                    start_price=start.price,
                    end_price=end.price,
                )
            )
            i += 2  # allow overlap
        else:
            i += 1
    return segs


def pivot_retest_signals(df_with_bands: pd.DataFrame) -> pd.DataFrame:
    """Second buy/sell (二买/二卖) as retests of pivot bands.

    - After a breakout above pivot_high, a subsequent dip below and then close back above pivot_high -> buy2
    - After a breakdown below pivot_low, a rally above and then close back below pivot_low -> sell2
    """
    rows: List[dict] = []
    close = df_with_bands["close"].values
    pl = df_with_bands["pivot_low"].values
    ph = df_with_bands["pivot_high"].values
    n = len(df_with_bands)
    above = False
    below = False
    for i in range(1, n):
        # track state relative to bands
        if pd.notna(ph[i]) and close[i] > ph[i]:
            above = True
        if pd.notna(pl[i]) and close[i] < pl[i]:
            below = True

        # buy2: was above, dipped below band then reclaimed
        if pd.notna(ph[i]) and pd.notna(ph[i - 1]):
            if above and close[i - 1] < ph[i - 1] and close[i] > ph[i]:
                rows.append({"index": i, "signal": "buy", "price": float(close[i]), "kind": "buy2"})
                above = False
        # sell2: was below, rallied above then lost it
        if pd.notna(pl[i]) and pd.notna(pl[i - 1]):
            if below and close[i - 1] > pl[i - 1] and close[i] < pl[i]:
                rows.append({"index": i, "signal": "sell", "price": float(close[i]), "kind": "sell2"})
                below = False
    if not rows:
        return pd.DataFrame(columns=["index", "signal", "price"])
    return pd.DataFrame(rows).drop_duplicates(subset=["index", "signal"]).sort_values("index").reset_index(drop=True)


def divergence_signals(
    df: pd.DataFrame,
    segments: List[Segment],
    *,
    min_price_ext_pct: float = 0.0,
    min_hist_delta: float = 0.0,
    require_hist_sign_consistency: bool = False,
) -> pd.DataFrame:
    """Third buy/sell (三买/三卖) via momentum divergence at segment turns.

    Use MACD histogram as momentum proxy:
    - buy3: new lower low at a down-turn with higher (less negative) MACD hist vs previous down-turn
    - sell3: new higher high at an up-turn with lower (less positive) MACD hist vs previous up-turn
    Signals placed at segment end index.
    """
    from . import indicators as ta

    if not segments:
        return pd.DataFrame(columns=["index", "signal", "price"])
    macd_line, sig, hist = ta.macd(df["close"])
    # fill NaNs conservatively
    hist = hist.fillna(0.0)

    rows: List[dict] = []
    last_down: Optional[Tuple[int, float, float]] = None  # (idx, price, hist)
    last_up: Optional[Tuple[int, float, float]] = None
    for seg in segments:
        idx = seg.end_idx
        price = seg.end_price
        h = float(hist.iloc[idx]) if 0 <= idx < len(df) else 0.0
        if seg.direction == "down":
            # potential buy3 at end of down segment
            if last_down is not None:
                prev_idx, prev_price, prev_h = last_down
                ext_pct = (prev_price - price) / max(abs(prev_price), 1e-12)
                hist_ok = (h - prev_h) > min_hist_delta
                sign_ok = True
                if require_hist_sign_consistency:
                    sign_ok = (prev_h <= 0) and (h <= 0)
                if (price < prev_price) and (ext_pct >= min_price_ext_pct) and hist_ok and sign_ok:
                    rows.append({"index": idx, "signal": "buy", "price": price, "kind": "buy3"})
            last_down = (idx, price, h)
        else:
            # potential sell3 at end of up segment
            if last_up is not None:
                prev_idx, prev_price, prev_h = last_up
                ext_pct = (price - prev_price) / max(abs(prev_price), 1e-12)
                hist_ok = (prev_h - h) > min_hist_delta
                sign_ok = True
                if require_hist_sign_consistency:
                    sign_ok = (prev_h >= 0) and (h >= 0)
                if (price > prev_price) and (ext_pct >= min_price_ext_pct) and hist_ok and sign_ok:
                    rows.append({"index": idx, "signal": "sell", "price": price, "kind": "sell3"})
            last_up = (idx, price, h)
    if not rows:
        return pd.DataFrame(columns=["index", "signal", "price"])
    return pd.DataFrame(rows).sort_values("index").reset_index(drop=True)


def analyze_full(
    df: pd.DataFrame,
    *,
    div_min_price_ext_pct: float = 0.0,
    div_min_hist_delta: float = 0.0,
    div_require_hist_sign_consistency: bool = False,
):
    """Fuller Chan pipeline with inclusion, stricter segmenting, Zhongshu, and extended signals.

    Steps:
    - 合并K线（包含关系）
    - 分型（基于合并K线）
    - 笔（严格交替）
    - 线段（三笔成段）
    - 中枢（笔价区间交叠）
    - 信号：线段拐点、枢纽突破、一/二买卖、背驰（三买/三卖）
    """
    merged = merge_kbars_inclusion(df)
    fr = find_fractals_on_merged(merged)
    pens = build_pens_full(fr)
    segs = build_segments_full(pens)
    pivots = build_pivots(pens)

    bands = annotate_pivot_bands(len(df), pivots)
    df_local = df.reset_index(drop=True).copy()
    df_local = pd.concat([df_local, bands], axis=1)

    seg_sigs = segment_signals(segs)
    if not seg_sigs.empty:
        seg_sigs = seg_sigs.assign(kind="turn")
    piv_sigs = pivot_breakout_signals(df_local)
    retest = pivot_retest_signals(df_local)
    div = divergence_signals(
        df_local,
        segs,
        min_price_ext_pct=div_min_price_ext_pct,
        min_hist_delta=div_min_hist_delta,
        require_hist_sign_consistency=div_require_hist_sign_consistency,
    )

    sigs = pd.concat([seg_sigs, piv_sigs, retest, div], ignore_index=True)
    if not sigs.empty:
        sigs = (
            sigs.sort_values(["index", "signal"])  # deterministic ordering
            .drop_duplicates(subset=["index", "signal"], keep="first")
            .reset_index(drop=True)
        )
    return {
        "fractals": fr,
        "pens": pens,
        "segments": segs,
        "pivots": pivots,
        "signals": sigs,
        "bands": bands,
    }


def analyze(
    df: pd.DataFrame,
    mode: str | None = None,
    *,
    # divergence tuning (only for full mode)
    div_min_price_ext_pct: float = 0.0,
    div_min_hist_delta: float = 0.0,
    div_require_hist_sign_consistency: bool = False,
):
    """Analyze with full Chan workflow by default.

    For backward-compatibility, passing mode="simple" runs the prior simplified logic.
    """
    if mode == "simple":
        return analyze.__wrapped__(df)  # type: ignore[attr-defined]
    return analyze_full(
        df,
        div_min_price_ext_pct=div_min_price_ext_pct,
        div_min_hist_delta=div_min_hist_delta,
        div_require_hist_sign_consistency=div_require_hist_sign_consistency,
    )


# Keep original analyze implementation bound for backward compatibility via __wrapped__
analyze.__wrapped__ = lambda df: {
    # replicate the original simplified call path
    **(lambda _df: (
        (lambda fr, pens, segs, pivots, bands, seg_sigs, df_local, piv_sigs, sigs: {
            "fractals": fr,
            "pens": pens,
            "segments": segs,
            "pivots": pivots,
            "signals": sigs,
            "bands": bands,
        })(
            fr := find_fractals(_df),
            pens := build_pens(fr),
            segs := build_segments(pens),
            pivots := build_pivots(pens),
            bands := annotate_pivot_bands(len(_df), pivots),
            seg_sigs := segment_signals(segs),
            df_local := pd.concat([_df.reset_index(drop=True).copy(), bands], axis=1),
            piv_sigs := pivot_breakout_signals(df_local),
            sigs := (lambda ss: ss.sort_values(["index", "signal"]).drop_duplicates(subset=["index"], keep="first").reset_index(drop=True) if not ss.empty else ss)(
                pd.concat([seg_sigs, piv_sigs], ignore_index=True)
            ),
        )
    ))(df)
}
