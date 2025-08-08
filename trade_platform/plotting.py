from __future__ import annotations

from typing import Optional, List

import pandas as pd


def _lazy_import_mpl():
    try:
        import matplotlib.pyplot as plt
        from matplotlib.collections import LineCollection
        return plt, LineCollection
    except Exception as e:
        raise RuntimeError("matplotlib is required for plotting. Install with `pip install matplotlib`." ) from e


def plot_kline(
    df: pd.DataFrame,
    out: Optional[str] = None,
    title: Optional[str] = None,
    width: int = 14,
    height: int = 6,
    pivot_low: Optional[pd.Series] = None,
    pivot_high: Optional[pd.Series] = None,
    pens: Optional[List] = None,
    segments: Optional[List] = None,
    signals: Optional[pd.DataFrame] = None,
    pivots: Optional[List] = None,
    theme: str = "light",
    label_segments: bool = False,
    label_pivots: bool = False,
):
    plt, LineCollection = _lazy_import_mpl()
    import numpy as np

    df = df.reset_index(drop=True)
    x = np.arange(len(df))

    fig, ax = plt.subplots(figsize=(width, height))

    # Theme setup
    if theme not in ("light", "dark", "minimal"):
        theme = "light"
    if theme == "dark":
        fig.patch.set_facecolor("#111")
        ax.set_facecolor("#111")
        grid_c = (1, 1, 1, 0.12)
        axis_c = "#ddd"
        up_c, dn_c = "#26a69a", "#ef5350"
        wick_c = "#aaa"
        seg_c, pen_c = "#ffb74d", "#b39ddb"
        band_c = "#42a5f5"
    elif theme == "minimal":
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")
        grid_c = (0, 0, 0, 0.06)
        axis_c = "#333"
        up_c, dn_c = "#2ca02c", "#d62728"
        wick_c = "#666"
        seg_c, pen_c = "#ff7f0e", "#9467bd"
        band_c = "#1f77b4"
    else:  # light
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")
        grid_c = (0, 0, 0, 0.1)
        axis_c = "#333"
        up_c, dn_c = "#2ca02c", "#d44"
        wick_c = "#666"
        seg_c, pen_c = "#ff7f0e", "#9467bd"
        band_c = "#1f77b4"

    ax.set_title(title or "Kline", color=axis_c)

    # Draw wicks
    wick_segments = [((xi, df.loc[i, "low"]), (xi, df.loc[i, "high"])) for i, xi in enumerate(x)]
    ax.add_collection(LineCollection(wick_segments, colors=wick_c, linewidths=0.6))

    # Draw bodies
    for i, xi in enumerate(x):
        o = df.loc[i, "open"]
        c = df.loc[i, "close"]
        lo = min(o, c)
        hi = max(o, c)
        color = dn_c if c < o else up_c
        ax.add_patch(
            plt.Rectangle((xi - 0.3, lo), 0.6, hi - lo if hi - lo != 0 else 0.001, color=color, alpha=0.8)
        )

    # Pivots as bands
    if pivot_low is not None and pivot_high is not None:
        pl = pivot_low.values
        ph = pivot_high.values
        in_band = False
        start = 0
        for i in range(len(x)):
            if pd.notna(pl[i]) and pd.notna(ph[i]):
                if not in_band:
                    start = i
                    in_band = True
            else:
                if in_band:
                    ax.fill_between(x[start:i], pl[start:i], ph[start:i], color=band_c, alpha=0.1)
                    in_band = False
        if in_band:
            ax.fill_between(x[start:len(x)], pl[start:len(x)], ph[start:len(x)], color=band_c, alpha=0.1)

    # Pens
    if pens:
        for p in pens:
            xs = [p.start.index, p.end.index]
            ys = [p.start.price, p.end.price]
            ax.plot(xs, ys, color=pen_c, linewidth=1.2, alpha=0.8)

    # Segments
    if segments:
        for s in segments:
            xs = [s.start_idx, s.end_idx]
            ys = [s.start_price, s.end_price]
            ax.plot(xs, ys, color=seg_c, linewidth=2.0, alpha=0.9)
            if label_segments:
                midx = (s.start_idx + s.end_idx) / 2
                midy = (s.start_price + s.end_price) / 2
                ax.text(midx, midy, f"{s.direction}", color=seg_c, fontsize=8, ha="center", va="bottom", alpha=0.9)

    # Signals
    if signals is not None and not signals.empty:
        buys = signals[signals.signal == "buy"]["index"].astype(int)
        sells = signals[signals.signal == "sell"]["index"].astype(int)
        ax.scatter(buys, df.loc[buys, "low"] * 0.999, marker="^", color=up_c, s=40, label="buy")
        ax.scatter(sells, df.loc[sells, "high"] * 1.001, marker="v", color=dn_c, s=40, label="sell")
        ax.legend(loc="upper left")

    # Pivot labels (if objects provided)
    if label_pivots and pivots:
        for i, z in enumerate(pivots):
            midx = (z.start_idx + z.end_idx) / 2
            midy = (z.low + z.high) / 2
            ax.text(midx, midy, f"P{i+1}", color=band_c, fontsize=8, ha="center", va="center", alpha=0.9)

    ax.set_xlim(-0.5, len(df) - 0.5)
    ax.set_xlabel("index", color=axis_c)
    ax.set_ylabel("price", color=axis_c)
    ax.grid(True, color=grid_c)
    for spine in ax.spines.values():
        spine.set_color(axis_c)
    ax.tick_params(colors=axis_c)

    fig.tight_layout()
    if out:
        fig.savefig(out, dpi=150)
    else:
        plt.show()
