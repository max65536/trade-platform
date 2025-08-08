from __future__ import annotations

from typing import Optional, List

import pandas as pd


def _lazy_import_mpl():
    try:
        import matplotlib.pyplot as plt
        from matplotlib.collections import LineCollection
        from matplotlib.lines import Line2D
        return plt, LineCollection, Line2D
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
    signals_faded: Optional[pd.DataFrame] = None,
    pivots: Optional[List] = None,
    theme: str = "light",
    label_segments: bool = False,
    label_pivots: bool = False,
    trades: Optional[pd.DataFrame] = None,
    label_trades: bool = True,
    show_signals: bool = True,
    show_faded_legend: bool = True,
):
    plt, LineCollection, Line2D = _lazy_import_mpl()
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

    # Signals (faded first for layering)
    if show_signals and signals_faded is not None and not signals_faded.empty:
        buys_f = signals_faded[signals_faded.signal == "buy"]["index"].astype(int)
        sells_f = signals_faded[signals_faded.signal == "sell"]["index"].astype(int)
        ax.scatter(buys_f, df.loc[buys_f, "low"] * 0.999, marker="^", color=up_c, s=28, alpha=0.25)
        ax.scatter(sells_f, df.loc[sells_f, "high"] * 1.001, marker="v", color=dn_c, s=28, alpha=0.25)
    if show_signals and signals is not None and not signals.empty:
        # If kind column is present, render 1/2/3-class signals differently
        if "kind" in signals.columns:
            handles, labels = ax.get_legend_handles_labels()
            def draw_group(mask, y_ref, marker, color, size, label=None, edge=None, lw=0.8):
                idxs = signals[mask]["index"].astype(int)
                if idxs.empty:
                    return
                y = df.loc[idxs, y_ref]
                sc = ax.scatter(
                    idxs,
                    y,
                    marker=marker,
                    s=size,
                    c=color,
                    edgecolors=edge if edge else None,
                    linewidths=lw if edge else None,
                    label=label,
                )
                if label:
                    handles.append(sc)
                    labels.append(label)
                return idxs

            def annotate_group(idxs, y_ref, text, color, dy=0.0):
                if idxs is None or len(idxs) == 0:
                    return
                for i in idxs:
                    yi = float(df.loc[i, y_ref]) if i in df.index else None
                    if yi is None:
                        continue
                    ax.text(i, yi + dy, text, color=color, fontsize=7, ha="center", va="center", alpha=0.9)

            # Buys
            b1 = draw_group((signals.signal == "buy") & (signals.kind == "buy1"), "low", "^", up_c, 50, label="buy1")
            b2 = draw_group((signals.signal == "buy") & (signals.kind == "buy2"), "low", "^", up_c, 40, label="buy2", edge=band_c)
            b3 = draw_group((signals.signal == "buy") & (signals.kind == "buy3"), "low", "^", "none", 60, label="buy3", edge=up_c, lw=1.2)
            # Sells
            s1 = draw_group((signals.signal == "sell") & (signals.kind == "sell1"), "high", "v", dn_c, 50, label="sell1")
            s2 = draw_group((signals.signal == "sell") & (signals.kind == "sell2"), "high", "v", dn_c, 40, label="sell2", edge=band_c)
            s3 = draw_group((signals.signal == "sell") & (signals.kind == "sell3"), "high", "v", "none", 60, label="sell3", edge=dn_c, lw=1.2)
            # Segment turns (neutral marker)
            draw_group(signals.kind == "turn", "close", "x", seg_c, 40, label="turn")

            # Numeric annotations near markers
            annotate_group(b1, "low", "1", up_c, dy=-(df["close"].median() * 0.0005))
            annotate_group(b2, "low", "2", up_c, dy=-(df["close"].median() * 0.0005))
            annotate_group(b3, "low", "3", up_c, dy=-(df["close"].median() * 0.0005))
            annotate_group(s1, "high", "1", dn_c, dy=(df["close"].median() * 0.0005))
            annotate_group(s2, "high", "2", dn_c, dy=(df["close"].median() * 0.0005))
            annotate_group(s3, "high", "3", dn_c, dy=(df["close"].median() * 0.0005))

            # Optionally add a legend entry for filtered (faded) signals
            if show_faded_legend and signals_faded is not None and not signals_faded.empty:
                filtered_proxy = Line2D([0], [0], marker='^', linestyle='None', color='none', markerfacecolor=up_c, alpha=0.25, markersize=6, label='filtered')
                handles.append(filtered_proxy)
                labels.append('filtered')
            if handles and labels:
                ax.legend(handles, labels, loc="upper left")
        else:
            # Fallback simple rendering if kind is absent
            buys = signals[signals.signal == "buy"]["index"].astype(int)
            sells = signals[signals.signal == "sell"]["index"].astype(int)
            ax.scatter(buys, df.loc[buys, "low"] * 0.999, marker="^", color=up_c, s=40, label="buy")
            ax.scatter(sells, df.loc[sells, "high"] * 1.001, marker="v", color=dn_c, s=40, label="sell")
            if show_faded_legend and signals_faded is not None and not signals_faded.empty:
                filtered_proxy = Line2D([0], [0], marker='^', linestyle='None', color='none', markerfacecolor=up_c, alpha=0.25, markersize=6, label='filtered')
                handles, labels = ax.get_legend_handles_labels()
                handles.append(filtered_proxy)
                labels.append('filtered')
                ax.legend(handles, labels, loc="upper left")
            else:
                ax.legend(loc="upper left")

    # Pivot labels (if objects provided)
    if label_pivots and pivots:
        for i, z in enumerate(pivots):
            midx = (z.start_idx + z.end_idx) / 2
            midy = (z.low + z.high) / 2
            ax.text(midx, midy, f"P{i+1}", color=band_c, fontsize=8, ha="center", va="center", alpha=0.9)

    # Trades overlay
    if trades is not None and not trades.empty:
        for _, tr in trades.iterrows():
            ei = int(tr["entry_idx"])
            xi = int(tr["exit_idx"])
            ep = float(tr["entry_px"])
            xp = float(tr["exit_px"])
            col = up_c if xp >= ep else dn_c
            ax.plot([ei, xi], [ep, xp], color=col, linewidth=1.5, alpha=0.9)
            ax.scatter([ei], [ep], color=up_c, marker="^", s=36)
            ax.scatter([xi], [xp], color=dn_c, marker="x", s=36)
            if label_trades and "ret" in tr:
                pct = tr["ret"] * 100.0
                ax.text(xi, xp, f"{pct:.1f}%", color=col, fontsize=8, ha="left", va="bottom")

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
