from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from http import HTTPStatus
from typing import Optional
from urllib.parse import parse_qs

import pandas as pd

from ..dataio import CandleFrame
from .. import chan
from .. import strategy
from ..backtest import simple_execute, execute_with_risk
from .. import plotting
from ..exchanges import ExchangeClient
from .paths import ensure_dirs, PLOTS_DIR, TRADES_DIR, STATS_DIR
from .pages import html_page, backtest_index


def handle_backtest(handler) -> None:
    length = int(handler.headers.get("Content-Length", "0"))
    raw = handler.rfile.read(length).decode("utf-8")
    qs = parse_qs(raw)
    getv = lambda k, default=None: (qs.get(k, [default])[0])

    try:
        input_path = (getv("input_path") or "").strip() or None
        start = (getv("start") or "").strip() or None
        end = (getv("end") or "").strip() or None
        fee = float(getv("fee", 0.0005) or 0.0005)
        rsi_min = _to_float(getv("rsi_min"))
        rsi_max = _to_float(getv("rsi_max"))
        min_atr_pct = _to_float(getv("min_atr_pct"))
        max_atr_pct = _to_float(getv("max_atr_pct"))
        stop_pct = _to_float(getv("stop_pct"))
        tp_pct = _to_float(getv("tp_pct"))
        position_size = float(getv("position_size", 1.0) or 1.0)
        slippage_bps = float(getv("slippage_bps", 0.0) or 0.0)
        theme = (getv("theme", "light") or "light").strip()
        limit = int(float(getv("limit", 0) or 0))
        show_macd = bool(getv("show_macd"))
        show_rsi = bool(getv("show_rsi"))
        show_atr = bool(getv("show_atr"))
        show_signals = bool(getv("show_signals", "on"))
        show_trades = bool(getv("show_trades", "on"))
        show_pivot_bands = bool(getv("show_pivot_bands"))
        show_segments = bool(getv("show_segments"))
        label_segments = bool(getv("label_segments"))

        ensure_dirs()
        if not input_path:
            return _send_html(handler, backtest_index("Please provide a CSV path."))
        if not os.path.exists(input_path):
            return _send_html(handler, backtest_index(f"CSV not found: {input_path}"))

        cf = CandleFrame.read_csv(input_path)
        df = cf.df.copy()
        if start:
            df = df[df["datetime"] >= pd.Timestamp(start)]
        if end:
            df = df[df["datetime"] <= pd.Timestamp(end)]
        df = df.reset_index(drop=True)

        out = chan.analyze(df)
        sigs = out["signals"]
        dfi = strategy.ensure_indicators(df)
        sigs = strategy.apply_signal_filters(
            sigs, dfi, rsi_min=rsi_min, rsi_max=rsi_max, min_atr_pct=min_atr_pct, max_atr_pct=max_atr_pct
        )

        if stop_pct is not None or tp_pct is not None:
            res = execute_with_risk(
                df,
                sigs,
                fee_rate=fee,
                stop_loss_pct=stop_pct,
                take_profit_pct=tp_pct,
                position_size=position_size,
                slippage_bps=slippage_bps,
            )
        else:
            res = simple_execute(
                df,
                sigs,
                fee_rate=fee,
                position_size=position_size,
                slippage_bps=slippage_bps,
            )

        run_id = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        trades_path = os.path.join(TRADES_DIR, f"{run_id}.csv")
        stats_path = os.path.join(STATS_DIR, f"{run_id}.json")
        cfg_path = os.path.join(STATS_DIR, f"{run_id}-config.json")
        plot_path = os.path.join(PLOTS_DIR, f"{run_id}.png")
        if not res.trades.empty:
            res.trades.to_csv(trades_path, index=False)
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(res.stats, f, indent=2)
        # Save config used for reproducibility
        cfg = {
            "input_path": input_path,
            "start": start,
            "end": end,
            "fee": fee,
            "rsi_min": rsi_min,
            "rsi_max": rsi_max,
            "min_atr_pct": min_atr_pct,
            "max_atr_pct": max_atr_pct,
            "stop_pct": stop_pct,
            "tp_pct": tp_pct,
            "position_size": position_size,
            "slippage_bps": slippage_bps,
            "theme": theme,
            "limit": limit,
            "show_macd": show_macd,
            "show_rsi": show_rsi,
            "show_atr": show_atr,
            "show_signals": show_signals,
            "show_trades": show_trades,
            "show_pivot_bands": show_pivot_bands,
            "show_segments": show_segments,
            "label_segments": label_segments,
        }
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)

        plot_df = df.copy()
        if limit and len(plot_df) > limit:
            plot_df = plot_df.tail(limit).reset_index(drop=True)

        plot_signals = sigs if show_signals else None
        pivot_low = None
        pivot_high = None
        segments = None
        if show_pivot_bands or show_segments:
            bands = out.get("bands")
            if show_pivot_bands and bands is not None and not bands.empty:
                pivot_low = bands.get("pivot_low")
                pivot_high = bands.get("pivot_high")
            if show_segments:
                segments = out.get("segments")

        plotting.plot_kline(
            plot_df,
            out=plot_path,
            title=f"Backtest: {os.path.basename(input_path)}",
            pivot_low=pivot_low,
            pivot_high=pivot_high,
            segments=segments,
            signals=plot_signals,
            trades=(res.trades if (show_trades and not res.trades.empty) else None),
            theme=theme,
            show_macd=show_macd,
            show_rsi=show_rsi,
            show_atr=show_atr,
            label_segments=label_segments,
            show_signals=show_signals,
        )

        preferred = [
            "trades","win_rate","cum_return","cagr","profit_factor",
            "best_trade","worst_trade","avg_ret","avg_hold_bars",
            "max_drawdown","max_dd_recovery_bars","volatility_ann","sharpe","sortino",
            "exposure_bars","exposure_pct",
        ]
        ordered = []
        for k in preferred:
            if k in res.stats:
                ordered.append((k, res.stats[k]))
        for k, v in res.stats.items():
            if k not in dict(ordered):
                ordered.append((k, v))
        stats_rows = "\n".join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in ordered)
        trades_link = (
            f"<a href=\"/static/trades/{os.path.basename(trades_path)}\">Download trades CSV</a>"
            if not res.trades.empty
            else "<em>No trades</em>"
        )
        body = f"""
        <div class=\"row\">
          <div class=\"col\">
            <div class=\"card\">
              <h3>Stats</h3>
              <table><tbody>{stats_rows}</tbody></table>
              <p>Stats JSON: <a href=\"/static/stats/{os.path.basename(stats_path)}\">{os.path.basename(stats_path)}</a></p>
              <p>Config JSON: <a href=\"/static/stats/{os.path.basename(cfg_path)}\">{os.path.basename(cfg_path)}</a></p>
              <p>{trades_link}</p>
              <p><a href=\"/\">⟵ Back</a></p>
            </div>
          </div>
          <div class=\"col\">
            <div class=\"card\">
              <h3>Plot</h3>
              <img src=\"/static/plots/{os.path.basename(plot_path)}\" />
            </div>
          </div>
        </div>
        """
        return _send_html(handler, html_page("Backtest Result", body))
    except Exception as e:
        try:
            return _send_html(handler, backtest_index(str(e)))
        except Exception:
            return _send_text(handler, f"Error: {e}", HTTPStatus.INTERNAL_SERVER_ERROR)


def api_ohlcv(handler, parsed) -> None:
    from .. import indicators as ta

    qs = parse_qs(parsed.query)
    getv = lambda k, d=None: (qs.get(k, [d])[0])
    try:
        input_path = (getv('input_path') or '').strip() or None
        exchange = (getv('exchange') or '').strip() or None
        symbol = (getv('symbol') or '').strip() or None
        timeframe = (getv('timeframe') or '').strip() or None
        limit = int(float(getv('limit', 500) or 500))
        since = (getv('since') or '').strip() or None
        sma_s = (getv('sma') or '').strip()
        ema_s = (getv('ema') or '').strip()
        sma_lengths = [int(x) for x in sma_s.split(',') if x.strip().isdigit()] if sma_s else []
        ema_lengths = [int(x) for x in ema_s.split(',') if x.strip().isdigit()] if ema_s else []
        macd_on = (getv('macd','1') in ('1','true'))
        rsi_on = (getv('rsi','0') in ('1','true'))
        atr_on = (getv('atr','0') in ('1','true'))
        bands_on = (getv('bands','1') in ('1','true'))
        segments_on = (getv('segments','1') in ('1','true'))
        signals_on = (getv('signals','1') in ('1','true'))

        # Load df
        if input_path:
            cf = CandleFrame.read_csv(input_path)
            df = cf.df.copy().reset_index(drop=True)
            sym = os.path.basename(input_path)
            tf = None
        elif exchange and symbol and timeframe:
            ex = ExchangeClient(exchange)
            ex.load_markets()
            ms = None
            if since:
                dt = datetime.fromisoformat(since)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                ms = int(dt.timestamp()*1000)
            ohlcv = ex.fetch_ohlcv_all(symbol=symbol, timeframe=timeframe, since=ms, limit=limit, max_bars=limit)
            cf = CandleFrame.from_ohlcv(ohlcv)
            df = cf.df.copy().reset_index(drop=True)
            sym, tf = symbol, timeframe
        else:
            return _send_text(handler, 'Bad Request', HTTPStatus.BAD_REQUEST)

        if limit and len(df) > limit:
            df = df.tail(limit).reset_index(drop=True)

        ts = df['timestamp'].astype(int).tolist() if 'timestamp' in df.columns else [int(pd.Timestamp(ts).value//10**6) for ts in df['datetime']]
        base = {
            'symbol': sym,
            'timeframe': tf,
            'columns': ['timestamp','open','high','low','close','volume'],
            'data': [[int(ts[i]), float(df.loc[i,'open']), float(df.loc[i,'high']), float(df.loc[i,'low']), float(df.loc[i,'close']), float(df.loc[i,'volume'] if 'volume' in df.columns else 0.0)] for i in range(len(df))],
        }

        ind = {}
        if sma_lengths:
            ind['sma'] = {}
            for ln in sma_lengths:
                s = df['close'].rolling(int(ln)).mean()
                ind['sma'][str(ln)] = [None if pd.isna(x) else float(x) for x in s]
        if ema_lengths:
            ind['ema'] = {}
            for ln in ema_lengths:
                s = df['close'].ewm(span=int(ln), adjust=False).mean()
                ind['ema'][str(ln)] = [None if pd.isna(x) else float(x) for x in s]
        if macd_on:
            macd, sig, hist = ta.macd(df['close'])
            ind['macd'] = {
                'macd': [None if pd.isna(x) else float(x) for x in macd],
                'signal': [None if pd.isna(x) else float(x) for x in sig],
                'hist': [None if pd.isna(x) else float(x) for x in hist],
            }
        if rsi_on:
            r = ta.rsi(df['close'], 14)
            ind.setdefault('rsi', {})['14'] = [None if pd.isna(x) else float(x) for x in r]
        if atr_on:
            a = ta.atr(df, 14)
            ind.setdefault('atr', {})['14'] = [None if pd.isna(x) else float(x) for x in a]
        if ind:
            base['indicators'] = ind

        if bands_on or segments_on or signals_on:
            out = chan.analyze(df)
            chan_payload = {}
            if bands_on:
                bands = out.get('bands')
                chan_payload['bands'] = {
                    'pivot_low': [None if pd.isna(x) else float(x) for x in bands['pivot_low']] if bands is not None and not bands.empty else None,
                    'pivot_high': [None if pd.isna(x) else float(x) for x in bands['pivot_high']] if bands is not None and not bands.empty else None,
                }
            if segments_on:
                segs = out.get('segments') or []
                chan_payload['segments'] = [
                    {
                        'start_idx': int(s.start_idx), 'end_idx': int(s.end_idx), 'direction': s.direction,
                        'start_price': float(s.start_price), 'end_price': float(s.end_price),
                    } for s in segs
                ]
            if signals_on:
                sigs = out.get('signals')
                chan_payload['signals'] = (
                    [ {'index': int(r['index']), 'signal': r['signal'], 'kind': r.get('kind') if 'kind' in sigs.columns else None, 'price': float(r['price'])} for _, r in sigs.iterrows() ]
                    if sigs is not None and not sigs.empty else []
                )
            base['chan'] = chan_payload

        return _send_json(handler, base)
    except Exception as e:
        return _send_text(handler, f"Error: {e}", HTTPStatus.INTERNAL_SERVER_ERROR)


def api_mtf(handler, parsed) -> None:
    from .. import indicators as ta
    from .. import multiframe as mtf

    qs = parse_qs(parsed.query)
    getv = lambda k, d=None: (qs.get(k, [d])[0])
    try:
        lower_input = (getv('lower_input') or '').strip() or None
        higher_input = (getv('higher_input') or '').strip() or None
        exchange = (getv('exchange') or '').strip() or None
        symbol = (getv('symbol') or '').strip() or None
        lower_tf = (getv('lower_tf') or '').strip() or None
        higher_tf = (getv('higher_tf') or '').strip() or None
        limit = int(float(getv('limit', 600) or 600))
        since = (getv('since') or '').strip() or None
        require_htf_breakout = (getv('require_htf_breakout','1') in ('1','true'))
        min_htf_run = int(float(getv('min_htf_run', 3) or 3))
        sma_s = (getv('sma') or '').strip()
        ema_s = (getv('ema') or '').strip()
        sma_lengths = [int(x) for x in sma_s.split(',') if x.strip().isdigit()] if sma_s else []
        ema_lengths = [int(x) for x in ema_s.split(',') if x.strip().isdigit()] if ema_s else []
        macd_on = (getv('macd','1') in ('1','true'))
        rsi_on = (getv('rsi','0') in ('1','true'))
        atr_on = (getv('atr','0') in ('1','true'))

        # LTF
        if lower_input:
            ldf = CandleFrame.read_csv(lower_input).df.copy().reset_index(drop=True)
        elif exchange and symbol and lower_tf:
            ex = ExchangeClient(exchange)
            ex.load_markets()
            ms = None
            if since:
                dt = datetime.fromisoformat(since)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                ms = int(dt.timestamp()*1000)
            ldf = CandleFrame.from_ohlcv(ex.fetch_ohlcv_all(symbol=symbol, timeframe=lower_tf, since=ms, limit=limit, max_bars=limit)).df.copy().reset_index(drop=True)
        else:
            return _send_text(handler, 'Missing LTF source', HTTPStatus.BAD_REQUEST)

        # HTF
        if higher_input:
            hdf = CandleFrame.read_csv(higher_input).df.copy().reset_index(drop=True)
        elif exchange and symbol and higher_tf:
            ex = ExchangeClient(exchange)
            ex.load_markets()
            ms = None
            if since:
                dt = datetime.fromisoformat(since)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                ms = int(dt.timestamp()*1000)
            hdf = CandleFrame.from_ohlcv(ex.fetch_ohlcv_all(symbol=symbol, timeframe=higher_tf, since=ms, limit=limit, max_bars=limit)).df.copy().reset_index(drop=True)
        else:
            return _send_text(handler, 'Missing HTF source', HTTPStatus.BAD_REQUEST)

        lo = chan.analyze(ldf)
        ho = chan.analyze(hdf)

        htf_ctx = mtf.align_htf_to_ltf(ldf, hdf, ho['segments'], ho['bands'])
        ldf2 = ldf.reset_index(drop=True).copy()
        for col in htf_ctx.columns:
            ldf2[col] = htf_ctx[col].values

        base_sigs = lo['signals']
        mtf_sigs = mtf.filter_signals_with_htf_opts(base_sigs, ldf2, require_htf_breakout=require_htf_breakout, min_htf_run=min_htf_run)

        ts = ldf2['timestamp'].astype(int).tolist() if 'timestamp' in ldf2.columns else [int(pd.Timestamp(t).value//10**6) for t in ldf2['datetime']]
        base = {
            'symbol': symbol or (lower_input and os.path.basename(lower_input)) or 'LTF',
            'timeframe': lower_tf,
            'columns': ['timestamp','open','high','low','close','volume'],
            'data': [[int(ts[i]), float(ldf2.loc[i,'open']), float(ldf2.loc[i,'high']), float(ldf2.loc[i,'low']), float(ldf2.loc[i,'close']), float(ldf2.loc[i,'volume'] if 'volume' in ldf2.columns else 0.0)] for i in range(len(ldf2))],
        }

        base['htf'] = {
            'bands': {
                'pivot_low': [None if pd.isna(x) else float(x) for x in ldf2['htf_pivot_low']] if 'htf_pivot_low' in ldf2.columns else None,
                'pivot_high': [None if pd.isna(x) else float(x) for x in ldf2['htf_pivot_high']] if 'htf_pivot_high' in ldf2.columns else None,
            }
        }

        ind = {}
        if sma_lengths:
            ind['sma'] = {}
            for ln in sma_lengths:
                s = ldf2['close'].rolling(int(ln)).mean()
                ind['sma'][str(ln)] = [None if pd.isna(x) else float(x) for x in s]
        if ema_lengths:
            ind['ema'] = {}
            for ln in ema_lengths:
                s = ldf2['close'].ewm(span=int(ln), adjust=False).mean()
                ind['ema'][str(ln)] = [None if pd.isna(x) else float(x) for x in s]
        if macd_on:
            m, sg, hi = ta.macd(ldf2['close'])
            ind['macd'] = {
                'macd': [None if pd.isna(x) else float(x) for x in m],
                'signal': [None if pd.isna(x) else float(x) for x in sg],
                'hist': [None if pd.isna(x) else float(x) for x in hi],
            }
        if rsi_on:
            r = ta.rsi(ldf2['close'], 14)
            ind.setdefault('rsi', {})['14'] = [None if pd.isna(x) else float(x) for x in r]
        if atr_on:
            a = ta.atr(ldf2, 14)
            ind.setdefault('atr', {})['14'] = [None if pd.isna(x) else float(x) for x in a]
        if ind:
            base['indicators'] = ind

        def _pack_sigs(df_sigs):
            if df_sigs is None or df_sigs.empty:
                return []
            out = []
            for _, r in df_sigs.iterrows():
                out.append({
                    'index': int(r['index']),
                    'signal': r['signal'],
                    'kind': r.get('kind') if 'kind' in df_sigs.columns else None,
                    'price': float(r['price']) if 'price' in df_sigs.columns else float(ldf2.loc[int(r['index']), 'close']),
                })
            return out

        base['signals_base'] = _pack_sigs(base_sigs)
        base['signals_mtf'] = _pack_sigs(mtf_sigs)

        return _send_json(handler, base)
    except Exception as e:
        return _send_text(handler, f"Error: {e}", HTTPStatus.INTERNAL_SERVER_ERROR)


# Helpers
def _to_float(s: Optional[str]) -> Optional[float]:
    if s is None:
        return None
    t = str(s).strip()
    if t == "" or t.lower() == "none":
        return None
    try:
        return float(t)
    except Exception:
        return None


def _send_text(handler, text: str, code: int = 200):
    data = text.encode("utf-8")
    handler.send_response(code)
    handler.send_header("Content-Type", "text/plain; charset=utf-8")
    handler.send_header("Content-Length", str(len(data)))
    handler.end_headers()
    handler.wfile.write(data)


def _send_json(handler, obj, code: int = 200):
    data = json.dumps(obj).encode("utf-8")
    handler.send_response(code)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(data)))
    handler.end_headers()
    handler.wfile.write(data)


def _send_html(handler, data: bytes, code: int = 200):
    handler.send_response(code)
    handler.send_header("Content-Type", "text/html; charset=utf-8")
    handler.send_header("Content-Length", str(len(data)))
    handler.end_headers()
    handler.wfile.write(data)
