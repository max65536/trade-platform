from __future__ import annotations

import html
import io
import json
import os
import sys
import uuid
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
from typing import Optional
from urllib.parse import parse_qs, urlparse

import pandas as pd
import numpy as np

from .dataio import CandleFrame
from . import indicators as ta
from . import chan
from . import strategy
from .backtest import simple_execute, execute_with_risk
from . import plotting
from .exchanges import ExchangeClient


BASE_DIR = os.path.abspath(os.path.join(os.getcwd(), "runs", "webui"))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
PLOTS_DIR = os.path.join(BASE_DIR, "plots")
TRADES_DIR = os.path.join(BASE_DIR, "trades")
STATS_DIR = os.path.join(BASE_DIR, "stats")


def ensure_dirs():
    for d in (BASE_DIR, UPLOAD_DIR, PLOTS_DIR, TRADES_DIR, STATS_DIR):
        os.makedirs(d, exist_ok=True)


def _html_page(title: str, body: str) -> bytes:
    doc = f"""
    <!doctype html>
    <html lang="en">
    <head>
      <meta charset="utf-8" />
      <meta name="viewport" content="width=device-width, initial-scale=1" />
      <title>{html.escape(title)}</title>
      <style>
      body {{ font-family: -apple-system, system-ui, Segoe UI, Roboto, Helvetica, Arial, 'Apple Color Emoji', 'Segoe UI Emoji'; margin: 24px; line-height: 1.4; }}
      input, select {{ padding: 6px; margin: 4px 0; }}
      .row {{ display: flex; gap: 16px; flex-wrap: wrap; }}
      .col {{ flex: 1 1 300px; min-width: 280px; }}
      .card {{ border: 1px solid #ddd; border-radius: 8px; padding: 16px; margin-bottom: 16px; }}
      .btn {{ padding: 8px 12px; background: #166ff6; color: #fff; border: none; border-radius: 6px; cursor: pointer; }}
      .btn:disabled {{ background: #aaa; }}
      table {{ border-collapse: collapse; }}
      td, th {{ border: 1px solid #ddd; padding: 6px 8px; }}
      img {{ max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 8px; }}
      </style>
    </head>
    <body>
      <div style="display:flex;align-items:center;gap:16px;">
        <h2 style="margin:0;">{html.escape(title)}</h2>
        <div style="margin-left:auto;font-size:14px;">
          <a href="/chart">Chart</a> · <a href="/">Backtest</a>
        </div>
      </div>
      {body}
    </body>
    </html>
    """
    return doc.encode("utf-8")


def chart_page(error: Optional[str] = None) -> bytes:
    err_html = f"<p style='color:#c00'>{html.escape(error)}</p>" if error else ""
    body = f"""
    {err_html}
    <div class="card">
      <form method="POST" action="/chart">
        <div class="row">
          <div class="col">
            <h3>Data Source</h3>
            <label>CSV Path<br/><input type="text" name="input_path" placeholder="data/BTCUSDT-1h.csv" style="width:100%"/></label><br/>
            <div style="opacity:0.8; font-size: 12px;">or Fetch via exchange:</div>
            <label>Exchange<br/><input type="text" name="exchange" placeholder="binance"/></label><br/>
            <label>Symbol<br/><input type="text" name="symbol" placeholder="BTC/USDT"/></label><br/>
            <label>Timeframe<br/><input type="text" name="timeframe" placeholder="1h"/></label><br/>
            <label>Limit<br/><input type="number" name="limit" value="400"/></label><br/>
            <label>Since (ISO, opt)<br/><input type="text" name="since" placeholder="2023-01-01"/></label>
          </div>
          <div class="col">
            <h3>Indicators</h3>
            <label>SMA lengths (comma)<br/><input type="text" name="sma" placeholder="20,50"/></label><br/>
            <label>EMA lengths (comma)<br/><input type="text" name="ema" placeholder="50,100"/></label><br/>
            <label><input type="checkbox" name="show_macd"/> MACD</label><br/>
            <label><input type="checkbox" name="show_rsi"/> RSI</label><br/>
            <label><input type="checkbox" name="show_atr"/> ATR</label><br/>
          </div>
          <div class="col">
            <h3>Chan</h3>
            <label><input type="checkbox" name="show_pivot_bands" checked/> Pivot bands</label><br/>
            <label><input type="checkbox" name="show_segments" checked/> Segments</label><br/>
            <label><input type="checkbox" name="label_segments"/> Label segments</label><br/>
            <label><input type="checkbox" name="show_signals" checked/> Signals</label><br/>
            <div style="opacity:0.8; font-size: 12px;">Divergence tuning (opt)</div>
            <label>Min price ext pct<br/><input type="number" step="0.0001" name="div_min_price_ext_pct"/></label><br/>
            <label>Min hist delta<br/><input type="number" step="0.0001" name="div_min_hist_delta"/></label><br/>
            <label><input type="checkbox" name="div_require_hist_sign_consistency"/> Require MACD sign consistency</label><br/>
          </div>
          <div class="col">
            <h3>Render</h3>
            <label>Theme
              <select name="theme">
                <option value="light" selected>light</option>
                <option value="dark">dark</option>
                <option value="minimal">minimal</option>
              </select>
            </label><br/>
            <label>Plot limit bars<br/><input type="number" name="plot_limit" value="300"/></label>
          </div>
        </div>
        <p><button type="submit" class="btn">Render Chart</button></p>
      </form>
    </div>
    """
    return _html_page("Trade Platform WebUI · Chart", body)


def index_page(error: Optional[str] = None) -> bytes:
    err_html = f"<p style='color:#c00'>{html.escape(error)}</p>" if error else ""
    body = f"""
    {err_html}
    <div class="card">
      <form method="POST" action="/backtest" enctype="multipart/form-data">
        <div class="row">
          <div class="col">
            <h3>Data</h3>
            <label>CSV Path<br/><input type="text" name="input_path" placeholder="data/BTCUSDT-1h.csv" style="width:100%"/></label><br/>
            <!-- Upload disabled: provide local CSV path accessible to server process. -->
            <label>Start (optional)<br/><input type="text" name="start" placeholder="2023-01-01"/></label><br/>
            <label>End (optional)<br/><input type="text" name="end" placeholder="2024-01-01"/></label><br/>
            <label>Limit bars (plot)<br/><input type="number" name="limit" value="0"/></label>
          </div>
          <div class="col">
            <h3>Strategy Filters</h3>
            <label>RSI min<br/><input type="number" step="0.1" name="rsi_min"/></label><br/>
            <label>RSI max<br/><input type="number" step="0.1" name="rsi_max"/></label><br/>
            <label>Min ATR %<br/><input type="number" step="0.0001" name="min_atr_pct"/></label><br/>
            <label>Max ATR %<br/><input type="number" step="0.0001" name="max_atr_pct"/></label><br/>
          </div>
          <div class="col">
            <h3>Risk & Fees</h3>
            <label>Fee rate<br/><input type="number" step="0.0001" name="fee" value="0.0005"/></label><br/>
            <label>Stop %<br/><input type="number" step="0.0001" name="stop_pct"/></label><br/>
            <label>TP %<br/><input type="number" step="0.0001" name="tp_pct"/></label><br/>
            <label>Position size<br/><input type="number" step="0.01" name="position_size" value="1.0"/></label><br/>
            <label>Slippage (bps)<br/><input type="number" step="0.1" name="slippage_bps" value="0"/></label><br/>
          </div>
          <div class="col">
            <h3>Plot</h3>
            <label>Theme
              <select name="theme">
                <option value="light" selected>light</option>
                <option value="dark">dark</option>
                <option value="minimal">minimal</option>
              </select>
            </label><br/>
            <label><input type="checkbox" name="show_signals" checked/> Show signals</label><br/>
            <label><input type="checkbox" name="show_trades" checked/> Show trades</label><br/>
            <label><input type="checkbox" name="show_pivot_bands"/> Show pivot bands</label><br/>
            <label><input type="checkbox" name="show_segments"/> Show segments</label><br/>
            <label><input type="checkbox" name="label_segments"/> Label segments</label><br/>
            <label><input type="checkbox" name="show_macd"/> Show MACD</label><br/>
            <label><input type="checkbox" name="show_rsi"/> Show RSI</label><br/>
            <label><input type="checkbox" name="show_atr"/> Show ATR</label><br/>
          </div>
        </div>
        <p><button type="submit" class="btn">Run Backtest</button></p>
      </form>
    </div>
    """
    return _html_page("Trade Platform WebUI", body)


def echarts_page() -> bytes:
    body = """
    <div class=card>
      <form id="form" onsubmit="ev=>ev.preventDefault()">
        <div class=row>
          <div class=col>
            <h3>Source</h3>
            <label>CSV <input id="input_path" style="width:100%" placeholder="data/BTCUSDT-1h.csv"/></label><br/>
            <div style="opacity:0.8;font-size:12px">or Exchange</div>
            <label>Exchange <input id="exchange" placeholder="binance"/></label>
            <label>Symbol <input id="symbol" placeholder="BTC/USDT"/></label>
            <label>Timeframe <input id="timeframe" placeholder="1h"/></label>
            <label>Limit <input id="limit" type=number value=500/></label>
            <label>Since <input id="since" placeholder="2023-01-01"/></label>
          </div>
          <div class=col>
            <h3>Indicators</h3>
            <label>SMA <input id="sma" placeholder="20,50"/></label>
            <label>EMA <input id="ema" placeholder="50,100"/></label><br/>
            <label><input id="macd" type=checkbox checked/> MACD</label>
            <label><input id="rsi" type=checkbox/> RSI</label>
            <label><input id="atr" type=checkbox/> ATR</label>
          </div>
          <div class=col>
            <h3>Chan</h3>
            <label><input id="bands" type=checkbox checked/> Bands</label>
            <label><input id="segments" type=checkbox checked/> Segments</label>
            <label><input id="signals" type=checkbox checked/> Signals</label>
          </div>
          <div class=col>
            <h3>View</h3>
            <label>Theme
              <select id="theme">
                <option value="light" selected>light</option>
                <option value="dark">dark</option>
                <option value="minimal">minimal</option>
              </select>
            </label>
            <label>Plot limit <input id="plot_limit" type=number value=400/></label>
            <button id="run" class="btn" type=button>Render</button>
          </div>
        </div>
      </form>
    </div>
    <div id="chart" style="width:100%;height:720px"></div>
    <script src="https://cdn.jsdelivr.net/npm/echarts@5/dist/echarts.min.js"></script>
    <script>
    const qs = new URLSearchParams(location.search);
    function val(id, def=''){ const v = document.getElementById(id).value||qs.get(id)||def; document.getElementById(id).value=v; return v; }
    function chk(id, def=false){ const el=document.getElementById(id); const v = qs.get(id); const b = v? (v==='1'||v==='true') : def; el.checked=b; return b; }
    async function run(){
      const params = new URLSearchParams();
      ['input_path','exchange','symbol','timeframe','limit','since','sma','ema','theme','plot_limit'].forEach(k=>{ const v=document.getElementById(k).value; if(v) params.set(k,v); });
      ['macd','rsi','atr','bands','segments','signals'].forEach(k=>{ params.set(k, document.getElementById(k).checked?'1':'0'); });
      history.replaceState(null,'','?'+params.toString());
      const res = await fetch('/api/ohlcv?'+params.toString());
      if(!res.ok){ alert('Request failed'); return; }
      const data = await res.json();
      renderChart(data, (document.getElementById('theme').value||'light'));
    }
    function renderChart(payload, theme){
      const root = document.getElementById('chart');
      const chart = echarts.init(root, null, {renderer:'canvas'});
      const ts = payload.data.map(r=>r[0]);
      const ohlc = payload.data.map(r=>[r[1],r[2],r[3],r[4]]);
      const vol = payload.data.map(r=>r[5]);
      const grid = [
        {top: 20, height: 360},
        ...(payload.indicators && payload.indicators.macd? [{top: 400, height: 140}] : []),
        ...(payload.indicators && payload.indicators.rsi? [{top: 550, height: 140}] : []),
      ];
      const xAxes = grid.map((g,i)=>({
        gridIndex: i,
        type: 'category',
        data: ts,
        boundaryGap: true,
        axisLine: { lineStyle: { color: theme==='dark'? '#ddd':'#666' } },
        axisLabel: { show: i===grid.length-1 }
      }));
      const yAxes = grid.map((g)=>({
        type: 'value', scale: true,
        axisLine: { lineStyle: { color: theme==='dark'? '#ddd':'#666' } },
        splitLine: { lineStyle: { color: theme==='dark'? '#333':'#eee' } }
      }));
      const series = [];
      // Candles
      series.push({
        type:'candlestick', name:'K', xAxisIndex:0, yAxisIndex:0,
        data: ohlc
      });
      // Bands
      if(payload.chan && payload.chan.bands){
        const pl = payload.chan.bands.pivot_low || [];
        const ph = payload.chan.bands.pivot_high || [];
        series.push({type:'line', name:'pivot_low', xAxisIndex:0, yAxisIndex:0, data:pl, symbol:'none', lineStyle:{width:1,color:'#1f77b4'}});
        series.push({type:'line', name:'pivot_high', xAxisIndex:0, yAxisIndex:0, data:ph, symbol:'none', lineStyle:{width:1,color:'#1f77b4'}});
      }
      // SMA/EMA
      const palette=['#1f77b4','#9467bd','#2ca02c','#d62728','#ff7f0e']; let pi=0;
      if(payload.indicators && payload.indicators.sma){
        Object.entries(payload.indicators.sma).forEach(([k,arr])=>{
          series.push({type:'line', name:'SMA'+k, xAxisIndex:0, yAxisIndex:0, data:arr, symbol:'none', lineStyle:{width:1.2,color:palette[pi++%palette.length]}});
        });
      }
      if(payload.indicators && payload.indicators.ema){
        Object.entries(payload.indicators.ema).forEach(([k,arr])=>{
          series.push({type:'line', name:'EMA'+k, xAxisIndex:0, yAxisIndex:0, data:arr, symbol:'none', lineStyle:{width:1.2,type:'dashed',color:palette[pi++%palette.length]}});
        });
      }
      // Signals (scatter)
      if(payload.chan && payload.chan.signals){
        const buys = payload.chan.signals.filter(s=>s.signal==='buy');
        const sells = payload.chan.signals.filter(s=>s.signal==='sell');
        series.push({type:'scatter', name:'buy', xAxisIndex:0, yAxisIndex:0, data:buys.map(s=>[ts[s.index], s.price]), symbol:'triangle', symbolSize:8, itemStyle:{color:'#2ca02c'}});
        series.push({type:'scatter', name:'sell', xAxisIndex:0, yAxisIndex:0, data:sells.map(s=>[ts[s.index], s.price]), symbol:'triangle', symbolRotate:180, symbolSize:8, itemStyle:{color:'#d62728'}});
      }
      // Segments (line segments)
      if(payload.chan && payload.chan.segments){
        payload.chan.segments.forEach(seg=>{
          series.push({type:'line', name:'seg', xAxisIndex:0, yAxisIndex:0, data:[[ts[seg.start_idx], seg.start_price],[ts[seg.end_idx], seg.end_price]], symbol:'none', lineStyle:{width:2, color: seg.direction==='up'? '#ff7f0e':'#9467bd'}});
        });
      }
      // MACD panel
      if(payload.indicators && payload.indicators.macd){
        const m = payload.indicators.macd;
        series.push({type:'bar', name:'hist', xAxisIndex:1, yAxisIndex:1, data:m.hist, itemStyle:{color:(val)=> val.value>=0? '#26a69a':'#ef5350'}});
        series.push({type:'line', name:'macd', xAxisIndex:1, yAxisIndex:1, data:m.macd, symbol:'none', lineStyle:{width:1,color:'#42a5f5'}});
        series.push({type:'line', name:'signal', xAxisIndex:1, yAxisIndex:1, data:m.signal, symbol:'none', lineStyle:{width:1,color:'#ab47bc'}});
      }
      // RSI panel
      if(payload.indicators && payload.indicators.rsi){
        const idx = (payload.indicators && payload.indicators.macd)? 2 : 1;
        series.push({type:'line', name:'RSI', xAxisIndex:idx, yAxisIndex:idx, data:payload.indicators.rsi["14"]||payload.indicators.rsi.default, symbol:'none', lineStyle:{width:1,color:'#ffb300'}});
      }
      // ATR (overlay on price grid for simplicity)
      if(payload.indicators && payload.indicators.atr){
        series.push({type:'line', name:'ATR', xAxisIndex:0, yAxisIndex:0, data:payload.indicators.atr["14"]||payload.indicators.atr.default, symbol:'none', lineStyle:{width:1,color:'#8d6e63',opacity:0.6}});
      }
      const option = {
        animation: false,
        tooltip: { trigger: 'axis', axisPointer: { type: 'cross' } },
        axisPointer: { link: [{xAxisIndex: 'all'}] },
        grid: grid,
        xAxis: xAxes,
        yAxis: yAxes,
        series: series,
        dataZoom: [{type:'inside', xAxisIndex: xAxes.map((_,i)=>i)},{type:'slider', xAxisIndex: xAxes.map((_,i)=>i)}]
      };
      chart.setOption(option);
      window.onresize=()=>chart.resize();
    }
    document.getElementById('run').onclick=run;
    // init with query
    ;['input_path','exchange','symbol','timeframe','limit','since','sma','ema','theme','plot_limit'].forEach(k=>val(k));
    ;['macd','rsi','atr','bands','segments','signals'].forEach(k=>chk(k, k!=='rsi' && k!=='atr'));
    if(location.search) run();
    </script>
    """
    return _html_page("Trade Platform WebUI · ECharts", body)


class ThreadingHTTPServer(ThreadingMixIn,):
    daemon_threads = True


class WebHandler(BaseHTTPRequestHandler):
    server_version = "TradeWebUI/0.1"

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/":
            return self._send_bytes(index_page(), content_type="text/html; charset=utf-8")
        if parsed.path == "/chart":
            return self._send_bytes(chart_page(), content_type="text/html; charset=utf-8")
        if parsed.path.startswith("/static/"):
            return self._serve_static(parsed.path)
        return self._send_text("Not Found", HTTPStatus.NOT_FOUND)

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path == "/backtest":
            return self._handle_backtest()
        if parsed.path == "/chart":
            return self._handle_chart()
        return self._send_text("Not Found", HTTPStatus.NOT_FOUND)

    def _serve_static(self, path: str):
        # /static/{kind}/{file}
        parts = path.split("/")
        if len(parts) < 4:
            return self._send_text("Bad request", HTTPStatus.BAD_REQUEST)
        kind = parts[2]
        name = "/".join(parts[3:])
        safe_name = os.path.normpath(name)
        if safe_name.startswith(".."):
            return self._send_text("Forbidden", HTTPStatus.FORBIDDEN)
        base = {"plots": PLOTS_DIR, "trades": TRADES_DIR, "stats": STATS_DIR}.get(kind)
        if not base:
            return self._send_text("Not Found", HTTPStatus.NOT_FOUND)
        fpath = os.path.join(base, safe_name)
        if not os.path.exists(fpath):
            return self._send_text("Not Found", HTTPStatus.NOT_FOUND)
        try:
            with open(fpath, "rb") as f:
                data = f.read()
            ctype = "application/octet-stream"
            if fpath.endswith(".png"):
                ctype = "image/png"
            elif fpath.endswith(".json"):
                ctype = "application/json"
            elif fpath.endswith(".csv"):
                ctype = "text/csv"
            return self._send_bytes(data, content_type=ctype)
        except Exception as e:
            return self._send_text(f"Error: {e}", HTTPStatus.INTERNAL_SERVER_ERROR)

    def _handle_backtest(self):
        ctype = self.headers.get("Content-Type", "")
        # Parse form (supports multipart/form-data and application/x-www-form-urlencoded)
        # Support only application/x-www-form-urlencoded to keep deps minimal
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length).decode("utf-8")
        qs = parse_qs(raw)
        getv = lambda k, default=None: (qs.get(k, [default])[0])

        try:
            input_path = getv("input_path", "").strip() if getv else ""
            start = (getv("start") or "").strip() or None
            end = (getv("end") or "").strip() or None
            fee = float(getv("fee", 0.0005) or 0.0005)
            rsi_min = self._to_float(getv("rsi_min"))
            rsi_max = self._to_float(getv("rsi_max"))
            min_atr_pct = self._to_float(getv("min_atr_pct"))
            max_atr_pct = self._to_float(getv("max_atr_pct"))
            stop_pct = self._to_float(getv("stop_pct"))
            tp_pct = self._to_float(getv("tp_pct"))
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
            used_path = input_path if input_path else None
            if not used_path:
                return self._send_bytes(index_page("Please provide a CSV path accessible to the server."), content_type="text/html; charset=utf-8")

            cf = CandleFrame.read_csv(used_path)
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

            # Save outputs
            run_id = datetime.utcnow().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]
            trades_path = os.path.join(TRADES_DIR, f"{run_id}.csv")
            stats_path = os.path.join(STATS_DIR, f"{run_id}.json")
            plot_path = os.path.join(PLOTS_DIR, f"{run_id}.png")
            if not res.trades.empty:
                res.trades.to_csv(trades_path, index=False)
            with open(stats_path, "w", encoding="utf-8") as f:
                json.dump(res.stats, f, indent=2)

            # Prepare plotting inputs
            plot_df = df.copy()
            if limit and len(plot_df) > limit:
                plot_df = plot_df.tail(limit).reset_index(drop=True)

            # Build signals for plotting: use filtered or hide
            plot_signals = sigs if show_signals else None

            # Optional context overlays
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
            # Draw
            plotting.plot_kline(
                plot_df,
                out=plot_path,
                title=f"Backtest: {os.path.basename(used_path)}",
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

            # Build result HTML
            stats_rows = "\n".join(
                f"<tr><td>{html.escape(str(k))}</td><td>{html.escape(str(v))}</td></tr>" for k, v in res.stats.items()
            )
            trades_link = f"<a href=\"/static/trades/{html.escape(os.path.basename(trades_path))}\">Download trades CSV</a>" if not res.trades.empty else "<em>No trades</em>"
            body = f"""
            <div class="row">
              <div class="col">
                <div class="card">
                  <h3>Stats</h3>
                  <table>
                    <tbody>
                    {stats_rows}
                    </tbody>
                  </table>
                  <p>Stats JSON: <a href="/static/stats/{html.escape(os.path.basename(stats_path))}">{html.escape(os.path.basename(stats_path))}</a></p>
                  <p>{trades_link}</p>
                  <p><a href="/">⟵ Back</a></p>
                </div>
              </div>
              <div class="col">
                <div class="card">
                  <h3>Plot</h3>
                  <img src="/static/plots/{html.escape(os.path.basename(plot_path))}" alt="plot" />
                </div>
              </div>
            </div>
            """
            return self._send_bytes(_html_page("Backtest Result", body), content_type="text/html; charset=utf-8")

        except Exception as e:
            return self._send_bytes(index_page(f"Error: {e}"), content_type="text/html; charset=utf-8")

    def _handle_chart(self):
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length).decode("utf-8")
        qs = parse_qs(raw)
        getv = lambda k, default=None: (qs.get(k, [default])[0])
        try:
            input_path = (getv("input_path") or "").strip() or None
            exchange = (getv("exchange") or "").strip() or None
            symbol = (getv("symbol") or "").strip() or None
            timeframe = (getv("timeframe") or "").strip() or None
            limit = int(float(getv("limit", 400) or 400))
            since = (getv("since") or "").strip() or None
            theme = (getv("theme", "light") or "light").strip()
            plot_limit = int(float(getv("plot_limit", 300) or 300))
            show_macd = bool(getv("show_macd"))
            show_rsi = bool(getv("show_rsi"))
            show_atr = bool(getv("show_atr"))
            show_pivot_bands = bool(getv("show_pivot_bands", "on"))
            show_segments = bool(getv("show_segments", "on"))
            label_segments = bool(getv("label_segments"))
            show_signals = bool(getv("show_signals", "on"))
            sma_s = (getv("sma") or "").strip()
            ema_s = (getv("ema") or "").strip()
            sma_lengths = [int(x) for x in sma_s.split(',') if x.strip().isdigit()] if sma_s else []
            ema_lengths = [int(x) for x in ema_s.split(',') if x.strip().isdigit()] if ema_s else []
            div_min_price_ext_pct = self._to_float(getv("div_min_price_ext_pct")) or 0.0
            div_min_hist_delta = self._to_float(getv("div_min_hist_delta")) or 0.0
            div_require_hist_sign_consistency = bool(getv("div_require_hist_sign_consistency"))

            ensure_dirs()
            # Load data
            if input_path:
                cf = CandleFrame.read_csv(input_path)
                df = cf.df.copy().reset_index(drop=True)
            elif exchange and symbol and timeframe:
                ex = ExchangeClient(exchange)
                ex.load_markets()
                ms = None
                if since:
                    from datetime import datetime, timezone
                    dt = datetime.fromisoformat(since)
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=timezone.utc)
                    ms = int(dt.timestamp() * 1000)
                ohlcv = ex.fetch_ohlcv_all(symbol=symbol, timeframe=timeframe, since=ms, limit=limit, max_bars=limit)
                cf = CandleFrame.from_ohlcv(ohlcv)
                df = cf.df.copy().reset_index(drop=True)
            else:
                return self._send_bytes(chart_page("Provide CSV path or exchange+symbol+timeframe"), content_type="text/html; charset=utf-8")

            # Chan analysis
            out = chan.analyze(
                df,
                div_min_price_ext_pct=div_min_price_ext_pct,
                div_min_hist_delta=div_min_hist_delta,
                div_require_hist_sign_consistency=div_require_hist_sign_consistency,
            )
            bands = out.get("bands")
            segments_obj = out.get("segments") if show_segments else None
            signals = out.get("signals") if show_signals else None

            # Trim for plotting
            if plot_limit and len(df) > plot_limit:
                df = df.tail(plot_limit).reset_index(drop=True)
                if bands is not None and not bands.empty:
                    bands = bands.tail(plot_limit).reset_index(drop=True)
                if signals is not None and not signals.empty:
                    signals = signals[signals["index"] >= len(out["signals"]) - plot_limit].reset_index(drop=True)

            pivot_low = bands.get("pivot_low") if (show_pivot_bands and bands is not None and not bands.empty) else None
            pivot_high = bands.get("pivot_high") if (show_pivot_bands and bands is not None and not bands.empty) else None

            # Render plot
            run_id = datetime.utcnow().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]
            plot_path = os.path.join(PLOTS_DIR, f"chart-{run_id}.png")
            plotting.plot_kline(
                df,
                out=plot_path,
                title=f"Chart: {symbol or os.path.basename(input_path)} {timeframe or ''}",
                pivot_low=pivot_low,
                pivot_high=pivot_high,
                segments=segments_obj,
                signals=signals,
                theme=theme,
                show_macd=show_macd,
                show_rsi=show_rsi,
                show_atr=show_atr,
                label_segments=label_segments,
                sma_lengths=sma_lengths,
                ema_lengths=ema_lengths,
            )

            body = f"""
            <div class=card>
              <p><a href="/static/plots/{html.escape(os.path.basename(plot_path))}">Open image</a> · <a href="/chart">⟵ Back</a></p>
              <img src="/static/plots/{html.escape(os.path.basename(plot_path))}" />
            </div>
            """
            return self._send_bytes(_html_page("Chart Rendered", body), content_type="text/html; charset=utf-8")
        except Exception as e:
            return self._send_bytes(chart_page(f"Error: {e}"), content_type="text/html; charset=utf-8")

    # Helpers
    def _to_float(self, s: Optional[str]) -> Optional[float]:
        if s is None:
            return None
        t = str(s).strip()
        if t == "" or t.lower() == "none":
            return None
        try:
            return float(t)
        except Exception:
            return None

    def _send_text(self, text: str, code: int = 200):
        data = text.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_bytes(self, data: bytes, *, content_type: str = "application/octet-stream", code: int = 200):
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def main(argv: Optional[list[str]] = None):
    import argparse
    parser = argparse.ArgumentParser(description="Trade Platform WebUI (minimal, no external deps)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)
    ensure_dirs()
    from http.server import HTTPServer

    class _ThreadingServer(ThreadingMixIn, HTTPServer):
        daemon_threads = True

    srv = _ThreadingServer((args.host, args.port), WebHandler)
    print(f"WebUI running: http://{args.host}:{args.port}")
    try:
        srv.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        srv.server_close()


if __name__ == "__main__":
    main()
