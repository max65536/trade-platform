from __future__ import annotations

# HTML snippets used by pages (kept minimal to avoid inline Python logic)

BACKTEST_FORM = """
<div class="card">
  <form method="POST" action="/backtest" enctype="application/x-www-form-urlencoded">
    <div class="row">
      <div class="col">
        <h3>Data</h3>
        <label>CSV Path<br/><input type="text" name="input_path" placeholder="data/BTCUSDT-1h.csv" style="width:100%"/></label><br/>
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


CHART_FORM = """
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


ECHARTS_SINGLE = """
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
  const grid = [
    {top: 20, height: 360},
    ...(payload.indicators && payload.indicators.macd? [{top: 400, height: 140}] : []),
    ...(payload.indicators && payload.indicators.rsi? [{top: 550, height: 140}] : []),
  ];
  const xAxes = grid.map((g,i)=>({ gridIndex: i, type: 'category', data: ts, boundaryGap: true, axisLabel: { show: i===grid.length-1 } }));
  const yAxes = grid.map((g)=>({ type: 'value', scale: true }));
  const series = [];
  series.push({ type:'candlestick', name:'K', xAxisIndex:0, yAxisIndex:0, data: ohlc });
  if(payload.chan && payload.chan.bands){
    const pl = payload.chan.bands.pivot_low || [];
    const ph = payload.chan.bands.pivot_high || [];
    series.push({type:'line', name:'pivot_low', xAxisIndex:0, yAxisIndex:0, data:pl, symbol:'none', lineStyle:{width:1,color:'#1f77b4'}});
    series.push({type:'line', name:'pivot_high', xAxisIndex:0, yAxisIndex:0, data:ph, symbol:'none', lineStyle:{width:1,color:'#1f77b4'}});
  }
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
  if(payload.chan && payload.chan.signals){
    const buys = payload.chan.signals.filter(s=>s.signal==='buy');
    const sells = payload.chan.signals.filter(s=>s.signal==='sell');
    series.push({type:'scatter', name:'buy', xAxisIndex:0, yAxisIndex:0, data:buys.map(s=>[ts[s.index], s.price]), symbol:'triangle', symbolSize:8, itemStyle:{color:'#2ca02c'}});
    series.push({type:'scatter', name:'sell', xAxisIndex:0, yAxisIndex:0, data:sells.map(s=>[ts[s.index], s.price]), symbol:'triangle', symbolRotate:180, symbolSize:8, itemStyle:{color:'#d62728'}});
  }
  if(payload.chan && payload.chan.segments){
    payload.chan.segments.forEach(seg=>{
      series.push({type:'line', name:'seg', xAxisIndex:0, yAxisIndex:0, data:[[ts[seg.start_idx], seg.start_price],[ts[seg.end_idx], seg.end_price]], symbol:'none', lineStyle:{width:2, color: seg.direction==='up'? '#ff7f0e':'#9467bd'}});
    });
  }
  if(payload.indicators && payload.indicators.macd){
    const m = payload.indicators.macd;
    series.push({type:'bar', name:'hist', xAxisIndex:1, yAxisIndex:1, data:m.hist, itemStyle:{color:(val)=> val.value>=0? '#26a69a':'#ef5350'}});
    series.push({type:'line', name:'macd', xAxisIndex:1, yAxisIndex:1, data:m.macd, symbol:'none', lineStyle:{width:1,color:'#42a5f5'}});
    series.push({type:'line', name:'signal', xAxisIndex:1, yAxisIndex:1, data:m.signal, symbol:'none', lineStyle:{width:1,color:'#ab47bc'}});
  }
  if(payload.indicators && payload.indicators.rsi){
    const idx = (payload.indicators && payload.indicators.macd)? 2 : 1;
    series.push({type:'line', name:'RSI', xAxisIndex:idx, yAxisIndex:idx, data:payload.indicators.rsi['14']||payload.indicators.rsi.default, symbol:'none', lineStyle:{width:1,color:'#ffb300'}});
  }
  if(payload.indicators && payload.indicators.atr){
    series.push({type:'line', name:'ATR', xAxisIndex:0, yAxisIndex:0, data:payload.indicators.atr['14']||payload.indicators.atr.default, symbol:'none', lineStyle:{width:1,color:'#8d6e63',opacity:0.6}});
  }
  const option = { animation:false, tooltip:{ trigger:'axis', axisPointer:{ type:'cross' } }, axisPointer:{ link: [{xAxisIndex: 'all'}] }, grid: grid, xAxis: xAxes, yAxis: yAxes, series: series, dataZoom: [{type:'inside', xAxisIndex: xAxes.map((_,i)=>i)},{type:'slider', xAxisIndex: xAxes.map((_,i)=>i)}] };
  chart.setOption(option);
  window.onresize=()=>chart.resize();
}
document.getElementById('run').onclick=run;
;['input_path','exchange','symbol','timeframe','limit','since','sma','ema','theme','plot_limit'].forEach(k=>val(k));
;['macd','rsi','atr','bands','segments','signals'].forEach(k=>chk(k, k!=='rsi' && k!=='atr'));
if(location.search) run();
</script>
"""


ECHARTS_MTF = """
<div class=card>
  <div style="margin-bottom:8px;opacity:0.8">Multi-timeframe (LTF + HTF alignment)</div>
  <form id="form2" onsubmit="ev=>ev.preventDefault()">
    <div class=row>
      <div class=col>
        <h3>LTF Source</h3>
        <label>CSV <input id="lower_input" style="width:100%" placeholder="data/BTCUSDT-4h.csv"/></label><br/>
        <div style="opacity:0.8; font-size:12px">or Exchange</div>
        <label>Exchange <input id="exchange" placeholder="binance"/></label>
        <label>Symbol <input id="symbol" placeholder="BTC/USDT"/></label>
        <label>LTF <input id="lower_tf" placeholder="4h"/></label>
        <label>Limit <input id="limit" type=number value=600/></label>
        <label>Since <input id="since" placeholder="2023-01-01"/></label>
      </div>
      <div class=col>
        <h3>HTF Source</h3>
        <label>CSV <input id="higher_input" style="width:100%" placeholder="data/BTCUSDT-1d.csv"/></label><br/>
        <div style="opacity:0.8; font-size:12px">or use Exchange + HTF</div>
        <label>HTF <input id="higher_tf" placeholder="1d"/></label>
      </div>
      <div class=col>
        <h3>MTF Filters</h3>
        <label><input id="require_htf_breakout" type=checkbox checked/> Require HTF breakout</label><br/>
        <label>Min HTF run <input id="min_htf_run" type=number value=3/></label><br/>
        <label><input id="show_base" type=checkbox checked/> Show base signals</label><br/>
        <label><input id="show_mtf" type=checkbox checked/> Show MTF-filtered signals</label>
      </div>
      <div class=col>
        <h3>Indicators & View</h3>
        <label>SMA <input id="sma" placeholder="20,50"/></label>
        <label>EMA <input id="ema" placeholder="50,100"/></label><br/>
        <label><input id="macd" type=checkbox checked/> MACD</label>
        <label><input id="rsi" type=checkbox/> RSI</label>
        <label><input id="atr" type=checkbox/> ATR</label><br/>
        <label>Theme
          <select id="theme">
            <option value="light" selected>light</option>
            <option value="dark">dark</option>
            <option value="minimal">minimal</option>
          </select>
        </label>
        <label>Plot limit <input id="plot_limit" type=number value=500/></label>
        <button id="run2" class="btn" type=button>Render</button>
      </div>
    </div>
  </form>
</div>
<div id="chart2" style="width:100%;height:760px"></div>
<script src="https://cdn.jsdelivr.net/npm/echarts@5/dist/echarts.min.js"></script>
<script>
const qs = new URLSearchParams(location.search);
function val2(id, def=''){ const v = document.getElementById(id).value||qs.get(id)||def; document.getElementById(id).value=v; return v; }
function chk2(id, def=false){ const el=document.getElementById(id); const v = qs.get(id); const b = v? (v==='1'||v==='true') : def; el.checked=b; return b; }
async function run(){
  const params = new URLSearchParams();
  ['lower_input','higher_input','exchange','symbol','lower_tf','higher_tf','limit','since','sma','ema','theme','plot_limit','min_htf_run'].forEach(k=>{ const v=document.getElementById(k).value; if(v) params.set(k,v); });
  ['require_htf_breakout','macd','rsi','atr','show_base','show_mtf'].forEach(k=>{ params.set(k, document.getElementById(k).checked?'1':'0'); });
  history.replaceState(null,'','?'+params.toString());
  const res = await fetch('/api/mtf?'+params.toString());
  if(!res.ok){ alert('Request failed'); return; }
  const p = await res.json();
  renderChart(p, (document.getElementById('theme').value||'light'));
}
function renderChart(p, theme){
  const root = document.getElementById('chart2');
  const chart = echarts.init(root, null, {renderer:'canvas'});
  const ts = p.data.map(r=>r[0]);
  const ohlc = p.data.map(r=>[r[1],r[2],r[3],r[4]]);
  const grid = [ {top:20, height:380}, ...(p.indicators && p.indicators.macd? [{top: 420, height: 140}] : []), ...(p.indicators && p.indicators.rsi? [{top: 570, height: 140}] : []) ];
  const xAxes = grid.map((g,i)=>({ gridIndex:i, type:'category', data:ts, boundaryGap:true, axisLabel:{show:i===grid.length-1} }));
  const yAxes = grid.map((g)=>({ type:'value', scale:true }));
  const series = [];
  series.push({type:'candlestick', name:'K', xAxisIndex:0, yAxisIndex:0, data:ohlc});
  if(p.htf && p.htf.bands){
    const pl = p.htf.bands.pivot_low || [];
    const ph = p.htf.bands.pivot_high || [];
    series.push({type:'line', name:'HTF pivot_low', xAxisIndex:0, yAxisIndex:0, data:pl, symbol:'none', lineStyle:{width:1,color:'#1f77b4'}});
    series.push({type:'line', name:'HTF pivot_high', xAxisIndex:0, yAxisIndex:0, data:ph, symbol:'none', lineStyle:{width:1,color:'#1f77b4'}});
  }
  const palette=['#1f77b4','#9467bd','#2ca02c','#d62728','#ff7f0e']; let pi=0;
  if(p.indicators && p.indicators.sma){
    Object.entries(p.indicators.sma).forEach(([k,arr])=>{ series.push({type:'line', name:'SMA'+k, xAxisIndex:0, yAxisIndex:0, data:arr, symbol:'none', lineStyle:{width:1.2,color:palette[pi++%palette.length]}}); });
  }
  if(p.indicators && p.indicators.ema){
    Object.entries(p.indicators.ema).forEach(([k,arr])=>{ series.push({type:'line', name:'EMA'+k, xAxisIndex:0, yAxisIndex:0, data:arr, symbol:'none', lineStyle:{width:1.2,type:'dashed',color:palette[pi++%palette.length]}}); });
  }
  if(p.signals_base && (new URLSearchParams(location.search)).get('show_base')!=='0'){
    const buys = p.signals_base.filter(s=>s.signal==='buy');
    const sells = p.signals_base.filter(s=>s.signal==='sell');
    series.push({type:'scatter', name:'base buy', xAxisIndex:0, yAxisIndex:0, data:buys.map(s=>[ts[s.index], s.price]), symbol:'triangle', symbolSize:7, itemStyle:{color:'#66bb6a',opacity:0.5}});
    series.push({type:'scatter', name:'base sell', xAxisIndex:0, yAxisIndex:0, data:sells.map(s=>[ts[s.index], s.price]), symbol:'triangle', symbolRotate:180, symbolSize:7, itemStyle:{color:'#ef5350',opacity:0.5}});
  }
  if(p.signals_mtf && (new URLSearchParams(location.search)).get('show_mtf')!=='0'){
    const buys = p.signals_mtf.filter(s=>s.signal==='buy');
    const sells = p.signals_mtf.filter(s=>s.signal==='sell');
    series.push({type:'scatter', name:'mtf buy', xAxisIndex:0, yAxisIndex:0, data:buys.map(s=>[ts[s.index], s.price]), symbol:'triangle', symbolSize:9, itemStyle:{color:'#2ca02c'}});
    series.push({type:'scatter', name:'mtf sell', xAxisIndex:0, yAxisIndex:0, data:sells.map(s=>[ts[s.index], s.price]), symbol:'triangle', symbolRotate:180, symbolSize:9, itemStyle:{color:'#d62728'}});
  }
  if(p.indicators && p.indicators.macd){
    const m = p.indicators.macd;
    series.push({type:'bar', name:'hist', xAxisIndex:1, yAxisIndex:1, data:m.hist, itemStyle:{color:(val)=> val.value>=0? '#26a69a':'#ef5350'}});
    series.push({type:'line', name:'macd', xAxisIndex:1, yAxisIndex:1, data:m.macd, symbol:'none', lineStyle:{width:1,color:'#42a5f5'}});
    series.push({type:'line', name:'signal', xAxisIndex:1, yAxisIndex:1, data:m.signal, symbol:'none', lineStyle:{width:1,color:'#ab47bc'}});
  }
  if(p.indicators && p.indicators.rsi){
    const idx = (p.indicators && p.indicators.macd)? 2 : 1;
    series.push({type:'line', name:'RSI', xAxisIndex:idx, yAxisIndex:idx, data:p.indicators.rsi['14']||p.indicators.rsi.default, symbol:'none', lineStyle:{width:1,color:'#ffb300'}});
  }
  if(p.indicators && p.indicators.atr){
    series.push({type:'line', name:'ATR', xAxisIndex:0, yAxisIndex:0, data:p.indicators.atr['14']||p.indicators.atr.default, symbol:'none', lineStyle:{width:1,color:'#8d6e63',opacity:0.6}});
  }
  const option = { animation:false, tooltip:{trigger:'axis', axisPointer:{type:'cross'}}, axisPointer:{link:[{xAxisIndex:'all'}]}, grid, xAxis:xAxes, yAxis:yAxes, series, dataZoom:[{type:'inside', xAxisIndex:xAxes.map((_,i)=>i)}, {type:'slider', xAxisIndex:xAxes.map((_,i)=>i)}] };
  chart.setOption(option);
  window.onresize=()=>chart.resize();
}
document.getElementById('run2').onclick=run;
;['lower_input','higher_input','exchange','symbol','lower_tf','higher_tf','limit','since','sma','ema','theme','plot_limit','min_htf_run'].forEach(k=>val2(k));
;['require_htf_breakout','macd','rsi','atr','show_base','show_mtf'].forEach(k=>chk2(k, k!=='rsi' && k!=='atr'));
if(location.search) run();
</script>
"""

