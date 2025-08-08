Trade Platform (ccxt + TA + Chan)

一个基于 ccxt 的最小可用交易/研究骨架：
- 统一拉取并存储K线数据（CSV）
- 常见技术指标（SMA/EMA/RSI/ATR/MACD）
- 缠论分析（完整）：包含关系→分型→笔→线段→中枢（Pivot Zone）
- 信号：线段拐点 + 枢纽突破/回测/背驰（买卖1/2/3，含 kind 标注）
- 多周期合成（MTF）：将高周期线段/中枢对齐到低周期，按方向与突破过滤信号
- 回测器与可视化（matplotlib）
- 命令行工具

快速开始（PDM）
- Python 3.10+
- 安装依赖：`pdm install`（如未安装 PDM，可用 `pipx install pdm`）
- 运行 CLI：`pdm run trade-cli --help`

命令行用法
- 拉取K线：
  - `pdm run trade-cli fetch --exchange binance --symbol BTC/USDT --timeframe 1h --limit 2000 --output data/BTCUSDT-1h.csv`

- 分析（含中枢与信号合并）：
  - `pdm run trade-cli analyze --input data/BTCUSDT-1h.csv --out data/BTCUSDT-1h-analyzed.csv`

- 回测：
  - `pdm run trade-cli backtest --input data/BTCUSDT-1h.csv --start 2023-01-01 --end 2024-01-01`

- 多周期合成（示例：4h + 1d）：
  - `pdm run trade-cli mtf --lower-input data/BTCUSDT-4h.csv --higher-input data/BTCUSDT-1d.csv --out data/BTCUSDT-4h-mtf.csv --require-htf-breakout --min-htf-run 3 --run-backtest`

- 可视化（主题/标注可选）：
  - `pdm run trade-cli plot --input data/BTCUSDT-4h-mtf.csv --use-mtf-bands --use-mtf-signals --theme dark --label-segments --save out/4h-mtf.png`
  - 说明：若 `signals` 含 `kind` 列，会按类别自动渲染：
    - `buy1/sell1` 实心箭头；`buy2/sell2` 实心箭头（带带宽色描边）；`buy3/sell3` 空心箭头（彩色描边）；`turn` 为 “x” 标记；并在箭头附近标注 1/2/3。

- 批量拉取：
  - `pdm run trade-cli batch --exchange binance --symbols BTC/USDT ETH/USDT --timeframes 4h 1d --output-dir data/spot --name-template {symbol_noslash}-{timeframe}.csv --max-bars 5000`

网络代理（Proxy）
- 说明：`trade_platform/exchanges.py` 已支持代理，优先级为 显式 `proxies` 参数（内部用） > `TRADE_*` 环境变量 > 系统级 `HTTP(S)_PROXY` 环境变量（requests 默认）。
- 推荐：使用 `TRADE_HTTP_PROXY` / `TRADE_HTTPS_PROXY` / `TRADE_NO_PROXY` 环境变量，或直接使用系统 `HTTP_PROXY` / `HTTPS_PROXY` / `NO_PROXY`。
- Linux/macOS 示例：
  - 一次性运行：
    - `TRADE_HTTP_PROXY=http://127.0.0.1:7890 TRADE_HTTPS_PROXY=http://127.0.0.1:7890 pdm run trade-cli fetch --exchange binance --symbol BTC/USDT --timeframe 1h --output data/BTCUSDT-1h.csv`
  - 临时导出：
    - `export TRADE_HTTP_PROXY=http://127.0.0.1:7890`
    - `export TRADE_HTTPS_PROXY=http://127.0.0.1:7890`
    - 可选跳过域：`export TRADE_NO_PROXY=localhost,127.0.0.1`
- Windows PowerShell 示例：
  - `$env:TRADE_HTTP_PROXY = 'http://127.0.0.1:7890'`
  - `$env:TRADE_HTTPS_PROXY = 'http://127.0.0.1:7890'`
  - 可选：`$env:TRADE_NO_PROXY = 'localhost,127.0.0.1'`
- 说明：如未设置 `TRADE_*`，requests 会自动读取系统级 `HTTP_PROXY` / `HTTPS_PROXY` / `NO_PROXY`。

测试
- 安装开发依赖：`pdm install -G dev`
- 运行：`pdm run pytest -q`

策略过滤与风控（示例）
- 单周期动量：
  - `pdm run trade-cli backtest --input data/BTCUSDT-1h.csv --rsi-min 55 --min-atr-pct 0.004 --stop-pct 0.02 --tp-pct 0.04`
- 多周期动量（4h + 1d）：
  - `pdm run trade-cli mtf --lower-input data/BTCUSDT-4h.csv --higher-input data/BTCUSDT-1d.csv --require-htf-breakout --min-htf-run 3 --rsi-min 55 --min-atr-pct 0.004 --stop-pct 0.02 --tp-pct 0.04 --run-backtest`
- 单周期均值回归：
  - `pdm run trade-cli backtest --input data/BTCUSDT-1h.csv --rsi-max 45 --max-atr-pct 0.02 --stop-pct 0.02 --tp-pct 0.03`

快捷脚本（PDM）
- `pdm run bt_momentum`：1h 动量回测
- `pdm run bt_meanrev`：1h 均值回归回测
- `pdm run mtf_momentum_4h_1d`：4h+1d 动量（方向+中枢突破）
- `pdm run mtf_meanrev_4h_1d`：4h+1d 均值回归（方向一致）
- `pdm run pipeline -- --exchange binance --symbols BTC/USDT --lower-tf 4h --higher-tf 1d --out-dir runs --require-htf-breakout --min-htf-run 3 --rsi-min 55 --min-atr-pct 0.004 --stop-pct 0.02 --tp-pct 0.04`

目录结构
- trade_platform/
  - exchanges.py       ccxt封装
  - dataio.py          CSV读取/写入
  - indicators.py      技术指标
 - chan.py            缠论核心（分型/笔/线段/中枢/信号）
  - multiframe.py      多周期合成与信号过滤
  - backtest.py        回测引擎
 - plotting.py        可视化
  - cli.py             命令行入口
 - pyproject.toml     PDM 项目配置（依赖/脚本）

说明
- 缠论分析默认采用工程化完整版本（包含关系合并、严格分笔、三笔成段、中枢与三类买卖点），同时保留简化实现用于兼容。
- 多周期通过 merge_asof 对齐高周期上下文（方向/中枢带）到低周期，支持方向一致与高周期突破与最小延续长度过滤。
- 回测为事件驱动、下根K线开盘成交的近似执行模型，用于快速验证策略逻辑。

高级：缠论完整模式
- 位置：`trade_platform/chan.py`
- 能力：
  - 合并K线（包含关系）
  - 分型（基于合并K线）、严格交替成笔
  - 三笔成段、笔价区间交叠构建中枢
  - 信号扩展：线段拐点、枢纽突破（一买/一卖）、回测（二买/二卖）、背驰（三买/三卖，基于 MACD 柱能量）
- 用法（Python）：
  - `from trade_platform import chan`
  - `out = chan.analyze(df)`（默认完整模式）
  - 返回结构：`fractals/pens/segments/pivots/signals/bands`；`signals` 含三类买卖点：`buy1/sell1`（突破）、`buy2/sell2`（回测）、`buy3/sell3`（背驰），其中信号的交易方向仍为 `signal=buy/sell`，类别标在 `kind` 列。
