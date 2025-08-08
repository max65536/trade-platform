Trade Platform (ccxt + TA + Chan)

一个基于 ccxt 的最小可用交易/研究骨架：
- 统一拉取并存储K线数据（CSV）
- 常见技术指标（SMA/EMA/RSI/ATR/MACD）
- 缠论分析（简化）：分型→笔→线段→中枢（Pivot Zone）
- 信号：线段拐点 + 中枢突破（支持合并与去重）
- 多周期合成（MTF）：将高周期线段/中枢对齐到低周期，按方向与突破过滤信号
- 回测器与可视化（matplotlib）
- 命令行工具

快速开始
- Python 3.10+
- pip install -r requirements.txt

命令行用法
- 拉取K线：
  - `python -m trade_platform.cli fetch --exchange binance --symbol BTC/USDT --timeframe 1h --limit 2000 --output data/BTCUSDT-1h.csv`

- 分析（含中枢与信号合并）：
  - `python -m trade_platform.cli analyze --input data/BTCUSDT-1h.csv --out data/BTCUSDT-1h-analyzed.csv`

- 回测：
  - `python -m trade_platform.cli backtest --input data/BTCUSDT-1h.csv --start 2023-01-01 --end 2024-01-01`

- 多周期合成（示例：4h + 1d）：
  - `python -m trade_platform.cli mtf --lower-input data/BTCUSDT-4h.csv --higher-input data/BTCUSDT-1d.csv --out data/BTCUSDT-4h-mtf.csv --require-htf-breakout --min-htf-run 3 --run-backtest`

- 可视化（主题/标注可选）：
  - `python -m trade_platform.cli plot --input data/BTCUSDT-4h-mtf.csv --use-mtf-bands --use-mtf-signals --theme dark --label-segments --save out/4h-mtf.png`

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

说明
- 缠论实现为工程化简化版本：分型/笔/线段/中枢（以笔价格区间交叠近似）与突破信号，便于在实盘/回测中稳定使用。
- 多周期通过 merge_asof 对齐高周期上下文（方向/中枢带）到低周期，支持方向一致与高周期突破与最小延续长度过滤。
- 回测为事件驱动、下根K线开盘成交的近似执行模型，用于快速验证策略逻辑。
